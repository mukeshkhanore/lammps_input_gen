#!/usr/bin/env python3
"""
Shell Model Processing for LAMMPS Structure and Setup File Generation from .pickle and GS.gulp file.
Author: Mukesh Khanore
Date: 12-Feb-2026
Version: 4.0 - Enhanced with robust error handling, logging, and validation
LAMMPS MD logic credit:  Mónica Elisabet Graf and Mauro António Pereira Gonçalves
"""
import sys
import os
import logging
import numpy as np
import copy
import pickle
from typing import Dict, List, Tuple, Any, Optional
import subprocess
from mendeleev import element
from dataclasses import dataclass, field
import pm__cell as pmc
import pm__chemical_order as pmco

# ============================================================================
# CONSTANTS
# ============================================================================
CORE_MASS_RATIO = 0.98
SHELL_MASS_RATIO = 0.02
DEFAULT_RMAX = 10.0
NEIGHBOR_DISTANCE = 1.0
PRECISION_DECIMALS = 4
DEFAULT_SUPERCELL_DIMS = [8, 8, 8]
DEFAULT_TEMPERATURE = 10.0
DEFAULT_T_STAT = 0.1
DEFAULT_P_STAT = 2.0
DEFAULT_SYMMETRY = "file"
DEFAULT_MODEL_FILE = "./potential.pickle"
DEFAULT_OUTPUT_FILE = "structure"

# LAMMPS simulation parameters
EQUILIBRATION_TEMP_STEPS = 20000
EQUILIBRATION_FINAL_STEPS = 30000
PRODUCTION_STEPS = 50000
TIMESTEP = 0.0002
THERMO_FREQ = 500
DUMP_FREQ = 500
EWALD_ACCURACY = 1.0e-6

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
def setup_logging(log_level: str = "INFO", log_file: str = "lammps_processing.log") -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler
    try:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    except IOError as e:
        print(f"Warning: Could not create log file {log_file}: {e}")
    
    logger.addHandler(console_handler)
    return logger

# Initialize logger
logger = setup_logging()

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================
class LAMMPSProcessingError(Exception):
    """Base exception for LAMMPS processing errors."""
    pass

class ModelLoadError(LAMMPSProcessingError):
    """Exception raised when model loading fails."""
    pass

class ValidationError(LAMMPSProcessingError):
    """Exception raised when validation fails."""
    pass

class ConfigurationError(LAMMPSProcessingError):
    """Exception raised when configuration is invalid."""
    pass

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================
@dataclass
class Config:
    """Configuration parameters for LAMMPS processing."""
    model_file: str = DEFAULT_MODEL_FILE
    supercell_dims: List[int] = field(default_factory=lambda: DEFAULT_SUPERCELL_DIMS.copy())
    symmetry: str = DEFAULT_SYMMETRY
    output_filename: str = DEFAULT_OUTPUT_FILE
    t_array: List[float] = field(default_factory=lambda: [DEFAULT_TEMPERATURE])
    t_stat: float = DEFAULT_T_STAT
    p_stat: float = DEFAULT_P_STAT
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ConfigurationError: If any parameter is invalid
        """
        # Validate model file
        if not self.model_file:
            raise ConfigurationError("Model file path cannot be empty")
        
        # Validate supercell dimensions
        if len(self.supercell_dims) != 3:
            raise ConfigurationError(
                f"Supercell dimensions must have exactly 3 values, got {len(self.supercell_dims)}"
            )
        
        if any(dim <= 0 for dim in self.supercell_dims):
            raise ConfigurationError(
                f"Supercell dimensions must be positive, got {self.supercell_dims}"
            )
        
        # Validate symmetry
        valid_symmetries = ["cubic", "random", "file"]
        if self.symmetry not in valid_symmetries:
            raise ConfigurationError(
                f"Symmetry must be one of {valid_symmetries}, got '{self.symmetry}'"
            )
        
        # Validate temperatures
        if not self.t_array or len(self.t_array) == 0:
            raise ConfigurationError("Temperature array cannot be empty")
        
        if any(t < 0 for t in self.t_array):
            raise ConfigurationError(
                f"Temperatures must be non-negative, got {self.t_array}"
            )
        
        # Validate thermostat damping
        if self.t_stat <= 0:
            raise ConfigurationError(
                f"Thermostat damping time must be positive, got {self.t_stat}"
            )
        
        # Validate barostat damping
        if self.p_stat <= 0:
            raise ConfigurationError(
                f"Barostat damping time must be positive, got {self.p_stat}"
            )
        
        logger.info("Configuration validation passed")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_mixing_type(mixtype: str, i: int, j: int, k: int) -> int:
    """
    Determine mixing type based on coordinates.
    
    Args:
        mixtype: Type of mixing ("homog", "G", or "14")
        i, j, k: Integer coordinates
        
    Returns:
        int: 0 for species 1, 1 for species 2
        
    Raises:
        ValidationError: If mixing type is not supported
    """
    if mixtype == "homog":
        return 0
    elif mixtype == "G":
        return 0 if pow(-1, i + j + k) == -1 else 1
    elif mixtype == "14":
        return 0 if (i % 2 == 0 and j % 2 == 0 and k % 2 == 0) else 1
    else:
        raise ValidationError(f"Ordering '{mixtype}' not defined. Valid options: homog, G, 14")

# ============================================================================
# SHELL MODEL DATA EXTRACTION
# ============================================================================
def extract_shell_model_data(model: Any) -> Tuple[Dict, Dict, List]:
    """
    Extract shell model data from the model.
    
    Args:
        model: The model object containing charges information
        
    Returns:
        Tuple[Dict, Dict, List]: Shell models data, springs, and potentials
        
    Raises:
        ValidationError: If model data is incomplete or invalid
    """
    try:
        shell_models_springs = {}
        shell_models_potentials = []
        shell_models_data = {'model': {}}
        
        if not hasattr(model, 'charges') or not model.charges:
            raise ValidationError("Model has no charge information")
        
        # First create structure for all species
        for charge in model.charges:
            species = charge.species
            if species not in shell_models_data['model']:
                shell_models_data['model'][species] = {
                    'core': {"mass": None, "charge": None},
                    'shell': {"mass": None, "charge": None}
                }
        
        # Then populate with charge values
        for charge in model.charges:
            species = charge.species
            part = charge.part
            # Normalize 'shel' to 'shell'
            if part == 'shel':
                part = 'shell'
            
            if part not in ['core', 'shell']:
                logger.warning(f"Unknown part type '{part}' for species {species}, skipping")
                continue
            
            shell_models_data['model'][species][part]['charge'] = charge.charge
            shell_models_data['model'][species][part]['mass'] = 0.0
        
        # Process springs
        if hasattr(model, 'springs'):
            for spring in model.springs:
                species = spring.species
                if not hasattr(spring, 'k2') or not hasattr(spring, 'k4'):
                    logger.warning(f"Spring for species {species} missing k2 or k4 parameters")
                    continue
                    
                shell_models_springs[species] = {
                    'k2': spring.k2,
                    'k4': spring.k4
                }
        
        # Process potentials
        if hasattr(model, 'potentials'):
            for pot in model.potentials:
                # Normalize 'shel' to 'shell'
                part1 = 'shell' if pot.part1 == 'shel' else pot.part1
                part2 = 'shell' if pot.part2 == 'shel' else pot.part2
                
                potential_entry = {
                    'kind': pot.kind,
                    'sp1': pot.sp1,
                    'part1': part1,
                    'sp2': pot.sp2,
                    'part2': part2,
                    'params': pot.params,
                    'cutoffs': pot.cutoffs
                }
                shell_models_potentials.append(potential_entry)
        
        logger.info(f"Extracted shell model data for {len(shell_models_data['model'])} species")
        logger.debug(f"Springs: {len(shell_models_springs)}, Potentials: {len(shell_models_potentials)}")
        
        return shell_models_data, shell_models_springs, shell_models_potentials
        
    except AttributeError as e:
        raise ValidationError(f"Model structure is invalid: {e}")

# ============================================================================
# SPECIES ID MAPPING
# ============================================================================
def create_species_id_map(cell: Any, model: Any) -> Dict:
    """
    Create mapping between species+part and numeric IDs.
    
    Args:
        cell: Cell object with species information
        model: Model object with species information
        
    Returns:
        Dict: Mapping from (species, part) to numeric ID
        
    Raises:
        ValidationError: If species data is invalid
    """
    try:
        species_id_map = {}
        species_list = []
        
        if not hasattr(model, 'AB_specie'):
            raise ValidationError("Model missing AB_specie attribute")
        
        # Build species list
        if "A" in model.AB_specie:
            species_list.extend(model.AB_specie["A"])
        if "B" in model.AB_specie:
            species_list.extend(model.AB_specie["B"])
        
        # Add oxygen if not already present
        if "O" not in species_list:
            species_list.append("O")
        
        # Create ID mapping
        for i, species in enumerate(species_list):
            species_id_map[(species, 'core')] = (i + 1)
            species_id_map[(species, 'shell')] = (len(cell.species_name) + i + 1)
        
        logger.info(f"Created species ID mapping for {len(species_list)} species")
        logger.debug(f"Species ID map: {species_id_map}")
        
        return species_id_map
        
    except (AttributeError, KeyError) as e:
        raise ValidationError(f"Failed to create species ID map: {e}")

# ============================================================================
# CELL CREATION AND MANIPULATION
# ============================================================================
def create_mapped_cell(original_cell: Any, species_id_map: Dict) -> Any:
    """
    Create a new cell with IDs mapped according to species_id_map.
    
    Args:
        original_cell: Original cell object
        species_id_map: Mapping from (species, part) to numeric ID
        
    Returns:
        Any: New cell with mapped IDs
    """
    new_cell = pmc.Cell(convention='zerolist', prescribe_N=0)
    new_cell.lattice = copy.deepcopy(original_cell.lattice)
    
    for atom in original_cell.atom:
        cs = "core" if atom.coreshell == 'core' else "shell"
        
        if (atom.name, cs) not in species_id_map:
            logger.warning(f"Species ({atom.name}, {cs}) not in species_id_map, skipping")
            continue
        
        new_cell.appendAtom(
            name=species_id_map[(atom.name, cs)],
            position_frac=atom.position_frac,
            coreshell=atom.coreshell
        )
    
    logger.info(f"Created mapped cell with {new_cell.N} atoms")
    return new_cell

def initialize_shell_models_data(cell: Any) -> Dict:
    """
    Initialize new shell models data structure with default values.
    
    Args:
        cell: Cell object with species information
        
    Returns:
        Dict: Initialized shell models data
    """
    shell_models_data = {'model': {}}
    for species in cell.species_name:
        shell_models_data['model'][species] = {
            'core': {"mass": 0.0, "charge": None},
            'shell': {"mass": 0.0, "charge": None}
        }
    
    logger.debug(f"Initialized shell models data for {len(cell.species_name)} species")
    return shell_models_data

def map_charges_to_new_model(shell_models_data: Dict, species_id_map: Dict, 
                              shell_models_data_new: Dict) -> Dict:
    """
    Map charges from original model to new model.
    
    Args:
        shell_models_data: Original shell models data
        species_id_map: Mapping from (species, part) to numeric ID
        shell_models_data_new: New shell models data structure
        
    Returns:
        Dict: Updated shell models data with mapped charges
        
    Raises:
        ValidationError: If species data is missing or invalid
    """
    for species in shell_models_data['model']:
        try:
            species_mass = element(species).atomic_weight
            logger.debug(f"Processing species: {species} with atomic weight: {species_mass}")
        except Exception as e:
            raise ValidationError(f"Failed to get atomic weight for species '{species}': {e}")
        
        for id_key, mapped_id in species_id_map.items():
            if id_key[0] == species:
                part = id_key[1]
                
                if mapped_id not in shell_models_data_new['model']:
                    logger.warning(f"Mapped ID {mapped_id} not in new model, skipping")
                    continue
                
                # Map charge
                shell_models_data_new['model'][mapped_id][part]['charge'] = \
                    shell_models_data['model'][species][part]['charge']
                
                # Calculate mass based on part
                if part == 'core':
                    mass = np.round(species_mass * CORE_MASS_RATIO, PRECISION_DECIMALS)
                else:
                    mass = np.round(species_mass * SHELL_MASS_RATIO, PRECISION_DECIMALS)
                
                shell_models_data_new['model'][mapped_id][part]['mass'] = mass
    
    logger.info("Successfully mapped charges and masses to new model")
    return shell_models_data_new

def validate_shell_model_data(shell_models_data: Dict) -> Dict:
    """
    Validate shell model data and ensure consistency.
    
    Args:
        shell_models_data: Shell models data to validate
        
    Returns:
        Dict: Validated shell models data
    """
    for species_id in shell_models_data['model']:
        for part, attributes in shell_models_data['model'][species_id].items():
            # If any attribute is None, set all to None for consistency
            if any(value is None for value in attributes.values()):
                for attr_name in attributes:
                    attributes[attr_name] = None
                logger.debug(f"Species {species_id} {part} has incomplete data, set to None")
    
    return shell_models_data

def create_string_named_cell(numeric_cell: Any) -> Any:
    """
    Create a cell with string atom names instead of numeric IDs.
    
    Args:
        numeric_cell: Cell with numeric IDs
        
    Returns:
        Any: Cell with string atom names
    """
    string_cell = pmc.Cell(convention='zerolist', prescribe_N=0)
    string_cell.lattice = copy.deepcopy(numeric_cell.lattice)
    
    for atom in numeric_cell.atom:
        string_cell.appendAtom(
            name=str(atom.name),
            position_frac=atom.position_frac,
            coreshell=atom.coreshell
        )
    
    string_cell.shell_models = copy.deepcopy(numeric_cell.shell_models)
    logger.debug(f"Created string-named cell with {string_cell.N} atoms")
    
    return string_cell

# ============================================================================
# SUPERCELL CREATION
# ============================================================================
def create_supercell(model: Any, supercell_dims: List[int], symmetry: str = "cubic") -> Any:
    """
    Create a supercell with specified dimensions and symmetry.
    
    Args:
        model: Model containing chemical order information
        supercell_dims: Dimensions [Nx, Ny, Nz] of the supercell
        symmetry: Type of symmetry ("cubic", "random", or "file")
        
    Returns:
        Any: Cell object representing the supercell
        
    Raises:
        FileNotFoundError: If symmetry is "file" and GS.gulp is not found
        ValidationError: If model data is incomplete
    """
    Nx, Ny, Nz = supercell_dims
    
    if symmetry == "file":
        logger.info("Reading GS.gulp file from working directory")
        gulp_file = "GS.gulp"
        
        if not os.path.exists(gulp_file):
            raise FileNotFoundError(
                f"Required file '{gulp_file}' not found in the working directory. "
                f"Please ensure the file exists or use a different symmetry option."
            )
        
        try:
            cell = pmc.Cell(gulp_file)
            cell.is_shell_model = True
            cell = cell.replicate([Nx // 2, Ny // 2, Nz // 2])
            cell.pairCoresShells()
            logger.info(f"Successfully created supercell from {gulp_file}")
            return cell
            
        except Exception as e:
            raise LAMMPSProcessingError(f"Error processing GULP file '{gulp_file}': {e}")
    
    else:
        # Validate model has required chemical order information
        if not hasattr(model, 'chemical_order'):
            raise ValidationError("Model missing chemical_order attribute")
        
        if "A" not in model.chemical_order or "B" not in model.chemical_order:
            raise ValidationError("Model chemical_order must contain both 'A' and 'B' sites")
        
        try:
            chemical_order_A = pmco.ChemicalOrder(model.chemical_order["A"]["configuration"])
            chemical_order_B = pmco.ChemicalOrder(model.chemical_order["B"]["configuration"])
        except Exception as e:
            raise ValidationError(f"Failed to create chemical order: {e}")
        
        # Prepare structure
        cell = pmc.Cell(convention='zerolist', prescribe_N=0)
        cell.is_shell_model = True
        cell.simplePerovskite(ABO_names=["A", "B", "O"], dimensions=[Nx, Ny, Nz], coreshell=True)
        
        # Apply chemical order
        try:
            cell = chemical_order_A.prescribeChemicalOrderForBox(
                cell, position="A", replacements=model.AB_specie["A"], 
                dimensions=[Nx, Ny, Nz], coreshell=True
            )
            cell = chemical_order_B.prescribeChemicalOrderForBox(
                cell, position="B", replacements=model.AB_specie["B"], 
                dimensions=[Nx, Ny, Nz], coreshell=True
            )
            cell.pairCoresShells()
        except Exception as e:
            raise LAMMPSProcessingError(f"Failed to apply chemical order: {e}")
        
        # Apply perturbation for random symmetry
        perturbation = np.zeros((cell.N, 3))
        if symmetry == "random":
            logger.info("Applying random perturbations to atomic positions")
            for i in range(cell.N):
                perturbation[i] = np.random.uniform(-0.05, 0.05, 3)
        
        # Order species
        species_order = []
        for charge in model.charges:
            if charge.species not in species_order:
                species_order.append(charge.species)
        
        # Finalize cell
        cell.countPresentSpecies()
        cell.pairCoresShells()
        cell.displaceCartesian(perturbation)
        
        logger.info(f"Created {symmetry} supercell with dimensions {supercell_dims}")
        logger.info(f"Total atoms in supercell: {cell.N}")
        
        return cell

# ============================================================================
# SHELL MODEL PROCESSING
# ============================================================================
def process_shell_model(model: Any, cell: Any, output_filename: str = "structure") -> Tuple[Any, Any]:
    """
    Process shell model and save to LAMMPS structure.
    
    Args:
        model: Model containing charges information
        cell: Cell object to process
        output_filename: Base name for output files
        
    Returns:
        Tuple[Any, Any]: Tuple containing mapped_cell and string_cell
        
    Raises:
        LAMMPSProcessingError: If processing fails
    """
    try:
        # Extract shell model data
        shell_models_data, shell_models_springs, shell_models_potentials = extract_shell_model_data(model)
        
        # Create species ID mapping
        species_id_map = create_species_id_map(cell, model)
        
        # Create new cell with mapped IDs
        mapped_cell = create_mapped_cell(cell, species_id_map)
        
        # Initialize new shell models data
        shell_models_data_new = initialize_shell_models_data(mapped_cell)
        
        # Map charges to new model
        shell_models_data_new = map_charges_to_new_model(
            shell_models_data, species_id_map, shell_models_data_new
        )
        
        # Validate shell model data
        shell_models_data_new = validate_shell_model_data(shell_models_data_new)
        
        # Assign shell models to mapped cell
        mapped_cell.shell_models = copy.deepcopy(shell_models_data_new)
        
        # Create a cell with string atom names for LAMMPS
        string_cell = create_string_named_cell(mapped_cell)
        
        # Write the structure
        try:
            string_cell.writeToLAMMPSStructure(output_filename)
            logger.info(f"Successfully wrote LAMMPS structure to {output_filename}")
        except Exception as e:
            raise LAMMPSProcessingError(f"Error writing LAMMPS structure: {e}")
        
        return mapped_cell, string_cell
        
    except Exception as e:
        logger.error(f"Shell model processing failed: {e}")
        raise

# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model(file_path: str) -> Any:
    """
    Load model from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Any: Loaded model
        
    Raises:
        ModelLoadError: If loading fails
    """
    if not os.path.exists(file_path):
        raise ModelLoadError(f"Model file not found: {file_path}")
    
    if not os.path.isfile(file_path):
        raise ModelLoadError(f"Path is not a file: {file_path}")
    
    try:
        logger.info(f"Loading model from {file_path}")
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        
        # Basic validation
        if not hasattr(model, 'charges'):
            raise ModelLoadError("Loaded model is missing required 'charges' attribute")
        
        logger.info("Model loaded successfully")
        return model
        
    except pickle.UnpicklingError as e:
        raise ModelLoadError(f"Failed to unpickle model file: {e}")
    except Exception as e:
        raise ModelLoadError(f"Error loading model from {file_path}: {e}")

# ============================================================================
# LAMMPS INPUT GENERATION
# ============================================================================
def generate_lammps_header(model_name: str) -> str:
    """Generate LAMMPS input file header section."""
    return f"""# ---------- Initialize Simulation --------------------- 
# Model: {model_name}
clear\t\t\t\t\t\t\t\t#This command deletes all atoms, restores all settings to their default values, and frees all memory allocated by LAMMPS 
units metal \t\t\t\t\t\t\t#Specific set of units
dimension 3 \t\t\t\t\t\t\t#Set dimensionality
boundary p p p \t\t\t\t\t\t\t#Boundary type, p: periodic
atom_style full \t\t\t\t\t\t#full to allow cores and shells
atom_modify map array\t\t\t\t\t\t#Lookup table for atoms

# ---------- Create Atoms --------------------- 
#see: https://lammps.sandia.gov/doc/Howto_coreshell.html
#https://lammps.sandia.gov/doc/Howto_coreshell.html
#https://lammps.sandia.gov/doc/Howto_triclinic.html
atom_style full
fix csinfo all property/atom i_CSID
read_data rstrt.dat fix csinfo NULL CS-Info
"""

def generate_charge_settings(shell_models_data: Dict, species_id_map: Dict) -> str:
    """Generate charge settings for each atom type."""
    lines = []
    charge_values = {}
    
    for species, parts in shell_models_data['model'].items():
        for part, params in parts.items():
            key = (species, part)
            if key in species_id_map and 'charge' in params and params['charge'] is not None:
                type_id = species_id_map[key]
                charge_values[type_id] = params['charge']
    
    for type_id, charge in sorted(charge_values.items()):
        species_part = next((key for key, val in species_id_map.items() if val == type_id), None)
        if species_part:
            species, part = species_part
            comment = f"#{species} {part}"
        else:
            comment = ""
        lines.append(f"set type {type_id} charge {charge:.7f} {comment}")
    
    return "\n".join(lines)

def generate_group_definitions(species_id_map: Dict) -> str:
    """Generate group definitions for cores and shells."""
    core_types = [id_val for (species, part), id_val in species_id_map.items() if part == 'core']
    shell_types = [id_val for (species, part), id_val in species_id_map.items() if part == 'shell']
    species_types = list(species for (species, part), _ in species_id_map.items() if part == 'core')
    
    core_types_str = ' '.join(str(t) for t in sorted(core_types))
    shell_types_str = ' '.join(str(t) for t in sorted(shell_types))
    species_types_str = ' '.join(str(t) for t in species_types)
    
    return f"""
unfix csinfo

group cores type {core_types_str} \t#{species_types_str}
group shells type {shell_types_str} \t#{species_types_str}

neighbor {NEIGHBOR_DISTANCE} bin
neigh_modify\tonce no # as per Tonda suggestion
comm_modify vel yes #comment again after test (Pavel has it commented)
"""

def generate_potential_section(model_name: str, shell_models_potentials: List, 
                                species_id_map: Dict) -> str:
    """Generate potential definitions section."""
    rmax = DEFAULT_RMAX
    if shell_models_potentials and 'cutoffs' in shell_models_potentials[0]:
        _, potential_rmax = shell_models_potentials[0]['cutoffs']
        # Use default rmax for now (each potential has its own cutoffs)
    
    lines = [f"""
# ---------- Define Interatomic Potential {model_name} --------------------- 
kspace_style ewald/disp {EWALD_ACCURACY}\t\t\t\t\t#needed for calculation of long-range forces
pair_style   buck/coul/long/cs {rmax} {rmax}\t\t\t# two cutoffs: non-coulombic and coulombic
pair_coeff   * *    0.0000 1.000000   0.00000\t\t\t# default interaction is zero
"""]
    
    # Add Buckingham potentials
    for pot in shell_models_potentials:
        if pot['kind'] == 'buck':
            try:
                type1 = species_id_map[(pot['sp1'], pot['part1'])]
                type2 = species_id_map[(pot['sp2'], pot['part2'])]
                A, rho, C = pot['params']
                rmin, rmax = pot['cutoffs']
                lines.append(
                    f"pair_coeff {type1} {type2} {A} {rho} {C} "
                    f"\t #{pot['sp1']}-{pot['sp2']} {pot['part1']}-{pot['part2']}"
                )
            except (KeyError, ValueError) as e:
                logger.warning(f"Could not process potential {pot}: {e}")
    
    return "\n".join(lines)

def generate_bond_section(shell_models_springs: Dict, species_id_map: Dict) -> str:
    """Generate bond (spring) definitions for core-shell pairs."""
    lines = ["\n\nbond_style class2\n"]
    
    for species, spring_params in shell_models_springs.items():
        if (species, 'core') in species_id_map and (species, 'shell') in species_id_map:
            core_id = species_id_map[(species, 'core')]
            shell_id = species_id_map[(species, 'shell')]
            k2 = spring_params['k2']
            k4 = spring_params['k4']
            lines.append(
                f"bond_coeff {core_id}  0.0 {k2/2.0} 0.0 {k4/24.0} "
                f"\t #{species} core-shell converted k2/2! and k4/4!"
            )
    
    return "\n".join(lines)

def generate_simulation_settings(t_0: float, t_stat: float, p_stat: float) -> str:
    """Generate simulation settings and initial run."""
    return f"""
#neigh_modify delay 10 check yes \t# ???
neigh_modify page 100000 one 10000\t\t\t\t#max neighbors of one atom set to 10000

## ---------- Define Settings --------------------- all as per Tonda recommendation
compute eng all pe/atom                                         #pe/atom: potential energy for each atom
compute eatoms all reduce sum c_eng                             #sum up all energies

# ------------------------ Initiating the run  -------------------------------

reset_timestep 0

thermo {THERMO_FREQ}
thermo_style custom step temp pe ke etotal enthalpy evdwl ecoul epair ebond elong etail  cella cellb cellc vol press pxx pyy pzz pxy pxz pyz

#thermo_modify source yes
thermo_modify format 4 %20.15g
thermo_modify format 5 %20.15g
thermo_modify format 6 %20.15g
thermo_modify format 7 %20.15g
thermo_modify format 8 %20.15g
thermo_modify format 9 %20.15g
thermo_modify format 10 %20.15g
thermo_modify format 11 %20.15g
thermo_modify format 12 %20.8g
thermo_modify format 13 %20.8g


#compute CStemp all temp/cs cores shells \t\t# compute temperature on centers of gravity of core/shell pairs and call it CStemp

compute CSequ all temp/cs cores shells
#compute thermo_press_lmp all pressure thermo_temp \t# press for correct kinetic scalar


#thermo_modify temp CStemp press thermo_press_lmp \t# output temp and press calculated in the CStemp and thermo-press_lmp way (COM = centre-of-mass)
thermo_modify temp CSequ

timestep {TIMESTEP}

#generate velocities
velocity all create {t_0}K 34 dist gaussian mom yes rot no bias yes temp CSequ

# ----------------------- Equilibration {t_0}K  ---------------------------------------------

fix npt_equ all npt temp {t_0} {t_0} {t_stat} tri 0.0 0.0 {p_stat}
fix_modify npt_equ temp CSequ
# ----------------------- Production {t_0}K ---------------------------------------------

fix npt all npt temp {t_0} {t_0} {t_stat} tri 0.0 0.0 {p_stat}
fix_modify npt temp CSequ

dump myDump all custom {DUMP_FREQ} trajectory_"$t_0"k.atom id type q x y z #fx fy fz
dump_modify myDump format 4 %20.6g
dump_modify myDump format 5 %20.6g
dump_modify myDump format 6 %20.6g

run {PRODUCTION_STEPS}
unfix npt
undump myDump
"""

def generate_temperature_ramps(t_array: List[float], t_stat: float, p_stat: float) -> str:
    """Generate temperature ramp sections for multi-temperature runs."""
    if len(t_array) <= 1:
        return ""
    
    lines = []
    for i, t in enumerate(t_array[1:]):
        t_00 = t_array[i]  # Previous temperature
        t_1 = t  # Current temperature
        
        lines.append(f"""
# ----------------------- Equilibration {t_1}K  ---------------------------------------------

fix npt_equ all npt temp {t_00} {t_1} {t_stat} tri 0.0 0.0 {p_stat}
fix_modify npt_equ temp CSequ
run {EQUILIBRATION_TEMP_STEPS}
unfix npt_equ 

fix npt_equ all npt temp {t_1} {t_1} {t_stat} tri 0.0 0.0 {p_stat}
fix_modify npt_equ temp CSequ
run {EQUILIBRATION_FINAL_STEPS}
unfix npt_equ 

# ----------------------- Production {t_1}K ---------------------------------------------
#
fix npt all npt temp {t_1} {t_1} {t_stat} tri 0.0 0.0 {p_stat}
fix_modify npt temp CSequ

dump myDump all custom {DUMP_FREQ} trajectory_{int(t_1)}k.atom id type q x y z #fx fy fz
dump_modify myDump format 4 %20.6g
dump_modify myDump format 5 %20.6g
dump_modify myDump format 6 %20.6g

run {PRODUCTION_STEPS}
unfix npt
undump myDump
""")
    
    return "\n".join(lines)

def generate_lammps_input(shell_models_data: Dict, shell_models_springs: Dict, 
                           shell_models_potentials: List, species_id_map: Dict, 
                           model_name: str, t_array: List[float], t_stat: float, 
                           p_stat: float) -> str:
    """
    Generate complete LAMMPS input file content.
    
    Args:
        shell_models_data: Shell model data dictionary
        shell_models_springs: Spring constants for core-shell pairs
        shell_models_potentials: List of potential parameters
        species_id_map: Mapping of species to type IDs
        model_name: Name of the model
        t_array: Array of temperatures for simulation
        t_stat: Thermostat damping time
        p_stat: Barostat damping time
        
    Returns:
        str: Complete LAMMPS input file content
    """
    try:
        t_0 = t_array[0]
        
        # Build the input file in sections
        sections = [
            generate_lammps_header(model_name),
            "\n",
            generate_charge_settings(shell_models_data, species_id_map),
            generate_group_definitions(species_id_map),
            generate_potential_section(model_name, shell_models_potentials, species_id_map),
            generate_bond_section(shell_models_springs, species_id_map),
            generate_simulation_settings(t_0, t_stat, p_stat),
            generate_temperature_ramps(t_array, t_stat, p_stat)
        ]
        
        content = "".join(sections)
        logger.info("Generated LAMMPS input file content")
        return content
        
    except Exception as e:
        logger.error(f"Failed to generate LAMMPS input: {e}")
        raise LAMMPSProcessingError(f"LAMMPS input generation failed: {e}")

def save_lammps_input(content: str, filename: str = "lammps.in") -> None:
    """
    Save LAMMPS input content to file with error handling.
    
    Args:
        content: LAMMPS input file content
        filename: Output filename
        
    Raises:
        LAMMPSProcessingError: If file writing fails
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Successfully wrote LAMMPS input to {filename}")
    except IOError as e:
        raise LAMMPSProcessingError(f"Error writing to {filename}: {e}")

# ============================================================================
# USER CONFIGURATION
# ============================================================================
def get_user_config() -> Config:
    """
    Get configuration parameters from user input.
    
    Returns:
        Config: Configuration object with validated parameters
    """
    logger.info("=" * 60)
    logger.info("Shell Model Processing for LAMMPS - Configuration")
    logger.info("=" * 60)
    
    # Model file
    model_file = input(f"\nEnter path to model pickle file\n[default: {DEFAULT_MODEL_FILE}]: ").strip()
    if not model_file:
        model_file = DEFAULT_MODEL_FILE
    
    # Supercell dimensions
    while True:
        try:
            dims_input = input(f"\nEnter supercell dimensions (Nx Ny Nz) [default: {' '.join(map(str, DEFAULT_SUPERCELL_DIMS))}]: ").strip()
            if not dims_input:
                supercell_dims = DEFAULT_SUPERCELL_DIMS.copy()
            else:
                supercell_dims = [int(x) for x in dims_input.split()]
                if len(supercell_dims) != 3:
                    logger.error("Error: Please enter exactly 3 dimensions")
                    continue
                if any(dim <= 0 for dim in supercell_dims):
                    logger.error("Error: All dimensions must be positive")
                    continue
            break
        except ValueError:
            logger.error("Error: Please enter integers only")
    
    # Symmetry
    symmetry = input(f"\nEnter symmetry type (cubic/random/file) [default: {DEFAULT_SYMMETRY}]: ").strip().lower()
    if symmetry not in ["cubic", "random", "file"]:
        symmetry = DEFAULT_SYMMETRY
    
    # Temperature array
    while True:
        try:
            t_array_input = input(f"\nEnter temperature array (space-separated values) [default: {DEFAULT_TEMPERATURE}]: ").strip()
            if not t_array_input:
                t_array = [DEFAULT_TEMPERATURE]
            else:
                t_array = [float(x) for x in t_array_input.split()]
                if any(t < 0 for t in t_array):
                    logger.error("Error: Temperatures must be non-negative")
                    continue
            break
        except ValueError:
            logger.error("Error: Please enter valid numbers only")
    
    # Thermostat damping
    while True:
        try:
            t_stat = float(input(f"\nEnter thermostat damping time (t_stat) in fs [default: {DEFAULT_T_STAT}]: ").strip() or str(DEFAULT_T_STAT))
            if t_stat <= 0:
                logger.error("Error: Thermostat damping must be positive")
                continue
            break
        except ValueError:
            logger.error("Error: Please enter a valid number")
    
    # Barostat damping
    while True:
        try:
            p_stat = float(input(f"\nEnter barostat damping time (p_stat) in fs [default: {DEFAULT_P_STAT}]: ").strip() or str(DEFAULT_P_STAT))
            if p_stat <= 0:
                logger.error("Error: Barostat damping must be positive")
                continue
            break
        except ValueError:
            logger.error("Error: Please enter a valid number")
    
    output_filename = DEFAULT_OUTPUT_FILE
    
    config = Config(
        model_file=model_file,
        supercell_dims=supercell_dims,
        symmetry=symmetry,
        output_filename=output_filename,
        t_array=t_array,
        t_stat=t_stat,
        p_stat=p_stat
    )
    
    # Display configuration summary
    logger.info("\n" + "=" * 70)
    logger.info("⚙️  CONFIGURATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Model file       : {config.model_file}")
    logger.info(f"  Supercell dims   : {config.supercell_dims}")
    logger.info(f"  Symmetry         : {config.symmetry}")
    logger.info(f"  Output filename  : {config.output_filename}")
    logger.info(f"  Temperatures [K] : {config.t_array}")
    logger.info(f"  T-stat damping   : {config.t_stat}")
    logger.info(f"  P-stat damping   : {config.p_stat}")
    logger.info("=" * 70 + "\n")
    
    return config

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main() -> None:
    """Main function to orchestrate the shell model processing."""
    try:
        logger.info("Starting LAMMPS shell model processing")
        
        # Get configuration from user
        config = get_user_config()
        
        # Validate configuration
        config.validate()
        
        # Load model
        model = load_model(config.model_file)
        
        # Extract model name
        model_name = model.header[0] if hasattr(model, 'header') and model.header else "Unknown"
        logger.info(f"Loaded model: {model_name}")
        
        # Create supercell
        cell = create_supercell(
            model,
            config.supercell_dims,
            config.symmetry
        )
        logger.info(f"Created supercell with dimensions {config.supercell_dims}")
        
        # Process shell model and generate LAMMPS structure
        mapped_cell, string_cell = process_shell_model(
            model,
            cell,
            config.output_filename
        )
        
        # Rename the structure file to rstrt.dat
        structure_file = f"{config.output_filename}.LAMMPSStructure"
        rstrt_file = 'rstrt.dat'
        
        if os.path.exists(structure_file):
            # Warn if rstrt.dat already exists
            if os.path.exists(rstrt_file):
                logger.warning(f"⚠️  File '{rstrt_file}' already exists - OVERWRITING")
            
            try:
                subprocess.run(['mv', structure_file, rstrt_file], check=True)
                logger.info(f"✓ Renamed '{structure_file}' to '{rstrt_file}'")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Error renaming structure file: {e}")
                logger.info("  → Attempting alternative rename method...")
                try:
                    os.rename(structure_file, rstrt_file)
                    logger.info(f"  ✓ Successfully renamed using os.rename()")
                except OSError as e2:
                    logger.error(f"  ✗ Failed to rename file: {e2}")
        else:
            logger.warning(f"⚠️  Structure file '{structure_file}' not found, skipping rename")
        
        # Create and save species_id_map
        species_id_map = create_species_id_map(cell, model)
        map_filename = "species_id_map.txt"
        try:
            with open(map_filename, "w") as f:
                for (species, part), id_value in species_id_map.items():
                    f.write(f"{species} {part} {id_value}\n")
            logger.info(f"Species ID map saved to {map_filename}")
        except IOError as e:
            logger.error(f"Failed to save species ID map: {e}")
        
        # Generate LAMMPS input
        shell_models_data, shell_models_springs, shell_models_potentials = extract_shell_model_data(model)
        lammps_content = generate_lammps_input(
            shell_models_data,
            shell_models_springs,
            shell_models_potentials,
            species_id_map,
            model_name,
            config.t_array,
            config.t_stat,
            config.p_stat
        )
        
        # Save to file
        save_lammps_input(lammps_content)
        
        logger.info("\n" + "=" * 70)
        logger.info(" 🎉 PROCESSING COMPLETED SUCCESSFULLY! ")
        logger.info("=" * 70)
        logger.info("\n📁 Generated Files:")
        logger.info("  ✓ rstrt.dat              (LAMMPS structure file)")
        logger.info("  ✓ lammps.in              (LAMMPS input script)")
        logger.info("  ✓ species_id_map.txt     (Species ID mapping)")
        logger.info("  ✓ lammps_processing.log  (Detailed processing log)")
        logger.info("\n" + "=" * 70 + "\n")
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except ModelLoadError as e:
        logger.error(f"Model loading error: {e}")
        sys.exit(1)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except LAMMPSProcessingError as e:
        logger.error(f"Processing error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
