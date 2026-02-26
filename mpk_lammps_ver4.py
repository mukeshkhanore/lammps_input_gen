#!/usr/bin/env python3
"""
Shell Model Processing for LAMMPS Structure and Setup File Generation

This script processes shell models for LAMMPS structure and setup file generation.
Mass ratio is set to 98% for core and 2% for shell.

Author: Mukesh Khanore
Date: 26-Feb-2026
Note: Mix is implemented for file and cubic modes; random mode is still work in progress
LAMMPS MD Logic: Mónica Elisabet Graf and Mauro António Pereira Gonçalves
Version: 4.3 - Fixed shell preservation in cubic/random modes; proper FILE mode handling
"""

import sys
import os
import logging
import numpy as np
import copy
import pickle
import subprocess
from typing import Dict, List, Tuple, Any, Optional
from mendeleev import element
from dataclasses import dataclass, field
import pm__cell as pmc
import pm__chemical_order as pmco
from datetime import datetime

current_time = datetime.now()
time_str = current_time.strftime("%H%M%S")
seed = int(time_str)

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
DEFAULT_RANDOM_SEED = seed
DEFAULT_T_STAT = 0.1
DEFAULT_P_STAT = 2.0
DEFAULT_SYMMETRY = "file"
DEFAULT_MODEL_FILE = "./potential.pickle"
DEFAULT_OUTPUT_FILE = "structure"

# LAMMPS simulation parameters
DEFAULT_EQUILIBRATION_TEMP_STEPS = 20000
DEFAULT_EQUILIBRATION_FINAL_STEPS = 30000
DEFAULT_PRODUCTION_STEPS = 50000
DEFAULT_TIMESTEP = 0.0002
DEFAULT_THERMO_FREQ = 500
DEFAULT_TRAJ_NAME = "trajectory"
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
        logger.warning(f"Could not create log file {log_file}: {e}")
    
    logger.addHandler(console_handler)
    return logger

# Initialize logger
logger = setup_logging()
logger.info(f"Generated seed from current time {current_time.strftime('%H:%M:%S')}: {seed}")

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
    thermo_freq: int = DEFAULT_THERMO_FREQ
    timestep: float = DEFAULT_TIMESTEP
    equilibration_temp_steps: int = DEFAULT_EQUILIBRATION_TEMP_STEPS
    equilibration_final_steps: int = DEFAULT_EQUILIBRATION_FINAL_STEPS
    production_steps: int = DEFAULT_PRODUCTION_STEPS
    traj_name: str = DEFAULT_TRAJ_NAME
    material_type: str = "pure"
    species_a: Optional[str] = None
    species_b: Optional[str] = None
    position: Optional[str] = None
    mix_ratio: Optional[float] = None
    
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
        
        # Validate THERMO_FREQ
        if self.thermo_freq <= 0:
            raise ConfigurationError(
                f"Thermo frequency must be positive, got {self.thermo_freq}"
            )
        
        # Validate TIMESTEP
        if self.timestep <= 0:
            raise ConfigurationError(
                f"Timestep must be positive, got {self.timestep}"
            )
        
        # Validate EQUILIBRATION_TEMP_STEPS
        if self.equilibration_temp_steps <= 0:
            raise ConfigurationError(
                f"Equilibration temperature steps must be positive, got {self.equilibration_temp_steps}"
            )
        
        # Validate EQUILIBRATION_FINAL_STEPS
        if self.equilibration_final_steps <= 0:
            raise ConfigurationError(
                f"Equilibration final steps must be positive, got {self.equilibration_final_steps}"
            )
        
        # Validate PRODUCTION_STEPS
        if self.production_steps <= 0:
            raise ConfigurationError(
                f"Production steps must be positive, got {self.production_steps}"
            )
        
        # Validate trajectory name
        if not self.traj_name:
            raise ConfigurationError("Trajectory name cannot be empty")
        
        # Validate material type
        if self.material_type not in ["pure", "mix"]:
            raise ConfigurationError(
                f"Material type must be 'pure' or 'mix', got '{self.material_type}'"
            )
        
        # Validate species based on material type
        if self.material_type == "pure":
            if not self.species_a or not self.species_b:
                raise ConfigurationError(
                    "For pure material, both species_a and species_b must be specified"
                )
        elif self.material_type == "mix":
            if not self.position or not self.mix_ratio:
                raise ConfigurationError(
                    "For mix material, position and mix_ratio must be specified"
                )
            if self.position not in ["A", "B"]:
                raise ConfigurationError(
                    f"Position must be 'A' or 'B', got '{self.position}'"
                )
        
        logger.info("✓ Configuration validation passed")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_mixing_type(mixing_type: str, i: int, j: int, k: int) -> int:
    """
    Determine mixing type based on coordinates.
    
    Args:
        mixing_type: Type of mixing ("homog", "G", or "14")
        i, j, k: Integer coordinates
        
    Returns:
        int: 0 for species 1, 1 for species 2
        
    Raises:
        ValidationError: If mixing type is not supported
    """
    if mixing_type == "homog":
        return 0
    elif mixing_type == "G":
        return 0 if pow(-1, i + j + k) == -1 else 1
    elif mixing_type == "14":
        return 0 if (i % 2 == 0 and j % 2 == 0 and k % 2 == 0) else 1
    else:
        raise ValidationError(f"Ordering '{mixing_type}' not defined. Valid options: homog, G, 14")

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
        
        logger.info(f"✓ Extracted shell model data for {len(shell_models_data['model'])} species")
        logger.debug(f"  Springs: {len(shell_models_springs)}, Potentials: {len(shell_models_potentials)}")
        
        return shell_models_data, shell_models_springs, shell_models_potentials
        
    except AttributeError as e:
        raise ValidationError(f"Model structure is invalid: {e}")

# ============================================================================
# SPECIES ID MAPPING
# ============================================================================
def create_species_id_map(cell: Any, model: Any, species_a: Optional[str] = None, species_b: Optional[str] = None) -> Dict:
    """
    Create mapping between species+part and numeric IDs.
    
    Supports user-specified species with backward compatibility:
      - If species_a and species_b are provided: builds map from user input
      - If not provided (None): falls back to model.AB_specie
    
    Args:
        cell: Cell object with species information
        model: Model object with species information
        species_a: User-specified A-site species (e.g., "Ba" or "Ba/Ca"), optional
        species_b: User-specified B-site species (e.g., "Ti" or "Zr/Sn"), optional
        
    Returns:
        Dict: Mapping from (species, part) to numeric ID, following ABO order
        
    Raises:
        ValidationError: If species data is invalid
    """
    try:
        species_id_map = {}
        species_list = []
        source_description = ""
        
        if not hasattr(model, 'AB_specie'):
            raise ValidationError("Model missing AB_specie attribute")
        
        # Determine whether to use user-specified species or model defaults
        if species_a is not None and species_b is not None:
            # Build species list from user input
            source_description = "user-specified"
            replacements_a = species_a
            replacements_b = species_b
            
            # Extract A-site species (handle single or mixed like "Ba" or "Ba/Ca")
            species_a_list = [s.strip() for s in replacements_a.split('/')]
            for sp in species_a_list:
                if sp and sp not in species_list:
                    species_list.append(sp)
            
            # Extract B-site species (handle single or mixed like "Ti" or "Zr/Sn")
            species_b_list = [s.strip() for s in replacements_b.split('/')]
            for sp in species_b_list:
                if sp and sp not in species_list:
                    species_list.append(sp)
            
            # Validate that all user-specified species exist in model.charges
            if hasattr(model, 'charges') and model.charges:
                available_species = {charge.species for charge in model.charges}
                for sp in species_list:
                    if sp != "O" and sp not in available_species:
                        logger.warning(
                            f"⚠️  Species '{sp}' specified by user not found in model.charges. "
                            f"It will have None for charge and spring constants. "
                            f"Available species: {sorted(available_species)}"
                        )
        else:
            # Fall back to model defaults (backward compatible)
            source_description = "model.AB_specie (backward compatible fallback)"
            
            if "A" in model.AB_specie:
                species_list.extend(model.AB_specie["A"])
            if "B" in model.AB_specie:
                species_list.extend(model.AB_specie["B"])
        
        # Add oxygen if not already present
        if "O" not in species_list:
            species_list.append("O")
        
        # Create ID mapping following ABO order: A-site → B-site → O
        for i, species in enumerate(species_list):
            species_id_map[(species, 'core')] = (i + 1)
            species_id_map[(species, 'shell')] = (len(species_list) + i + 1)
        
        logger.info(f"✓ Created species ID mapping from {source_description} for {len(species_list)} species: {species_list}")
        logger.debug(f"  Species ID map: {species_id_map}")
        
        return species_id_map
        
    except (AttributeError, KeyError) as e:
        raise ValidationError(f"Failed to create species ID map: {e}")

# ============================================================================
# CELL CREATION AND MANIPULATION
# ============================================================================
def create_mapped_cell(original_cell: Any, species_id_map: Dict) -> Any:
    """
    Create a new cell with IDs mapped according to species_id_map.
    
    Atoms are sorted in ABO order with each core immediately followed by its shell:
    A1_core1, A1_shell1, A1_core2, A1_shell2, ..., A2_core1, A2_shell1, ..., B1_core1, B1_shell1, ..., O_core1, O_shell1, ...
    
    Args:
        original_cell: Original cell object
        species_id_map: Mapping from (species, part) to numeric ID
        
    Returns:
        Any: New cell with mapped IDs, atoms sorted in ABO order with core-shell pairs
    """
    new_cell = pmc.Cell(convention='zerolist', prescribe_N=0)
    new_cell.lattice = copy.deepcopy(original_cell.lattice)
    
    # Calculate number of species (cores and shells have separate IDs)
    num_species = len(species_id_map) // 2
    
    # Separate cores and shells, build lookup structures
    cores_by_species = {}  # {species_name: [(idx, atom), ...]}
    shells_by_species = {}  # {species_name: [(idx, atom), ...]}
    
    for idx, atom in enumerate(original_cell.atom):
        # Normalize coreshell attribute
        coreshell_raw = str(atom.coreshell).lower().strip() if atom.coreshell else ""
        is_core = coreshell_raw in ('core', 'cor')
        
        species_name = atom.name
        
        if is_core:
            if species_name not in cores_by_species:
                cores_by_species[species_name] = []
            cores_by_species[species_name].append((idx, atom))
        else:
            if species_name not in shells_by_species:
                shells_by_species[species_name] = []
            shells_by_species[species_name].append((idx, atom))
    
    # Build list of atoms to add, pairing cores with their nearest shell
    atoms_to_add = []
    
    # Get species order from species_id_map (already in ABO order)
    species_order = []
    for (species, part), _ in sorted(species_id_map.items(), key=lambda x: x[1]):
        if part == 'core' and species not in species_order:
            species_order.append(species)
    
    for species_name in species_order:
        cores = cores_by_species.get(species_name, [])
        shells = shells_by_species.get(species_name, [])
        
        # Track which shells have been paired
        used_shell_indices = set()
        
        for core_idx, core_atom in cores:
            # Find the nearest unpaired shell for this core
            core_pos = np.array(core_atom.position_frac)
            best_shell = None
            best_dist = float('inf')
            best_shell_list_idx = None
            
            for shell_list_idx, (shell_idx, shell_atom) in enumerate(shells):
                if shell_list_idx in used_shell_indices:
                    continue
                shell_pos = np.array(shell_atom.position_frac)
                # Calculate distance (considering periodic boundaries approximately)
                diff = core_pos - shell_pos
                # Apply minimum image convention for periodic boundaries
                diff = diff - np.round(diff)
                dist = np.linalg.norm(diff)
                if dist < best_dist:
                    best_dist = dist
                    best_shell = (shell_idx, shell_atom)
                    best_shell_list_idx = shell_list_idx
            
            # Get mapped IDs
            core_mapped_id = species_id_map.get((species_name, 'core'))
            shell_mapped_id = species_id_map.get((species_name, 'shell'))
            
            if core_mapped_id is None:
                logger.warning(f"Species ({species_name}, core) not in species_id_map, skipping")
                continue
            
            # Calculate species_index for sorting
            species_index = core_mapped_id - 1
            
            # Add core first
            atoms_to_add.append({
                'mapped_id': core_mapped_id,
                'position_frac': core_atom.position_frac,
                'coreshell': core_atom.coreshell,
                'species_index': species_index,
                'pair_order': len(atoms_to_add)  # Order within this species
            })
            
            # Add paired shell immediately after core
            if best_shell is not None and shell_mapped_id is not None:
                used_shell_indices.add(best_shell_list_idx)
                _, shell_atom = best_shell
                atoms_to_add.append({
                    'mapped_id': shell_mapped_id,
                    'position_frac': shell_atom.position_frac,
                    'coreshell': shell_atom.coreshell,
                    'species_index': species_index,
                    'pair_order': len(atoms_to_add)
                })
        
        # Handle any remaining unpaired shells
        for shell_list_idx, (shell_idx, shell_atom) in enumerate(shells):
            if shell_list_idx not in used_shell_indices:
                shell_mapped_id = species_id_map.get((species_name, 'shell'))
                if shell_mapped_id is not None:
                    species_index = shell_mapped_id - num_species - 1
                    atoms_to_add.append({
                        'mapped_id': shell_mapped_id,
                        'position_frac': shell_atom.position_frac,
                        'coreshell': shell_atom.coreshell,
                        'species_index': species_index,
                        'pair_order': len(atoms_to_add)
                    })
                    logger.warning(f"Unpaired shell found for {species_name}")
    
    # Add sorted atoms to the new cell (already in correct order from above)
    for atom_data in atoms_to_add:
        new_cell.appendAtom(
            name=atom_data['mapped_id'],
            position_frac=atom_data['position_frac'],
            coreshell=atom_data['coreshell']
        )
    
    logger.info(f"✓ Created mapped cell with {new_cell.N} atoms (ABO order, core followed by its shell)")
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
        except (ValueError, AttributeError, KeyError) as e:
            raise ValidationError(f"Failed to get atomic weight for species '{species}': {e}") from e
        
        for id_key, mapped_id in species_id_map.items():
            if id_key[0] == species:
                part = id_key[1]
                
                # Validate key existence before access
                if species not in shell_models_data.get('model', {}):
                    raise ValidationError(f"Species '{species}' not found in shell model data")
                if part not in shell_models_data['model'][species]:
                    raise ValidationError(f"Part '{part}' not found for species '{species}'")
                
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
    
    logger.info("✓ Successfully mapped charges and masses to new model")
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
# SUPERCELL CREATION - MAIN FUNCTION
# ============================================================================
def create_supercell(model: Any, supercell_dims: List[int], symmetry: str = "cubic", material_type: str = "mix", 
                     species_a: Optional[str] = None, species_b: Optional[str] = None, position: str = "A", mix_ratio: float = 0.5) -> Any:
    """
    Create a supercell with specified dimensions and symmetry.
    
    Supports three symmetry modes:
      - "file": Reads structure from GS.gulp file
      - "cubic": Uses chemical order without random mixing
      - "random": Uses chemical order with random species assignment
    
    Args:
        model: Model containing chemical order information
        supercell_dims: Dimensions [Nx, Ny, Nz] of the supercell
        symmetry: Type of symmetry ("file", "cubic", or "random")
        material_type: Type of material ("pure", "mix", etc.)
        species_a: First species in the material (e.g., "Ba")
        species_b: Second species in the material (e.g., "Ti")
        position: Position where mixing occurs ("A" or "B")
        mix_ratio: Mixing ratio (fraction of first species)
        
    Returns:
        Any: Cell object representing the supercell
        
    Raises:
        FileNotFoundError: If symmetry is "file" and GS.gulp is not found
        ValidationError: If model data is incomplete or invalid
        LAMMPSProcessingError: If structure creation fails
    """
    nx, ny, nz = supercell_dims
    
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
            cell = cell.replicate([nx // 2, ny // 2, nz // 2])
            cell.pairCoresShells()
            logger.info(f"✓ Successfully created supercell from {gulp_file}")
            return cell
            
        except (OSError, IOError, ValueError, AttributeError, TypeError) as e:
            raise LAMMPSProcessingError(f"Error processing GULP file '{gulp_file}': {e}") from e
    
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
        
        # Determine species to use: user-provided or model's default (backward compatible)
        # IMPORTANT: prescribeChemicalOrderForBox() expects a list format, not string
        if species_a is None:
            if not hasattr(model, 'AB_specie') or "A" not in model.AB_specie:
                raise ValidationError("No species_a provided and model.AB_specie['A'] not found")
            replacements_a = model.AB_specie["A"]  # Already a list from model
            logger.debug(f"Using species from model for A-site: {replacements_a}")
        else:
            # Parse user input string into list format that prescribeChemicalOrderForBox() expects
            if '/' in species_a:
                replacements_a = [s.strip() for s in species_a.split('/')]
            else:
                replacements_a = [species_a.strip()]
            logger.info(f"Using user-specified species for A-site: {replacements_a}")
        
        if species_b is None:
            if not hasattr(model, 'AB_specie') or "B" not in model.AB_specie:
                raise ValidationError("No species_b provided and model.AB_specie['B'] not found")
            replacements_b = model.AB_specie["B"]  # Already a list from model
            logger.debug(f"Using species from model for B-site: {replacements_b}")
        else:
            # Parse user input string into list format that prescribeChemicalOrderForBox() expects
            if '/' in species_b:
                replacements_b = [s.strip() for s in species_b.split('/')]
            else:
                replacements_b = [species_b.strip()]
            logger.info(f"Using user-specified species for B-site: {replacements_b}")
        
        # Validate that custom species exist in model charges
        if not hasattr(model, 'charges') or not model.charges:
            raise ValidationError("Model has no charge information to validate species")
        
        available_species = {charge.species for charge in model.charges}
        
        # Validate species_a (can be single like "Ba" or mixed like "Ba/Ca")
        if species_a is not None:
            species_a_list = [s.strip() for s in species_a.split('/')]
            for sp in species_a_list:
                if sp not in available_species:
                    raise ValidationError(
                        f"Custom species '{sp}' (from species_a='{species_a}') not found in model. "
                        f"Available species: {sorted(list(available_species))}"
                    )
        
        # Validate species_b (can be single like "Ti" or mixed like "Zr/Sn")
        if species_b is not None:
            species_b_list = [s.strip() for s in species_b.split('/')]
            for sp in species_b_list:
                if sp not in available_species:
                    raise ValidationError(
                        f"Custom species '{sp}' (from species_b='{species_b}') not found in model. "
                        f"Available species: {sorted(list(available_species))}"
                    )
        
        # Prepare structure
        cell = pmc.Cell(convention='zerolist', prescribe_N=0)
        cell.is_shell_model = True
        cell.simplePerovskite(ABO_names=["A", "B", "O"], dimensions=[nx, ny, nz], coreshell=True)
        
        # Apply chemical order with selected species
        try:
            cell = chemical_order_A.prescribeChemicalOrderForBox(
                cell, position="A", replacements=replacements_a, 
                dimensions=[nx, ny, nz], coreshell=True
            )
            cell = chemical_order_B.prescribeChemicalOrderForBox(
                cell, position="B", replacements=replacements_b, 
                dimensions=[nx, ny, nz], coreshell=True
            )
            cell.pairCoresShells()
        except (AttributeError, ValueError, TypeError, KeyError) as e:
            raise LAMMPSProcessingError(f"Failed to apply chemical order: {e}") from e
        
        # Apply perturbation for random symmetry
        perturbation = np.zeros((cell.N, 3))
        if symmetry == "random":
            logger.info("Applying random perturbations to atomic positions")
            for i in range(cell.N):
                perturbation[i] = np.random.uniform(-0.05, 0.05, 3)
        
        # Order species in ABO order: all A-site species, then all B-site species, then oxygen (like gs.gulp)
        species_order = []
        
        # Add A-site species
        if isinstance(replacements_a, (list, tuple)):
            for sp in replacements_a:
                if sp not in species_order:
                    species_order.append(sp)
        else:
            if replacements_a not in species_order:
                species_order.append(replacements_a)
        
        # Add B-site species
        if isinstance(replacements_b, (list, tuple)):
            for sp in replacements_b:
                if sp not in species_order:
                    species_order.append(sp)
        else:
            if replacements_b not in species_order:
                species_order.append(replacements_b)
        
        # Add oxygen
        if "O" not in species_order:
            species_order.append("O")
        
        logger.info(f"Using ABO species order: A-site={replacements_a}, B-site={replacements_b}, O=oxygen")
        
        # Finalize cell
        cell.countPresentSpecies()
        cell.pairCoresShells()
        cell.displaceCartesian(perturbation)
        
        logger.info(f"✓ Created {symmetry} supercell with dimensions {supercell_dims}")
        logger.info(f"  Total atoms in supercell: {cell.N}")
        
        return cell

# ============================================================================
# SHELL MODEL PROCESSING
# ============================================================================
def process_shell_model(model: Any, cell: Any, output_filename: str = "structure", 
                        species_a: Optional[str] = None, species_b: Optional[str] = None) -> Tuple[Any, Any, Dict, Dict, List, Dict]:
    """
    Process shell model and save to LAMMPS structure.
    
    Args:
        model: Model containing charges information
        cell: Cell object to process
        output_filename: Base name for output files
        species_a: User-specified A-site species (e.g., "Ba" or "Ba/Ca"), optional
        species_b: User-specified B-site species (e.g., "Ti" or "Zr/Sn"), optional
        
    Returns:
        Tuple containing:
            - mapped_cell: Cell with numeric IDs
            - string_cell: Cell with string atom names
            - shell_models_data: Shell model data dictionary
            - shell_models_springs: Spring constants dictionary
            - shell_models_potentials: List of potential parameters
            - species_id_map: Mapping of (species, part) to numeric ID
        
    Raises:
        LAMMPSProcessingError: If processing fails
    """
    try:
        # Extract shell model data
        shell_models_data, shell_models_springs, shell_models_potentials = extract_shell_model_data(model)
        
        # Create species ID mapping using user-specified species (or fall back to model defaults)
        species_id_map = create_species_id_map(cell, model, species_a, species_b)
        
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
            logger.info(f"✓ Successfully wrote LAMMPS structure to {output_filename}")
        except Exception as e:
            raise LAMMPSProcessingError(f"Error writing LAMMPS structure: {e}") from e
        
        return mapped_cell, string_cell, shell_models_data_new, shell_models_springs, shell_models_potentials, species_id_map
        
    except (ValidationError, LAMMPSProcessingError):
        logger.error(f"Shell model processing failed", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in shell model processing: {e}", exc_info=True)
        raise LAMMPSProcessingError(f"Shell model processing failed: {e}") from e
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
        
        logger.info("✓ Model loaded successfully")
        return model
        
    except pickle.UnpicklingError as e:
        raise ModelLoadError(f"Failed to unpickle model file: {e}") from e
    except (OSError, IOError, ValueError, AttributeError) as e:
        raise ModelLoadError(f"Error loading model from {file_path}: {e}") from e

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
    
    # Validate that we have both core and shell types
    if not core_types:
        raise ValidationError("No core types found in species ID map")
    if not shell_types:
        raise ValidationError("No shell types found in species ID map")
    
    species_types = list(dict.fromkeys([species for (species, part), _ in species_id_map.items() if part == 'core']))
    
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
    # Validate that we have potentials to process
    if not shell_models_potentials:
        logger.warning("No shell model potentials provided - using default section")
    
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

def generate_simulation_settings(initial_temp: float, t_stat: float, p_stat: float, 
                                  thermo_freq: int, timestep: float,
                                  equilibration_temp_steps: int,
                                  equilibration_final_steps: int,
                                  production_steps: int, traj_name: str) -> str:
    """Generate simulation settings and initial run."""
    return f"""
neigh_modify page 100000 one 10000\t\t\t\t#max neighbors of one atom set to 10000

## ---------- Define Settings ---------------------
compute eng all pe/atom                                         #pe/atom: potential energy for each atom
compute eatoms all reduce sum c_eng                             #sum up all energies

# ------------------------ Initiating the run  -------------------------------

reset_timestep 0

thermo {thermo_freq}
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

compute CSequ all temp/cs cores shells

thermo_modify temp CSequ

timestep {timestep}

#generate velocities
velocity all create {initial_temp}K 34 dist gaussian mom yes rot no bias yes temp CSequ

# ----------------------- Equilibration {initial_temp}K  ---------------------------------------------

fix npt_equ all npt temp {initial_temp} {initial_temp} {t_stat} tri 0.0 0.0 {p_stat}
fix_modify npt_equ temp CSequ
run {equilibration_temp_steps}
unfix npt_equ

fix npt_equ all npt temp {initial_temp} {initial_temp} {t_stat} tri 0.0 0.0 {p_stat}
fix_modify npt_equ temp CSequ
run {equilibration_final_steps}
unfix npt_equ

# ----------------------- Production {initial_temp}K ---------------------------------------------

fix npt all npt temp {initial_temp} {initial_temp} {t_stat} tri 0.0 0.0 {p_stat}
fix_modify npt temp CSequ

dump myDump all custom {DUMP_FREQ} {traj_name}_${{int(initial_temp)}}k.atom id type q x y z #fx fy fz
dump_modify myDump format 4 %20.6g
dump_modify myDump format 5 %20.6g
dump_modify myDump format 6 %20.6g

run {production_steps}
unfix npt
undump myDump
"""

def generate_temperature_ramps(t_array: List[float], t_stat: float, p_stat: float,
                                thermo_freq: int, timestep: float,
                                equilibration_temp_steps: int,
                                equilibration_final_steps: int,
                                production_steps: int, traj_name: str) -> str:
    """Generate temperature ramp sections for multi-temperature runs."""
    if len(t_array) <= 1:
        return ""
    
    lines = []
    for i, t in enumerate(t_array[1:]):
        prev_temp = t_array[i]
        curr_temp = t
        
        lines.append(f"""
# ----------------------- Equilibration {curr_temp}K  ---------------------------------------------

fix npt_equ all npt temp {prev_temp} {curr_temp} {t_stat} tri 0.0 0.0 {p_stat}
fix_modify npt_equ temp CSequ
run {equilibration_temp_steps}
unfix npt_equ 

fix npt_equ all npt temp {curr_temp} {curr_temp} {t_stat} tri 0.0 0.0 {p_stat}
fix_modify npt_equ temp CSequ
run {equilibration_final_steps}
unfix npt_equ 

# ----------------------- Production {curr_temp}K ---------------------------------------------

fix npt all npt temp {curr_temp} {curr_temp} {t_stat} tri 0.0 0.0 {p_stat}
fix_modify npt temp CSequ

dump myDump all custom {DUMP_FREQ} {traj_name}_${{int(curr_temp)}}k.atom id type q x y z #fx fy fz
dump_modify myDump format 4 %20.6g
dump_modify myDump format 5 %20.6g
dump_modify myDump format 6 %20.6g

run {production_steps}
unfix npt
undump myDump
""")
    
    return "\n".join(lines)

def generate_lammps_input(shell_models_data: Dict, shell_models_springs: Dict, 
                           shell_models_potentials: List, species_id_map: Dict, 
                           model_name: str, t_array: List[float], t_stat: float, 
                           p_stat: float, thermo_freq: int, timestep: float,
                           equilibration_temp_steps: int,
                           equilibration_final_steps: int,
                           production_steps: int, traj_name: str) -> str:
    """
    Generate complete LAMMPS input file content.
    """
    try:
        initial_temp = t_array[0]
        
        sections = [
            generate_lammps_header(model_name),
            "\n",
            generate_charge_settings(shell_models_data, species_id_map),
            generate_group_definitions(species_id_map),
            generate_potential_section(model_name, shell_models_potentials, species_id_map),
            generate_bond_section(shell_models_springs, species_id_map),
            generate_simulation_settings(initial_temp, t_stat, p_stat, thermo_freq, timestep,
                                        equilibration_temp_steps, equilibration_final_steps,
                                        production_steps, traj_name),
            generate_temperature_ramps(t_array, t_stat, p_stat, thermo_freq, timestep,
                                      equilibration_temp_steps, equilibration_final_steps,
                                      production_steps, traj_name)
        ]
        
        content = "".join(sections)
        logger.info("✓ Generated LAMMPS input file content")
        return content
        
    except (ValidationError, LAMMPSProcessingError):
        logger.error(f"Failed to generate LAMMPS input", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in LAMMPS input generation: {e}", exc_info=True)
        raise LAMMPSProcessingError(f"LAMMPS input generation failed: {e}") from e

def save_lammps_input(content: str, filename: str = "lammps.in") -> None:
    """
    Save LAMMPS input content to file with error handling.
    """
    try:
        if os.path.exists(filename):
            logger.warning(f"⚠️  File '{filename}' already exists - OVERWRITING")
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"✓ Successfully wrote LAMMPS input to {filename}")
    except IOError as e:
        raise LAMMPSProcessingError(f"Error writing to {filename}: {e}")

# ============================================================================
# USER CONFIGURATION
# ============================================================================
def get_user_config() -> Config:
    """
    Get configuration parameters from user input.
    """
    logger.info("=" * 70)
    logger.info("🔧 Shell Model Processing for LAMMPS - Configuration")
    logger.info("=" * 70)
    
    # Model file
    model_file = input(f"\nEnter path to model pickle file\n[default: {DEFAULT_MODEL_FILE}]: ").strip()
    if not model_file:
        model_file = DEFAULT_MODEL_FILE
    
    # Material type
    while True:
        material_type = input("\nEnter material type (mix/pure) [default: pure]: ").strip().lower()
        if not material_type:
            material_type = "pure"
        
        if material_type not in ["mix", "pure"]:
            logger.error("❌ Error: Please enter either 'mix' or 'pure'")
            continue
        
        logger.info(f"✓ Material type selected: {material_type}")
        break
    
    position = None
    species_a = None
    species_b = None
    mix_ratio = None
    
    if material_type == "mix":
        while True:
            position = input("\nEnter position (A/B) [default: A]: ").strip().upper()
            if not position:
                position = "A"
            
            if position not in ["A", "B"]:
                logger.error("❌ Error: Please enter either 'A' or 'B'")
                continue
            
            logger.info(f"✓ Position selected: {position}")
            break
        
        while True:
            mix_species = input(f"\nEnter the two species to mix for position {position} (separated by space) [e.g., Ba Ca]: ").strip()
            if not mix_species or ' ' not in mix_species:
                logger.error("❌ Error: Please enter two species separated by a space")
                continue
            
            parts = [s.strip() for s in mix_species.split()]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                logger.error("❌ Error: Please enter exactly two valid species separated by a space")
                continue
            
            if ',' in parts[0] or '/' in parts[0] or ',' in parts[1] or '/' in parts[1]:
                logger.error("❌ Error: Please enter only atomic symbols without commas or slashes")
                continue
                
            species_1, species_2 = parts[0], parts[1]
            logger.info(f"✓ Mixed species selected: {species_1} and {species_2}")
            break
            
        while True:
            mix_ratio_input = input(f"\nEnter fraction of {species_1} (0.0 to 1.0) [e.g., 0.5 for 50:50]: ").strip()
            if not mix_ratio_input:
                logger.error("❌ Error: Mix ratio cannot be empty")
                continue
            
            try:
                mix_ratio = float(mix_ratio_input)
                if mix_ratio < 0.0 or mix_ratio > 1.0:
                    logger.error("❌ Error: Fraction must be between 0.0 and 1.0")
                    continue
                
                logger.info(f"✓ Mix fraction selected: {mix_ratio} for {species_1} (and {1.0 - mix_ratio:.3g} for {species_2})")
                break
                
            except ValueError:
                logger.error("❌ Error: Please enter a valid decimal number")
        
        if position == "A":
            species_a = f"{species_1}/{species_2}"
            while True:
                species_b = input("\nEnter single species for remaining B site [e.g., Ti, Zr, Sn]: ").strip()
                if not species_b:
                    logger.error("❌ Error: Species B cannot be empty")
                    continue
                
                if ' ' in species_b or ',' in species_b:
                    logger.error("❌ Error: Please enter only ONE species for B site")
                    continue
                
                logger.info(f"✓ Species B selected: {species_b}")
                break
        else:
            species_b = f"{species_1}/{species_2}"
            while True:
                species_a = input("\nEnter single species for remaining A site [e.g., Ba, Ca, Pb]: ").strip()
                if not species_a:
                    logger.error("❌ Error: Species A cannot be empty")
                    continue
                
                if ' ' in species_a or ',' in species_a:
                    logger.error("❌ Error: Please enter only ONE species for A site")
                    continue
                
                logger.info(f"✓ Species A selected: {species_a}")
                break
    
    else:
        while True:
            species_a = input("\nEnter single species for A site [e.g., Ba, Ca, Pb]: ").strip()
            if not species_a:
                logger.error("❌ Error: Species A cannot be empty")
                continue
            
            if ' ' in species_a or ',' in species_a:
                logger.error("❌ Error: Please enter only ONE species for A site")
                continue
            
            logger.info(f"✓ Species A selected: {species_a}")
            break
        
        while True:
            species_b = input("\nEnter single species for B site [e.g., Ti, Zr, Sn]: ").strip()
            if not species_b:
                logger.error("❌ Error: Species B cannot be empty")
                continue
            
            if ' ' in species_b or ',' in species_b:
                logger.error("❌ Error: Please enter only ONE species for B site")
                continue
            
            logger.info(f"✓ Species B selected: {species_b}")
            break

    # Supercell dimensions
    while True:
        try:
            dims_input = input(f"\nEnter supercell dimensions (Nx Ny Nz) [default: {' '.join(map(str, DEFAULT_SUPERCELL_DIMS))}]: ").strip()
            if not dims_input:
                supercell_dims = DEFAULT_SUPERCELL_DIMS.copy()
            else:
                supercell_dims = [int(x) for x in dims_input.split()]
                if len(supercell_dims) != 3:
                    logger.error("❌ Error: Please enter exactly 3 dimensions")
                    continue
                if any(dim <= 0 for dim in supercell_dims):
                    logger.error("❌ Error: All dimensions must be positive")
                    continue
            break
        except ValueError:
            logger.error("❌ Error: Please enter integers only")
    
    # Symmetry
    symmetry = input(f"\nEnter symmetry type (cubic/random/file) [default: {DEFAULT_SYMMETRY}]: ").strip().lower()
    if symmetry not in ["cubic", "random", "file"]:
        symmetry = DEFAULT_SYMMETRY
    
    # Temperature array
    while True:
        try:
            t_array_input = input(f"\nEnter temperature array (space-separated) [default: {DEFAULT_TEMPERATURE}]: ").strip()
            if not t_array_input:
                t_array = [DEFAULT_TEMPERATURE]
            else:
                t_array = [float(x) for x in t_array_input.split()]
                if any(t < 0 for t in t_array):
                    logger.error("❌ Error: Temperatures must be non-negative")
                    continue
            break
        except ValueError:
            logger.error("❌ Error: Please enter valid numbers only")
    
    # Thermostat damping
    while True:
        try:
            t_stat = float(input(f"\nEnter thermostat damping time (fs) [default: {DEFAULT_T_STAT}]: ").strip() or str(DEFAULT_T_STAT))
            if t_stat <= 0:
                logger.error("❌ Error: Thermostat damping must be positive")
                continue
            break
        except ValueError:
            logger.error("❌ Error: Please enter a valid number")
    
    # Barostat damping
    while True:
        try:
            p_stat = float(input(f"\nEnter barostat damping time (fs) [default: {DEFAULT_P_STAT}]: ").strip() or str(DEFAULT_P_STAT))
            if p_stat <= 0:
                logger.error("❌ Error: Barostat damping must be positive")
                continue
            break
        except ValueError:
            logger.error("❌ Error: Please enter a valid number")
    
    # THERMO_FREQ
    while True:
        try:
            thermo_freq = int(input(f"\nEnter thermo frequency [default: {DEFAULT_THERMO_FREQ}]: ").strip() or str(DEFAULT_THERMO_FREQ))
            if thermo_freq <= 0:
                logger.error("❌ Error: Thermo frequency must be positive")
                continue
            break
        except ValueError:
            logger.error("❌ Error: Please enter a valid integer")
    
    # TIMESTEP
    while True:
        try:
            timestep = float(input(f"\nEnter timestep (fs) [default: {DEFAULT_TIMESTEP}]: ").strip() or str(DEFAULT_TIMESTEP))
            if timestep <= 0:
                logger.error("❌ Error: Timestep must be positive")
                continue
            break
        except ValueError:
            logger.error("❌ Error: Please enter a valid number")
    
    # EQUILIBRATION_TEMP_STEPS
    while True:
        try:
            equilibration_temp_steps = int(input(f"\nEnter equilibration temperature steps [default: {DEFAULT_EQUILIBRATION_TEMP_STEPS}]: ").strip() or str(DEFAULT_EQUILIBRATION_TEMP_STEPS))
            if equilibration_temp_steps <= 0:
                logger.error("❌ Error: Must be positive")
                continue
            break
        except ValueError:
            logger.error("❌ Error: Please enter a valid integer")
    
    # EQUILIBRATION_FINAL_STEPS
    while True:
        try:
            equilibration_final_steps = int(input(f"\nEnter equilibration final steps [default: {DEFAULT_EQUILIBRATION_FINAL_STEPS}]: ").strip() or str(DEFAULT_EQUILIBRATION_FINAL_STEPS))
            if equilibration_final_steps <= 0:
                logger.error("❌ Error: Must be positive")
                continue
            break
        except ValueError:
            logger.error("❌ Error: Please enter a valid integer")
    
    # PRODUCTION_STEPS
    while True:
        try:
            production_steps = int(input(f"\nEnter production steps [default: {DEFAULT_PRODUCTION_STEPS}]: ").strip() or str(DEFAULT_PRODUCTION_STEPS))
            if production_steps <= 0:
                logger.error("❌ Error: Must be positive")
                continue
            break
        except ValueError:
            logger.error("❌ Error: Please enter a valid integer")
    
    # Trajectory name
    traj_name = input(f"\nEnter trajectory filename prefix [default: {DEFAULT_TRAJ_NAME}]: ").strip()
    if not traj_name:
        traj_name = DEFAULT_TRAJ_NAME
    
    output_filename = DEFAULT_OUTPUT_FILE
    
    config = Config(
        model_file=model_file,
        supercell_dims=supercell_dims,
        symmetry=symmetry,
        output_filename=output_filename,
        t_array=t_array,
        t_stat=t_stat,
        p_stat=p_stat,
        thermo_freq=thermo_freq,
        timestep=timestep,
        equilibration_temp_steps=equilibration_temp_steps,
        equilibration_final_steps=equilibration_final_steps,
        production_steps=production_steps,
        traj_name=traj_name,
        material_type=material_type,
        species_a=species_a,
        species_b=species_b,
        position=position,
        mix_ratio=mix_ratio
    )
    
    # Display configuration summary
    logger.info("\n" + "=" * 70)
    logger.info("⚙️  CONFIGURATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Material type              : {material_type}")
    if material_type == "mix":
        logger.info(f"  Position                   : {position}")
        if position == 'A':
            logger.info(f"  Species (A mix)            : {species_a}")
            logger.info(f"  Species (B pure)           : {species_b}")
        else:
            logger.info(f"  Species (A pure)           : {species_a}")
            logger.info(f"  Species (B mix)            : {species_b}")
        logger.info(f"  Mix ratio                  : {mix_ratio}")
    else:
        logger.info(f"  Species A site             : {species_a}")
        logger.info(f"  Species B site             : {species_b}")
    logger.info(f"  Model file                 : {config.model_file}")
    logger.info(f"  Supercell dims             : {config.supercell_dims}")
    logger.info(f"  Symmetry                   : {config.symmetry}")
    logger.info(f"  Temperatures [K]           : {config.t_array}")
    logger.info(f"  T-stat damping             : {config.t_stat} fs")
    logger.info(f"  P-stat damping             : {config.p_stat} fs")
    logger.info(f"  THERMO_FREQ                : {config.thermo_freq}")
    logger.info(f"  TIMESTEP [fs]              : {config.timestep}")
    logger.info(f"  EQUILIBRATION_TEMP_STEPS   : {config.equilibration_temp_steps}")
    logger.info(f"  EQUILIBRATION_FINAL_STEPS  : {config.equilibration_final_steps}")
    logger.info(f"  PRODUCTION_STEPS           : {config.production_steps}")
    logger.info(f"  Trajectory name            : {config.traj_name}")
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

        # ---- Validate user-specified species against the potential file ----
        user_species: set = set()
        for field_val in (config.species_a, config.species_b):
            if field_val:
                for sp in field_val.split('/'):
                    sp = sp.strip()
                    if sp:
                        user_species.add(sp)

        model_charge_species = (
            {charge.species for charge in model.charges}
            if hasattr(model, 'charges') else set()
        )
        missing_species = user_species - model_charge_species
        if missing_species:
            for sp in sorted(missing_species):
                logger.warning(
                    f"\u26a0\ufe0f  Species '{sp}' entered by user was NOT found in the potential "
                    f"pickle file. It will lack shell-model parameters (charge, spring constants). "
                    f"Species available in the model: {sorted(model_charge_species)}"
                )
        else:
            logger.info(
                f"\u2713 All user-specified species {sorted(user_species)} are present in the model."
            )
        
        # Create supercell
        cell = create_supercell(
            model,
            config.supercell_dims,
            config.symmetry,
            config.material_type,
            config.species_a,
            config.species_b,
            config.position,
            config.mix_ratio
        )
        logger.info(f"✓ Created supercell with dimensions {config.supercell_dims}")
        
        # Process shell model with user-specified species
        (
            mapped_cell, string_cell,
            shell_models_data, shell_models_springs,
            shell_models_potentials, species_id_map,
        ) = process_shell_model(model, cell, config.output_filename, 
                               config.species_a, config.species_b)

        # Rename structure file
        structure_file = f"{config.output_filename}.LAMMPSStructure"
        rstrt_file = 'rstrt.dat'

        if os.path.exists(structure_file):
            if os.path.exists(rstrt_file):
                logger.warning(f"⚠️  File '{rstrt_file}' already exists - OVERWRITING")
            try:
                subprocess.run(['mv', structure_file, rstrt_file], check=True)
                logger.info(f"✓ Renamed '{structure_file}' to '{rstrt_file}'")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Error renaming file: {e}")
                try:
                    os.rename(structure_file, rstrt_file)
                    logger.info(f"  ✓ Successfully renamed using os.rename()")
                except OSError as e2:
                    logger.error(f"  ✗ Failed to rename file: {e2}")
        else:
            logger.warning(f"⚠️  Structure file '{structure_file}' not found")

        # Save species ID map
        map_filename = "species_id_map.txt"
        try:
            with open(map_filename, "w") as f:
                for (species, part), id_value in species_id_map.items():
                    f.write(f"{species} {part} {id_value}\n")
            logger.info(f"✓ Species ID map saved to {map_filename}")
        except IOError as e:
            logger.error(f"Failed to save species ID map: {e}")

        # Generate and save LAMMPS input
        lammps_content = generate_lammps_input(
            shell_models_data,
            shell_models_springs,
            shell_models_potentials,
            species_id_map,
            model_name,
            config.t_array,
            config.t_stat,
            config.p_stat,
            config.thermo_freq,
            config.timestep,
            config.equilibration_temp_steps,
            config.equilibration_final_steps,
            config.production_steps,
            config.traj_name
        )

        save_lammps_input(lammps_content)
        
        logger.info("\n" + "=" * 70)
        logger.info(" 🎉 PROCESSING COMPLETED SUCCESSFULLY! ")
        logger.info("=" * 70)
        logger.info("\n📁 Generated Files:")
        logger.info("  ✓ rstrt.dat              (LAMMPS structure file)")
        logger.info("  ✓ lammps.in              (LAMMPS input script)")
        logger.info("  ✓ species_id_map.txt     (Species ID mapping)")
        logger.info("  ✓ lammps_processing.log  (Detailed processing log)")
        logger.info("  ✓ supercell_structure.poscar  (Supercell structure)")
        logger.info("\n" + "=" * 70 + "\n")
        
    except ConfigurationError as e:
        logger.error(f"❌ Configuration error: {e}")
        sys.exit(1)
    except ModelLoadError as e:
        logger.error(f"❌ Model loading error: {e}")
        sys.exit(1)
    except ValidationError as e:
        logger.error(f"❌ Validation error: {e}")
        sys.exit(1)
    except LAMMPSProcessingError as e:
        logger.error(f"❌ Processing error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\n⏸️  Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()