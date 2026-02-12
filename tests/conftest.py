"""
Pytest configuration and fixtures for LAMMPS script tests.
"""
import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pickle

# Add parent directory to path to import the module under test
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# PICKLABLE MODEL CLASSES
# ============================================================================
class SimpleModel:
    """Simple model class for pickling in tests."""
    def __init__(self):
        self.header = ["Test Model"]
        self.charges = []
        self.AB_specie = {}
        self.springs = []
        self.potentials = []


class SimpleCharge:
    """Simple charge class for pickling in tests."""
    def __init__(self, species, part, charge):
        self.species = species
        self.part = part
        self.charge = charge


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture
def mock_model():
    """Create a mock model object with typical structure."""
    model = Mock()
    
    # Mock header
    model.header = ["Test Model"]
    
    # Mock charges
    charge1 = Mock()
    charge1.species = "Sr"
    charge1.part = "core"
    charge1.charge = 2.0
    
    charge2 = Mock()
    charge2.species = "Sr"
    charge2.part = "shell"
    charge2.charge = -0.8
    
    charge3 = Mock()
    charge3.species = "Ti"
    charge3.part = "core"
    charge3.charge = 4.0
    
    charge4 = Mock()
    charge4.species = "Ti"
    charge4.part = "shell"
    charge4.charge = -1.2
    
    charge5 = Mock()
    charge5.species = "O"
    charge5.part = "core"
    charge5.charge = 0.8
    
    charge6 = Mock()
    charge6.species = "O"
    charge6.part = "shell"
    charge6.charge = -2.8
    
    model.charges = [charge1, charge2, charge3, charge4, charge5, charge6]
    
    # Mock springs
    spring1 = Mock()
    spring1.species = "Sr"
    spring1.k2 = 100.0
    spring1.k4 = 10.0
    
    spring2 = Mock()
    spring2.species = "Ti"
    spring2.k2 = 150.0
    spring2.k4 = 15.0
    
    spring3 = Mock()
    spring3.species = "O"
    spring3.k2 = 200.0
    spring3.k4 = 20.0
    
    model.springs = [spring1, spring2, spring3]
    
    # Mock potentials
    pot1 = Mock()
    pot1.kind = "buck"
    pot1.sp1 = "Sr"
    pot1.part1 = "core"
    pot1.sp2 = "O"
    pot1.part2 = "shell"
    pot1.params = [1000.0, 0.3, 0.0]
    pot1.cutoffs = [0.0, 10.0]
    
    pot2 = Mock()
    pot2.kind = "buck"
    pot2.sp1 = "Ti"
    pot2.part1 = "core"
    pot2.sp2 = "O"
    pot2.part2 = "shell"
    pot2.params = [1500.0, 0.25, 0.0]
    pot2.cutoffs = [0.0, 10.0]
    
    model.potentials = [pot1, pot2]
    
    # Mock AB_specie
    model.AB_specie = {
        "A": ["Sr"],
        "B": ["Ti"]
    }
    
    # Mock chemical_order
    model.chemical_order = {
        "A": {"configuration": "homog"},
        "B": {"configuration": "homog"}
    }
    
    return model


@pytest.fixture
def mock_cell():
    """Create a mock cell object."""
    cell = Mock()
    cell.species_name = ["Sr", "Ti", "O"]
    cell.N = 10
    
    # Mock atoms
    atoms = []
    for i in range(5):
        atom = Mock()
        atom.name = "Sr"
        atom.position_frac = [0.0, 0.0, 0.0]
        atom.coreshell = "core"
        atoms.append(atom)
    
    for i in range(5):
        atom = Mock()
        atom.name = "Sr"
        atom.position_frac = [0.0, 0.0, 0.0]
        atom.coreshell = "shell"
        atoms.append(atom)
    
    cell.atom = atoms
    
    # Mock lattice
    cell.lattice = Mock()
    
    return cell


@pytest.fixture
def sample_pickle_file(temp_dir):
    """Create a sample pickle file for testing."""
    model = SimpleModel()
    model.charges = [
        SimpleCharge("Sr", "core", 2.0),
        SimpleCharge("Sr", "shell", -0.8),
    ]
    model.AB_specie = {"A": ["Sr"], "B": ["Ti"]}
    
    filepath = os.path.join(temp_dir, "test_model.pickle")
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    return filepath


@pytest.fixture
def sample_config():
    """Create a sample configuration dictionary."""
    from mpk_lammps_ver4 import Config
    return Config(
        model_file="./test_model.pickle",
        supercell_dims=[4, 4, 4],
        symmetry="cubic",
        output_filename="test_structure",
        t_array=[10.0, 50.0, 100.0],
        t_stat=0.1,
        p_stat=2.0
    )


@pytest.fixture
def mock_gulp_file(temp_dir):
    """Create a mock GS.gulp file."""
    gulp_content = """# Mock GULP file for testing
vectors
  4.0 0.0 0.0
  0.0 4.0 0.0
  0.0 0.0 4.0
cartesian
Sr core 0.0 0.0 0.0
O core 2.0 0.0 0.0
"""
    filepath = os.path.join(temp_dir, "GS.gulp")
    with open(filepath, "w") as f:
        f.write(gulp_content)
    return filepath
