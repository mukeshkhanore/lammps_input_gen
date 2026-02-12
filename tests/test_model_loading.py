"""
Unit tests for model loading functionality.
"""
import pytest
import sys
import os
import pickle
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpk_lammps_ver4 import load_model, ModelLoadError
from unittest.mock import Mock


# Define picklable model classes for testing
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


class TestLoadModel:
    """Test the load_model function."""
    
    def test_load_valid_model(self, sample_pickle_file):
        """Test loading a valid model from pickle file."""
        model = load_model(sample_pickle_file)
        assert model is not None
        assert hasattr(model, 'charges')
    
    def test_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises ModelLoadError."""
        with pytest.raises(ModelLoadError, match="Model file not found"):
            load_model("/nonexistent/path/model.pickle")
    
    def test_directory_instead_of_file_raises_error(self, temp_dir):
        """Test that providing a directory instead of file raises error."""
        with pytest.raises(ModelLoadError, match="not a file"):
            load_model(temp_dir)
    
    def test_corrupted_pickle_raises_error(self, temp_dir):
        """Test that corrupted pickle file raises error."""
        corrupted_file = os.path.join(temp_dir, "corrupted.pickle")
        with open(corrupted_file, "wb") as f:
            f.write(b"This is not a valid pickle file")
        
        with pytest.raises(ModelLoadError, match="Failed to unpickle"):
            load_model(corrupted_file)
    
    def test_model_without_charges_raises_error(self, temp_dir):
        """Test that model without charges attribute raises error."""
        model = SimpleModel()
        # Don't add charges attribute by manually deleting it
        del model.charges
        model.header = ["Test"]
        
        invalid_model_file = os.path.join(temp_dir, "invalid.pickle")
        with open(invalid_model_file, "wb") as f:
            pickle.dump(model, f)
        
        with pytest.raises(ModelLoadError, match="missing required 'charges' attribute"):
            load_model(invalid_model_file)
    
    def test_model_with_empty_charges(self, temp_dir):
        """Test loading model with empty charges list (should succeed)."""
        model = SimpleModel()
        model.charges = []  # Empty but attribute exists
        model.header = ["Test"]
        
        model_file = os.path.join(temp_dir, "empty_charges.pickle")
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        
        loaded = load_model(model_file)
        assert loaded.charges == []
