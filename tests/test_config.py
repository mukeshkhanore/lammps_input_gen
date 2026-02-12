"""
Unit tests for Configuration class and validation.
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpk_lammps_ver4 import Config, ConfigurationError


class TestConfigClass:
    """Test the Config dataclass."""
    
    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = Config()
        assert config.model_file == "./potential.pickle"
        assert config.supercell_dims == [8, 8, 8]
        assert config.symmetry == "file"
        assert config.t_array == [10.0]
        assert config.t_stat == 0.1
        assert config.p_stat == 2.0
    
    def test_config_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = Config(
            model_file="custom.pickle",
            supercell_dims=[6, 6, 6],
            symmetry="cubic",
            t_array=[100.0, 200.0],
            t_stat=0.5,
            p_stat=3.0
        )
        assert config.model_file == "custom.pickle"
        assert config.supercell_dims == [6, 6, 6]
        assert config.symmetry == "cubic"
        assert config.t_array == [100.0, 200.0]
        assert config.t_stat == 0.5
        assert config.p_stat == 3.0
    
    def test_valid_config_passes_validation(self):
        """Test that a valid config passes validation."""
        config = Config(
            model_file="test.pickle",
            supercell_dims=[4, 4, 4],
            symmetry="cubic",
            t_array=[50.0],
            t_stat=0.2,
            p_stat=1.5
        )
        # Should not raise any exception
        config.validate()
    
    def test_empty_model_file_fails_validation(self):
        """Test that empty model file fails validation."""
        config = Config(model_file="")
        with pytest.raises(ConfigurationError, match="Model file path cannot be empty"):
            config.validate()
    
    def test_wrong_number_of_dimensions_fails(self):
        """Test that wrong number of dimensions fails validation."""
        config = Config(supercell_dims=[4, 4])  # Only 2 dimensions
        with pytest.raises(ConfigurationError, match="exactly 3 values"):
            config.validate()
    
    def test_negative_dimensions_fail(self):
        """Test that negative dimensions fail validation."""
        config = Config(supercell_dims=[4, -4, 4])
        with pytest.raises(ConfigurationError, match="must be positive"):
            config.validate()
    
    def test_zero_dimension_fails(self):
        """Test that zero dimension fails validation."""
        config = Config(supercell_dims=[0, 4, 4])
        with pytest.raises(ConfigurationError, match="must be positive"):
            config.validate()
    
    def test_invalid_symmetry_type(self):
        """Test that invalid symmetry type fails validation."""
        config = Config(symmetry="invalid")
        with pytest.raises(ConfigurationError, match="Symmetry must be one of"):
            config.validate()
    
    def test_valid_symmetry_types(self):
        """Test that all valid symmetry types pass."""
        for sym in ["cubic", "random", "file"]:
            config = Config(symmetry=sym)
            config.validate()  # Should not raise
    
    def test_empty_temperature_array_fails(self):
        """Test that empty temperature array fails validation."""
        config = Config(t_array=[])
        with pytest.raises(ConfigurationError, match="Temperature array cannot be empty"):
            config.validate()
    
    def test_negative_temperature_fails(self):
        """Test that negative temperature fails validation."""
        config = Config(t_array=[100.0, -50.0])
        with pytest.raises(ConfigurationError, match="must be non-negative"):
            config.validate()
    
    def test_zero_temperature_is_valid(self):
        """Test that zero temperature is valid (absolute zero)."""
        config = Config(t_array=[0.0])
        config.validate()  # Should not raise
    
    def test_negative_t_stat_fails(self):
        """Test that negative thermostat damping fails."""
        config = Config(t_stat=-0.1)
        with pytest.raises(ConfigurationError, match="Thermostat damping.*must be positive"):
            config.validate()
    
    def test_zero_t_stat_fails(self):
        """Test that zero thermostat damping fails."""
        config = Config(t_stat=0.0)
        with pytest.raises(ConfigurationError, match="Thermostat damping.*must be positive"):
            config.validate()
    
    def test_negative_p_stat_fails(self):
        """Test that negative barostat damping fails."""
        config = Config(p_stat=-2.0)
        with pytest.raises(ConfigurationError, match="Barostat damping.*must be positive"):
            config.validate()
    
    def test_zero_p_stat_fails(self):
        """Test that zero barostat damping fails."""
        config = Config(p_stat=0.0)
        with pytest.raises(ConfigurationError, match="Barostat damping.*must be positive"):
            config.validate()
    
    def test_multiple_temperatures_valid(self):
        """Test that multiple temperatures are valid."""
        config = Config(t_array=[10.0, 50.0, 100.0, 200.0])
        config.validate()  # Should not raise
