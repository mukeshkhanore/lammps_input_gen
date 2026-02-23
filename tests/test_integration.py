"""
Integration tests for complete LAMMPS processing workflow.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpk_lammps_ver4 import (
    load_model,
    extract_shell_model_data,
    create_species_id_map,
    generate_lammps_input,
    save_lammps_input,
    Config
)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""
    
    def test_load_and_generate_workflow(self, sample_pickle_file, temp_dir, mock_cell):
        """Test loading model and generating LAMMPS input."""
        # Load model
        model = load_model(sample_pickle_file)
        assert model is not None
        
        # Extract shell model data
        shell_data, springs, potentials = extract_shell_model_data(model)
        assert len(shell_data['model']) > 0
        
        # Create species ID map
        species_map = create_species_id_map(mock_cell, model)
        assert len(species_map) > 0
        
        # Generate LAMMPS input
        content = generate_lammps_input(
            shell_data, springs, potentials, species_map,
            "Test Model", [100.0], 0.1, 2.0,
            500, 0.0002, 20000, 30000, 50000, "traj"
        )
        assert len(content) > 0
        assert "clear" in content
        
        # Save to file
        output_file = os.path.join(temp_dir, "test_lammps.in")
        save_lammps_input(content, output_file)
        assert os.path.exists(output_file)
        
        # Verify file contents
        with open(output_file, 'r') as f:
            saved = f.read()
        assert saved == content


class TestConfigurationWorkflow:
    """Test configuration-based workflow."""
    
    def test_config_validation_and_usage(self):
        """Test creating and validating configuration."""
        config = Config(
            model_file="test.pickle",
            supercell_dims=[6, 6, 6],
            symmetry="cubic",
            t_array=[50.0, 100.0],
            t_stat=0.2,
            p_stat=1.5
        )
        
        # Validation should pass
        config.validate()
        
        # Config values should be accessible
        assert config.supercell_dims == [6, 6, 6]
        assert len(config.t_array) == 2
    
    def test_invalid_config_fails_early(self):
        """Test that invalid config fails during validation."""
        from mpk_lammps_ver4 import ConfigurationError
        
        config = Config(supercell_dims=[4, -2, 4])
        
        with pytest.raises(ConfigurationError):
            config.validate()


class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    def test_missing_model_file_handled(self):
        """Test that missing model file is handled gracefully."""
        from mpk_lammps_ver4 import ModelLoadError
        
        with pytest.raises(ModelLoadError, match="not found"):
            load_model("/nonexistent/model.pickle")
    
    def test_invalid_model_structure_handled(self, temp_dir):
        """Test that invalid model structure is caught."""
        import pickle
        from mpk_lammps_ver4 import ModelLoadError
        
        # Create invalid model (missing charges)
        invalid_model = {}
        filepath = os.path.join(temp_dir, "invalid.pickle")
        with open(filepath, 'wb') as f:
            pickle.dump(invalid_model, f)
        
        with pytest.raises(ModelLoadError, match="missing required"):
            load_model(filepath)
