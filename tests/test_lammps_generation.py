"""
Unit tests for LAMMPS input file generation.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpk_lammps_ver4 import (
    generate_lammps_header,
    generate_charge_settings,
    generate_group_definitions,
    generate_temperature_ramps,
    generate_lammps_input,
    save_lammps_input,
    LAMMPSProcessingError
)


class TestGenerateLammpsHeader:
    """Test LAMMPS header generation."""
    
    def test_header_contains_model_name(self):
        """Test that generated header contains the model name."""
        header = generate_lammps_header("Test Model")
        assert "Test Model" in header
    
    def test_header_contains_initialization(self):
        """Test that header contains key initialization commands."""
        header = generate_lammps_header("Model")
        assert "clear" in header
        assert "units metal" in header
        assert "dimension 3" in header
        assert "boundary p p p" in header
        assert "atom_style full" in header
    
    def test_header_contains_read_data(self):
        """Test that header contains read_data command."""
        header = generate_lammps_header("Model")
        assert "read_data rstrt.dat" in header


class TestGenerateChargeSettings:
    """Test charge settings generation."""
    
    def test_charge_settings_for_species(self):
        """Test generating charge settings for species."""
        shell_data = {
            'model': {
                'Sr': {
                    'core': {'charge': 2.0},
                    'shell': {'charge': -0.8}
                }
            }
        }
        species_map = {('Sr', 'core'): 1, ('Sr', 'shell'): 4}
        
        settings = generate_charge_settings(shell_data, species_map)
        
        assert "set type 1 charge 2.0000000" in settings
        assert "set type 4 charge -0.8000000" in settings
        assert "#Sr core" in settings
        assert "#Sr shell" in settings
    
    def test_skip_none_charges(self):
        """Test that None charges are skipped."""
        shell_data = {
            'model': {
                'Sr': {
                    'core': {'charge': None}
                }
            }
        }
        species_map = {('Sr', 'core'): 1}
        
        settings = generate_charge_settings(shell_data, species_map)
        assert "set type 1" not in settings


class TestGenerateGroupDefinitions:
    """Test group definition generation."""
    
    def test_group_definitions_separate_cores_shells(self):
        """Test that cores and shells are in separate groups."""
        species_map = {
            ('Sr', 'core'): 1,
            ('Ti', 'core'): 2,
            ('O', 'core'): 3,
            ('Sr', 'shell'): 4,
            ('Ti', 'shell'): 5,
            ('O', 'shell'): 6
        }
        
        groups = generate_group_definitions(species_map)
        
        assert "group cores type 1 2 3" in groups
        assert "group shells type 4 5 6" in groups
    
    def test_neighbor_settings_included(self):
        """Test that neighbor settings are included."""
        species_map = {('Sr', 'core'): 1, ('Sr', 'shell'): 2}
        
        groups = generate_group_definitions(species_map)
        
        assert "neighbor" in groups
        assert "neigh_modify" in groups


class TestGenerateTemperatureRamps:
    """Test temperature ramp generation."""
    
    def test_single_temperature_no_ramps(self):
        """Test that single temperature produces no ramps."""
        ramps = generate_temperature_ramps([100.0], 0.1, 2.0)
        assert ramps == ""
    
    def test_multiple_temperatures_create_ramps(self):
        """Test that multiple temperatures create ramp sections."""
        ramps = generate_temperature_ramps([10.0, 50.0, 100.0], 0.1, 2.0)
        
        assert "50" in ramps
        assert "100" in ramps
        assert "Equilibration 50.0K" in ramps  # Note: float format
        assert "Equilibration 100.0K" in ramps
        assert "Production 50.0K" in ramps
        assert "Production 100.0K" in ramps
    
    def test_ramp_includes_equilibration_steps(self):
        """Test that ramps include equilibration steps."""
        ramps = generate_temperature_ramps([10.0, 50.0], 0.1, 2.0)
        
        # Should have ramping from 10 to 50 (note: float format)
        assert "temp 10.0" in ramps  # Starting temp
        assert "temp 50.0 50.0" in ramps  # Final equilibration


class TestGenerateLammpsInput:
    """Test complete LAMMPS input generation."""
    
    def test_complete_input_generation(self):
        """Test generating complete LAMMPS input."""
        shell_data = {
            'model': {
                'Sr': {
                    'core': {'charge': 2.0},
                    'shell': {'charge': -0.8}
                }
            }
        }
        springs = {'Sr': {'k2': 100.0, 'k4': 10.0}}
        potentials = [{
            'kind': 'buck',
            'sp1': 'Sr',
            'part1': 'core',
            'sp2': 'O',
            'part2': 'shell',
            'params': [1000.0, 0.3, 0.0],
            'cutoffs': [0.0, 10.0]
        }]
        species_map = {('Sr', 'core'): 1, ('Sr', 'shell'): 2}
        
        content = generate_lammps_input(
            shell_data, springs, potentials, species_map,
            "Test Model", [100.0], 0.1, 2.0
        )
        
        assert "clear" in content
        assert "Test Model" in content
        assert "set type 1 charge" in content
        assert "pair_coeff" in content
        assert "bond_coeff" in content
        assert "run" in content


class TestSaveLammpsInput:
    """Test saving LAMMPS input to file."""
    
    def test_save_to_valid_file(self, temp_dir):
        """Test saving content to a valid file."""
        content = "# Test LAMMPS input\nclear\n"
        filepath = os.path.join(temp_dir, "test.in")
        
        save_lammps_input(content, filepath)
        
        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            saved_content = f.read()
        assert saved_content == content
    
    def test_save_to_invalid_path_raises_error(self):
        """Test that saving to invalid path raises error."""
        content = "# Test\n"
        invalid_path = "/nonexistent/directory/file.in"
        
        with pytest.raises(LAMMPSProcessingError, match="Error writing"):
            save_lammps_input(content, invalid_path)
