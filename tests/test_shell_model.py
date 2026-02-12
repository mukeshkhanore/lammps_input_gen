"""
Unit tests for shell model data extraction and processing.
"""
import pytest
import sys
import os
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpk_lammps_ver4 import (
    extract_shell_model_data, 
    create_species_id_map,
    validate_shell_model_data,
    ValidationError,
    CORE_MASS_RATIO,
    SHELL_MASS_RATIO
)


class TestExtractShellModelData:
    """Test shell model data extraction."""
    
    def test_extract_from_valid_model(self, mock_model):
        """Test extracting data from a valid model."""
        shell_data, springs, potentials = extract_shell_model_data(mock_model)
        
        assert 'model' in shell_data
        assert 'Sr' in shell_data['model']
        assert 'Ti' in shell_data['model']
        assert 'O' in shell_data['model']
        
        # Check springs
        assert 'Sr' in springs
        assert springs['Sr']['k2'] == 100.0
        assert springs['Sr']['k4'] == 10.0
        
        # Check potentials
        assert len(potentials) == 2
        assert potentials[0]['kind'] == 'buck'
    
    def test_model_without_charges_raises_error(self):
        """Test that model without charges raises ValidationError."""
        model = Mock()
        model.charges = None
        
        with pytest.raises(ValidationError, match="no charge information"):
            extract_shell_model_data(model)
    
    def test_model_with_empty_charges_raises_error(self):
        """Test that model with empty charges raises ValidationError."""
        model = Mock()
        model.charges = []
        
        with pytest.raises(ValidationError, match="no charge information"):
            extract_shell_model_data(model)
    
    def test_shel_normalized_to_shell(self):
        """Test that 'shel' is normalized to 'shell'."""
        model = Mock()
        charge = Mock()
        charge.species = "Sr"
        charge.part = "shel"  # Old notation
        charge.charge = 1.0
        model.charges = [charge]
        model.springs = []
        model.potentials = []
        
        shell_data, _, _ = extract_shell_model_data(model)
        # Should have 'shell' not 'shel'
        assert 'shell' in shell_data['model']['Sr']
    
    def test_unknown_part_type_is_skipped(self):
        """Test that unknown part types are skipped with warning."""
        model = Mock()
        charge1 = Mock()
        charge1.species = "Sr"
        charge1.part = "unknown_part"
        charge1.charge = 1.0
        
        charge2 = Mock()
        charge2.species = "Sr"
        charge2.part = "core"
        charge2.charge = 2.0
        
        model.charges = [charge1, charge2]
        model.springs = []
        model.potentials = []
        
        shell_data, _, _ = extract_shell_model_data(model)
        # Should still work, just skip the unknown part
        assert 'Sr' in shell_data['model']


class TestCreateSpeciesIdMap:
    """Test species ID mapping creation."""
    
    def test_create_mapping_for_valid_cell_and_model(self, mock_cell, mock_model):
        """Test creating species ID map from valid cell and model."""
        species_map = create_species_id_map(mock_cell, mock_model)
        
        # Check that we have mappings for A-site, B-site, and O
        assert ('Sr', 'core') in species_map
        assert ('Sr', 'shell') in species_map
        assert ('Ti', 'core') in species_map
        assert ('Ti', 'shell') in species_map
        assert ('O', 'core') in species_map
        assert ('O', 'shell') in species_map
        
        # Check that IDs are unique
        ids = list(species_map.values())
        assert len(ids) == len(set(ids))  # All unique
    
    def test_model_without_ab_specie_raises_error(self, mock_cell):
        """Test that model without AB_specie raises ValidationError."""
        model = Mock(spec=['header'])  # Only allow header attribute
        # Missing AB_specie attribute
        
        with pytest.raises(ValidationError, match="missing AB_specie"):
            create_species_id_map(mock_cell, model)
    
    def test_oxygen_always_added_to_species_list(self, mock_cell):
        """Test that oxygen is always added even if not in AB_specie."""
        model = Mock()
        model.AB_specie = {"A": ["Sr"], "B": ["Ti"]}
        
        species_map = create_species_id_map(mock_cell, model)
        
        assert ('O', 'core') in species_map
        assert ('O', 'shell') in species_map


class TestValidateShellModelData:
    """Test shell model data validation."""
    
    def test_complete_data_unchanged(self):
        """Test that complete data passes validation unchanged."""
        shell_data = {
            'model': {
                1: {
                    'core': {'mass': 87.62, 'charge': 2.0},
                    'shell': {'mass': 1.79, 'charge': -0.8}
                }
            }
        }
        
        result = validate_shell_model_data(shell_data)
        assert result['model'][1]['core']['mass'] == 87.62
        assert result['model'][1]['core']['charge'] == 2.0
    
    def test_incomplete_data_set_to_none(self):
        """Test that incomplete data is set to None for consistency."""
        shell_data = {
            'model': {
                1: {
                    'core': {'mass': 87.62, 'charge': None},  # Missing charge
                    'shell': {'mass': None, 'charge': -0.8}   # Missing mass
                }
            }
        }
        
        result = validate_shell_model_data(shell_data)
        
        # All attributes for core should be None
        assert result['model'][1]['core']['mass'] is None
        assert result['model'][1]['core']['charge'] is None
        
        # All attributes for shell should be None
        assert result['model'][1]['shell']['mass'] is None
        assert result['model'][1]['shell']['charge'] is None
