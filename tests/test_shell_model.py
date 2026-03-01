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
    revise_species_id_map,
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


class TestReviseSpeciesIdMap:
    """Tests for revise_species_id_map — removes species absent from model potential."""

    def _make_model(self, species_list):
        """Return a mock model whose .charges covers exactly species_list."""
        model = Mock()
        charges = []
        for sp in species_list:
            for part in ("core", "shell"):
                c = Mock()
                c.species = sp
                c.part = part
                c.charge = 1.0
                charges.append(c)
        model.charges = charges
        return model

    def test_valid_species_unchanged(self):
        """All species in the map that are in the model are preserved."""
        model = self._make_model(["Sr", "Ti", "O"])
        species_id_map = {
            ('Sr', 'core'): 1, ('Ti', 'core'): 2, ('O', 'core'): 3,
            ('Sr', 'shell'): 4, ('Ti', 'shell'): 5, ('O', 'shell'): 6,
        }
        revised = revise_species_id_map(species_id_map, model)

        assert ('Sr', 'core') in revised
        assert ('Ti', 'core') in revised
        assert ('O', 'core') in revised
        assert ('Sr', 'shell') in revised
        assert ('Ti', 'shell') in revised
        assert ('O', 'shell') in revised

    def test_invalid_species_removed_with_warning(self, caplog):
        """Species not in model.charges are dropped and a WARNING is logged."""
        import logging
        model = self._make_model(["Sr", "Ti", "O"])
        # Hf is NOT in the model
        species_id_map = {
            ('Hf', 'core'): 1, ('Sr', 'core'): 2, ('O', 'core'): 3,
            ('Hf', 'shell'): 4, ('Sr', 'shell'): 5, ('O', 'shell'): 6,
        }
        with caplog.at_level(logging.WARNING):
            revised = revise_species_id_map(species_id_map, model)

        # Hf must be gone
        assert ('Hf', 'core') not in revised
        assert ('Hf', 'shell') not in revised
        # Valid species must still be present
        assert ('Sr', 'core') in revised
        assert ('O', 'core') in revised
        # A WARNING mentioning 'Hf' must have been emitted
        assert any("Hf" in msg for msg in caplog.messages), (
            "Expected WARNING mentioning 'Hf'"
        )

    def test_ids_are_reassigned_sequentially(self):
        """After dropping species the returned IDs must be 1-based and sequential."""
        model = self._make_model(["Sr", "O"])
        # Map originally has Ba (absent from model), Sr, and O
        species_id_map = {
            ('Ba', 'core'): 1, ('Sr', 'core'): 2, ('O', 'core'): 3,
            ('Ba', 'shell'): 4, ('Sr', 'shell'): 5, ('O', 'shell'): 6,
        }
        revised = revise_species_id_map(species_id_map, model)

        # Only Sr and O remain → core IDs should be 1,2 and shell IDs 3,4
        core_ids = sorted(v for (_, part), v in revised.items() if part == 'core')
        shell_ids = sorted(v for (_, part), v in revised.items() if part == 'shell')
        assert core_ids == list(range(1, len(core_ids) + 1))
        assert shell_ids == list(range(len(core_ids) + 1, len(core_ids) + len(shell_ids) + 1))

    def test_empty_model_charges_returns_only_oxygen(self):
        """When model has no charges only 'O' (always valid) survives."""
        model = Mock()
        model.charges = []
        species_id_map = {
            ('Sr', 'core'): 1, ('O', 'core'): 2,
            ('Sr', 'shell'): 3, ('O', 'shell'): 4,
        }
        revised = revise_species_id_map(species_id_map, model)
        # 'O' is excluded from the "missing" check so it stays
        assert ('O', 'core') in revised
        assert ('Sr', 'core') not in revised

    def test_no_warning_when_all_species_valid(self, caplog):
        """No WARNING should be emitted when every species is in the model."""
        import logging
        model = self._make_model(["Sr", "Ti", "O"])
        species_id_map = {
            ('Sr', 'core'): 1, ('Ti', 'core'): 2, ('O', 'core'): 3,
            ('Sr', 'shell'): 4, ('Ti', 'shell'): 5, ('O', 'shell'): 6,
        }
        with caplog.at_level(logging.WARNING):
            revise_species_id_map(species_id_map, model)

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) == 0, (
            f"No warnings expected for valid species. Got: {[r.message for r in warning_msgs]}"
        )


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

    def test_incomplete_data_sets_all_to_none(self):
        """When any value in a core/shell dict is None, ALL are set to None.

        validate_shell_model_data contract: if any value is None, set all
        values in that part to None for a consistent sentinel.
        """
        shell_data = {
            'model': {
                1: {
                    'core':  {'mass': 87.62, 'charge': None},   # charge=None → all become None
                    'shell': {'mass': 1.79,  'charge': -0.8}    # fully populated → unchanged
                },
                2: {
                    'core':  {'mass': 47.87, 'charge': 4.0},    # fully populated → unchanged
                    'shell': {'mass': None,  'charge': -1.2}    # mass=None → all become None
                }
            }
        }

        result = validate_shell_model_data(shell_data)

        # ID 1 / core: had charge=None → both mass and charge become None
        assert result['model'][1]['core']['charge'] is None
        assert result['model'][1]['core']['mass'] is None

        # ID 1 / shell: was complete → still complete
        assert result['model'][1]['shell']['charge'] == -0.8
        assert result['model'][1]['shell']['mass'] == 1.79

        # ID 2 / core: was complete → still complete
        assert result['model'][2]['core']['charge'] == 4.0
        assert result['model'][2]['core']['mass'] == 47.87

        # ID 2 / shell: had mass=None → both mass and charge become None
        assert result['model'][2]['shell']['mass'] is None
        assert result['model'][2]['shell']['charge'] is None


# =============================================================================
# TESTS: Species-not-in-model edge cases (Option B behaviour)
# =============================================================================

class TestSpeciesNotInModel:
    """
    Test Option B behaviour: when user specifies a species not in the model,
    functions should warn clearly and continue with None values rather than
    crashing or silently producing wrong data.
    """

    def _make_model_with_limited_species(self):
        """Return a mock model that only knows about Sr, Ti, O."""
        from unittest.mock import Mock
        model = Mock()
        model.AB_specie = {"A": ["Sr"], "B": ["Ti"]}

        def _charge(sp, part, val):
            c = Mock()
            c.species = sp
            c.part = part
            c.charge = val
            return c

        model.charges = [
            _charge("Sr", "core",  2.0),
            _charge("Sr", "shell", -0.8),
            _charge("Ti", "core",  4.0),
            _charge("Ti", "shell", -1.2),
            _charge("O",  "core",  0.8),
            _charge("O",  "shell", -2.8),
        ]
        model.springs = []
        model.potentials = []
        return model

    # ------------------------------------------------------------------
    # 1. create_species_id_map: warns but still builds map (Option B)
    # ------------------------------------------------------------------
    def test_create_species_id_map_warns_for_unknown_species(self, mock_cell, caplog):
        """Unknown species should trigger a WARNING but not raise an exception."""
        import logging
        model = self._make_model_with_limited_species()

        with caplog.at_level(logging.WARNING):
            # Hf is NOT in the model — Option B: should warn, not raise
            result = create_species_id_map(mock_cell, model,
                                           species_a="Hf", species_b="Ti")

        # Map was still returned
        assert isinstance(result, dict)
        assert ("Hf", "core") in result or len(result) > 0

        # A WARNING was emitted
        assert any("Hf" in msg for msg in caplog.messages), (
            "Expected a WARNING mentioning 'Hf' for the unknown species"
        )

    def test_create_species_id_map_no_warning_for_valid_species(self, mock_cell, caplog):
        """Known species should not produce any WARNING."""
        import logging
        model = self._make_model_with_limited_species()

        with caplog.at_level(logging.WARNING):
            create_species_id_map(mock_cell, model,
                                  species_a="Sr", species_b="Ti")

        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) == 0, (
            f"No warnings expected for known species, got: {[m.message for m in warning_msgs]}"
        )

    # ------------------------------------------------------------------
    # 2. map_charges_to_new_model: skips with warning for unknown element
    # ------------------------------------------------------------------
    def test_map_charges_warns_and_skips_invalid_element(self, caplog):
        """
        If shell_models_data contains a species symbol that mendeleev cannot
        look up, the function should warn and skip (not raise ValidationError).
        """
        import logging
        from mpk_lammps_ver4 import map_charges_to_new_model

        shell_models_data = {
            'model': {
                'XX': {  # 'XX' is not a real element
                    'core':  {'mass': 0.0, 'charge': None},
                    'shell': {'mass': 0.0, 'charge': None},
                }
            }
        }
        species_id_map = {('XX', 'core'): 1, ('XX', 'shell'): 4}
        shell_models_data_new = {
            'model': {
                1: {'core': {'mass': 0.0, 'charge': None}, 'shell': {'mass': 0.0, 'charge': None}},
                4: {'core': {'mass': 0.0, 'charge': None}, 'shell': {'mass': 0.0, 'charge': None}},
            }
        }

        with caplog.at_level(logging.WARNING):
            result = map_charges_to_new_model(
                shell_models_data, species_id_map, shell_models_data_new
            )

        # Should NOT raise — Option B
        assert result is not None

        # A WARNING mentioning 'XX' must have been emitted
        assert any("XX" in msg for msg in caplog.messages), (
            "Expected a WARNING mentioning 'XX' for the invalid element symbol"
        )

        # Values should remain None (not overwritten with garbage)
        assert result['model'][1]['core']['charge'] is None

    # ------------------------------------------------------------------
    # 3. create_string_named_cell: shell_models keys remain integers
    # ------------------------------------------------------------------
    def test_create_string_named_cell_keeps_integer_keys(self):
        """
        create_string_named_cell converts atom *names* to strings but intentionally
        KEEPS the shell_models['model'] keys as integers so that
        pm__cell.writeToLAMMPSStructure (which accesses them via integer index
        ii+1) does not break.
        """
        from mpk_lammps_ver4 import create_string_named_cell
        from unittest.mock import MagicMock, patch
        import copy

        # Build a minimal numeric_cell mock
        numeric_cell = MagicMock()
        numeric_cell.lattice = MagicMock()
        numeric_cell.N = 2
        numeric_cell.shell_models = {
            'model': {
                1: {'core': {'mass': 85.47, 'charge': 2.0}, 'shell': {'mass': 1.74, 'charge': -0.5}},
                2: {'core': {'mass': 47.87, 'charge': 4.0}, 'shell': {'mass': 0.97, 'charge': -1.2}},
            }
        }
        atom1 = MagicMock()
        atom1.name = 1
        atom1.position_frac = [0.0, 0.0, 0.0]
        atom1.coreshell = "core"
        atom2 = MagicMock()
        atom2.name = 2
        atom2.position_frac = [0.5, 0.5, 0.5]
        atom2.coreshell = "core"
        numeric_cell.atom = [atom1, atom2]

        with patch('mpk_lammps_ver4.pmc') as mock_pmc:
            str_cell_mock = MagicMock()
            str_cell_mock.N = 2
            str_cell_mock.atom = []
            mock_pmc.Cell.return_value = str_cell_mock

            create_string_named_cell(numeric_cell)

        # The source dict's keys must still be integers (the function deep-copies
        # without converting keys, intentionally).
        original_keys = list(numeric_cell.shell_models['model'].keys())
        for key in original_keys:
            assert isinstance(key, int), (
                f"Shell model key {key!r} should remain an integer "
                f"(writeToLAMMPSStructure accesses them by integer index)"
            )
        assert 1 in numeric_cell.shell_models['model']
        assert 2 in numeric_cell.shell_models['model']

    # ------------------------------------------------------------------
    # 4. validate_shell_model_data: must emit WARNING for None values
    # ------------------------------------------------------------------
    def test_validate_shell_model_data_emits_warning_for_none_data(self, caplog):
        """
        Species with None charge/mass should trigger a WARNING (not just DEBUG)
        so the user sees that a species is missing model parameters.
        """
        import logging
        from mpk_lammps_ver4 import validate_shell_model_data

        shell_data = {
            'model': {
                # ID 99 simulates an unknown species with no parameters
                99: {
                    'core':  {'mass': None, 'charge': None},
                    'shell': {'mass': None, 'charge': None},
                }
            }
        }

        with caplog.at_level(logging.WARNING):
            validate_shell_model_data(shell_data)

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) > 0, (
            "Expected at least one WARNING for species with None parameters"
        )
        assert any("99" in r.message for r in warning_records), (
            "WARNING message should identify the problematic species ID (99)"
        )

    # ------------------------------------------------------------------
    # 5. create_mapped_cell: warns when species has NO atoms in cell
    # ------------------------------------------------------------------
    def test_create_mapped_cell_warns_for_species_with_no_atoms(self, caplog):
        """
        If a species appears in species_id_map but has zero atoms in the
        actual cell, a WARNING should be emitted (Option B: don't crash).
        """
        import logging
        from mpk_lammps_ver4 import create_mapped_cell
        from unittest.mock import MagicMock, patch

        # species_id_map contains 'Hf' but the cell has no Hf atoms
        species_id_map = {
            ('Hf', 'core'): 1, ('Hf', 'shell'): 2,
            ('O',  'core'): 3, ('O',  'shell'): 4,
        }

        original_cell = MagicMock()
        original_cell.lattice = MagicMock()

        # Only O atoms — no Hf
        o_core = MagicMock()
        o_core.name = "O"
        o_core.coreshell = "core"
        o_core.position_frac = [0.5, 0.5, 0.5]
        original_cell.atom = [o_core]

        with patch('mpk_lammps_ver4.pmc') as mock_pmc:
            new_cell_mock = MagicMock()
            new_cell_mock.N = 1
            mock_pmc.Cell.return_value = new_cell_mock

            with caplog.at_level(logging.WARNING):
                create_mapped_cell(original_cell, species_id_map)

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Hf" in msg for msg in warning_msgs), (
            f"Expected a WARNING about 'Hf' having no atoms in the cell. Got: {warning_msgs}"
        )
