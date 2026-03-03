# Test Suite for LAMMPS Shell Model Processing Script

## Overview

This directory contains comprehensive tests for `mpk_lammps_ver4.py` (**v4.5**),
covering configuration validation, model loading, shell model processing, and
LAMMPS input generation.

The test suite consists of **70 tests** with 100% pass rate, validating all
critical workflows and edge cases.

## Test Structure

```text
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Pytest fixtures and mocked external dependencies
├── test_config.py           # Configuration validation (17 tests)
├── test_integration.py      # End-to-end integration tests (5 tests)
├── test_lammps_generation.py # LAMMPS input generation (16 tests)
├── test_model_loading.py    # Model file loading (6 tests)
├── test_shell_model.py      # Shell model data extraction (21 tests)
└── test_utilities.py        # Utility function tests (6 tests)
```

### Installation

1. **Activate the Python environment:**

   ```bash
   source ~/python_env/py311/bin/activate
   ```

2. **Install test dependencies:**

   ```bash
   pip install -r requirements-test.txt
   ```

   Or install individually:

   ```bash
   pip install pytest pytest-cov pytest-mock
   ```

## Running Tests

### Run all tests

```bash
pytest -v
```

### Run with verbose output and coverage

```bash
pytest -v --cov=mpk_lammps_ver4 --cov-report=term-missing
```

### Run specific test file

```bash
pytest tests/test_config.py -v
```

### Run specific test class

```bash
pytest tests/test_config.py::TestConfigClass -v
```

### Run specific test

```bash
pytest tests/test_config.py::TestConfigClass::test_valid_config_passes_validation -v
```

### Run with coverage report (HTML)

```bash
pytest --cov=mpk_lammps_ver4 --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Run only unit tests

```bash
pytest -m unit
```

### Run only integration tests

```bash
pytest -m integration
```

## Test Categories

### Unit Tests

- **`test_config.py`** (17 tests): Configuration validation
  - Default and custom values
  - All parameter validations (dimensions, symmetry, temperature, etc.)
  - Material type constraints (pure vs. mix)
  - Error conditions and edge cases

- **`test_utilities.py`** (6 tests): Utility functions
  - Mixing type determination (`get_mixing_type()`)
  - Support for homogeneous, G-type, and 1/4 ordering
  - Error handling for invalid inputs

- **`test_model_loading.py`** (6 tests): Model file loading
  - Valid model loading from pickle files
  - Error handling (missing files, corrupted data)
  - Model structure validation
  - Handling of missing or empty charges

- **`test_shell_model.py`** (21 tests): Shell model processing
  - Data extraction from model objects
  - Species ID mapping (`create_species_id_map`)
  - **Species ID map revision** (`revise_species_id_map` — 5 tests):
    - Valid species are preserved with their original ordering
    - Invalid species (not in `model.charges`) are removed with a `WARNING` log
    - IDs are reassigned sequentially after removals (no gaps)
    - Empty model charges returns only oxygen
    - No warning emitted when all species are valid
  - Data validation: when any value is `None`, all values in that part become `None`
  - Option B: warns for unknown species rather than crashing
  - String cell creation: `shell_models['model']` keys stay as **integers**
  - Nomenclature normalization (`shel` → `shell`)
  - Handling of unknown particle types
  - `_sanitize_shell_model_for_writing` is exercised indirectly through
    the integration test and `process_shell_model` call chain

- **`test_lammps_generation.py`** (16 tests): LAMMPS input generation
  - Header generation with model metadata
  - Charge settings: `generate_charge_settings(cell, species_id_map)` reads from
    `cell.shell_models['model']` (integer-keyed by type ID)
  - Group definitions (cores vs. shells)
  - Temperature ramps with equilibration stages
  - Neighbor list settings
  - Complete input: `generate_lammps_input(cell, ...)` — first arg is a cell object
  - File I/O validation

### Integration Tests

- **`test_integration.py`** (5 tests): End-to-end workflows
  - Complete processing pipeline (load → extract → generate → save)
  - Configuration-based workflow execution
  - Configuration validation with actual usage
  - Error handling in realistic scenarios

## Test Fixtures

Located in `conftest.py`:

- **`temp_dir`**: Temporary directory for test files
- **`mock_model`**: Mock model object with typical structure
- **`mock_cell`**: Mock cell object for testing
- **`sample_pickle_file`**: Sample pickle file for testing
- **`sample_config`**: Sample configuration object
- **`mock_gulp_file`**: Mock GS.gulp file

> **Note on `sys.path` (v4.5 change):** The following eight `sys.path.append`
> lines were removed from `mpk_lammps_ver4.py` in v4.5:
>
> ```python
> sys.path.append(r'../lib/pm__py/pm__cell/')
> sys.path.append(r'../lib/pm__py/pm__constants/')
> sys.path.append(r'../lib/pm__py/pm__data_from_file/')
> sys.path.append(r'../lib/pm__py/pm__utilities/')
> sys.path.append(r"../lib/mk__py/")
> sys.path.append(r"../lib/pm__py/pm__chemical_order/")
> sys.path.append(r"../lib/transformations/")
> sys.path.append(r"../lib/pm__py/pm__shell_model_kit/")
> ```
>
> These deployment-specific path hacks no longer exist in the source.
> The `pm__` packages must now be installed in your environment or present on
> `PYTHONPATH`. The test suite is unaffected: `pm__cell`, `pm__chemical_order`,
> and `pm__shell_model_kit` continue to be mocked at the `sys.modules` level
> in `conftest.py`, so no physical paths are needed to run the tests.

## Test Coverage

The test suite comprehensively covers:

- ✅ Configuration validation (all parameters and constraints)
- ✅ Material type handling (pure and mixed compositions)
- ✅ Utility functions (mixing type determination)
- ✅ Model loading (valid files and error cases)
- ✅ Shell model data extraction and validation
- ✅ Species ID mapping, revision, and normalization
- ✅ LAMMPS input generation (all sections)
- ✅ File I/O operations
- ✅ Error handling and custom exceptions
- ✅ Integration workflows (end-to-end processing)

## Writing New Tests

### Example Unit Test

```python
import pytest
from mpk_lammps_ver4 import Config, ConfigurationError

def test_negative_temperature_fails():
    """Test that negative temperature fails validation."""
    config = Config(
        t_array=[100.0, -50.0],
        material_type="pure",
        species_a="Sr",
        species_b="Ti"
    )
    with pytest.raises(ConfigurationError, match="must be non-negative"):
        config.validate()
```

### Example Integration Test

```python
def test_complete_workflow(sample_pickle_file, temp_dir, mock_cell):
    """Test complete processing workflow."""
    from unittest.mock import Mock

    # Load and process model
    model = load_model(sample_pickle_file)
    shell_data, springs, potentials = extract_shell_model_data(model)

    # Create species map
    species_map = create_species_id_map(mock_cell, model)

    # generate_lammps_input expects a *cell* object with .shell_models
    mock_input_cell = Mock()
    mock_input_cell.shell_models = shell_data

    content = generate_lammps_input(
        mock_input_cell, springs, potentials, species_map,
        "Test Model", [100.0], 0.1, 2.0,
        500, 0.0002, 20000, 30000, 50000, "traj"
    )

    # Save and verify
    output_file = os.path.join(temp_dir, "test_lammps.in")
    save_lammps_input(content, output_file)
    assert os.path.exists(output_file)
```

## Continuous Integration

To run tests with comprehensive reporting:

```bash
# Run all tests with coverage and missing line report
pytest --cov=mpk_lammps_ver4 --cov-report=term-missing

# Run with strict mode (all warnings as errors)
pytest --strict-warnings -v

# Generate both terminal and HTML coverage reports
pytest --cov=mpk_lammps_ver4 --cov-report=term-missing --cov-report=html

# Run tests with detailed failure information
pytest -v --tb=long

# Run tests in parallel for faster execution (if pytest-xdist is installed)
pytest -n auto
```

### GitHub Actions / CI/CD Integration

Add to your CI/CD pipeline:

```yaml
- name: Run Tests
  run: pytest -v --cov=mpk_lammps_ver4 --cov-report=term-missing

- name: Generate Coverage Report
  run: pytest --cov=mpk_lammps_ver4 --cov-report=html
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure the parent directory is in Python path:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### Module Name Issues

Note: The module name in imports should match your file name:

- File: `mpk_lammps_ver4.py`
- Import as: `mpk_lammps_ver4` (underscore replaces hyphen)

### Fixture Not Found

If pytest can't find fixtures, ensure `conftest.py` is in the `tests/` directory.

## Best Practices

1. **Test Independence**: Each test should be independent
2. **Use Fixtures**: Use fixtures for common setup
3. **Clear Names**: Test names should describe what they test
4. **One Assertion**: Focus each test on one behavior
5. **Mock External Dependencies**: Mock file I/O, network calls, etc.

## Future Enhancements

Potential additions:

- [ ] Performance benchmarks
- [ ] Property-based testing with hypothesis
- [ ] Mutation testing with mutpy
- [ ] Integration with CI/CD pipeline
- [ ] Snapshot testing for generated files
