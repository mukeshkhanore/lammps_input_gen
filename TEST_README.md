# Test Suite for LAMMPS Shell Model Processing Script

## Overview

This directory contains comprehensive tests for `mpk_lammps_ver4.py`, covering configuration validation, model loading, shell model processing, and LAMMPS input generation.

## Test Structure

```text
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Pytest fixtures and configuration
├── test_config.py           # Configuration class tests
├── test_utilities.py        # Utility function tests
├── test_model_loading.py    # Model loading tests
├── test_shell_model.py      # Shell model processing tests
├── test_lammps_generation.py # LAMMPS input generation tests
└── test_integration.py      # End-to-end integration tests
```

## Installation

### Install test dependencies

```bash
pip install -r requirements-test.txt
```

```bash
pip install -r requirements-test.txt
```

Or install individually:

```bash
pip install pytest pytest-cov pytest-mock
```

### Run all tests

```bash
pytest
```

### Run with verbose output

```bash
pytest -v
```

### Run specific test file

```bash
pytest tests/test_config.py
```

### Run specific test class

```bash
pytest tests/test_config.py::TestConfigClass
```

### Run specific test

```bash
pytest tests/test_config.py::TestConfigClass::test_valid_config_passes_validation
```

### Run with coverage report

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

- **`test_config.py`**: Tests for `Config` class
  - Default values
  - Custom values
  - Validation of all parameters
  - Error conditions

- **`test_utilities.py`**: Tests for utility functions
  - `get_mixing_type()` with different patterns
  - Error handling for invalid inputs

- **`test_model_loading.py`**: Tests for model loading
  - Valid model loading
  - Error handling (missing files, corrupted data)
  - Model validation

- **`test_shell_model.py`**: Tests for shell model processing
  - Data extraction
  - Species ID mapping
  - Data validation
  - Normalization (`shel` → `shell`)

- **`test_lammps_generation.py`**: Tests for LAMMPS input generation
  - Header generation
  - Charge settings
  - Group definitions
  - Temperature ramps
  - Complete input file generation

### Integration Tests

- **`test_integration.py`**: End-to-end workflow tests
  - Complete processing pipeline
  - Configuration-based workflow
  - Error handling in realistic scenarios

## Test Fixtures

Located in `conftest.py`:

- **`temp_dir`**: Temporary directory for test files
- **`mock_model`**: Mock model object with typical structure
- **`mock_cell`**: Mock cell object for testing
- **`sample_pickle_file`**: Sample pickle file for testing
- **`sample_config`**: Sample configuration object
- **`mock_gulp_file`**: Mock GS.gulp file

## Test Coverage

The test suite covers:

- ✅ Configuration validation (all parameters)
- ✅ Utility functions (mixing types)
- ✅ Model loading (valid and invalid cases)
- ✅ Shell model data extraction
- ✅ Species ID mapping
- ✅ LAMMPS input generation (all sections)
- ✅ Error handling and custom exceptions
- ✅ Integration workflows

## Writing New Tests

### Example Unit Test

```python
import pytest
from mpk_lammps_final_ver4 import Config, ConfigurationError

def test_negative_temperature_fails():
    """Test that negative temperature fails validation."""
    config = Config(t_array=[100.0, -50.0])
    with pytest.raises(ConfigurationError, match="must be non-negative"):
        config.validate()
```

### Example Integration Test

```python
def test_complete_workflow(sample_pickle_file, temp_dir):
    """Test complete processing workflow."""
    model = load_model(sample_pickle_file)
    shell_data, springs, potentials = extract_shell_model_data(model)
    # ... continue workflow
    assert result_is_valid
```

## Continuous Integration

To run tests automatically (make sure py311 environment is activated):

```bash

# Run tests and generate coverage
pytest --cov=mpk_lammps_ver4 --cov-report=term-missing

# Run with strict mode (warnings as errors)
pytest --strict-warnings
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
