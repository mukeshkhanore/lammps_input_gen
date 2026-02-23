# LAMMPS Shell Model Processing Script

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust Python script for processing shell models and generating LAMMPS structure and setup files from `.pickle` and `GS.gulp` files.

## 📋 Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#%EF%B8%8F-configuration)
- [Output Files](#-output-files)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Credits](#-credits)
- [License](#-license)

## ✨ Features

- **Shell Model Processing**: Extracts and processes shell model data for perovskite materials
- **Supercell Generation**: Creates supercells with configurable dimensions and symmetry
- **LAMMPS Input Generation**: Automatically generates complete LAMMPS input scripts
- **Robust Error Handling**: Comprehensive validation and error reporting
- **Logging System**: Detailed logging for debugging and tracking
- **Multiple Temperature Support**: Generate temperature ramps for multi-stage simulations
- **File Overwrite Protection**: Warnings before overwriting existing files
- **Comprehensive Testing**: 62 unit and integration tests for reliability

## 📦 Requirements

### Python Version

- Python 3.11 or higher

### Dependencies

```txt
numpy
mendeleev
pm__cell
pm__chemical_order
pm__shell_model_kit
```

### Testing Dependencies (Optional)

```txt
pytest
pytest-cov
pytest-mock
```

## 🚀 Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd lammps_mpk_script
   ```

2. **Activate your Python environment:**

   ```bash
   source ~/python_env/py311/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install numpy mendeleev
   # Install pm__ modules as per your system configuration
   ```

4. **Install testing dependencies (optional):**

   ```bash
   pip install -r requirements-test.txt
   ```

## 💻 Usage

### Basic Usage

Run the script interactively:

```bash
python mpk_lammps_ver4.py
```

The script will prompt you for:

- Model file path (`.pickle` file)
- Supercell dimensions (e.g., `8 8 8`)
- Symmetry type (`cubic`, `random`, or `file`)
- Output filename (default: `structure`)
- Temperature array (e.g., `10 50 100 200`)
- Thermostat damping time (default: `0.1`)
- Barostat damping time (default: `2.0`)

### Example Session

```
⚙️  CONFIGURATION SUMMARY
======================================================================
  Model file       : ./potential.pickle
  Supercell dims   : [8, 8, 8]
  Symmetry         : file
  Output filename  : structure
  Temperatures [K] : [10.0, 50.0, 100.0, 200.0]
  T-stat damping   : 0.1
  P-stat damping   : 2.0
======================================================================

✓ Renamed 'structure.LAMMPSStructure' to 'rstrt.dat'
✓ LAMMPS input saved to 'lammps.in'

🎉 PROCESSING COMPLETED SUCCESSFULLY!

📁 Generated Files:
  ✓ rstrt.dat              (LAMMPS structure file)
  ✓ lammps.in              (LAMMPS input script)
  ✓ species_id_map.txt     (Species ID mapping)
  ✓ lammps_processing.log  (Detailed processing log)
```

## ⚙️ Configuration

### Symmetry Options

1. **`cubic`**: Standard cubic perovskite arrangement
2. **`random`**: Random perturbations applied to atomic positions
3. **`file`**: Read structure from `GS.gulp` file (must exist in working directory)

### Constants (Configurable in Code)

```python
CORE_MASS_RATIO = 0.98        # 98% of mass to core
SHELL_MASS_RATIO = 0.02       # 2% of mass to shell
DEFAULT_RMAX = 10.0           # Cutoff radius
EQUILIBRATION_TEMP_STEPS = 20000   # Temperature equilibration steps
EQUILIBRATION_FINAL_STEPS = 30000  # Final equilibration steps
PRODUCTION_STEPS = 50000           # Production run steps
TIMESTEP = 0.0002                  # MD timestep
```

## 📄 Output Files

### 1. `rstrt.dat`

LAMMPS structure file containing:

- Atom coordinates
- Cell parameters
- Core-shell connectivity
- Species information

### 2. `lammps.in`

Complete LAMMPS input script with:

- Initialization commands
- Force field definitions (Buckingham potentials)
- Core-shell springs
- Temperature ramps
- NPT ensemble settings
- Dump configurations

### 3. `species_id_map.txt`

Maps species names to numeric IDs:

```
Sr core 1
Ti core 2
O core 3
Sr shell 4
Ti shell 5
O shell 6
```

### 4. `lammps_processing.log`

Detailed processing log with timestamps and debug information

## 🧪 Testing

The project includes a comprehensive test suite with 62 tests.

### Run All Tests

```bash

pytest -v
```

### Run Specific Test Categories

```bash
# Configuration tests
pytest tests/test_config.py -v

# Utility function tests
pytest tests/test_utilities.py -v

# Integration tests
pytest tests/test_integration.py -v
```

### Generate Coverage Report

```bash
pytest --cov=mpk_lammps_ver4 --cov-report=html
```

View the report: `open htmlcov/index.html`

### Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_config.py           # Configuration validation (26 tests)
├── test_utilities.py        # Utility functions (6 tests)
├── test_model_loading.py    # Model loading (6 tests)
├── test_shell_model.py      # Shell model processing (9 tests)
├── test_lammps_generation.py # LAMMPS input generation (12 tests)
└── test_integration.py      # End-to-end workflows (3 tests)
```

See [TEST_README.md](TEST_README.md) for detailed testing documentation.

## 📁 Project Structure

```
lammps_mpk_script/
├── mpk_lammps_ver4.py           # Main script
├── README.md                     # This file
├── TEST_README.md                # Testing documentation
├── requirements-test.txt         # Testing dependencies
├── pytest.ini                    # Pytest configuration
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_utilities.py
│   ├── test_model_loading.py
│   ├── test_shell_model.py
│   ├── test_lammps_generation.py
│   └── test_integration.py
└── examples/                     # Example files (if available)
    ├── example.pickle
    └── GS.gulp
```

## 🏗️ Code Architecture

### Key Components

1. **Configuration Management** (`Config` class)
   - Validates all user inputs
   - Centralizes configuration parameters
   - Ensures consistency

2. **Model Processing**
   - Loads pickle files
   - Extracts shell model data
   - Maps species to IDs

3. **Supercell Generation**
   - Creates perovskite structures
   - Applies chemical ordering
   - Handles different symmetries

4. **LAMMPS Input Generation**
   - Generates header sections
   - Defines force fields
   - Creates temperature ramps
   - Configures MD settings

5. **Error Handling**
   - Custom exception hierarchy
   - Comprehensive validation
   - Detailed error messages

### Design Principles

- **Type Safety**: All functions have type hints
- **Logging**: Comprehensive logging at all levels
- **Validation**: Input validation at every stage
- **Modularity**: Functions have single responsibilities
- **Testing**: Extensive unit and integration tests

## 👥 Credits

- **Author**: Mukesh Khanore
- **LAMMPS MD Logic**: Mónica Elisabet Graf and Mauro António Pereira Gonçalves
- **Date**: 23-Feb-2026
- **Version**: 4.1

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'pm\_\_cell'**

```
Solution: Install the pm__ packages in your Python environment
```

**2. File 'GS.gulp' not found**

```
Solution: Ensure GS.gulp exists in the working directory when using symmetry='file'
```

**3. Invalid pickle file**

```
Solution: Verify your pickle file contains required attributes:
- charges
- AB_specie
- header
```

**4. Tests failing with import errors**

```
Solution: Activate your py311 environment before running tests:
source ~/python_env/py311/bin/activate
```

## 📚 Additional Documentation

- [Testing Guide](TEST_README.md) - Comprehensive testing documentation
- [Pytest Explanation](pytest_explanation.md) - Understanding the test suite

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📧 Contact

For questions or issues, please open an issue on the repository or contact the author.

---

**Note**: This script requires specific `pm__` packages that are part of the perovskite modeling toolkit. Ensure these are properly installed in your environment before use.
