# LAMMPS Shell Model Processing Script

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
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
- **Material Type Support**: Handle both pure and mixed perovskite materials
- **Chemical Ordering**: Support for multiple ordering types (homog, G, 1/4)
- **Robust Error Handling**: Comprehensive validation and error reporting
- **Logging System**: Detailed logging for debugging and tracking
- **Multiple Temperature Support**: Generate temperature ramps for multi-stage simulations
- **Advanced MD Parameters**: Configurable equilibration and production steps
- **File Overwrite Protection**: Warnings before overwriting existing files
- **Comprehensive Testing**: 70 unit and integration tests for reliability

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
   ```

   The `pm__` libraries (`pm__cell`, `pm__chemical_order`, `pm__shell_model_kit`)
   must be available in your Python environment. Since v4.5 the script **no
   longer auto-adds `../lib/...` paths** at startup — ensure the packages are
   installed or add the lib directory to `PYTHONPATH` before running:

   ```bash
   # Option A — packages are pip-installed (recommended)
   pip install pm__cell pm__chemical_order pm__shell_model_kit

   # Option B — packages live in ../lib/ (use PYTHONPATH)
   export PYTHONPATH="${PYTHONPATH}:../lib/pm__py/pm__cell:../lib/pm__py/pm__chemical_order:../lib/pm__py/pm__shell_model_kit"
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
- Symmetry type (`cubic`, `random`, or `file`)
- Material mode (`pure` or `mix`) for `cubic`/`random` symmetry
- Species input (`species_a`, `species_b`, optional mixed-site pair)
- Mix settings (`position`, `mix_ratio`) when `material_type=mix`
- Supercell dimensions (e.g., `8 8 8`)
- Temperature array (e.g., `10 50 100 200`)
- Thermostat damping time (default: `0.1`)
- Barostat damping time (default: `2.0`)
- Runtime control parameters: `THERMO_FREQ`, `TIMESTEP`,
  `EQUILIBRATION_TEMP_STEPS`, `EQUILIBRATION_FINAL_STEPS`,
  `PRODUCTION_STEPS`, and trajectory prefix

### Interactive Configuration Flow

The CLI question flow is conditional and follows `get_user_config()`:

1. Model pickle path
2. Symmetry (`file` / `cubic` / `random`)
3. If symmetry is `file`: skip species and material prompts
4. If symmetry is `cubic` or `random`:
   - Ask material type (`pure` / `mix`)
   - If `mix`: ask mixed-site position (`A`/`B`), two species, and `mix_ratio`
   - If `pure`: ask single `species_a` and single `species_b`
5. Ask simulation controls for all modes:
   - supercell dimensions
   - temperature array
   - thermostat/barostat damping
   - `THERMO_FREQ`, `TIMESTEP`
   - equilibration and production steps
   - trajectory filename prefix

Validation rules enforced during prompts include:

- `mix_ratio` must be in the range `[0.0, 1.0]`
- dimensions and all step/frequency values must be positive
- temperatures must be non-negative
- species entries must be non-empty and syntactically valid for selected mode

### Example Session

```
⚙️  CONFIGURATION SUMMARY
======================================================================
   Symmetry                   : cubic
   Material type              : mix
   Position                   : A
   Species (A mix)            : Ba/Ca
   Species (B pure)           : Ti
  Model file       : ./potential.pickle
   Supercell dims             : [2, 2, 2]
   Temperatures [K]           : [10.0, 50.0, 100.0, 200.0]
   T-stat damping             : 0.1 fs
   P-stat damping             : 2.0 fs
   THERMO_FREQ                : 500
   TIMESTEP [fs]              : 0.0002
   EQUILIBRATION_TEMP_STEPS   : 20000
   EQUILIBRATION_FINAL_STEPS  : 30000
   PRODUCTION_STEPS           : 50000
   Trajectory name            : trajectory
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

### Material Types

The script now supports two material types:

1. **`pure`** (default): Single perovskite composition
   - Requires: `species_a` and `species_b` to be specified
   - Example: ABO₃ perovskite (e.g., SrTiO₃)

2. **`mix`**: Mixed perovskite composition
   - Requires: `position` ("A" or "B"), `mix_ratio` (0.0-1.0)
   - Example: (Sr,Ba)TiO₃ or SrTi(Mo,W)O₃

### Symmetry Options

1. **`cubic`**: Standard cubic perovskite arrangement
2. **`random`**: Random perturbations applied to atomic positions
3. **`file`**: Read structure from `GS.gulp` file (must exist in working directory)

### Simulation Parameters (Configurable in Code)

```python
# Physical Constants
CORE_MASS_RATIO = 0.98        # 98% of mass to core
SHELL_MASS_RATIO = 0.02       # 2% of mass to shell
DEFAULT_RMAX = 10.0           # Cutoff radius

# Simulation Control Parameters
TIMESTEP = 0.0002                  # MD timestep (fs)
THERMO_FREQ = 500                  # Thermodynamic output frequency
EWALD_ACCURACY = 1.0e-6            # Ewald summation accuracy

# MD Equilibration Steps
EQUILIBRATION_TEMP_STEPS = 20000   # Temperature equilibration steps
EQUILIBRATION_FINAL_STEPS = 30000  # Final equilibration steps

# Production Run
PRODUCTION_STEPS = 50000           # Production simulation steps

# Thermostat/Barostat
DEFAULT_T_STAT = 0.1               # Thermostat damping time constant
DEFAULT_P_STAT = 2.0               # Barostat damping time constant
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

The project includes a comprehensive test suite with **70 tests** covering all major components.

### Run All Tests

```bash
pytest -v
```

### Run with Coverage Report

```bash
pytest --cov=mpk_lammps_ver4 --cov-report=html
```

View the report: `open htmlcov/index.html`

### Run Specific Test Categories

```bash
# Configuration validation tests (17 tests)
pytest tests/test_config.py -v

# Integration tests (5 tests)
pytest tests/test_integration.py -v

# LAMMPS generation tests (16 tests)
pytest tests/test_lammps_generation.py -v

# Model loading tests (6 tests)
pytest tests/test_model_loading.py -v

# Shell model tests (21 tests)
pytest tests/test_shell_model.py -v

# Utilities tests (6 tests)
pytest tests/test_utilities.py -v
```

### Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and mocked dependencies
├── test_config.py           # Configuration validation (17 tests)
├── test_integration.py      # End-to-end workflows (5 tests)
├── test_lammps_generation.py # LAMMPS input generation (16 tests)
├── test_model_loading.py    # Model loading (6 tests)
├── test_shell_model.py      # Shell model processing (21 tests)
└── test_utilities.py        # Utility functions (6 tests)
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
   - Validates all user inputs with comprehensive error checking
   - Supports both pure and mixed material configurations
   - Centralizes all simulation parameters
   - Ensures consistency across the workflow

2. **Model Processing**
   - Loads pickle files with validation
   - Extracts shell model data (charges, springs, potentials)
   - Maps species to numeric IDs via `create_species_id_map`
   - **Revises the ID map** via `revise_species_id_map`: removes species not
     present in `model.charges`, reassigns sequential IDs, and emits clear
     `WARNING` messages (Option B — warn + continue rather than crash)
   - Calls `initialize_shell_models_data` → `map_charges_to_new_model` →
     `validate_shell_model_data` to populate all integer-keyed charge/mass entries
   - **Option B fall-back** via `_sanitize_shell_model_for_writing`: replaces any
     remaining `None` charge/mass values with `0.0` before calling
     `writeToLAMMPSStructure`, preventing `TypeError` for unknown species
   - Normalizes nomenclature (`shel` → `shell`)
   - Note: `create_string_named_cell` converts atom _names_ to strings but
     intentionally keeps `shell_models['model']` keys as **integers** so
     that `writeToLAMMPSStructure` (which accesses them via `ii+1`) works correctly

3. **Supercell Generation**
   - Creates perovskite structures with specified dimensions
   - Applies chemical ordering (homogeneous, G-type, 1/4 ordering)
   - Handles different symmetries (cubic, random, file-based)
   - Supports both pure and mixed compositions

4. **LAMMPS Input Generation**
   - Generates complete LAMMPS input scripts with:
     - Initialization commands
     - Force field definitions (Buckingham potentials)
     - Core-shell spring constants
     - Temperature ramps with equilibration stages
     - NPT ensemble MD settings
     - Dump file configurations

5. **Error Handling**
   - Custom exception hierarchy for different error types
   - Comprehensive input validation at every stage
   - Detailed error messages to guide users
   - Graceful failure with informative logging

### Design Principles

- **Type Safety**: All functions have type hints for better IDE support and error checking
- **Logging**: Comprehensive logging at all operational levels (debug, info, warning, error)
- **Validation**: Multi-level validation ensures data integrity throughout the pipeline
- **Modularity**: Functions have single, well-defined responsibilities
- **Testing**: Extensive test coverage (70 tests) with mocked external dependencies
- **Documentation**: Detailed docstrings and comprehensive readme documentation

## 👥 Credits

- **Author**: Mukesh Khanore
- **LAMMPS MD Logic**: Mónica Elisabet Graf and Mauro António Pereira Gonçalves
- **Date**: 03-Mar-2026
- **Version**: 4.5

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
