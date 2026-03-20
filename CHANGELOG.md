# Changelog

All notable changes to `mpk_lammps_ver4.py` are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Changed

- **Generated-file summary corrected (Option B)** — removed the stale
  `supercell_structure.poscar` line from the runtime success message in
  `main()`. The script now reports only files that are actually produced.

### Documentation

- **README prompt flow updated** — usage and configuration sections now match
  the current interactive `get_user_config()` branching and validation rules.
- **Python requirement consistency** — badge and text now both state Python
  `3.11+`.
- **TEST_README coverage mapping clarified** — explicitly distinguishes direct
  tests for `revise_species_id_map` from indirect coverage of
  `_sanitize_shell_model_for_writing`.

## [4.5] – 2026-03-03

### Fixed

- **Shell preservation in `cubic` and `random` symmetry modes** — Shells are now
  correctly carried through the supercell creation pipeline when `symmetry` is
  `"cubic"` or `"random"`. Previously, core-shell pairing was lost during the
  `pm__cell` call chain, producing cells with unmatched or missing shells.

- **`file` mode structure reading** — The `GS.gulp` read path in
  `create_supercell(symmetry="file")` now correctly handles atom ordering and
  passes the lattice to subsequent steps without dropping shell atoms.

### Removed

- **Runtime `sys.path.append` block** — The following eight lines were removed
  from the module header:

  ```python
  sys.path.append(r'../lib/pm__py/pm__cell/')
  sys.path.append(r'../lib/pm__py/pm__constants/')
  sys.path.append(r'../lib/pm__py/pm__data_from_file/')
  sys.path.append(r'../lib/pm__py/pm__utilities/')
  sys.path.append(r"../lib/mk__py/")
  sys.path.append(r"../lib/pm__py/pm__chemical_order/")
  sys.path.append(r"../lib/transformations/")
  sys.path.append(r"../lib/pm__py/pm__shell_model_kit/")
  ```

  These were deployment-specific path hacks that assumed a fixed `../lib/`
  directory layout. The script now relies on the `pm__` packages being properly
  installed in the active Python environment or already present on `PYTHONPATH`.
  See the updated [Installation](#-installation) section in `README.md`.

### Changed

- **Version string** updated from `4.4` to `4.5`; **date** updated to
  `03-03-2026`.

### Tests (70 tests, 100% pass)

No new tests were added in this patch release; all 70 existing tests continue
to pass unchanged. The test suite was never affected by the `sys.path.append`
removal because `pm__cell`, `pm__chemical_order`, and `pm__shell_model_kit` are
mocked at the `sys.modules` level in `conftest.py`.

---

## [4.4] – 2026-03-01

### Added

- **`revise_species_id_map(species_id_map, model)`** — new function that
  filters the species ID map produced by `create_species_id_map` so it
  contains only species present in `model.charges`. Missing species are
  logged at `WARNING` level (Option B: warn + continue, not crash), and IDs
  are reassigned sequentially so there are no gaps.

- **`_sanitize_shell_model_for_writing(string_cell, ...)`** — new private
  helper that replaces any remaining `None` charge/mass values in
  `string_cell.shell_models` with a numeric default (`0.0`) before
  `writeToLAMMPSStructure` is called, preventing a `TypeError` for
  unknown/missing species.

- **Option B warning in `create_mapped_cell`** — a `WARNING` is now emitted
  when a species is present in `species_id_map` but has zero atoms in the
  cell (e.g. zero-concentration species in a mixed composition).

- **Option B warnings in `create_supercell` (pure mode)** — when `material_type="pure"`
  and user-specified A-site or B-site species are not found in `model.charges`,
  a batched `WARNING` is logged rather than raising an exception.

- **`initialize_shell_models_data`** — accepts an optional `species_id_map`
  keyword argument to support future ID-map-based initialisation (the
  alternative code path is currently commented out; the legacy
  `cell.species_name` path remains the default).

### Changed

- **`DEFAULT_SYMMETRY`** changed from `"file"` to `"cubic"`. The previous
  default caused a `FileNotFoundError` at startup when `GS.gulp` was absent;
  `"cubic"` is the safer default for most users.

- **`create_species_id_map`** — the per-species warning loop has been
  replaced with a single batched `WARNING` that lists _all_ missing species
  at once, making the log output cleaner and easier to read.

- **`initialize_shell_models_data`** — default `mass` initialisation changed
  from `0.0` to `None`. This allows `validate_shell_model_data` to detect
  un-mapped entries reliably (any leftover `None` after
  `map_charges_to_new_model` signals a missing species).

- **`generate_charge_settings(cell, species_id_map)`** — first argument
  renamed from `shell_models_data: Dict` to `cell: Any`. The function now
  reads charge information directly from `cell.shell_models['model']`
  (integer-keyed by type ID) instead of from the raw extraction dict.
  This keeps the charge source in sync with the cell that will be written.

- **`generate_lammps_input(cell, ...)`** — first positional argument renamed
  from `shell_models_data: Dict` to `cell: Any` to match the updated
  `generate_charge_settings` signature. Call sites in `main()` have been
  updated accordingly (now passes `string_cell`).

- **`process_shell_model`** — now calls `revise_species_id_map` between
  `create_species_id_map` and `create_mapped_cell`. Also passes `id_list`
  and `user_species_set` to `map_charges_to_new_model` for accurate
  per-species charge/mass resolution, and calls
  `_sanitize_shell_model_for_writing` before writing the LAMMPS structure.

- **Version string** updated from `4.3` to `4.4`; **date** updated from
  `26-Feb-2026` to `01-Mar-2026`.

### Tests (70 tests, 100% pass)

| File                        |  Tests |
| --------------------------- | -----: |
| `test_config.py`            |     17 |
| `test_integration.py`       |      5 |
| `test_lammps_generation.py` |     16 |
| `test_model_loading.py`     |      6 |
| `test_shell_model.py`       |     21 |
| `test_utilities.py`         |      6 |
| **Total**                   | **70** |

All tests introduced or modified to cover v4.4 changes:

- `TestReviseSpeciesIdMap` (5 tests) — covers valid/invalid species,
  sequential ID reassignment, empty-charges edge case, and no-warning path.
- `TestGenerateChargeSettings` — updated to pass a mock `cell` object with
  `.shell_models` instead of a raw dict.
- `TestGenerateLammpsInput` — updated first argument to a mock cell object.
- `TestEndToEndWorkflow` — integration test builds `mock_input_cell` with
  `.shell_models` before calling `generate_lammps_input(cell, ...)`.

---

## [4.3] – 2026-02-26

### Added

- **`revise_species_id_map`** architecture note in README (documentation only
  for changes already present in v4.3 codebase; full implementation landed
  in v4.4).
- **`create_string_named_cell`** — keeps `shell_models['model']` keys as
  integers (not stringified) so `writeToLAMMPSStructure` (integer-indexed)
  works correctly.
- **`validate_shell_model_data`** — all-`None` semantics: if _any_ attribute
  for a species/part is `None`, _all_ attributes for that entry become `None`
  (consistent exclusion from LAMMPS output).
- Comprehensive test suite established: 70 tests across 6 files, covering
  config validation, model loading, shell model processing, LAMMPS generation,
  utilities, and integration workflows.

### Changed

- `Config` defaults: `supercell_dims` → `[2, 2, 2]`, `symmetry` → `"cubic"`.
- `generate_charge_settings` / `generate_lammps_input` signatures updated to
  accept a cell object (first introduced as v4.3 signature change).
- `create_species_id_map` emits `WARNING` for user-specified species absent
  from `model.charges`.

---

## [4.2] – 2026-02-23

### Changed

- Refactored script for consistency, readability, and robustness.
- Standardised variable names throughout.
- Added comprehensive type hints and docstrings.
- External dependencies (`pm__cell`, `pm__chemical_order`) mocked in tests.
- Fixed type errors in LAMMPS template generation (positional arguments).

---

## [4.1] – 2026-02-20

### Added

- Initial public release of `mpk_lammps_ver4.py`.
- Shell model processing pipeline for LAMMPS structure and setup file
  generation.
- Support for pure and mixed perovskite compositions.
- Multiple symmetry modes: `cubic`, `random`, `file`.
- Temperature-ramp LAMMPS input generation.
- Logging system and custom exception hierarchy.
