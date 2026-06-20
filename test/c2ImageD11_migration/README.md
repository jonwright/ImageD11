# c2ImageD11 Migration Testing

Test harness for comparing the original f2py-based C backend (`ImageD11._cImageD11`)
with the new c2py23-based backend (`c2ImageD11._cImageD11`).

## Architecture

The `ImageD11/cImageD11.py` module has been modified to support conditional
backend selection via the `IMAGED11_USE_C2` environment variable:

- **Unset (default)**: Uses the original f2py-compiled `ImageD11._cImageD11`.
  Identical behavior to the `master` branch. No c2ImageD11 dependency.

- **`IMAGED11_USE_C2=1`**: Uses the c2py23-compiled `c2ImageD11._cImageD11`.
  Requires c2ImageD11 and c2py23 to be installed. The c2ImageD11 package is
  expected to provide an API-compatible module — any missing compatibility
  will result in test failures.

## Quick Start

```bash
# 1. Set up venvs and install dependencies
bash test/c2ImageD11_migration/setup_venvs.sh

# 2. Run all test configurations
bash test/c2ImageD11_migration/run_tests.sh

# 3. Build docs for both backends
bash test/c2ImageD11_migration/build_docs.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| `setup_venvs.sh` | Creates `venv-old/` (ImageD11 only) and `venv-new/` (ImageD11 + c2py23 + c2ImageD11). Clones c2ImageD11 with submodules into `c2ImageD11_clone/`. |
| `run_tests.sh` | Runs the ImageD11 test suite in 3 configurations: (A) old backend, (B) new backend with env var, (C) c2ImageD11 equivalence tests. Outputs to timestamped `results_*/` directory. |
| `build_docs.sh` | Generates pydoc and Sphinx HTML for both backends. Outputs to `results_docs_*/`. |

## Three Test Configurations

| Config | Env | Backend | Status |
|--------|-----|---------|--------|
| A | (unset) | f2py (`ImageD11._cImageD11`) | 196/196 pass |
| B | `IMAGED11_USE_C2=1` | c2py23 (`c2ImageD11._cImageD11`) | 4 collection errors (API gaps) |
| C | — | Raw C equivalence (`test_equivalence.py`) | 47/47 pass |
| C-full | — | Full c2ImageD11 test suite | 86 pass, 5 skipped |

API gaps in Config B are documented in `c2ImageD11_issues.md`.

## Results

All test output goes to `results_<timestamp>/` directories.
These are gitignored.

## Notes

- c2py23 is not yet on PyPI; it's installed from `github.com/jonwright/c2py23`.
- c2ImageD11 uses `--no-build-isolation` for pip install since c2py23 must be pre-installed.
- The `c2ImageD11_clone/` directory is gitignored.
- This is a testing-only workflow. Production ImageD11 users should not set `IMAGED11_USE_C2`.
