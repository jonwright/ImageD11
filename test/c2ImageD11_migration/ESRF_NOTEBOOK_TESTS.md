# Running notebook end-to-end tests with c2ImageD11

These instructions are for running the papermill notebook tests at ESRF
with the new c2ImageD11 C backend, using the venv that already has all
dependencies installed from git.

## Prerequisites

The `venv-new/` at the repo root already has:
- ImageD11 (editable install, `c2ImageD11-testing` branch)
- c2py23 (from git)
- c2ImageD11 (from git, with submodules)
- All `[full]` extras including papermill, jupyter, etc.

**Critical**: The `IMAGED11_USE_C2` env var is only recognized on the
`c2ImageD11-testing` branch. On `master` (or any other branch),
`ImageD11/cImageD11.py` has no env var support — it always uses the
f2py backend regardless of the variable. The notebook tests will run
but with the old backend. Verify with `git branch` before starting.

## How the notebook resolution works

The papermill test script resolves ImageD11 from the git checkout on disk:

```
test/papermill_test_notebooks.py
  → detects checkout_folder = parent of this repo
  → detects checkout_name = "ImageD11"                      (repo dir name)
  → sees build/ already exists → skips git clone + build_ext
  → inserts checkout into sys.path[0]
  → notebooks inherit PYTHONPATH with checkout at front
  → cImageD11.py checks IMAGED11_USE_C2 → picks c2ImageD11
```

No additional build step needed. The editable install in venv-new already
has everything. Setting IMAGED11_USE_C2=1 activates the new backend.

## Quick run (full suite, hours)

```bash
cd /path/to/ImageD11
git checkout c2ImageD11-testing          # required — env var only recognized here
source venv-new/bin/activate
IMAGED11_USE_C2=1 python test/papermill_test_notebooks.py /path/to/results
```

## Run only the fast route (~20 min)

The simplest route is `test_tomographic_route`. Edit the end of
`test/papermill_test_notebooks.py` — comment out the other routes
and add:

```python
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("destination_folder")
    parser.add_argument("-s", "--system", action="store_true")
    opts = parser.parse_args()
    test_tomographic_route(opts.destination_folder)
```

Then run:

```bash
source venv-new/bin/activate
IMAGED11_USE_C2=1 python test/papermill_test_notebooks.py /path/to/results
```

## Verify backend before running

```bash
source venv-new/bin/activate
IMAGED11_USE_C2=1 python -c "
from ImageD11 import cImageD11
print('Backend:', 'c2ImageD11' if 'returns (index, value) tuple' in (cImageD11.closest.__doc__ or '') else 'f2py')
print('check_multiprocessing:', hasattr(cImageD11, 'check_multiprocessing'))
print('OPENMP:', cImageD11.OPENMP if hasattr(cImageD11, 'OPENMP') else 'N/A')
"
```

Should print: `Backend: c2ImageD11`, `check_multiprocessing: True`.

## What the notebook tests cover that pytest doesn't

| Function | Exercise path |
|----------|--------------|
| `closest` | Indexing assignment (hkl→observed peak matching) |
| `compute_geometry` | Real geometry parameters, int→float cast path |
| `compute_xlylzl_xpos_variable` | Point-by-point detector geometry |
| `coverlaps` | Sparse peak filtering during tomo mapping |
| `put_incr64` | Sinogram reconstruction (tomo_2_map) |
| `quickorient` | Fast orientation in unitcell.py |
| `refine_assigned` | Indexing refinement loop |
| `score` / `score_and_refine` | Core indexing — not directly unit-tested |
| `sparse_blob2Dproperties` | Sparse peak processing in tomo route |
| `splat` | 3D grain plot (4_visualise notebook) |
| `tosparse_f32` / `tosparse_u16` | Lima segmenter + sparseframe path |

## Compare results

To compare notebook outputs between old and new backends, run each route
twice (with and without the env var) into separate output directories,
then diff the key output files:

```bash
# Old backend
python test/papermill_test_notebooks.py /path/to/results_old

# New backend
IMAGED11_USE_C2=1 python test/papermill_test_notebooks.py /path/to/results_new

# Compare (example — adjust to output format)
diff -r results_old/tomo_route/processed results_new/tomo_route/processed
```
