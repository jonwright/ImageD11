# c2ImageD11 API compatibility issues

Found by running the ImageD11 test suite with `IMAGED11_USE_C2=1` (which does
`from c2ImageD11 import *` in `ImageD11/cImageD11.py`).

All issues below should be fixed on the c2ImageD11 side.

Tested with:
- c2py23: commit `ef54f59` (main HEAD)
- c2ImageD11: commit `88e6c4f` (master HEAD, "fix API compatibility round 2")
- ImageD11: local editable, branch `c2ImageD11-testing`


## Resolved (round 1: `a7739e9`)

- [x] **`check_multiprocessing`** — alias added
- [x] **`tosparse_*` bool masks** — `msk.format == '?'` added
- [x] **`closest` tuple output** — wrapper added
- [x] **`score_and_refine` tuple output** — wrapper added
- [x] **`blobproperties` alloc output** — wrapper added

## Resolved (round 2: `88e6c4f`)

- [x] **`array_stats` tuple output** — wrapper added (1-arg f2py + 5-arg c2py23)
- [x] **`array_mean_var_cut` tuple output** — wrapper added
- [x] **`bloboverlaps` arg count** — `verbose` default added in `.c2py`
- [x] **`localmaxlabel` dtype strictness** — wrapper casts int→float32
- [x] **`compute_geometry` dtype strictness** — wrapper casts int→float64
- [x] **`compute_xlylzl` dtype strictness** — wrapper casts int→float64


## Issue 1: `put_incr` race condition with OpenMP

**Severity**: Medium — 1 test failure (`test_put_twice`), non-deterministic

**Error**:
```
assert (data == np.array( [0, 10] + [0]*8 , float)).all()
E       assert np.False_
# data[1] == 8.0 instead of 10.0 (2 accumulations lost)
```

**Call site** (`test/test_put_incr.py:19-25`):
```python
data = np.zeros(10,np.float32)
ind  = np.ones(10,np.intp)       # all 10 indices = 1
vals = np.ones(10,np.float32)    # all 10 values = 1
cImageD11.put_incr(data, ind, vals)
assert (data == np.array( [0, 10] + [0]*8 , float)).all()
```

All 10 entries write to `data[1]`. Expected: `data[1] == 10.0`.
Got: `data[1] == 8.0` (2 accumulations lost to race).

**Root cause**: c2ImageD11's SIMD `put_incr64` uses `#pragma omp parallel for`
with `gil_release: true`. When multiple indices point to the same output
position, the parallel loop has a data race — no atomic add. The f2py
version is sequential (correct by construction).

**Reproduction**: Passes when run in isolation, fails in full test suite
under load (thread scheduling varied). Always `data[1] < 10` when it fails.

**Fix**: Either:
1. Add `#pragma omp atomic` to the accumulation statement in the SIMD kernels
   (`src/imageproc/simd/put_incr64_kernel.c` and `put_incr32_kernel.c`)
2. Or drop `pragma omp parallel for` from put_incr kernels (make them sequential)
   since collisions are common in this workload (histogram/frequency array)


## Results

| Config | Pass/Fail | Notes |
|--------|-----------|-------|
| A (old, f2py) | **196/196** | Baseline |
| B (new, c2py23) | **195/196** | 1 race-condition failure |
| C (equivalence) | 47/47 | |
| C-full | 86 pass, 5 skip | |
