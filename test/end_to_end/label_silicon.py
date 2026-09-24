"""Label the Si_cube sparse f2scan and record the omega/dty bins and peaks.

This is the worker script. It is meant to be called, in a subprocess, with the
ImageD11 to test on sys.path (repo -> new code, jupyter-slurm site-packages ->
old code). It runs only under Python 3.

    python label_silicon.py <dest_folder> <output_dir>

- Spins up a DataSet from the sparse pixels found by fetch_data.get_test_data.
- Calls DataSet.import_from_sparse (which is what works out the omega/dty bins
  and the grid) as part of the test.
- Runs properties.main (single core) and writes the pks2d table.
- Writes <output_dir>/peaks.h5 and <output_dir>/bins.json.
- Prints the ImageD11 __file__, the timing and the peak counts to stdout so a
  driver can capture and compare two runs (old vs new).

It is guarded with if __name__ == '__main__' because properties.main forks a
multiprocessing pool, whose children re-import this module.
"""
import os
import sys
import time
import json
import shutil
import tempfile

import numpy as np


def find_sparse(dest):
    from ImageD11 import fetch_data
    ds = fetch_data.si_cube_s3dxrd_dataset(dest, allow_download=False)
    if ds is None:
        raise RuntimeError("get_test_data returned no dataset for %s" % dest)
    sp = getattr(ds, "sparsefile", None)
    if not sp or not os.path.exists(sp):
        raise RuntimeError("sparse file not found: %r" % (sp,))
    return sp


def pin_to_one_core():
    """Pin to the first core in the allowed set from sched_getaffinity."""
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None
    if not allowed:
        return None
    os.sched_setaffinity(0, {allowed[0]})
    return allowed[0]


def _f(x):
    return None if x is None else float(x)


def _lst(x):
    return None if x is None else [float(v) for v in x]


def bins_json(ds):
    """The omega and dty binning that import_from_sparse worked out."""
    return {
        "shape": [int(ds.shape[0]), int(ds.shape[1])],
        "nscans": int(len(ds.scans)),
        "omega_wraps": bool(getattr(ds, "omega_wraps", False)),
        "obinedges": _lst(getattr(ds, "obinedges", None)),
        "obincens": _lst(getattr(ds, "obincens", None)),
        "omin": _f(getattr(ds, "omin", None)),
        "omax": _f(getattr(ds, "omax", None)),
        "ostep": _f(getattr(ds, "ostep", None)),
        "ybinedges": _lst(getattr(ds, "ybinedges", None)),
        "ybincens": _lst(getattr(ds, "ybincens", None)),
        "ymin": _f(getattr(ds, "ymin", None)),
        "ymax": _f(getattr(ds, "ymax", None)),
        "ystep": _f(getattr(ds, "ystep", None)),
    }


def run(dest, outdir):
    import ImageD11
    import ImageD11.sinograms.dataset as D
    from ImageD11.sinograms import properties

    print("ImageD11 imported from: %s" % ImageD11.__file__)
    print("python executable: %s" % sys.executable)
    core = pin_to_one_core()
    print("cpu affinity pinned to: %s" % core)

    sparse = find_sparse(dest)
    print("sparse file: %s" % sparse)

    os.makedirs(outdir, exist_ok=True)
    pksfile = os.path.join(outdir, "peaks.h5")
    binsfile = os.path.join(outdir, "bins.json")

    tmp = tempfile.mkdtemp()
    try:
        ds = D.DataSet(dataroot=tmp, analysisroot=tmp, sample="Si_cube",
                       dset="S3DXRD_nt_moves_dty")
        t0 = time.time()
        ds.import_from_sparse(sparse)
        t_import = time.time() - t0
        print("import_from_sparse: %.3f s" % t_import)

        dsfile = os.path.join(tmp, "ds.h5")
        ds.save(dsfile)
        with open(binsfile, "w") as fh:
            json.dump(bins_json(ds), fh, indent=2)

        t0 = time.time()
        properties.main(dsfile, sparsefile=sparse, pksfile=pksfile,
                        options={"nproc": 1})
        t_label = time.time() - t0
        print("properties.main: %.3f s" % t_label)

        pkst = properties.pks_table.load(pksfile)
        print("peaks: %d" % int(pkst.npk[:, 0].sum()))
        print("pairs_ii: %d; pairs_ij: %d"
              % (int(pkst.npk[:, 1].sum()), int(pkst.npk[:, 2].sum())))
        print("nlabel: %d" % int(pkst.nlabel))
        print("glabel saved: %s" % (pkst.glabel is not None))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: label_silicon.py <dest_folder> <output_dir>")
    run(sys.argv[1], sys.argv[2])
