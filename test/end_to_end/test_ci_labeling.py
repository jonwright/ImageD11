"""End-to-end labelling of the fetched Si_cube sparse as a CI-configured test.

Runs the single-core labelling (properties.main) through the label_silicon.py
worker in a subprocess, records the timing and the number of 2D and 4D peaks to
a log file, and fails if those counts drift from the reference (old-code) values.

It only runs on a machine where the Si_cube sparse data is present (e.g. a
machine that has already fetched it via test_fetch_data); otherwise it skips, so
public CI without the data does not fail. Runs only under Python 3.
"""
from __future__ import print_function

import os
import sys
import time
import shutil
import tempfile
import subprocess
import datetime
import unittest

import h5py

try:
    import ImageD11.sinograms.dataset  # noqa
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEST = os.path.join(REPO, "test", "test_fetch_data")
WORKER = os.path.join(HERE, "label_silicon.py")
LOG = os.path.join(HERE, "ci_labeling.log")

# Reference counts from the old-code labelling of the Si_cube sparse.
PK2D = 9134
PK4D = 2621

_KEYS = ("ImageD11 imported from:", "import_from_sparse:", "properties.main:",
         "peaks:", "nlabel:", "pairs_ii:")


def _sparsefile():
    if not os.path.exists(DEST):
        return None
    from ImageD11 import fetch_data
    ds = fetch_data.si_cube_s3dxrd_dataset(DEST, allow_download=False)
    if ds is None:
        return None
    sp = getattr(ds, "sparsefile", None)
    if sp is None or not os.path.exists(sp):
        return None
    return sp


class TestCiLabelling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.version_info[0] < 3:
            raise unittest.SkipTest(
                "subprocess.run is Python 3 only; cannot run the labelling worker")
        cls.sparse = _sparsefile()
        if cls.sparse is None:
            raise unittest.SkipTest(
                "Si_cube sparse not found at %s; run test_fetch_data to get it"
                % DEST)

    def test_labelling_reference_counts(self):
        outdir = tempfile.mkdtemp(prefix="id11_ci_label_")
        try:
            env = dict(os.environ, PYTHONPATH=REPO)
            start = time.time()
            proc = subprocess.run(
                [sys.executable, WORKER, DEST, outdir],
                env=env, cwd="/tmp",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            wall = time.time() - start
            out = proc.stdout.decode()

            info = {}
            for line in out.splitlines():
                for key in _KEYS:
                    if line.startswith(key):
                        info[key.rstrip(":")] = line[len(key):].strip()

            with h5py.File(os.path.join(outdir, "peaks.h5"), "r") as h:
                g = h["pks2d"]
                pk2d = int(g["pk_props"].shape[1])
                pk4d = int(g.attrs.get("nlabel", 0))

            with open(LOG, "a") as f:
                f.write("=== %s ===\n" % datetime.datetime.now().isoformat())
                f.write("ImageD11: %s\n" % info.get("ImageD11 imported from", "?"))
                f.write("subprocess wall: %.3f s\n" % wall)
                f.write("import_from_sparse: %s\n"
                        % info.get("import_from_sparse", "?"))
                f.write("properties.main: %s\n" % info.get("properties.main", "?"))
                f.write("pk2d: %d\n" % pk2d)
                f.write("pk4d: %d\n" % pk4d)
                f.write("\n")
                f.write(out)
                f.write("\n")

            self.assertEqual(proc.returncode, 0,
                             "label worker failed:\n" + out)
            self.assertEqual(pk2d, PK2D,
                             "pk2d changed: %d != %d\n%s" % (pk2d, PK2D, out))
            self.assertEqual(pk4d, PK4D,
                             "pk4d changed: %d != %d\n%s" % (pk4d, PK4D, out))
        finally:
            shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
