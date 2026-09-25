"""Trim the real (unversioned) f2scan files to a regular sinogram grid.

test/f2scans/ is not committed (the three ~40MB master files live on disk, not
in git), so this test skips when they are absent. When present it checks that
DataSet.guess_shape trims each real f2scan to the expected regular grid - every
rotation exactly round(360/step) frames, no partial turn left over - which is
the behaviour the f2scan import was rewritten to give.
"""
from __future__ import print_function

import os
import re
import unittest

import ImageD11.sinograms.dataset as D

HERE = os.path.dirname(os.path.abspath(__file__))
F2SCANS = os.path.join(HERE, "f2scans")

# file -> expected (n_rotations, frames_per_rotation)
CASES = {
    "s4_RT_small_preload_s3dxrd_z22.h5": (389, 1440),
    "AA2024_T8_CR50_posth2_align_z_redo1.h5": (250, 3600),
    "Cu_refine_scanning_z_redo2.h5": (551, 1636),
}


def _slice_len(scan):
    m = re.match(r".*\[(\d+):(\d+)\]", scan)
    return None if m is None else int(m.group(2)) - int(m.group(1))


def _build(fname):
    path = os.path.join(F2SCANS, fname)
    ds = D.DataSet(dataroot=".", analysispath="/tmp", sample="s", dset="d",
                   detector="eiger", omegamotor="diffrz_cen360",
                   dtymotor="diffty")
    ds.masterfile = path
    ds.import_scans()
    ds.imageshape = (1, 1)
    ds.import_motors_from_master()
    ds.guess_shape()
    ds.guessbins()
    return ds


class TestRealF2scanTrim(unittest.TestCase):
    def test_real_f2scans_trim_to_regular(self):
        present = [f for f in CASES if os.path.exists(os.path.join(F2SCANS, f))]
        if not present:
            raise unittest.SkipTest(
                "real f2scan masterfiles not present in %s" % F2SCANS)
        for fname in present:
            ds = _build(fname)
            s1 = ds.shape[1]
            lens = [_slice_len(s) for s in ds.scans]
            self.assertEqual(len(ds.scans), ds.shape[0])
            self.assertTrue(all(l == s1 for l in lens),
                            "%s: slices not all of length %d" % (fname, s1))
            self.assertEqual(ds.omega.shape, tuple(ds.shape))
            self.assertEqual(ds.dty.shape, tuple(ds.shape))


if __name__ == "__main__":
    unittest.main()
