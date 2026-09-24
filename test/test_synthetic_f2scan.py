"""Synthetic f2scan cases trimmed to a regular sinogram grid.

The minimal import path (DataSet.guess_shape) detects each turn's boundaries
from the omega column - a mod-360 motor column (usual 3DXRD) or an absolute
one, folded mod-360 as needed - and trims every turn to exactly
round(360/step) frames, dropping any partial turn, so the sinogram grid is
regular and the existing labelling/pairing code needs no change.

These drive straight-line fits to the three real f2scan files (forward scan),
a reverse-scan case, and both a mod-360 and an absolute omega column.
"""
from __future__ import print_function

import os
import re
import shutil
import tempfile
import unittest

import numpy as np

import synthetic_f2scan as S

try:
    import ImageD11.sinograms.dataset
except Exception:  # pragma: no cover
    pass


def slice_len(scan):
    """Length of a "1.1::[a:b]" slice, or None if it is a plain scan."""
    m = re.match(r".*\[(\d+):(\d+)\]", scan)
    return None if m is None else int(m.group(2)) - int(m.group(1))


class TestTrimToRegular(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="id11_f2scan_trim_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _case(self, name, omegamotor="diffrz_cen360"):
        """Build one named REAL-case synthetic master and import it."""
        om0, om1, npoints, dt0, dt1, setpoint = S.REAL[name]
        omega_abs, omega_col, dty_arr, _ = S.linspace_case(
            (om0, om1), (dt0, dt1), npoints, setpoint)
        path = os.path.join(self.tmp, "%s_synth.h5" % name)
        S.make_master(path, npoints, omega_abs, omega_col, dty_arr, setpoint)
        return S.build_dataset(path, self.tmp, omegamotor=omegamotor)

    def _assert_regular(self, ds):
        """Every slice holds exactly shape[1] frames; grids match the shape."""
        lens = [slice_len(s) for s in ds.scans]
        self.assertEqual(len(lens), ds.shape[0])
        self.assertTrue(all(l == ds.shape[1] for l in lens),
                        "slices not all of length %d: %s" % (ds.shape[1], lens[:5]))
        self.assertEqual(tuple(ds.omega.shape), tuple(ds.shape))
        self.assertEqual(tuple(ds.dty.shape), tuple(ds.shape))

    def test_s4(self):
        # a clean scan: step divides 360, so every turn is 1440 frames.
        ds = self._case("s4")
        self.assertEqual(ds.shape[1], 1440)
        self.assertGreater(ds.shape[0], 380)
        self._assert_regular(ds)

    def test_AA_drops_partial_last_turn(self):
        # 250 turns of 3600 plus a partial 251st (3565) which must be dropped.
        ds = self._case("AA")
        self.assertEqual(ds.shape, (250, 3600))
        self._assert_regular(ds)

    def test_Cu_clips_overlong_turns(self):
        # step does not divide 360, so some turns run 1637; clip to 1636.
        ds = self._case("Cu")
        self.assertEqual(ds.shape, (550, 1636))
        self._assert_regular(ds)

    def test_absolute_omega_column_is_folded(self):
        # diffrz_trig is the absolute motor; the trim mod-360 folds it itself.
        ds = self._case("AA", omegamotor="diffrz_trig")
        self.assertEqual(ds.shape, (250, 3600))
        self._assert_regular(ds)

    def test_reverse_scan(self):
        # omega descends (reverse scan): the wrap rises at each turn boundary.
        ds = S.make_case((720.0, 0.0), (10.0, 10.0), 13, 60.0, self.tmp,
                         name="rev")
        self.assertEqual(ds.shape, (2, 6))
        self._assert_regular(ds)


if __name__ == "__main__":
    unittest.main()
