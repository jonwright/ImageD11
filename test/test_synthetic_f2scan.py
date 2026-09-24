"""Synthetic f2scan cases trimmed to a regular sinogram grid.

The minimal import path (DataSet.guess_shape) reads the f2scan omega column,
averages the per-frame step into omegastep, and places turn cuts on bin edges:
a frame is in turn floor((i + 0.5) * omegastep / 360). Each row is R frames
where R is the shortest interior turn; an over-long turn sheds its edge
frame(s) to reach R and a short trailing turn (first/last) is dropped. A scan
that doesn't start at omega 0 is handled because the first angle cancels in
the turn assignment, and a reverse scan works because the normalised step
carries the sign.

These cover the three real-case shapes (which reproduce s4/AA/Cu exactly),
forward and reverse scans, a non-zero start, and both the mod-360 motor column
(diffrz_cen360) and the absolute one (diffrz_trig).
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
    import ImageD11.sinograms.dataset as IMGD
except Exception:  # pragma: no cover
    IMGD = None


def slice_len(scan):
    """Length of a "1.1::[a:b]" slice, or None if it is a plain scan."""
    m = re.match(r".*\[(\d+):(\d+)\]", scan)
    return None if m is None else int(m.group(2)) - int(m.group(1))


class TestGetRotationsImages(unittest.TestCase):
    """Core: per-turn frame counts from the omega column."""

    def test_forward_nonzero_start(self):
        # 4 turns of 6, starting at 100.5 (not 0, not a multiple of 60).
        self.assertEqual(IMGD.get_rotations_images(
            100.5 + 60.0 * np.arange(24)).tolist(), [6, 6, 6, 6])

    def test_reverse_nonzero_start(self):
        # reverse scan from 700, 4 turns of 6.
        self.assertEqual(IMGD.get_rotations_images(
            700.0 - 60.0 * np.arange(24)).tolist(), [6, 6, 6, 6])

    def test_dividing_step(self):
        # 30 frames of 60 deg = 1800 deg = 5 full turns of 6.
        self.assertEqual(IMGD.get_rotations_images(
            np.arange(30) * 60.0).tolist(), [6] * 5)

    def test_not_dividing_step(self):
        # 3273 frames at 0.22 deg step (like the real Cu case) gives exactly
        # two turns of 1636 and 1637 frames respectively.
        counts = IMGD.get_rotations_images(np.arange(3273) * 0.22)
        self.assertEqual(set(counts.tolist()), {1636, 1637})


class TestTrimToRegular(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="id11_f2scan_trim_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _case(self, name, omegamotor="diffrz_cen360"):
        """Build one named REAL-case synthetic master and import it."""
        om0, om1, npoints, dt0, dt1, setpoint = S.REAL[name]
        omega_edges, omega_col, dty_arr, _ = S.linspace_case(
            (om0, om1), (dt0, dt1), npoints, setpoint)
        path = os.path.join(self.tmp, "%s_synth.h5" % name)
        S.make_master(path, npoints, omega_edges, omega_col, dty_arr, setpoint)
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
        # step divides 360, so every turn is 1440 frames; nothing to drop.
        ds = self._case("s4")
        self.assertEqual(ds.shape, (389, 1440))
        self._assert_regular(ds)

    def test_AA_drops_partial_last_turn(self):
        # 250 turns of 3600 plus a partial 251st (3565) which must be dropped.
        ds = self._case("AA")
        self.assertEqual(ds.shape, (250, 3600))
        self._assert_regular(ds)

    def test_Cu_clips_overlong_turns(self):
        # step does not divide 360, so some turns run 1637; drop one each to 1636.
        ds = self._case("Cu")
        self.assertEqual(ds.shape, (551, 1636))
        self._assert_regular(ds)

    def test_absolute_omega_column(self):
        # the absolute motor column (diffrz_trig) gives the same row size.
        ds = self._case("AA", omegamotor="diffrz_trig")
        self.assertEqual(ds.shape, (250, 3600))
        self._assert_regular(ds)

    def test_reverse_scan(self):
        # omega descends (reverse scan).
        ds = S.make_case((720.0, 0.0), (10.0, 10.0), 13, 60.0, self.tmp, name="rev")
        self.assertEqual(ds.shape, (2, 6))
        self._assert_regular(ds)


if __name__ == "__main__":
    unittest.main()
