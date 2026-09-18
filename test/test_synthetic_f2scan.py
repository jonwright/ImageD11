"""Synthetic f2scan cases driven through the current code, to see which
shapes/bin grids come out and which fail. Distinguishes the real cases:

    s4 : clean   - step divides 360, whole turns -> rectangular sinogram.
    AA : partial - step divides 360, but a partial last turn.
    Cu : missed  - step does not divide 360, plus a partial last turn.

And the extra axes the generator can set: sigma direction, dty direction,
and whether the stored rotation column is mod 360 or absolute.
"""
from __future__ import print_function

import shutil
import sys
import tempfile
import unittest

import numpy as np

import synthetic_f2scan as S

try:
    import ImageD11.sinograms.dataset
except Exception as e:  # pragma: no cover
    pass


def occupancy(ds):
    io = np.digitize(ds.omega_for_bins, ds.obinedges) - 1
    ok = (io >= 0) & (io < len(ds.obincens))
    c = np.bincount(io[ok], minlength=len(ds.obincens)) if ok.any() else np.array([])
    return c, ok


class TestRealCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="id11_f2scan_real_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fields(self, name):
        om0, om1, n, dt0, dt1, setp = S.REAL[name]
        return (om0, om1), (dt0, dt1), n, setp

    def test_s4_is_a_clean_rectangle(self):
        ds = S.case("s4", self.tmp)
        self.assertEqual(ds.shape, (389, 1440))
        c, ok = occupancy(ds)
        self.assertEqual(int(c.min()), 389)
        self.assertEqual(int(c.max()), 389)
        self.assertEqual(int((c == 0).sum()), 0)
        self.assertTrue(ok.all())
        self.assertEqual(ds.sinohist(weights=np.ones(ds.omega.size)).sum(),
                         ds.omega.size)

    def test_aa_is_a_partial_turn(self):
        """npoints is not a whole number of turns, so guess_shape proposes a
        shape that does not fit the data and the build fails."""
        omega, dty, npoints, setpoint = self._fields("AA")
        s1 = int(np.round(360.0 / setpoint))
        self.assertNotEqual((npoints // s1) * s1, npoints)
        # current code: reshape fails -> building the dataset raises
        with self.assertRaises(ValueError):
            S.case("AA", self.tmp)

    def test_cu_step_does_not_divide_360(self):
        """360/step is not an integer, so there is no integer frames-per-turn
        and the scan does not close a circle evenly."""
        omega, dty, npoints, setpoint = self._fields("Cu")
        fpt = 360.0 / setpoint
        self.assertFalse(float(fpt).is_integer())
        self.assertNotEqual((npoints // int(round(fpt))) * int(round(fpt)),
                            npoints)


class TestSyntheticCases(unittest.TestCase):
    """Cases from the parameter table, in the directions of interest."""
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="id11_f2scan_cases_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _occupancy_result(self, **kw):
        try:
            ds = S.make_case(**kw, tmpdir=self.tmp)
        except Exception as e:
            return ("FAIL", type(e).__name__, str(e)[:60])
        return ("OK", tuple(ds.shape), len(ds.obincens))

    def test_negative_omega(self):
        """same scan but rotating the other way (negative omega step)."""
        res = self._occupancy_result(
            omega=(360.0, 360.0 - 140039.7445), dty=(11.60, 10.8218),
            npoints=560160, setpoint_step=0.2500098418, name="neg_om")
        self.assertEqual(res[0], "OK")

    def test_negative_dty(self):
        res = self._occupancy_result(
            omega=(0.0, 140039.7445), dty=(10.8218, 11.60),
            npoints=560160, setpoint_step=0.2500098418, name="neg_dty")
        self.assertEqual(res[0], "OK")

    def test_absolute_rot_column(self):
        """store the rotation column absolute (not resetting at 360)."""
        res = self._occupancy_result(
            omega=(0.0, 140039.7445), dty=(11.60, 10.8218), npoints=560160,
            setpoint_step=0.2500098418, omega_mod360=False, name="abs_rot")
        self.assertEqual(res[0], "OK")

    def test_irrational_step(self):
        """step = 1/pi deg interleaves frames, so the scan is irregular and
        the current code cannot build a rectangular sinogram from it."""
        npoints = 560160
        step = 1.0 / np.pi
        omega_end = npoints * step
        res = self._occupancy_result(
            omega=(0.0, omega_end), dty=(11.60, 10.8218), npoints=npoints,
            setpoint_step=step, name="irrat")
        # currently the shape does not fit -> the build fails
        self.assertEqual(res[0], "FAIL")


if __name__ == "__main__":
    unittest.main()
