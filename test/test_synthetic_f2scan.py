"""Synthetic f2scan cases driven through the real import/guess_shape path.

The three real cases are reproduced exactly by the generator:

    s4 : clean   - step divides 360, whole turns -> rectangular sinogram.
    AA : partial - step divides 360, but a partial last turn.
    Cu : missed  - step does not divide 360, plus a partial last turn, so a
                   part of a turn overflows into a 1637th frame that the grid
                   cannot hold (unplaced).

These check the cell map (cell_frame), the raw arrays it is built on, the
derived inverse maps, the neighbour primitive, the save/load round trip, and
that pk2d still indexes the grid by frm.
"""
from __future__ import print_function

import os
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


# Building a 560k-frame dataset from its synthetic master is the expensive part
# of these tests (~0.2-0.3s each). Cache one per case per run so the 10+ tests
# that inspect the same three cases do not rebuild them over and over.
_CACHE_DIR = None
_CASES = {}


def _case(name):
    global _CACHE_DIR
    if _CACHE_DIR is None:
        import atexit
        _CACHE_DIR = tempfile.mkdtemp(prefix="id11_f2scan_case_")
        atexit.register(
            lambda: shutil.rmtree(_CACHE_DIR, ignore_errors=True))
    if name not in _CASES:
        _CASES[name] = S.case(name, _CACHE_DIR)
    return _CASES[name]


class TestCellMap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="id11_f2scan_cell_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shapes_and_unplaced(self):
        # shape and how many frames the grid drops (over-long turns)
        self.assertEqual(_case("s4").shape, (389, 1440))
        self.assertEqual(_case("AA").shape, (251, 3600))
        self.assertEqual(_case("Cu").shape, (551, 1636))
        self.assertEqual(len(_case("Cu").unplaced_frames), 201)
        self.assertEqual(len(_case("AA").unplaced_frames), 0)
        # s4 straight-line is clean, so only the round-off boundary drops one
        self.assertLess(len(_case("s4").unplaced_frames), 3)

    def test_omega_wraps_is_set_for_multi_turn(self):
        self.assertTrue(_case("s4").omega_wraps)
        self.assertTrue(_case("Cu").omega_wraps)

    def test_cell_frame_shape_matches_dataset(self):
        ds = _case("Cu")
        self.assertEqual(ds.cell_frame.shape, ds.shape)

    def test_frame_to_cell_roundtrip(self):
        ds = _case("Cu")
        cf = ds.cell_frame
        i, j = np.nonzero(cf >= 0)
        flat = i * ds.shape[1] + j
        self.assertTrue(np.all(ds.frame_to_cell[cf[i, j]] == flat))

    def test_grids_equal_raw_at_real_cells(self):
        ds = _case("Cu")
        cf = ds.cell_frame
        real = cf >= 0
        # omega is the frame's own value at each real cell
        self.assertTrue(np.allclose(
            ds.omega[real], ds.omega_raw[cf[real]]))
        # dty is the row mean: each row is one dty value (single dty bin)
        for r in range(ds.shape[0]):
            row = cf[r]
            nz = row[row >= 0]
            if nz.size:
                self.assertAlmostEqual(
                    ds.dty[r, 0], ds.dty_raw[nz].mean())
                self.assertTrue(np.allclose(ds.dty[r], ds.dty[r, 0]))
        self.assertEqual(np.prod(ds.shape), ds.cell_frame.size)

    def test_unplaced_frames_stay_in_raw(self):
        ds = _case("Cu")
        up = ds.unplaced_frames
        self.assertLess(np.max(up), len(ds.omega_raw))
        # every unplaced frame has no cell
        self.assertTrue(np.all(ds.frame_to_cell[up] < 0))

    def test_save_load_roundtrip(self):
        ds = _case("Cu")
        path = self.tmp + "/ds.h5"
        ds.save(path)
        loaded = ImageD11.sinograms.dataset.load(path)
        self.assertTrue(np.array_equal(loaded.cell_frame, ds.cell_frame))
        self.assertTrue(np.array_equal(
            loaded.unplaced_frames, ds.unplaced_frames))
        self.assertEqual(loaded.shape, ds.shape)
        self.assertTrue(np.allclose(loaded.omega[loaded.cell_frame >= 0],
                                    ds.omega[ds.cell_frame >= 0]))


class TestGridNeighbours(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="id11_f2scan_nb_")
        self.ds = _case("Cu")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_interior_connectivity4(self):
        s1 = self.ds.shape[1]
        k = 1 * s1 + 5
        got = self.ds.grid_neighbours(k, 4)
        want = {1 * s1 + 4, 1 * s1 + 6, 0 * s1 + 5, 2 * s1 + 5}
        self.assertEqual(set(got.tolist()), want)

    def test_connectivity8_includes_diagonals(self):
        s1 = self.ds.shape[1]
        k = 1 * s1 + 5
        got = set(self.ds.grid_neighbours(k, 8).tolist())
        self.assertIn(0 * s1 + 4, got)
        self.assertIn(2 * s1 + 6, got)
        self.assertEqual(len(got) - 2, 6)  # 8 minus the two off-grid

    def test_omega_seam_wraps(self):
        s1 = self.ds.shape[1]
        k = 2 * s1 + (s1 - 1)
        got = set(self.ds.grid_neighbours(k, 4).tolist())
        self.assertIn(2 * s1 + 0, got)  # +omega wraps to column 0

    def test_empty_cells_are_not_neighbours(self):
        empties = np.nonzero(self.ds.cell_frame.ravel() < 0)[0]
        self.assertGreaterEqual(len(empties), 1)
        for k in empties[:5]:
            nb = self.ds.grid_neighbours(k, 4)
            self.assertTrue(np.all(
                self.ds.cell_frame.ravel()[nb] >= 0))


class TestPk2dIndexing(unittest.TestCase):
    """frm stays the grid index, so pk2d(ds.omega, ds.dty) still works."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="id11_f2scan_pk_")
        self.ds = _case("Cu")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pk2d_uses_grid_index(self):
        from ImageD11.sinograms.properties import pks_table
        cf = self.ds.cell_frame
        i, j = np.nonzero(cf >= 0)
        flat = (i * self.ds.shape[1] + j)
        # pick three real cells to be our peaks
        idx = flat[:3]
        pk_props = np.zeros((5, 3), np.int64)
        pk_props[0] = [5, 7, 9]     # s1
        pk_props[1] = [100, 200, 300]  # sI
        pk_props[4] = idx            # frm = grid index
        pkst = pks_table()
        pkst.pk_props = pk_props
        out = pkst.pk2d(self.ds.omega_for_bins, self.ds.dty)
        # the omega taken from the grid at the frame's own cell
        self.assertTrue(np.allclose(
            out["omega"], self.ds.omega_for_bins.flat[idx]))
        self.assertTrue(np.allclose(
            out["dty"], self.ds.dty.flat[idx]))
        # and the bliss frame number is one indirection away
        bliss = self.ds.cell_frame.flat[idx]
        self.assertTrue(np.all(bliss >= 0))


class TestPropsMask(unittest.TestCase):
    """Masked frames are skipped at labelling, not dropped from the SparseScan."""

    def setUp(self):
        import tempfile
        import h5py
        self.tmp = tempfile.mkdtemp(prefix="id11_f2scan_propsmask_")
        path = os.path.join(self.tmp, "sp.h5")
        nnz = np.array([2, 3, 1], np.uint32)
        row = np.array([1, 1, 2, 2, 2, 3], np.uint16)
        col = np.array([1, 2, 1, 2, 3, 1], np.uint16)
        inten = np.array([10., 20., 5., 6., 7., 8.], np.float32)
        with h5py.File(path, "w") as h:
            g = h.create_group("1.1")
            g.attrs["nframes"] = 3
            g.attrs["shape0"] = 8
            g.attrs["shape1"] = 8
            g.create_dataset("nnz", data=nnz)
            g.create_dataset("row", data=row)
            g.create_dataset("col", data=col)
            g.create_dataset("intensity", data=inten)
        self.path = path
        import ImageD11.sparseframe as SF
        import ImageD11.sinograms.dataset as D
        self.SF = SF
        ds = D.DataSet(sample="s", dset="d")
        ds.shape = (2, 4)
        ds.cell_frame = np.array([[0, 1, 2, 3], [4, 5, 6, 7]]).astype(np.int64)
        ds.omega = np.arange(8.).reshape(2, 4)
        ds.dty = np.zeros((2, 4))
        ds.omega_raw = np.arange(8.0)
        ds.masked_frames = np.array([], np.int64)
        ds.omega_wraps = False
        self.ds = ds

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _props(self, masked=None):
        import ImageD11.sinograms.properties as P
        s = self.SF.SparseScan(self.path, "1.1")
        s.motors["omega"] = self.ds.omega[0]
        return P.props(s, 0, ds=self.ds, masked=masked)

    def test_no_mask_all_frames(self):
        r, _ = self._props()
        self.assertEqual(r.shape[1], 3)
        self.assertEqual(r[4].tolist(), [0, 1, 2])

    def test_masked_frame_dropped_no_shift(self):
        r, _ = self._props(masked=[1])
        self.assertEqual(r.shape[1], 2)
        self.assertEqual(r[4].tolist(), [0, 2])  # frame 1 gone, others unchanged

    def test_masking_does_not_alter_scan(self):
        s = self.SF.SparseScan(self.path, "1.1")
        self.assertEqual(s.nnz.tolist(), [2, 3, 1])
        self.assertEqual(len(s.row), 6)


if __name__ == "__main__":
    unittest.main()
