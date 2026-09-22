# tests for the Friedel pair module (ImageD11.friedel_pairs)
"""
Covers:
  * PairMode mode resolution, legacy aliases and column names
  * the per-mode search-space (g, eta) transforms
  * the per-mode physical pair distances
  * the eta-bin subset pairing (diagonal vs horizontal)
  * end-to-end match_friedel_pairs for the three relationships
  * legacy column names written alongside the new canonical ones
"""
import numpy as np
import pytest

from ImageD11 import columnfile
from ImageD11 import friedel_pairs as fp
from ImageD11.friedel_pairs import (
    PairMode,
    _resolve_mode,
    _pair_column,
    _legacy_column,
    _sort_column,
    _resolve_chunk,
    _search_space,
    _physical_pair_distance,
    _wrap_eta,
    PeakSubsets,
    FriedelPairIndexer,
    get_pairs,
    find_pairs,
    locate_pairs_affine,
    locate_pairs,
    fit_y0,
    match_box_beam,
    calc_tth_eta,
)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
class FakeDS:
    """minimal stand-in for a sinograms dataset used by PeakSubsets"""

    obincens = np.array([15.0, 45.0])
    obinedges = np.array([0.0, 30.0, 60.0])
    ybincens = np.linspace(-10.0, 10.0, 21)
    ybinedges = np.linspace(-11.0, 11.0, 22)
    ymin = -10.0
    ymax = 10.0
    ystep = 1.0
    ostep = 1.0
    y0 = None
    dsname = "fake"

    def save(self, *a, **k):
        pass

    def correct_bins_for_half_scan(self, *a, **k):
        pass


def _make_cf(arrays):
    cf = columnfile.colfile_from_dict(arrays)
    cf.nrows = len(arrays[list(arrays)[0]])
    return cf


def _gen_synthetic(mode, n=12, seed=7):
    """
    Build a columnfile containing n reference peaks and n partners, where each
    partner satisfies the given Friedel relationship (plus a small noise so the
    KDTree search does not see exactly-degenerate residuals).
    """
    rng = np.random.default_rng(seed)
    g_ref = rng.normal(size=(n, 3))
    g_ref /= np.linalg.norm(g_ref, axis=1, keepdims=True)
    e_ref = rng.uniform(30, 150, size=n)
    om_ref = rng.uniform(10, 170, size=n)
    I_ref = rng.uniform(50, 200, size=n)
    noise_g = rng.normal(scale=0.002, size=(n, 3))
    noise_e = rng.normal(scale=0.05, size=n)
    I_part = I_ref * (1 + rng.normal(scale=0.003, size=n))

    if mode == PairMode.VERTICAL.value:
        g_part = -g_ref + noise_g
        e_part = _wrap_eta(180 - e_ref) + noise_e
        om_part = om_ref + 180
    elif mode == PairMode.DIAGONAL.value:
        g_part = -g_ref + noise_g
        e_part = _wrap_eta(180 + e_ref) + noise_e
        om_part = om_ref + rng.uniform(-2, 2, size=n)
    else:  # horizontal
        g_part = g_ref + noise_g
        e_part = _wrap_eta(-e_ref) + noise_e
        om_part = om_ref + rng.uniform(1.0, 3.0, size=n)

    return _make_cf({
        "gx": np.concatenate([g_ref[:, 0], g_part[:, 0]]),
        "gy": np.concatenate([g_ref[:, 1], g_part[:, 1]]),
        "gz": np.concatenate([g_ref[:, 2], g_part[:, 2]]),
        "eta": np.concatenate([e_ref, e_part]),
        "omega": np.concatenate([om_ref, om_part]),
        "sum_intensity": np.concatenate([I_ref, I_part]),
    })


# ─────────────────────────────────────────────────────────────────────────────
# mode resolution / naming
# ─────────────────────────────────────────────────────────────────────────────
def test_resolve_mode_aliases():
    assert _resolve_mode("omega") == "vertical_pair"
    assert _resolve_mode("omega_pair") == "vertical_pair"
    assert _resolve_mode("eta") == "diagonal_pair"
    assert _resolve_mode("eta_pair") == "diagonal_pair"
    assert _resolve_mode("horizontal") == "horizontal_pair"
    assert _resolve_mode("horizontal_pair") == "horizontal_pair"
    assert _resolve_mode("entry_exit") == "horizontal_pair"
    assert _resolve_mode(PairMode.VERTICAL) == "vertical_pair"
    with pytest.raises(ValueError):
        _resolve_mode("bogus")


def test_pair_and_legacy_columns():
    assert _pair_column("omega") == "vertical_pair_id"
    assert _pair_column("eta") == "diagonal_pair_id"
    assert _pair_column("horizontal") == "horizontal_pair_id"
    assert _legacy_column("omega") == "omega_pair_id"
    assert _legacy_column("eta") == "eta_pair_id"
    assert _legacy_column("horizontal") is None
    assert _sort_column("omega") == "omega"
    assert _sort_column("eta") == "eta"
    assert _sort_column("horizontal") == "eta"


def test_resolve_chunk():
    assert _resolve_chunk("scans") == ("vertical_pair", "scans")
    assert _resolve_chunk("frames") == ("vertical_pair", "frames")
    assert _resolve_chunk("eta_bins") == ("diagonal_pair", "eta_bins")
    assert _resolve_chunk("horizontal_bins") == ("horizontal_pair", "eta_bins")
    assert _resolve_chunk("minus_eta_bins") == ("horizontal_pair", "eta_bins")
    with pytest.raises(ValueError):
        _resolve_chunk("bogus")


# ─────────────────────────────────────────────────────────────────────────────
# search space transforms
# ─────────────────────────────────────────────────────────────────────────────
def test_search_space_flips():
    cf = _make_cf({
        "gx": np.array([1.0, -1.0, 1.0, -1.0]),
        "gy": np.array([0.2, -0.2, -0.2, 0.2]),
        "gz": np.array([0.1, -0.1, 0.1, -0.1]),
        "eta": np.array([60.0, 60.0, -60.0, -60.0]),
        "sum_intensity": np.array([100.0, 100.0, 100.0, 100.0]),
    })
    # index 1: g=(-1,-0.2,-0.1), eta=60
    s_v, _ = _search_space(cf, [1], flip="omega")
    np.testing.assert_allclose(s_v[0, :3], [1.0, 0.2, 0.1], atol=1e-8)
    np.testing.assert_allclose(s_v[0, 3], 120.0, atol=1e-8)

    s_d, _ = _search_space(cf, [1], flip="eta")
    np.testing.assert_allclose(s_d[0, :3], [1.0, 0.2, 0.1], atol=1e-8)
    np.testing.assert_allclose(s_d[0, 3], -120.0, atol=1e-8)

    # index 3: g=(-1,0.2,-0.1), eta=-60 ; horizontal keeps g and flips eta
    s_h, _ = _search_space(cf, [3], flip="horizontal")
    np.testing.assert_allclose(s_h[0, :3], [-1.0, 0.2, -0.1], atol=1e-8)
    np.testing.assert_allclose(s_h[0, 3], 60.0, atol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# physical pair distances
# ─────────────────────────────────────────────────────────────────────────────
def test_physical_pair_distance():
    # pair (0,1): eta 30 & 60; g vectors opposite for vertical/diagonal,
    # same-ish for horizontal.
    cf = _make_cf({
        "gx": np.array([1.0, -1.0]),
        "gy": np.array([0.2, -0.2]),
        "gz": np.array([0.1, -0.1]),
        "eta": np.array([30.0, 60.0]),
        "sum_intensity": np.array([100.0, 90.0]),
    })
    dgv, deta, dlog = _physical_pair_distance(cf, [0], [1], pair_type="omega")
    np.testing.assert_allclose(dgv, [0.0], atol=1e-8)
    np.testing.assert_allclose(deta, [90.0], atol=1e-8)

    dgv, deta, dlog = _physical_pair_distance(cf, [0], [1], pair_type="diagonal")
    np.testing.assert_allclose(dgv, [0.0], atol=1e-8)
    np.testing.assert_allclose(deta, [150.0], atol=1e-8)

    dgv, deta, dlog = _physical_pair_distance(cf, [0], [1], pair_type="horizontal")
    np.testing.assert_allclose(dgv, [2.049], atol=1e-3)
    np.testing.assert_allclose(deta, [90.0], atol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# eta-bin subset pairing
# ─────────────────────────────────────────────────────────────────────────────
def test_eta_bins_pairing_diagonal_and_horizontal():
    # half acquisition so scans/frames are skipped, only eta bins built
    cf = _make_cf({
        "gx": np.array([1.0]), "gy": np.array([0.0]), "gz": np.array([0.0]),
        "eta": np.array([0.0]), "omega": np.array([10.0]),
        "sum_intensity": np.array([1.0]),
    })
    psub = PeakSubsets(cf, FakeDS(), n_eta_bins=360, pairing="diagonal")
    diag = psub.eta_bins_subsets
    # diagonal: eta bin i pairs with bin i+180 -> exactly 180 pairs for 360 bins
    assert len(diag) == 180
    for p in diag[:5]:
        # partner must be ~180 deg away
        assert abs(p.eta_hi - (p.eta_lo + 180)) < 1e-6 or \
               abs(p.eta_hi - (p.eta_lo - 180)) < 1e-6

    psub = PeakSubsets(cf, FakeDS(), n_eta_bins=360, pairing="horizontal")
    horiz = psub.eta_bins_subsets
    # horizontal: eta bin i pairs with bin -i -> exactly 180 pairs
    assert len(horiz) == 180
    for p in horiz[:5]:
        assert abs(p.eta_hi + p.eta_lo) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# end-to-end pairing
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["vertical_pair", "diagonal_pair", "horizontal_pair"])
def test_match_friedel_pairs(mode):
    n = 12
    cf = _gen_synthetic(mode, n=n)
    ds = FakeDS()
    idx = FriedelPairIndexer(cf, ds, tol_gv=0.05, tol_eta=1.0, tol_logI=1.0,
                             weights={"gx": 1, "gy": 1, "gz": 1, "eta": 1, "I": 1},
                             n_steps=8)
    out = idx.match_friedel_pairs(pair_type=mode, drop_unpaired=False,
                                  filter_mode="relaxed", doplot=False)
    col = mode + "_id"
    assert col in out.titles
    # every peak is found in a pair (n ref + n partner)
    assert int((out.getcolumn(col) > -1).sum()) == 2 * n
    i1, i2 = get_pairs(out, mode)
    assert len(i1) == n


def test_match_friedel_pairs_legacy_names_still_work():
    cf = _gen_synthetic("vertical_pair", n=6)
    ds = FakeDS()
    idx = FriedelPairIndexer(cf, ds, tol_gv=0.05, tol_eta=1.0, tol_logI=1.0,
                             weights={"gx": 1, "gy": 1, "gz": 1, "eta": 1, "I": 1},
                             n_steps=8)
    out = idx.match_friedel_pairs(pair_type="omega", drop_unpaired=False,
                                  filter_mode="relaxed", doplot=False)
    # legacy column name present and shares the buffer with the canonical one
    assert "omega_pair_id" in out.titles
    assert "vertical_pair_id" in out.titles
    assert out.getcolumn("omega_pair_id") is out.getcolumn("vertical_pair_id")


# ─────────────────────────────────────────────────────────────────────────────
# ring-level helpers
# ─────────────────────────────────────────────────────────────────────────────
def test_find_pairs_diagonal():
    n = 10
    rng = np.random.default_rng(5)
    g = rng.normal(size=(n, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    e = rng.uniform(30, 150, n)
    g_part = -g
    e_part = _wrap_eta(180 + e)
    cf = _make_cf({
        "gx": np.concatenate([g[:, 0], g_part[:, 0]]),
        "gy": np.concatenate([g[:, 1], g_part[:, 1]]),
        "gz": np.concatenate([g[:, 2], g_part[:, 2]]),
        "eta": np.concatenate([e, e_part]),
        "omega": np.concatenate([np.full(n, 30.0), np.full(n, 210.0)]),
        "dty": np.concatenate([rng.uniform(-8, 8, n), rng.uniform(-8, 8, n)]),
        "sum_intensity": np.concatenate([rng.uniform(50, 200, n), rng.uniform(50, 200, n)]),
    })
    ip, im = find_pairs(cf, gvtol=0.05, mode="diagonal_pair")
    assert len(ip) == n
    assert (cf.eta[ip] > 0).all()
    assert (cf.eta[im] < 0).all()


def test_locate_pairs_and_affine_equivalent_to_eta_pairs():
    n = 10
    rng = np.random.default_rng(5)
    g = rng.normal(size=(n, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    e = rng.uniform(30, 150, n)
    om = rng.uniform(20, 160, n)
    cf = _make_cf({
        "gx": np.concatenate([g[:, 0], -g[:, 0]]),
        "gy": np.concatenate([g[:, 1], -g[:, 1]]),
        "gz": np.concatenate([g[:, 2], -g[:, 2]]),
        "eta": np.concatenate([e, _wrap_eta(180 + e)]),
        "omega": np.concatenate([om, om + 1.0]),
        "dty": np.concatenate([rng.uniform(-8, 8, n), rng.uniform(-8, 8, n)]),
        "sum_intensity": np.concatenate([rng.uniform(50, 200, n), rng.uniform(50, 200, n)]),
    })
    ip = np.flatnonzero(cf.eta > 0)
    im = np.flatnonzero(cf.eta < 0)
    s0, v = locate_pairs_affine(cf, (ip, im))
    assert s0.shape == (2, n) and v.shape == (2, n)
    # locate_pairs(y0) must equal the y0-shifted affine form
    sx, sy = locate_pairs(cf, (ip, im), y0=1.0)
    np.testing.assert_allclose(sx, s0[0] - 1.0 * v[0])
    np.testing.assert_allclose(sy, s0[1] - 1.0 * v[1])
    # locate_eta_pairs writes identical sx/sy into cf
    from ImageD11.friedel_pairs import locate_eta_pairs
    e_cf = cf.copy()
    locate_eta_pairs(e_cf, (ip, im), y0=1.0)
    np.testing.assert_allclose(e_cf.getcolumn('sx')[ip], sx)


def test_fit_y0():
    n = 60
    rng = np.random.default_rng(2)
    g = rng.normal(size=(n, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    e = rng.uniform(30, 150, n)
    om = rng.uniform(20, 160, n)
    cf = _make_cf({
        "gx": np.concatenate([g[:, 0], -g[:, 0]]),
        "gy": np.concatenate([g[:, 1], -g[:, 1]]),
        "gz": np.concatenate([g[:, 2], -g[:, 2]]),
        "eta": np.concatenate([e, _wrap_eta(180 + e)]),
        "omega": np.concatenate([om, om + 1.0]),
        "dty": np.concatenate([rng.uniform(-8, 8, n), rng.uniform(-8, 8, n)]),
        "sum_intensity": np.concatenate([rng.uniform(50, 200, n), rng.uniform(50, 200, n)]),
    })
    ip = np.flatnonzero(cf.eta > 0); im = np.flatnonzero(cf.eta < 0)
    best, y0s, std = fit_y0(cf, (ip, im), np.linspace(-3, 3, 25), npks=40, nbx=64, nby=64)
    assert np.isfinite(best)
    assert std.shape == y0s.shape
    assert len(y0s) == 25


def test_match_box_beam():
    n = 6
    rng = np.random.default_rng(11)
    om = rng.uniform(10, 170, n); e = rng.uniform(30, 150, n)
    tth = np.full(n, 10.0); I = rng.uniform(50, 200, n)
    xl = rng.uniform(-1, 1, n); yl = rng.uniform(-1, 1, n); zl = rng.uniform(-1, 1, n)
    cf = _make_cf({
        "omega": np.concatenate([om, (om + 180) % 360]),
        "eta": np.concatenate([e, (180 - e) % 360]),
        "tth": np.concatenate([tth, tth]),
        "sum_intensity": np.concatenate([I, I]),
        "xl": np.concatenate([xl, xl]),
        "yl": np.concatenate([yl, yl]),
        "zl": np.concatenate([zl, zl]),
        "ds": np.concatenate([np.full(n, 1.0), np.full(n, 1.0)]),
    })
    cf.parameters.set('wavelength', 0.2)
    cpair = match_box_beam(cf, pair_type='vertical_pair', doplot=False)
    assert cpair.nrows == 2 * cf.nrows  # every peak is used on both tree sides
    assert 'gx' in cpair.titles
    assert np.isfinite(cpair.gx).all()


def test_legacy_alias_survives_filter_roundtrip(tmp_path):
    """canonical + legacy pair-id columns keep sharing one buffer through save/filter/copy."""
    out = _make_cf({
        "gx": np.array([1.0, -1.0]), "gy": np.array([0.0, 0.0]),
        "gz": np.array([0.0, 0.0]), "eta": np.array([40.0, -140.0]),
        "omega": np.array([30.0, 210.0]), "sum_intensity": np.array([100.0, 98.0]),
    })
    out.addcolumn(np.array([0, 0]), 'vertical_pair_id')
    out.addcolumn(out.getcolumn('vertical_pair_id'), 'omega_pair_id')
    assert out.getcolumn('vertical_pair_id') is out.getcolumn('omega_pair_id')
    fn = str(tmp_path / "cf.h5")
    columnfile.colfile_to_hdf(out, fn)
    loaded = columnfile.columnfile(fn)
    assert loaded.getcolumn('vertical_pair_id') is loaded.getcolumn('omega_pair_id')
    loaded.filter(np.array([True, True]))
    assert loaded.getcolumn('vertical_pair_id') is loaded.getcolumn('omega_pair_id')
    copied = loaded.copy()
    assert copied.getcolumn('vertical_pair_id') is copied.getcolumn('omega_pair_id')
    rows = loaded.copyrows(np.arange(2))
    assert rows.getcolumn('vertical_pair_id') is rows.getcolumn('omega_pair_id')


# ─────────────────────────────────────────────────────────────────────────────
# self-contained Si_cube cf_4d fixture
# ─────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path

FIXTURE = Path(__file__).parent / "data" / "Si_cube_friedel_test.cf_4d.h5"
FIXTURE_UBI = Path(__file__).parent / "data" / "Si_cube_friedel_test.ubi"
SI_A = 5.43094  # silicon cell length (Angstrom)


def _allowed_si_rings(gmax):
    """|g| of the allowed (centrosymmetric FCC) silicon reflections <= gmax."""
    import itertools
    s = set()
    for h, k, l in itertools.product(range(-12, 13), repeat=3):
        if h == k == l == 0:
            continue
        if (h + k) % 2 or (k + l) % 2 or (l + h) % 2:
            continue
        s.add(round(np.sqrt(h * h + k * k + l * l) / SI_A, 3))
    return np.array(sorted(v for v in s if v <= gmax))


def _ring_residual(gg, rings):
    j = np.argmin(np.abs(rings - gg))
    return abs(rings[j] - gg) / gg * 100.0


@pytest.mark.skipif(not FIXTURE.exists(), reason="Si_cube cf_4d fixture not present")
def test_si_cube_cf4d_fixture_geometry_and_friedel_pairs():
    """The self-contained cf_4d round-trips its geometry and yields Friedel pairs
    that sit on the allowed silicon reflections."""
    c = columnfile.columnfile(str(FIXTURE))
    # geometry + cell restored from the HDF5 attributes
    assert abs(c.parameters.get('wavelength') - 0.1897) < 1e-3
    assert abs(c.parameters.get('distance') - 151015.75) < 1.0
    assert abs(float(c.parameters.get('cell__a')) - SI_A) < 1e-4
    # the raw peak columns are present, the computed geometry is not stored
    for col in ('sc', 'fc', 'omega', 'dty', 'sum_intensity', 'Number_of_pixels'):
        assert col in c.titles
    assert 'gx' not in c.titles and 'tth' not in c.titles

    c.updateGeometry()
    assert 'gx' in c.titles and 'tth' in c.titles

    g = np.sqrt(c.gx ** 2 + c.gy ** 2 + c.gz ** 2)
    rings = _allowed_si_rings(g.max())
    ip, im = find_pairs(c, gvtol=0.005, mode='diagonal_pair')
    assert len(ip) > 0
    # both members of a pair are on (near) the same silicon ring
    sag = np.abs(g[ip] - g[im]) / np.maximum(g[ip], g[im]) * 100.0
    assert np.median(sag) < 0.5
    r1 = np.array([_ring_residual(x, rings) for x in g[ip]])
    r2 = np.array([_ring_residual(x, rings) for x in g[im]])
    both = (r1 < 1.0) & (r2 < 1.0)
    assert both.mean() > 0.9

    # cross-check against the indexed grain orientation: pairs are real Si
    # reflections (integer hkl under h = UB.g) that flip hkl -> -h,-k,-l.
    if FIXTURE_UBI.exists():
        UB = np.loadtxt(str(FIXTURE_UBI)).reshape(3, 3)
        gmat = np.column_stack((c.gx, c.gy, c.gz))     # (n,3)
        hkl = np.dot(UB, gmat.T)                       # (3,n); hkl = UB.g
        ni = np.abs(hkl - np.round(hkl)).max(axis=0)
        iok = (ni[ip] < 0.1) & (ni[im] < 0.1)
        assert iok.mean() > 0.9, "pairs should index to silicon hkl"
        flip = np.linalg.norm(hkl[:, ip] + hkl[:, im], axis=0)
        assert (flip[iok] < 0.1).all(), "indexed pairs must be hkl -> -h,-k,-l"

