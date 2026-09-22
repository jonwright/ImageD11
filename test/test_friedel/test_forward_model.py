#!/usr/bin/env python
from __future__ import print_function
"""
Friedel-pair tests built on a *forward model*.

A single grain's reflections (all allowed hkl with d* < dsmax) are simulated
for an ideal scan and expressed purely as observed-style detector coordinates
(sc, fc, omega).  The tests then:

  * check the three Friedel relationships pair *perfectly* on ideal data
    (horizontal = same g at +-eta, vertical/diagonal = hkl -> -h,-k,-l),
  * show how each of three instrument mismatches degrades the match, so the
    tolerance must be relaxed: wedge=0.5 deg, tilt_x=0 (true is non-zero) and
    a grain translation of (3,2,1) px = (225,150,75) um,
  * validate the forward model against the real Si_cube observations by
    assigning hkl from the UBI to both sets and comparing sc/fc/omega of the
    matched [h,k,l,sign(eta)] peaks.
"""
import os
import numpy as np
import pytest

from ImageD11 import friedel_pairs as fp
from ImageD11 import columnfile

from forward_model import (simulated_columnfile, compute_geometry,
                           all_forward_peaks)

SPACEGROUP = 227  # Fd-3m
DSMAX = 3.20      # d* (1/Angstrom) cap; for Si the first rings sit below 3.2
TTH_CAP = 60.0    # keep reflections with tth < 60 deg


def _data_path(name):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", name))


def _load():
    obs = columnfile.columnfile(_data_path("Si_cube_friedel_test.cf_4d.h5"))
    pars = dict(obs.parameters.get_parameters())
    ubi = np.loadtxt(_data_path("Si_cube_friedel_test.ubi")).reshape(3, 3)
    return ubi, pars


def _hkl_array(cf):
    return np.column_stack((cf.h, cf.k, cf.l))


def _ideal_pars():
    return _load()[1]


# -----------------------------------------------------------------------------
# 1) ideal forward model pairs perfectly in all three modes
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ideal_sim():
    ubi, pars = _load()
    sim = simulated_columnfile(ubi, SPACEGROUP, pars, dsmax=DSMAX, tth_cap=TTH_CAP)
    compute_geometry(sim, pars)  # interpret with the (same) instrument params
    return sim


def test_forward_model_yields_many_peaks(ideal_sim):
    # 8 rings of Si -> 10^3-10^4 peaks for pairing
    assert ideal_sim.nrows > 1000, ideal_sim.nrows


@pytest.mark.parametrize("mode", ["horizontal_pair", "vertical_pair", "diagonal_pair"])
def test_ideal_pairing_is_perfect(ideal_sim, mode):
    sim = ideal_sim
    ip, im = fp.find_pairs(sim, gvtol=1e-3, mode=mode)
    assert len(ip) > 0, "no pairs found for %s" % mode
    gref = np.column_stack((sim.gx[ip], sim.gy[ip], sim.gz[ip]))
    gpart = np.column_stack((sim.gx[im], sim.gy[im], sim.gz[im]))
    sign = 1.0 if mode == "horizontal_pair" else -1.0
    dist = np.sqrt(((gref - sign * gpart) ** 2).sum(axis=1))
    assert dist.max() < 1e-6, "ideal pairs should be exactly related"
    # vertical/diagonal are hkl -> -h,-k,-l; horizontal keeps the same hkl
    flip = ((sim.h[ip] + sim.h[im]) == 0) & ((sim.k[ip] + sim.k[im]) == 0) & \
           ((sim.l[ip] + sim.l[im]) == 0)
    if mode == "horizontal_pair":
        assert flip.mean() < 0.05
    else:
        assert flip.mean() > 0.99


# -----------------------------------------------------------------------------
# 2) instrument mismatches degrade the pairing (tolerance must be relaxed)
# -----------------------------------------------------------------------------
_PERTURBATIONS = {
    "wedge_0p5deg":  lambda p: dict(p, wedge=0.5),
    "tilt_x_zero":   lambda p: dict(p, tilt_x=0.0),
    "translation_3_2_1px": lambda p: dict(p, t_x=225.0, t_y=150.0, t_z=75.0),
}


def _diagonal_pairs(ubi, pars_sim, pars_interpret, gvtol):
    sim = simulated_columnfile(ubi, SPACEGROUP, pars_sim, dsmax=DSMAX, tth_cap=TTH_CAP)
    compute_geometry(sim, pars_interpret)
    ip, _ = fp.find_pairs(sim, gvtol=gvtol, mode="diagonal_pair")
    return len(ip)


def test_perturbations_need_a_larger_tolerance():
    ubi, pars = _load()
    n_ideal = _diagonal_pairs(ubi, pars, pars, 2e-3)
    for name, mutate in _PERTURBATIONS.items():
        sim_pars = mutate(pars)
        n_tight = _diagonal_pairs(ubi, sim_pars, pars, 2e-3)
        n_loose = _diagonal_pairs(ubi, sim_pars, pars, 0.05)
        assert n_tight < n_ideal, "%s should lose pairs at a tight tolerance" % name
        # the mismatch is recoverable by relaxing the g-vector tolerance
        assert n_loose >= 0.9 * n_ideal, "%s not recoverable even at 0.05" % name


# -----------------------------------------------------------------------------
# 3) forward model matches the observed Si_cube peaks
# -----------------------------------------------------------------------------
def test_forward_model_matches_observed():
    ubi, pars = _load()
    obs = columnfile.columnfile(_data_path("Si_cube_friedel_test.cf_4d.h5"))
    compute_geometry(obs, pars)
    g = np.column_stack((obs.gx, obs.gy, obs.gz))
    hkl = np.dot(ubi, g.T).T
    ni = np.abs(hkl - np.round(hkl)).max(axis=1)
    indexed = ni < 0.05
    hs = np.round(hkl[indexed]).astype(int)
    eta_obs = np.degrees(np.arctan2(-g[indexed, 1], g[indexed, 2]))

    pred = all_forward_peaks(ubi, SPACEGROUP, pars, dsmax=DSMAX, tth_cap=TTH_CAP)
    eta_pred = pred["eta_true"]
    lookup = {}
    for i in range(len(pred["h"])):
        key = (int(pred["h"][i]), int(pred["k"][i]), int(pred["l"][i]),
               int(np.sign(eta_pred[i])))
        lookup[key] = i

    dsc, dfc, dom = [], [], []
    for j in range(hs.shape[0]):
        key = (int(hs[j, 0]), int(hs[j, 1]), int(hs[j, 2]), int(np.sign(eta_obs[j])))
        if key in lookup:
            i = lookup[key]
            dsc.append(obs.sc[indexed][j] - pred["sc"][i])
            dfc.append(obs.fc[indexed][j] - pred["fc"][i])
            dom.append(obs.omega[indexed][j] - pred["omega"][i])
    dsc = np.array(dsc); dfc = np.array(dfc); dom = np.array(dom)

    # almost all indexed observed peaks should have a forward prediction
    assert len(dsc) > 0.9 * indexed.sum()
    # the idealised geometry predicts the measured positions to sub-pixel /
    # a fraction of a degree (the geometry was already calibrated).
    assert np.mean(np.abs(dsc)) < 1.0
    assert np.mean(np.abs(dfc)) < 1.0
    assert np.mean(np.abs(dom)) < 0.2
    assert np.abs(dsc).max() < 3.0
    assert np.abs(dfc).max() < 3.0
    assert np.abs(dom).max() < 0.5
