#!/usr/bin/env python
from __future__ import print_function
"""
Forward-model helper for the Friedel-pair tests.

Simulates an ideal single-grain diffraction scan from a UBI matrix and the
instrument geometry, returning peaks expressed purely as *observed-style*
detector coordinates (sc, fc, omega).  The grain travels through the Ewald
sphere, so each reflection is recorded at the two entry/exit angles (the
+-eta pair).  No geometry is precomputed here, so callers can process the
output exactly as if it were measured data (e.g. ``columnfile.updateGeometry``)
with whatever instrument parameters they want to test.

The UBI convention is the indexer one (``hkl = UBI . g``); the physical
grain matrix is ``UB = inv(UBI)``, so ``g = UB . hkl = inv(UBI) . hkl``.
"""
import numpy as np

from ImageD11 import transform, unitcell, indexing
from ImageD11.columnfile import colfile_from_dict
from ImageD11.parameters import parameters


def unit_cell_from_ubi(UBI, spacegroup):
    """Unit cell implied by an indexer-convention UBI (hkl = UBI . g)."""
    return unitcell.unitcell(indexing.ubitocellpars(UBI), spacegroup)


def forward_hkl(UBI, spacegroup, dsmax=3.20):
    """Integer hkl of every allowed reflection with d* < dsmax."""
    cell = unit_cell_from_ubi(UBI, spacegroup)
    return np.array([hk for _, hk in cell.gethkls(dsmax)])


def all_forward_peaks(UBI, spacegroup, pars, dsmax=3.20, tth_cap=60.0):
    """
    Project every allowed reflection (d* < dsmax) for a grain to detector
    coordinates, keeping both Ewald solutions whose tth < tth_cap and that
    have a real eta/omega (those on the rotation axis are dropped).

    Returns a dict of numpy arrays: sc, fc, omega, tth, h, k, l, eta_true.
    ``pars`` are the instrument parameters used for the *simulation*.
    """
    hkl = forward_hkl(UBI, spacegroup, dsmax)
    g = np.dot(np.linalg.inv(UBI), hkl.T)
    wvln = pars['wavelength']
    wedge = pars['wedge']
    chi = pars['chi']
    tth, (eta1, eta2), (omega1, omega2) = transform.uncompute_g_vectors(
        g, wvln, wedge=wedge, chi=chi)

    sc_p, fc_p, om_p, tth_p, e_p = [], [], [], [], []
    h_p, k_p, l_p = [], [], []
    for eta, omega in ((eta1, omega1), (eta2, omega2)):
        fc, sc = transform.compute_xyz_from_tth_eta(tth, eta, omega, **pars)
        good = (tth > 0) & (np.abs(eta) > 0) & (tth < tth_cap)
        sc_p.append(sc[good])
        fc_p.append(fc[good])
        om_p.append(omega[good])
        tth_p.append(tth[good])
        e_p.append(eta[good])
        h_p.append(hkl[good, 0])
        k_p.append(hkl[good, 1])
        l_p.append(hkl[good, 2])

    return {
        'sc': np.concatenate(sc_p),
        'fc': np.concatenate(fc_p),
        'omega': np.concatenate(om_p),
        'tth': np.concatenate(tth_p),
        'eta_true': np.concatenate(e_p),
        'h': np.concatenate(h_p).astype(int),
        'k': np.concatenate(k_p).astype(int),
        'l': np.concatenate(l_p).astype(int),
    }


def simulated_columnfile(UBI, spacegroup, pars, dsmax=3.20, tth_cap=60.0):
    """
    Columnfile of simulated peaks carrying only the observed-style coordinates
    (sc, fc, omega) plus ground-truth h, k, l, eta_true.  The geometry
    parameters are attached so ``updateGeometry`` can recompute tth/eta/g.
    """
    d = all_forward_peaks(UBI, spacegroup, pars, dsmax=dsmax, tth_cap=tth_cap)
    cf = colfile_from_dict(d)
    cf.parameters = parameters(**pars)
    return cf


def compute_geometry(cf, pars):
    """Interpret the (sc, fc, omega) peaks with the given instrument parameters
    (as if they were observed data) and write tth/eta/gx/gy/gz onto cf."""
    cf.parameters = parameters(**pars)
    cf.updateGeometry()
    return cf
