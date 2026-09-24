"""Build synthetic f2scan master files and drive them through the real
import/guess_shape/guessbins path.

Every case is described by (omega_start, omega_end, npoints, dty_start,
dty_end, setpoint_step): the rotation and translation are linear, so
np.linspace reproduces them, and setpoint_step is the value recorded in
fscan_parameters (which may differ slightly from the measured step).

REAL holds straight-line fits to three real f2scan files (from the
diffrz_trig / diffty_trig arrays):

    s4 : step 0.25   -> 360/step = 1440,   389 whole turns.   A clean scan.
    AA : step 0.10   -> 360/step = 3600,   but 250 turns + a partial 251st.
    Cu : step 0.22   -> 360/step = 1636.4, a step that does not divide 360,
         and a partial last turn.           The "misses periodic" case.
"""
from __future__ import print_function

import os

import h5py
import numpy as np

import ImageD11.sinograms.dataset as IMGD

# (omega_start, omega_end, npoints, dty_start, dty_end, setpoint_step)
REAL = {
    "s4": (0.0, 140039.7445, 560160, 11.60, 10.8218, 0.2500098418),
    "AA": (0.0, 90356.4008, 903565, 11.2431, 10.7411, 0.1000038362),
    "Cu": (0.0, 198359.6872, 901636, 11.7539, 10.6519, 0.2200085735),
}


def linspace_case(omega, dty, npoints, setpoint_step, omega_mod360=True):
    """Build the (omega_abs, omega_col, dty, title) for one case."""
    omega_abs = np.linspace(omega[0], omega[1], npoints)
    omega_col = np.mod(omega_abs, 360.0) if omega_mod360 else omega_abs
    dty_arr = np.linspace(dty[0], dty[1], npoints)
    dty_slope = 0.0 if npoints < 2 else (dty[1] - dty[0]) / (npoints - 1)
    title = (
        "f2scan diffrz 0 %.6f diffty %.4f, %.6e %d 0.002 0.00200017"
        % (setpoint_step, dty[0], dty_slope, npoints)
    )
    return omega_abs, omega_col, dty_arr, title


def make_master(path, npoints, omega_abs, omega_col, dty, setpoint_step):
    """A minimal bliss-like f2scan master for npoints frames."""
    with h5py.File(path, "w") as h:
        g = h.create_group("1.1")
        g["title"] = ("f2scan diffrz 0 %.6f diffty 0, 0 %d 0.002 0.00200017"
                      % (setpoint_step, npoints))
        g.create_dataset("measurement/eiger",
                         data=np.zeros((npoints, 1, 1), np.int32))
        g.create_dataset("measurement/diffrz_cen360", data=omega_col)
        g.create_dataset("measurement/diffrz_trig", data=omega_abs)
        g.create_dataset("measurement/diffty_trig", data=dty)
        g.create_dataset("instrument/positioners/diffty", data=dty)
        p = g.create_group("instrument/fscan_parameters")
        p["step_size"] = float(setpoint_step)
        p["npoints"] = npoints
    return path


def build_dataset(path, tmpdir):
    """Import the master through the real pipeline (skipping the image and
    nnz parts, which a synthetic file does not carry)."""
    ds = IMGD.DataSet(dataroot=".", analysispath=tmpdir, sample="s", dset="d",
                      detector="eiger", omegamotor="diffrz_cen360",
                      dtymotor="diffty")
    ds.masterfile = path
    ds.import_scans()
    ds.imageshape = (1, 1)
    ds.import_motors_from_master()
    ds.guess_shape()
    ds.guessbins()
    return ds


def build_direct(omega, dty, shape, omega_wraps):
    """A DataSet made by hand, with omega/dty/shape/omega_wraps given. Bypasses
    guess_shape (and the master file), so the omega_wraps flag is exactly what
    was supplied rather than being set by the scan type. omega here is whatever
    the motor column holds, e.g. already mod-360 positions."""
    ds = IMGD.DataSet(sample="S", dset="d")
    ds.shape = tuple(shape)
    ds.omega = np.asarray(omega, float).reshape(shape)
    ds.dty = np.asarray(dty, float).reshape(shape)
    ds.omega_wraps = omega_wraps
    ds.guessbins()
    return ds


def regular_mod360(nturns, npts_turn, dty_start=0.0, dty_step=0.002,
                   omega_start=0.0):
    """Omega and dty arrays that lie exactly on a regular grid.

    omega is stored as mod-360 positions (it resets each turn), with a step
    that divides 360, so every turn samples the same positions and each
    (turn, position) cell holds one frame. dty is constant along a turn.
    """
    om_mod = np.tile(omega_start + np.arange(npts_turn) * (360.0 / npts_turn),
                     (nturns, 1))
    dty = np.repeat(dty_start + np.arange(nturns) * dty_step, npts_turn)
    dty = dty.reshape(nturns, npts_turn)
    return om_mod, dty, (nturns, npts_turn)


def case(name, tmpdir):
    """Build the named real-case synthetic, return the DataSet it produces."""
    om0, om1, npoints, dt0, dt1, setpoint = REAL[name]
    omega = (om0, om1)
    dty = (dt0, dt1)
    omega_abs, omega_col, dty_arr, _ = linspace_case(omega, dty, npoints,
                                                    setpoint)
    path = os.path.join(tmpdir, "%s_synth.h5" % name)
    make_master(path, npoints, omega_abs, omega_col, dty_arr, setpoint)
    return build_dataset(path, tmpdir)


def make_case(omega, dty, npoints, setpoint_step, tmpdir, name="case",
              omega_mod360=True):
    """Generic generator from an (omega, dty, npoints, setpoint_step) table."""
    omega_abs, omega_col, dty_arr, _ = linspace_case(omega, dty, npoints,
                                                     setpoint_step,
                                                     omega_mod360=omega_mod360)
    path = os.path.join(tmpdir, "%s_synth.h5" % name)
    make_master(path, npoints, omega_abs, omega_col, dty_arr, setpoint_step)
    return build_dataset(path, tmpdir)


def fscan2d_zigzag(n_dty, n_omega, dty_start, dty_step, omega_start, omega_step):
    """Flattened (omega, dty) arrays for a regular fscan2d scan.

    dty is the slow axis (one value per row, step dty_step), omega the fast axis
    flying as a zig-zag: each row sweeps the same omega range, alternating
    direction, so the same grain lands in the same omega column whatever the
    direction. Returns (omega, dty, shape) with omega/dty dty-major.
    """
    omega = np.empty((n_dty, n_omega))
    for r in range(n_dty):
        row = omega_start + np.arange(n_omega) * omega_step
        omega[r] = row if r % 2 == 0 else row[::-1]
    dty = np.repeat(dty_start + np.arange(n_dty) * dty_step, n_omega)
    return omega.ravel(), dty, (n_dty, n_omega)


def make_fscan2d_master(path, omega, dty, n_dty, n_omega, dty_start, dty_step,
                        omega_start, omega_step, omegamotor="diffrz_cen360",
                        dtymotor="diffty"):
    """A minimal fscan2d master: dty slow axis, omega fast axis (zig-zag)."""
    with h5py.File(path, "w") as h:
        g = h.create_group("1.1")
        g["title"] = ("fscan2d dty %.6f %.6f %d rot %.6f %.6f %d 0.002 0.00200017"
                      % (dty_start, dty_step, n_dty, omega_start, omega_step,
                         n_omega))
        g.create_dataset("measurement/eiger",
                         data=np.zeros((len(omega), 1, 1), np.int32))
        g.create_dataset("measurement/%s" % omegamotor, data=omega)
        g.create_dataset("instrument/positioners/%s" % dtymotor, data=dty)
        p = g.create_group("instrument/fscan_parameters")
        p["slow_npoints"] = n_dty
        p["fast_npoints"] = n_omega
        p["step_size"] = float(omega_step)
    return path


def make_fscan2d_case(n_dty, n_omega, dty_start, dty_step, omega_start,
                      omega_step, tmpdir, name="f2d", outlier=None):
    """Build and import an fscan2d zig-zag dataset.

    outlier = (frame_index, dty_value) to overwrite one frame's dty, exercising
    the regular-sinogram dty handling (row median ignores it, projection_shifts
    reports it).
    """
    omega, dty, shape = fscan2d_zigzag(n_dty, n_omega, dty_start, dty_step,
                                       omega_start, omega_step)
    if outlier is not None:
        dty = dty.copy()
        dty[outlier[0]] = outlier[1]
    path = os.path.join(tmpdir, "%s_synth.h5" % name)
    make_fscan2d_master(path, omega, dty, n_dty, n_omega, dty_start, dty_step,
                        omega_start, omega_step)
    return build_dataset(path, tmpdir)
