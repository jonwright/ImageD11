from __future__ import print_function, division

import os, h5py, numpy as np
import fast_histogram
import logging

import ImageD11.grain
import ImageD11.unitcell
import ImageD11.sinograms.properties
from ImageD11.blobcorrector import get_corrector
from ImageD11.columnfile import colfile_from_dict

"""
TO DO: 

- debug / convert this to work on f2scan and/or fscan2d data
- treat 360 scans starting near the center (monitor/hits)
- send pyFAI style integration jobs
- mca harvesting for fluo tomo
- scalar reconstructions (counters, roi, etc)
- 1d reconstructions (fluoct xrdct)
- run peaksearch/segmentations
- peak sinograms
- linking or merging peaks across dty/omega
- indexing and reconstructing grains

"""


# POSSIBLE_DETECTOR_NAMES = ("frelon3", "eiger")


def guess_chunks(name, shape):
    if name == "omega":
        return (shape[0], 1)
    if name == "dty":
        return (1, shape[1])
    if name in ("omega_raw", "dty_raw", "monitor_raw", "nnz_raw"):
        # long 1-D master-length arrays: chunk along the single axis
        return (max(1, shape[0] // 1000),)
    return shape


def guess_omega_step( omega, rptcut=0.02 ):
    """
    Estimates the step in omega angle data for multi-turn

    Case 1: more turns try to interleave intentionally (irrationals)
        - step should reduce smoothly as more frames are added
        - example: step = pi or sqrt(2) or 180 * (3 - np.sqrt(5)) or any irrational
        - data improve resolution 'forever'
    Case 2: more turns repeat the same angles over and over
        - step stays the same when more frames are added
        - example: 0.25, 0.3, 0.5, 1.0, etc
    Case 1&2: Prime rationals. Frames start interleaving, eventually repeat.
        - example: 17 degree step. Needs 17 turns, then starts repeating.
        - example: 1/3607 degree step. Needs a lot of turns, then starts repeating.

    rptcut = tolerance to decide if frames are the same angle.
             A fraction of the largest step found.
    """
    omega = np.asarray(omega)
    v = (omega - omega.min()) % 360    # Values mod 360 to put frames in order
    v.sort()           # Adjacent frames in angle
    dv = v[1:]-v[:-1]  # Step from one frame to the next
    # If we have (many) intentional repeats the pattern is
    # 0,0,0,step,0,0,0,step,0,0,0,step,...
    # The max is the largest step we might want to use
    # The min may be zero
    guess = dv[dv>(dv.max()*rptcut)].mean()
    # print('mx, avg',dv.max(), dv.mean(), guess)
    return guess


def bin_phase(values, step):
    """Where to put the first bin centre so that the bins line up with values.

    Returns the circular mean of values modulo step, in [0, step). Circular
    because the residuals pile up at both ends when the bins are nearly right,
    and a linear mean would then land half a step away from all of them.
    """
    r = (np.asarray(values).ravel() % step) * (2 * np.pi / step)
    phase = np.arctan2(np.sin(r).mean(), np.cos(r).mean()) * step / (2 * np.pi)
    return phase % step


def cell_frame_from_raw(omega_seg, dty_seg, step, base=0):
    """Grid occupied by one continuous f2scan.

    omega_seg / dty_seg: the absolute (or mod-360) motor positions of one
        continuous rotation, in acquisition order.
    step: the setpoint step in deg (instrument/fscan_parameters/step_size).
    base: the offset of this scan's first frame in the dataset's raw arrays.

    Returns (cell_frame, unplaced, n_turns, s1):

        cell_frame: (n_turns, s1) int array, value = base + index of the raw
            frame in the cell, or -1 where the turn is short.
        unplaced: integer array of the raw indices (within this segment) of
            frames an over-long turn cannot hold. These are real data that the
            grid drops; they stay in omega_raw/dty_raw.
        n_turns: number of rows (distinct turns).
        s1: columns, round(360 / step).

    A turn is floor(unwrapped_omega / 360). A frame whose position within its
    turn is >= s1 belongs to an over-long turn and is dropped; a short turn
    leaves trailing -1 cells. See the plan's section 1.
    """
    omega_seg = np.asarray(omega_seg, float).ravel()
    # unwrap to a continuous angle: mod-360 positions come back continuous,
    # absolute positions are untouched by unwrap (no 180-deg jumps).
    unwrapped = np.unwrap(omega_seg * np.pi / 180.0) * 180.0 / np.pi
    turn = np.floor(unwrapped / 360.0).astype(np.int64)
    if turn.size:
        turn = turn - turn.min()  # a negative-omega scan starts at turn 0
    s1 = int(np.round(360.0 / step))
    n_turns = int(turn.max()) + 1 if turn.size else 0
    cell_frame = np.full((n_turns, s1), -1, np.int64)
    # position within a turn: frames are contiguous and in acquisition order,
    # so it is the offset from the first frame of that turn.
    # turn is monotonic non-decreasing (continuous rotation), so the first frame
    # of each turn is found by a single diff, not a per-turn search.
    starts = np.full(n_turns, -1, np.int64)
    if turn.size:
        starts[0] = 0
        nxt = np.flatnonzero(np.diff(turn)) + 1  # first frame of each new turn
        starts[turn[nxt]] = nxt
    pos = np.arange(len(omega_seg)) - starts[turn]
    frames = np.arange(len(omega_seg))
    valid = pos < s1
    cell_frame[turn[valid], pos[valid]] = base + frames[valid]
    unplaced = frames[~valid]
    return cell_frame, unplaced, n_turns, s1


def grid_from_cells(flat, cell_frame):
    """Spread a master-length array onto the grid, filling empty cells.

    flat: full-length array in raw frame order (e.g. omega_raw, dty_raw, nnz).
    cell_frame: (n_turns, s1) int, value = raw frame index or -1.

    Real cells take the raw value. Empty cells are filled from the nearest real
    value in the same column (interpolated over rows), so statistics like
    omega.min()/dty.max() stay sane. A fully empty column copies the previous
    filled column; the first fully empty column is filled with the row pattern
    from the nearest real column.
    """
    cell_frame = np.asarray(cell_frame)
    s0, s1 = cell_frame.shape
    grid = np.empty((s0, s1), dtype=np.asarray(flat).dtype)
    real = cell_frame >= 0
    grid[real] = flat[cell_frame[real]]
    # fill holes column by column, only where there is a hole (rare on a
    # regular f2scan), so we do not scan every column.
    holes = ~real
    cols_with_holes = np.nonzero(holes.any(axis=0))[0]
    for j in cols_with_holes:
        rows = np.nonzero(real[:, j])[0]
        if len(rows) == 0:
            # fully empty column: handled after the loop
            continue
        vals = grid[rows, j]
        hole = np.nonzero(holes[:, j])[0]
        if len(rows) == 1:
            grid[hole, j] = vals[0]
        else:
            grid[hole, j] = np.interp(hole, rows, vals)
    # fully empty columns
    empty_cols = np.nonzero(~real.any(axis=0))[0]
    for j in empty_cols:
        src = j - 1
        while src >= 0 and (src in empty_cols):
            src -= 1
        if src >= 0:
            grid[:, j] = grid[:, src]
        else:
            # no previous filled column : pick any row value as a stand-in
            k = np.nonzero(real.any(axis=1))[0]
            if k.size:
                grid[:, j] = grid[k[0], :].mean()
            else:
                grid[:, j] = 0
    return grid


def row_mean_grid(flat, cell_frame):
    """The dset.dty grid: each row constant, the mean of the real cells in it.

    A f2scan's dty drifts by about one ystep per turn, so binned by measured
    dty a turn can straddle two dty rows. For the sinogram we want each row (=
    one turn = one scan) to sit in a single dty bin, so the row takes the mean
    of the raw dty over its real cells. This is the "dset.dty =
    gridded(dty_raw).mean(axis=rotation)" fix.
    """
    cell_frame = np.asarray(cell_frame)
    s0, s1 = cell_frame.shape
    means = np.empty(s0, dtype=np.asarray(flat).dtype)
    for r in range(s0):
        row = cell_frame[r]
        nz = row[row >= 0]
        means[r] = flat[nz].mean() if nz.size else 0.0
    return np.repeat(means, s1).reshape(s0, s1)


class DataSet:
    """One DataSet instance per detector!"""

    # simple strings or ints
    ATTRNAMES = (
        "dataroot",
        "analysisroot",
        "sample",
        "dset",
        "shape",
        "dsname",
        "datapath",
        "analysispath",
        "masterfile",
        "limapath",
        "detector",
        "omegamotor",
        "dtymotor",
        "monitorname",
        "monitor_ref",
        "pksfile",
        "sparsefile",
        "parfile",
        "e2dxfile",
        "e2dyfile",
        "detectorh5",
        "splinefile",
        "maskfile",
        "bgfile",
        "pksfile",
        "col4dfile",
        "col3dfile",
        "col2dfile",
        "grainsfile",
        "sparsefile",
        "icolfile",
        "pbpfile",
        "y0",
        "omega_wraps",
    )
    STRINGLISTS = ("scans", "imagefiles", "sparsefiles")
    # sinograms
    NDNAMES = (
        "omega",
        "omega_for_bins",
        "dty",
        "nnz",
        "frames_per_file",
        "nlm",
        "frames_per_scan",
        "monitor",
        "ybinedges", "ybincens",
        "obinedges", "obincens",
        "ybin_real_mask",
        # full master-length raw motors, kept alongside the grids so nothing
        # is lost for the frames an irregular scan's grid drops (see the plan)
        "omega_raw",
        "dty_raw",
        "monitor_raw",
        "nnz_raw",
        "scan_frame_offset",
        # 1D address of each raw frame on the sinogram (row*ncols + omega bin),
        # -1 for frames an irregular scan leaves off the grid (unplaced_frames)
        "frame_location",
        # argsort(Frame_location): sinogram address -> raw frame, -1 if empty
        "bins_to_frames",
        # bliss frame numbers the user wants dropped entirely at labelling
        "masked_frames",
    )

    def __init__(
        self,
        dataroot=".",
        analysisroot=".",
        sample="sample",
        dset="dataset",
        detector="eiger",
        omegamotor="rot_center",
        dtymotor="dty",
        filename=None,
        analysispath=None,
        omega_wraps=False,
    ):
        """The things we need to know to process data"""

        # defaults to eiger and nanoscope station, can be overwritten with init parameters detector, omegamotor and dtymotor

        self.detector = detector  # frelon3
        self.limapath = None  # where is the data in the Lima files

        self.omegamotor = omegamotor  # diffrz
        self.dtymotor = dtymotor  # diffty

        self.dataroot = dataroot  # folder to find {sample}/{sample}_{dset}
        self.analysisroot = analysisroot  # where to write or find sparse data
        self.sample = sample  # from bliss path
        self.dset = dset  # from bliss path

        self.dsname = "_".join((self.sample, self.dset))

        # paths for raw data

        self.datapath = os.path.join(self.dataroot, self.sample, self.dsname)
        self.masterfile = os.path.join(self.datapath, self.dsname + ".h5")

        # These are in order ! The order of the lists is important - all things should match.
        self.scans = None  # read from master or load from analysis
        self.frames_per_scan = (
            None  # how many frames (and motor positions) in each scan row.
        )
        self.imagefiles = None  # List of strings. w.r.t self.datapath
        self.frames_per_file = None  # how many frames in this file (Lima files)
        self.sparsefiles = None  # maps sparse files to self.imagefiles

        self.shape = (0, 0)
        self.omega = None
        self.dty = None
        self.monitor = None
        self.omega_raw = None
        self.dty_raw = None
        self.monitor_raw = None
        self.nnz_raw = None
        self.scan_frame_offset = None
        self.frame_location = None
        self.bins_to_frames = None
        self.masked_frames = None
        self.monitorname = None
        self.monitor_ref = None
        self.ybinedges = None
        self.ybincens = None
        self.obinedges = None
        self.obincens = None
        # Does the scan turn far enough that omega has to be averaged on the
        # circle? Default False: a 4D peak is merged linearly. Set it True for
        # a multi-turn scan (e.g. f2scan), or pass omega_wraps=True here. It is
        # saved and loaded with the dataset.
        self.omega_wraps = omega_wraps

        self._peaks_table = None
        self._pk2d = None
        self._pk4d = None

        self.dsfile = None

        # paths for processed data
        self.analysispath = None # default
        # Loaded
        if filename is not None:
            self.dsfile = filename
            self.load(filename)
        # Supplied overwrites
        if analysispath is not None: 
            # Can be loaded with the dataset 
            self.analysispath = analysispath

        self.update_paths()

    def update_paths(self, force=False, verbose = False):
        # paths for processed data
        # root of analysis for this dataset for this sample:
        self.analysispath_default = os.path.join(
            self.analysisroot, self.sample, self.dsname
        )
        if self.analysispath is None:
            self.analysispath = self.analysispath_default

        self.dsfile_default = os.path.join(
            self.analysispath, self.dsname + "_dataset.h5"
        )
        # at the moment, set self.dsfile to be the default
        # if save or load is ever called, this will be replaced
        if self.dsfile is None:
            self.dsfile = self.dsfile_default
        # They should be saved / loaded with the dataset.
        for name, extn in [
            ("pksfile", "_peaks_table.h5"),
            ("col4dfile", "_peaks_4d.h5"),
            ("col3dfile", "_peaks_3d.h5"),
            ("col2dfile", "_peaks_2d.h5"),
            ("grainsfile", "_grains.h5"),
            ("sparsefile", "_sparse.h5"),
            ("icolfile", "_icolf.h5"),  # subset peaks selected for indexing (pbp)
            ("pbpfile", "_pbp.txt"),  # point by point raw output
            ("refmanfile", "_refine_manager.h5"),  # PBPRefine object for refinement
            ("refpeaksfile", "_refine_peaks.h5"),  # icolf for refinement
            ("refmapfile", "_refine_map_in.h5"),  # input pbp map for refinement
            ("refoutfile", "_refine_map_out.h5"),  # output pbp map from refinement
        ]:
            # If the user has got a different name (via loading or edit), we keep that
            if (getattr(self, name, None) is None) or force:
                # Otherwise, these are the defaults.
                setattr(self, name, os.path.join(self.analysispath, self.dsname + extn))
                if verbose:
                    print('updated', getattr( self, name, None ) )
            else:
                if verbose:
                    print('not updated', getattr( self, name, None ) )

    def __repr__(self):
        r = []
        for name in "dataroot analysisroot sample dset".split():
            r.append('%s = "%s"' % (name, getattr(self, name)))
        r.append("shape = ( %d, %d)" % tuple(self.shape))
        if self.scans is not None:
            r.append(
                "# scans %d from %s to %s"
                % (len(self.scans), self.scans[0], self.scans[-1])
            )
        return "\n".join(r)

    def compare(self, other):
        """Try to see if the load/save is working"""
        from types import FunctionType

        sattrs = set([name for name in vars(self) if name[0] != "_"])
        oattrs = set([name for name in vars(self) if name[0] != "_"])
        if sattrs != oattrs:
            logging.info("Attribute mismatch " + str(sattrs) + " != " + str(oattrs))
            return False
        for a in sattrs:
            s = getattr(self, a)
            if isinstance(s, FunctionType):
                continue
            o = getattr(other, a)
            t = type(s)
            if type(o) != type(s):
                logging.info("Type mismatch %s %s" % (str(t), str(a)))
                return False
            if t == np.ndarray:
                if s.shape != o.shape:
                    logging.info("Shape mismatch %s %s" % (str(s.shape), str(o.shape)))
                    return False
                if (s != o).all():
                    logging.info("Data mismatch " + str(a))
                    return False
            else:
                if s != o:
                    logging.info("Data mismatch ")
                    return False
        logging.info("Dataset objects seem to match!")
        return True

    def report(self):
        print(self)
        print("# Collected %d missing %d" % (self.check_images()))
        print("# Segmented %d missing %d" % (self.check_sparse()))

    def import_all(self,
                   scans=None, shape=None,
                   guess_y0=True
                  ):
        # collect the data
        self.import_scans(scans=scans)
        # lima frames
        self.import_imagefiles()
        # motor positions
        self.import_motors_from_master()
        self.guess_shape()
        self.guessbins()
        if guess_y0:
            self.guess_y0()
        # pixels per frame
        try:
            self.import_nnz()
        except:
            logging.info("nnz not available. Segmentation done?")

    def import_from_sparse(self, hname, scans=None, shape=None):
        """
        hname = hdf5 file containing sparse pixels (and motors)
        dataset = a dataset instance to import into
        scans = defaults to reading all "%d.1" scans in the file
                give a list to read in some other order or a subset
        """
        self.sparsefile = hname
        if scans is None:
            with h5py.File(hname, "r") as hin:
                # Read all in numerical order
                scans = list(hin["/"])
                order = np.argsort([float(v) for v in scans if v.endswith(".1")])
                self.scans = [scans[i] for i in order]
        else:
            self.scans = scans
        self.masterfile = hname  # hacky, motors come from the sparsefile
        self.import_nnz_from_sparse()  # must exist
        self.import_motors_from_master()
        # the sparse file encodes a regular grid, so shape comes from nnz
        if shape is not None:
            self.shape = tuple(shape)
        else:
            self.shape = self.nnz.shape
        s0, s1 = self.shape
        # per-scan (row) start offsets into the master-length raw arrays
        self.scan_frame_offset = np.asarray(self._scan_raw_offset[:-1], np.int64)
        # locate every frame on the sinogram by binning its omega
        self._bin_frames()
        if len(scans) == 1 and self.shape[0] > 1:
            file_nums = np.arange(self.shape[0] * self.shape[1]).reshape(self.shape)
            self.scans = [
                "%s::[%d:%d]" % (self.scans[0], row[0], row[-1] + 1)
                for row in file_nums
            ]

    def _bin_frames(self, row_of_frame=None):
        """Locate every raw frame onto the sinogram and build the aligned grids.

        frame_location[frame] = row*ncols + omega_bin(frame): the 1D address of
        that frame on the sinogram. bins_to_frames is its argsort, so
        bins_to_frames[address] = frame. On the regular path the row is the scan
        (one row per scan, len(scans) == rows, and the scans must already be
        sorted by dty - which is enforced here), and the column is the frame's
        omega bin, so argsort(frame_location) lexsorts the (dty, omega) grid
        rather than assuming frame index = column. Placing a frame by its
        measured omega is what a zig-zag f2scan needs: the same grain diffracts
        at the same omega in every row, so it lands in the same column whatever
        direction the scan swept.

        dset.dty is the median of dty_raw over each row (== over each scan),
        applied unconditionally. A single dty outlier therefore does not move a
        frame's row and is absorbed by the median; it is not lost, because
        dset.projection_shifts reports dty_raw - dset.dty per frame. Requires
        exactly one frame per bin and scans sorted by dty.

        TODO: the row median is only the right dty while a row is one scan. An
        f2scan whose dty drifts across the rotation spans bins and would need the
        same treatment applied on the f2scan (cell_blocks) path.
        """
        s0, s1 = self.shape
        nframes = len(self.omega_raw)
        if row_of_frame is None:
            # a regular grid is stored dty-major (one row per scan, s1 columns),
            # so a frame's row is its position in the linear stream / s1. This
            # does not depend on frames_per_scan, which a fscan2d master cannot
            # keep in step when it is split into per-turn rotations.
            row = np.arange(nframes, dtype=np.int64) // s1
        else:
            row = np.asarray(row_of_frame, np.int64)
            if len(row) != nframes:  # pragma: no cover
                raise ValueError("row_of_frame length does not match raw frames")
        # the row dty is the median of dty_raw over the row (== the scan), so an
        # outlier cannot pull it around and the grid keeps one dty per row.
        row_dty = np.full(s0, np.nan)
        for r in range(s0):
            sel = row == r
            if sel.any():
                row_dty[r] = float(np.median(self.dty_raw[sel]))
        # rows are scans, so an unsorted dty means the scan/row ordering cannot
        # be trusted: that is a hard error, not a warning.
        if s0 > 1:
            d = np.diff(row_dty)
            if not (np.all(d > 0) or np.all(d < 0)):
                raise ValueError(
                    "scans are not sorted by dty (row dty medians are not "
                    "monotonic); a regular sinogram needs one scan per row in "
                    "dty order.")
        # omega bins from the measured omega (no 2-D reshape, so ragged scans
        # of different lengths do not need to all match)
        self.omin = float(self.omega_raw.min())
        self.omax = float(self.omega_raw.max())
        if self.omega_wraps:
            self.ostep = 360.0 / s1
            phase = bin_phase(self.omega_raw % 360, self.ostep)
            self.obincens = phase + np.arange(s1) * self.ostep
            edge0 = phase - self.ostep / 2
            self.obinedges = edge0 + np.arange(s1 + 1) * self.ostep
        else:
            self.ostep = (self.omax - self.omin) / (s1 - 1) if s1 > 1 else 1.0
            self.obincens = np.linspace(self.omin, self.omax, s1)
            self.obinedges = np.linspace(
                self.omin - self.ostep / 2, self.omax + self.ostep / 2, s1 + 1)
        col = np.digitize(self.omega_raw, self.obinedges) - 1
        col = np.clip(col, 0, s1 - 1)
        self.frame_location = (row * s1 + col).astype(np.int64)
        counts = np.bincount(self.frame_location, minlength=s0 * s1)
        bad = int((counts != 1).sum())
        if bad:
            raise ValueError(
                "sinogram is not one frame per bin: %d cells have 0 or >1 frames"
                % bad)
        self.bins_to_frames = np.argsort(self.frame_location, kind="stable")
        self.unplaced_frames = np.array([], np.int64)
        # grids spread back through the sorted map (column = omega bin)
        self.omega = self.omega_raw[self.bins_to_frames].reshape(s0, s1)
        self.omega_for_bins = self.omega
        # dset.dty is the median over each row (scan), constant along the row
        self.dty = np.repeat(row_dty, s1).reshape(s0, s1)
        # nnz is read after guess_shape in import_all, so it may not exist yet
        if self.nnz_raw is not None:
            self.nnz = self.nnz_raw[self.bins_to_frames].reshape(s0, s1)
        # dty bins from the measured dty
        self.ymin = float(self.dty_raw.min())
        self.ymax = float(self.dty_raw.max())
        self.ybincens = np.linspace(self.ymin, self.ymax, s0)
        self.ystep = (self.ymax - self.ymin) / (s0 - 1) if s0 > 1 else 1.0
        self.ybinedges = np.linspace(
            self.ymin - self.ystep / 2, self.ymax + self.ystep / 2, s0 + 1)

    def import_scans(self, scans=None, hname=None):
        """Reads in the scans from the bliss master file"""
        # we need to work out what detector we have at this point
        # self.guess_detector()
        if hname is None:
            hname = self.masterfile
        frames_per_scan = []
        with h5py.File(hname, "r") as hin:
            if scans is None:
                scans = [
                    scan
                    for scan in list(hin["/"])
                    if (
                        scan.endswith(".1")
                        and ("measurement" in hin[scan])
                        and (self.detector in hin[scan]["measurement"])
                        and (self.omegamotor in hin[scan]["measurement"])
                    )
                ]
            goodscans = []
            for scan in scans:
                # Make sure that this scan has a measurement from our detector
                if self.detector not in hin[scan]["measurement"]:
                    print("Bad scan", scan)
                else:
                    try:
                        frames = hin[scan]["measurement"][self.detector]
                    except KeyError as e:  # Thrown by h5py
                        print("Bad scan", scan, ", h5py error follows:")
                        print(e)
                        continue
                    if len(frames.shape) == 3:  # need 1D series of frames
                        goodscans.append(scan)
                        frames_per_scan.append(frames.shape[0])
                    else:
                        print("Bad scan", scan)

        self.scans = goodscans
        self.frames_per_scan = frames_per_scan

        logging.info("imported %d scans from %s" % (len(self.scans), hname))
        return self.scans

    def import_imagefiles(self):
        """Get the Lima file names from the bliss master file, also scan_npoints"""
        # self.import_scans() should always be called before this function, so we know the detector
        self.imagefiles = []
        self.frames_per_file = []
        with h5py.File(self.masterfile, "r") as hin:
            bad = []
            for i, scan in enumerate(self.scans):
                if ("measurement" not in hin[scan]) or (
                    self.detector not in hin[scan]["measurement"]
                ):
                    print("Bad scan", scan)
                    bad.append(scan)
                    continue
                frames = hin[scan]["measurement"][self.detector]
                self.imageshape = frames.shape[1:]
                for vsrc in frames.virtual_sources():
                    self.imagefiles.append(vsrc.file_name)
                    self.frames_per_file.append(
                        vsrc.src_space.shape[0]
                    )  # not sure about this
                    # check limapath
                    if self.limapath is None:
                        self.limapath = vsrc.dset_name
                    assert self.limapath == vsrc.dset_name
        self.frames_per_file = np.array(self.frames_per_file, int)
        self.sparsefiles = [
            os.path.join(
                "sparsefiles", name.replace("/", "_").replace(".h5", "_sparse.h5")
            )
            for name in self.imagefiles
        ]
        logging.info("imported %d lima filenames" % (np.sum(self.frames_per_file)))

    def import_motors_from_master(self):
        """read the motors from the lima file
        you need to import the imagefiles first
        these will be the motor positions to accompany the images
        # could also get these from sparse files if saved

        Builds the per-scan lists self.omega / self.dty (used by guess_shape to
        make the grids) and the full master-length raw arrays self.omega_raw /
        self.dty_raw along with scan_frame_offset, so frames the grid drops on
        an irregular f2scan are not lost.
        """
        # self.guess_motornames()
        self.omega = [
            None,
        ] * len(self.scans)
        self.dty = [
            None,
        ] * len(self.scans)
        omega_raw = []
        dty_raw = []
        with h5py.File(self.masterfile, "r") as hin:
            bad = []
            for i, scan in enumerate(self.scans):
                # Should always be there, if not, filter scans before you get to here
                om = hin[scan]["measurement"][self.omegamotor][()]
                if len(om) == self.frames_per_scan[i]:
                    self.omega[i] = om
                else:  # hope the first point was good ? Probably corrupted MUSST data.
                    self.omega[i] = [
                        om[0],
                    ]
                    bad.append(i)
                # this can be an array or a scalar
                # read from h5:
                dty = hin[scan]["instrument/positioners"][self.dtymotor]
                if len(dty.shape) == 0:
                    self.dty[i] = np.full(self.frames_per_scan[i], dty[()])
                elif dty.shape[0] == self.frames_per_scan[i]:
                    self.dty[i] = dty[:]
                else:
                    # corrupted MUSST?
                    self.dty[i] = np.full(self.frames_per_scan[i], dty[0])
                # keep the true motor positions, full master length, so a frame
                # the grid drops keeps its number and its motors.
                omega_raw.append(np.asarray(self.omega[i], float).ravel())
                dty_raw.append(np.asarray(self.dty[i], float).ravel())
        for b in bad:
            dom = [
                (abs(self.omega[i][0] - self.omega[b])[0], i)  # always length-1 arrays, take first element
                for i in range(len(self.scans))
                if i not in bad
            ]
            # dom is a list of tuples of (first omega value, i)
            # make it into an array
            dom = np.array(dom)

            if len(dom) > 0:
                j = int(dom[np.argmin(dom[:,0])][1])  # get argmin of omega column of dom, go there, then take the corresponding i
                self.omega[b] = self.omega[j]  # best match
                # the raw array for this scan now matches the replacement it got
                omega_raw[b] = np.asarray(self.omega[j], float).ravel()
                print(
                    "replace bad scan omega", b, self.scans[b], "with", j, self.scans[j]
                )
        self.omega_raw = np.concatenate(omega_raw) if omega_raw else np.array([], float)
        self.dty_raw = np.concatenate(dty_raw) if dty_raw else np.array([], float)
        # per-master-scan start offsets into the raw arrays (cumulative). Used
        # by guess_shape to slice each scan's segment; scan_frame_offset (per
        # grid row) is built there.
        self._scan_raw_offset = np.zeros(len(self.scans) + 1, np.int64)
        np.cumsum([len(o) for o in omega_raw], out=self._scan_raw_offset[1:])
        logging.info("imported omega/dty (%d raw frames)" % (len(self.omega_raw)))

    def guess_shape(self):
        """Reshape the raw motor arrays into the sinogram grid.

        f2scan is one continuous rotation split into turns of round(360/step)
        frames, but a turn is not exactly that many frames: an over-long turn
        drops the frame that has already crossed into the next turn, and a
        short turn leaves trailing empty cells. bins_to_frames records which raw
        frame occupies each cell, so nothing is lost and no frame is double
        counted. Every other scan type keeps the regular assumption.
        """
        npts = np.sum(self.frames_per_scan)
        cell_blocks = []       # (cell_frame, scan_base) per f2scan master scan
        f2slice_flag = [False]  # did we see an f2scan?
        if os.path.exists(self.masterfile):
            # strip [i::j] from self.scans if already there:
            seen = set()
            scans = []
            snames = [ s.split('::')[0] for s in self.scans ]
            for s in snames:
                if s not in seen:
                    seen.add( s )
                    scans.append( s )
            # number of turns
            rotations = []
            for i, scan in enumerate(scans):
                with h5py.File(self.masterfile, "r") as hin:
                    s = hin[scan]
                    title = s["title"].asstr()[()]
                    # print("Scan title", title)
                    if title.split()[0] == "fscan2d":
                        s0 = s["instrument/fscan_parameters/slow_npoints"][()]
                        s1 = s["instrument/fscan_parameters/fast_npoints"][()]
                        if s0 > 1:
                            file_nums = np.arange(s0 * s1).reshape((s0, s1))
                            # slice notation means last frame+1 to be inclusive
                            rotations += [
                                "%s::[%d:%d]" % (scan, row[0], row[-1] + 1)
                                for row in file_nums
                                ]
                        else:
                            rotations += [ scan, ]
                    elif title.split()[0] == "f2scan":
                        # one continuous rotation split into turns: the merged
                        # omega must be averaged on the circle. The turn
                        # boundaries come from the data, not a fixed count.
                        f2slice_flag[0] = True
                        step = s["instrument/fscan_parameters/step_size"][()]
                        # the raw segment for this master scan
                        idx = self.scans.index(scan)
                        lo = int(self._scan_raw_offset[idx])
                        hi = int(self._scan_raw_offset[idx + 1])
                        cf, unplaced, nturns, s1 = cell_frame_from_raw(
                            self.omega_raw[lo:hi], self.dty_raw[lo:hi], step,
                            base=lo)
                        cell_blocks.append((cf, unplaced, lo))
                        if nturns > 1:
                            self.omega_wraps = True
                        # per-turn slices (variable length): real turn boundaries
                        for t in range(nturns):
                            row = cf[t]
                            nonneg = row[row >= 0]
                            if nonneg.size:
                                a = int(nonneg.min()); b = int(nonneg.max()) + 1
                                rotations.append("%s::[%d:%d]" % (scan, a, b))
                            else:
                                rotations.append("%s::[%d:%d]" % (scan, lo, lo))
                    else:
                        s0 = 1
                        s1 = npts
                        rotations.append( scan )
            self.scans = rotations
        # build the grid
        if cell_blocks:
            # f2scan: stack the turn blocks and use the derived (nturns, s1)
            cell_frame = cell_blocks[0][0]
            unplaced = cell_blocks[0][1] + cell_blocks[0][2]
            for cf, up, base in cell_blocks[1:]:
                cell_frame = np.vstack([cell_frame, cf])
                unplaced = np.concatenate([unplaced, up + base])
            s0, s1 = cell_frame.shape
            self._unplaced_in_scan = unplaced
            self.shape = (s0, s1)
            if np.prod(self.shape) != npts:
                print("Warning: irregular scan - might be bugs in here")
                print(npts, len(self.scans))
            flat = cell_frame.ravel()
            self.bins_to_frames = flat
            self.frame_location = np.full(len(self.omega_raw), -1, np.int64)
            real = flat >= 0
            self.frame_location[flat[real]] = np.nonzero(real)[0]
            self.unplaced_frames = np.nonzero(self.frame_location < 0)[0]
            self.omega = grid_from_cells(self.omega_raw, cell_frame)
            self.dty = row_mean_grid(self.dty_raw, cell_frame)
            # per-grid-row starting raw frame index
            starts = np.full(s0, -1, np.int64)
            for r in range(s0):
                row = cell_frame[r]
                nonneg = row[row >= 0]
                starts[r] = int(nonneg.min()) if nonneg.size else 0
            self.scan_frame_offset = starts
            if len(self.unplaced_frames):
                logging.info(
                    "f2scan: %d frames over-long turns leave unplaced (grid %dx%d)"
                    % (len(self.unplaced_frames), s0, s1))
        else:
            if len(self.scans) >= 1:
                s0 = len(self.scans)
                s1 = int(npts // s0) if s0 else 0
            else:
                s0 = 0; s1 = 0
            self.shape = (s0, s1)
            if np.prod(self.shape) != npts:
                print("Warning: irregular scan - might be bugs in here")
                print(npts, len(self.scans))
            # regular multi-scan: bin every frame onto the sinogram by omega
            self.scan_frame_offset = np.asarray(self._scan_raw_offset[:-1], np.int64)
            self._bin_frames()
        logging.info(
                "sinogram shape = ( %d , %d ) imageshape = ( %d , %d)"
                % (self.shape[0], self.shape[1], self.imageshape[0], self.imageshape[1])
            )

    def guessbins(self):
        """
        Attempts to estimate the step size in the data by looking at the numbers
        in self.omega and self.dty that should already have self.shape reflecting
        the length of the individual scans.

        Perhaps this is the wrong approach. But we don't have it in the bliss data.

        The data might not be on a regular grid.
        """
        ny, nomega = self.shape
        if self.obincens is None:
            self.omin = self.omega.min()
            self.omax = self.omega.max()
            if self.omega_wraps:
                # Multi-turn scan: every turn falls into one set of bins
                # covering the circle.
                # One bin per frame in a turn, because that is the shape the
                # sinogram has to come out in. A whole number of steps then
                # closes the circle: binning 0 and 360 is the same angle twice
                # and the extra bin pushes a frame per turn into its neighbour.
                nbins = nomega
                self.ostep = 360.0 / nbins
                # assume the first scan is representative
                # if you have different steps in different scans ... that is bad
                measured = guess_omega_step(self.omega[0])
                if abs(measured - self.ostep) > 0.01 * self.ostep:
                    logging.warning(
                        "omega step from the data is %f but %d frames per turn "
                        "makes it %f. Is the scan shape %s right?"
                        % (measured, nomega, self.ostep, (self.shape,)))
                # Put the bins on the data. A frame is an exposure centre, so
                # it is offset from the requested start by half a step, and
                # bins anchored at zero then have their edges running through
                # the middle of the frames.
                phase = bin_phase(self.omega % 360, self.ostep)
                self.obincens = phase + np.arange(nbins) * self.ostep
                self.omin = self.obincens[0]
                self.omax = self.obincens[-1]
                # Fold onto the bins rather than onto 0-360, so that a frame
                # just below the first edge comes back at the top instead of
                # falling outside.
                edge0 = phase - self.ostep / 2
                self.omega_for_bins = (self.omega - edge0) % 360 + edge0
                if self.obinedges is None:
                    self.obinedges = edge0 + np.arange(nbins + 1) * self.ostep
            else:
                self.omega_for_bins = self.omega
                self.ostep = (self.omax - self.omin) / (nomega - 1)
                self.obincens = np.linspace(self.omin, self.omax, nomega)
        else: # self.obincens was loaded
            self.omin = self.obincens[0]
            self.omax = self.obincens[-1]
            self.ostep = np.mean(self.obincens[1:] - self.obincens[:-1])
            if self.omega_wraps:
                # Fold onto the same bins built by guessbins, whose first edge
                # is the fold point (see bin_phase), not a hardcoded 0.
                edge0 = self.omin - self.ostep / 2
                self.omega_for_bins = (self.omega - edge0) % 360 + edge0
            else:
                self.omega_for_bins = self.omega
        if self.obinedges is None: # catches last 3 else here.
            self.obinedges = np.linspace(
               self.omin - self.ostep / 2, self.omax + self.ostep / 2, nomega + 1
            )
        # values 0, 1, 2
        # shape = 3
        # step = 1
        if self.ybincens is not None:
            self.ymin = self.ybincens[0]
            self.ymax = self.ybincens[-1]
        else:
            self.ymin = self.dty.min()
            self.ymax = self.dty.max()
            self.ybincens = np.linspace(self.ymin, self.ymax, ny)
        if ny > 1:
            self.ystep = (self.ymax - self.ymin) / (ny - 1)
        else:
            self.ystep = 1
        if self.ybinedges is None:
            self.ybinedges = np.linspace(
                self.ymin - self.ystep / 2, self.ymax + self.ystep / 2, ny + 1
            )

    def guess_y0(self):
        """Guess y0 from y bins. We assume the scan is symmetric across y0.
        Should be good for an initial guess."""
        y0 = (self.ymax + self.ymin)/2
        self.y0 = y0
    
    def correct_bins_for_half_scan(self, y0 = 0.0):
        """
        Pad self.ybincens / self.ybinedges around the bin nearest to y0
        so that the dataset becomes symmetric. 
        The original measured bins are never moved; only virtual bins are added
        on whichever side is shorter. A boolean mask records which bins are real.
 
        Sets self.ybin_real_mask : bool array on self.ybincens, False on virtual bins
        """
        ystep = self.ystep
        # Recover original measured bins
        if hasattr(self, 'ybin_real_mask'):
            yc_orig = np.asarray(self.ybincens, dtype=float)[self.ybin_real_mask]
        else:
            yc_orig = np.asarray(self.ybincens, dtype=float).copy()
        # get the bin closest to y0
        central_bin    = int(np.argmin(np.abs(yc_orig - y0)))
        central_val    = yc_orig[central_bin]
        # Distance from central_val to each end of the measured range
        lo_dist = central_val - yc_orig[0]
        hi_dist = yc_orig[-1] - central_val
        half    = np.ceil(max(lo_dist, hi_dist) / ystep) * ystep
        # Build symmetric grid centred on central_val, with the same ystep
        n_half    = int(round(half / ystep))
        new_cens  = central_val + np.arange(-n_half, n_half + 1) * ystep
        new_edges = central_val - ystep / 2 + np.arange(-n_half, n_half + 2) * ystep
        # Real-bin mask: True where new_cens matches an orig_cens within ystep/4
        real_mask = np.array([
            np.any(np.abs(yc_orig - cv) < ystep / 4)
            for cv in new_cens])
        # Update bins
        self.ybincens       = new_cens
        self.ybinedges      = new_edges
        self.ybin_real_mask = real_mask
        # ymin / ymax are otherwise only set in guessbins, and the padding above can
        # move ybincens[0]. Anything reading ds.ymin (recon_bins, guess_y0, dty_to_dtyi)
        # would then be out by the width of the low-side padding.
        self.ymin           = float(new_cens[0])
        self.ymax           = float(new_cens[-1])
        n_pad = int((~real_mask).sum())
        print(
            "[correct_bins_for_half_scan]  y0 = {:.4f}, central bin value = {:.4f}, central bin id: {}, "
            "halfrange={:.4f}, n_bins={} "
            "({} real + {} padded).".format(y0, central_val, central_bin, half,
                                            len(new_cens), real_mask.sum(), n_pad))

    def get_ring_current_per_scan(self):
        """Gets the ring current for each scan (i.e rotation/y-step)
        Stores it inside self.ring_currents_per_scan and a scaled version inside self.ring_currents_per_scan_scaled"""
        if not hasattr(self, "ring_currents_per_scan"):
            ring_currents = []
            with h5py.File(self.masterfile, "r") as h5in:
                for scan in self.scans:
                    ring_current = float(h5in[scan]["instrument/machine/current"][()])
                    ring_currents.append(ring_current)

            self.ring_currents_per_scan = np.array(ring_currents)
            self.ring_currents_per_scan_scaled = np.array(
                ring_currents / np.max(ring_currents)
            )

    def get_monitor(self, name="fpico6"):
        # masterfile or sparsefile
        hname = self.masterfile
        if hasattr(self, "sparsefile") and os.path.exists(self.sparsefile):
            hname = self.sparsefile
        monitor = []
        with h5py.File(hname, "r") as hin:
            for scan in self.scans:
                if scan.find("::") > -1:
                    snum, slc = scan.split("::")
                    lo, hi = [int(v) for v in slc[1:-1].split(":")]
                    mon = hin[snum]["measurement"][name][lo:hi]
                else:
                    mon = hin[scan]["measurement"][name][:]
                monitor.append(mon)

        # full master-length monitor, so a frame the grid drops keeps it
        self.monitor_raw = np.concatenate(monitor)
        # the 2-D grid in frame_location order, the shape get_monitor_pk2d expects
        if getattr(self, "bins_to_frames", None) is not None:
            self.monitor = grid_from_cells(
                self.monitor_raw, self.bins_to_frames.reshape(self.shape))
        else:
            self.monitor = self.monitor_raw.reshape(self.shape)
        return self.monitor
    
    def reset_peaks_cache(self):
        """
        Clear cached peaks table - relevant if you set a monitor which will change intensities in columnfiles.
        """
        import warnings
        if self._pk2d is not None:
            # we have an existing 2D peaks table
            warnings.warn("Clearing cached pk2d")
            self._pk2d = None
        
        if self._pk4d is not None:
            # we have an existing 4D peaks table
            warnings.warn("Clearing cached pk4d")
            self._pk4d = None
        
        if os.path.exists(self.col2dfile):
             warnings.warn("I found an existing 2D colfile on disk - you probably want to remake this with ds.get_cf_2d(ignore_existing=True)")
                
        if os.path.exists(self.col4dfile):
             warnings.warn("I found an existing 4D colfile on disk - you probably want to remake this with ds.get_cf_4d(ignore_existing=True)")
    
    def set_monitor(self, name="fpico6", ref_value_func=np.mean):
        """
        Sets self.monitor and self.monitor_ref after calling self.get_monitor()
        Clears cached pk2d and pk4d so they can be re-computed
        
        ref_value_func: function to apply to self.monitor to generate a reference value
        when we normalise, we multiply by ref_value_func(self.monitor)/self.monitor
        we suggest np.mean as an example...
        hint: if you want to return a constant, use this:
        ref_value_func=lambda x: 1e5
        """
        self.monitor = self.get_monitor(name=name)
        self.monitor_ref = ref_value_func(self.monitor)
        
        self.reset_peaks_cache()
        
    
    def get_monitor_pk2d(self, pk2d, name='fpico6'):
        """
        To be used to normalise the peaks 2d
        """
        if self.monitor is None:
            monitor = self.get_monitor(name)
        else:
            monitor = self.monitor
        iy = np.digitize( pk2d['dty'], self.ybinedges ) - 1
        io = np.digitize( pk2d['omega'], self.obinedges ) - 1 
        #pk2d['iy'] = iy  # cache these too ?
        #pk2d['io'] = io
        return monitor[ iy, io ]

    def bliss_frame(self, ir, ic):
        """The bliss/master frame number of cell (ir, ic), or -1 if empty."""
        flat = self.bins_to_frames.reshape(self.shape)
        return int(flat[ir, ic])

    def grid_from_raw(self, name):
        """Re-spread a master-length raw array onto the grid via bins_to_frames.

        name: 'omega', 'dty', 'monitor' or 'nnz' (the grid attribute and the
        *_raw counterpart). Makes the grid agree with the frame map after the
        raw array or the map was edited. This is the helper the plan mentions
        for a user who edits dty in place, regenerating the grid from the raw.
        """
        raw = getattr(self, name + "_raw")
        cf = self.bins_to_frames.reshape(self.shape)
        if name == 'nnz':
            grid = np.zeros(self.shape, raw.dtype)
            real = cf >= 0
            grid[real] = raw[cf[real]]
        elif name == 'dty':
            # dty grid is the per-row mean (each turn sits in one dty bin)
            grid = row_mean_grid(raw, cf)
        else:
            grid = grid_from_cells(raw, cf)
        return grid

    def grid_neighbours(self, k, connectivity=4):
        """The grid neighbours of the cell at flat index k.

        Pure index arithmetic on the (nrotations, nframes) grid, no omega/dty
        comparison: a cell's neighbours are k +/- 1 (same row, omega direction)
        and k +/- s1 (adjacent row, dty direction). Columns wrap when
        omega_wraps, so a frame at the omega seam pairs with its neighbour.
        connectivity: 4 (square), 6 (hexagonal), 8 (square + diagonals). Only
        cells that exist and hold a frame (bins_to_frames >= 0) are returned.

        Returns a sorted int array of flat grid indices.
        """
        s0, s1 = self.shape
        r, c = divmod(int(k), s1)
        if connectivity == 4:
            offs = ((-1, 0), (1, 0), (0, -1), (0, 1))
        elif connectivity == 8:
            offs = ((-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1))
        elif connectivity == 6:
            # hexagonal : two column neighbours and four diagonal ones
            offs = ((0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
        else:
            raise ValueError("connectivity must be 4, 6 or 8")
        offs = np.asarray(offs, np.int64)
        rr = r + offs[:, 0]
        cc = c + offs[:, 1]
        if self.omega_wraps:
            cc = cc % s1
        inrow = (rr >= 0) & (rr < s0)
        incol = (cc >= 0) & (cc < s1)
        ok = inrow & incol
        rr = rr[ok]; cc = cc[ok]
        idx = rr * s1 + cc
        real = self.bins_to_frames[idx] >= 0
        return np.unique(idx[real])

    @property
    def projection_shifts(self):
        """Per-frame difference between the measured dty and its row's median.

        projection_shifts[frame] = dty_raw[frame] - dset.dty[row, 0], i.e. how
        far the frame's dty sits from the median dty of the row (scan) it was
        binned into. A frame whose dty is faithful to its scan gives ~0; a dty
        outlier (one frame far from the rest of its scan) is reported here, since
        dset.dty is the row median and so ignores it.

        Only defined once the frame map exists; frames that do not sit on the
        grid (f2scan unplaced frames) are reported as 0, since a missing cell
        carries no deviation.
        """
        if self.dty_raw is None or self.dty is None or self.frame_location is None:
            return None
        s1 = self.shape[1]
        floc = np.asarray(self.frame_location, np.int64)
        row = np.zeros(len(floc), np.int64)
        placed = floc >= 0
        row[placed] = floc[placed] // s1
        row_median = np.asarray(self.dty, float)[:, 0]
        out = np.zeros(len(floc), float)
        out[placed] = (np.asarray(self.dty_raw, float)[placed]
                       - row_median[row[placed]])
        return out

    def guess_detector(self):
        """Guess which detector we are using from the masterfile"""

        # open the masterfile
        hname = self.masterfile
        detectors_seen = []
        scan = "1.1"

        with h5py.File(hname, "r") as hin:
            # go through the first scan, and see what detectors we can see
            for measurement in list(hin[scan]["measurement"]):
                if measurement.attrs.get("interpretation") == "image":
                    detectors_seen.append(measurement)

        if len(detectors_seen) != 1:
            raise ValueError(
                "More than one detector seen! Can't work out which one to process."
            )
        else:
            self.detector = detectors_seen[0]

    #     def guess_motornames(self):
    #         '''Guess which station we were using (Nanoscope or 3DXRD) from which motors are in instrument/positioners'''
    #         from ImageD11.sinograms.assemble_label import HEADERMOTORS_NSCOPE, HEADERMOTORS_TDXRD, HEADERMOTORS
    #         # open the masterfile
    #         hname = self.masterfile
    #         motors_seen = []
    #         scan = "1.1"

    #         with h5py.File( hname, 'r' ) as hin:
    #             # go through the first scan, and see what motors we can see
    #             for positioner in list(hin[scan]['instrument/positioners']):
    #                 if positioner in HEADERMOTORS:
    #                     motors_seen.append(positioner)

    #         using_nscope = False
    #         using_tdxrd = False
    #         for motor in motors_seen:
    #             if motor in HEADERMOTORS_NSCOPE:
    #                 using_nscope = True
    #             elif motor in HEADERMOTORS_TDXRD:
    #                 using_tdxrd = True

    #         if using_nscope and using_tdxrd:
    #             raise ValueError("Found both nscope and tdxrd motors in positioners, not sure which one we were using!")

    #         if using_nscope:
    #             self.omegamotor = 'rot_center'
    #             self.dtymotor = 'dty'
    #         elif using_tdxrd:
    #             self.omegamotor = 'diffrz'
    #             self.dtymotor = 'diffty'

    def sinohist(self, weights=None, omega=None, dty=None, method="fast", return_edges=False):
        """Bin some data onto the sinogram histogram"""
        omin = self.omega_for_bins.min()
        omax = self.omega_for_bins.max()
        omin_edge = omin - self.ostep / 2
        omax_edge = omax + self.ostep / 2

        bins = len(self.obincens), len(self.ybincens)
        rng = (
            (omin_edge, omax_edge),
            (self.ybinedges[0], self.ybinedges[-1]),
        )
        if isinstance(weights, np.ndarray):
            wt = weights.ravel()
        else:
            wt = weights
        if omega is None:
            omega = self.omega_for_bins
        if dty is None:
            dty = self.dty
            
        if self.omega_wraps:
            # Fold onto the same bins guessbins built, not onto 0-360: the
            # bins do not start at 0 (see bin_phase), and omega here is
            # normally already self.omega_for_bins, so this must agree with
            # how that was folded or the histogram range and the data part
            # company. Idempotent when it already has been.
            edge0 = self.obinedges[0]
            om_mod = (omega - edge0) % 360 + edge0
        else:
            om_mod = omega
        
        if method == "numpy":
            ret = np.histogram2d(
                om_mod.ravel(), dty.ravel(), weights=wt, bins=bins, range=rng
            )
            histo = ret[0]
        elif method == "fast":
            histo = fast_histogram.histogram2d(
                om_mod.ravel(), dty.ravel(), weights=wt, bins=bins, range=rng
            )
        if return_edges:
            om_edges = np.linspace(rng[0][0], rng[0][1], bins[0])
            dty_edges = np.linspace(rng[1][0], rng[1][1], bins[1])
            return histo, om_edges, dty_edges
        else:
            return histo

    def get_phases_from_disk(self):
        if not hasattr(self, "parfile") or self.parfile is None:
            raise AttributeError("Need self.parfile to load phases!")
        return ImageD11.unitcell.Phases(self.parfile)

    @property
    def peaks_table(self):
        if self._peaks_table is None:
            self._peaks_table = ImageD11.sinograms.properties.pks_table.load(
                self.pksfile
            )
        return self._peaks_table

    @property
    def pk2d(self):
        if self._pk2d is None:
            if self.monitor is not None:
                # we normalise
                scale_factor = self.monitor_ref/self.monitor
                self._pk2d = self.peaks_table.pk2d(self.omega_for_bins, self.dty, scale_factor=scale_factor)
            else:
                # don't normalise
                self._pk2d = self.peaks_table.pk2d(self.omega_for_bins, self.dty)
        return self._pk2d

    @property
    def pk4d(self):
        if self._pk4d is None:
            # obinedges[0] is where omega_for_bins was folded to (see
            # guessbins/bin_phase, usually not 0): the merged omega must come
            # back in that same frame, or it lands a full turn from its own
            # frames right at the seam. See pk2dmerge.
            omega0 = self.obinedges[0] if self.omega_wraps else 0.0
            if self.monitor is not None:
                # we normalise
                scale_factor = self.monitor_ref/self.monitor
                self._pk4d = self.peaks_table.pk2dmerge(
                    self.omega_for_bins, self.dty, scale_factor=scale_factor,
                    omega_wraps=bool(self.omega_wraps), omega0=omega0)
            else:
                # don't normalise
                self._pk4d = self.peaks_table.pk2dmerge(
                    self.omega_for_bins, self.dty,
                    omega_wraps=bool(self.omega_wraps), omega0=omega0)
        return self._pk4d

    def get_spatial_corrector(self):
        """Return a spatial corrector callable built from whichever distortion
        source is set on self, in priority order: detectorh5, then e2dx/e2dy
        files, then splinefile. Returns None if none is configured.
    
        The returned callable takes a columnfile and adds the
        'sc'/'fc' corrected coordinate columns:
            corrector = self.get_spatial_corrector()
            if corrector is not None:
                cf = corrector(cf)
        """
        return get_corrector(
            spline_file=getattr(self, "splinefile", None),
            dxfile=getattr(self, "e2dxfile", None),
            dyfile=getattr(self, "e2dyfile", None),
            h5file=getattr(self, "detectorh5", None),
            detector=getattr(self, "detector", "eiger"),
        )
    
    def get_colfile_from_peaks_dict(self, peaks_dict=None, corrector=None):
        """Converts a dictionary of peaks (peaks_dict) into an ImageD11 columnfile
        adds on the geometric computations (tth, eta, gvector, etc)
        Uses self.pk2d if no peaks_dict provided"""
        # TODO add optional peaks mask
        if peaks_dict is None:
            peaks_dict = self.pk2d
        cf = colfile_from_dict(peaks_dict)

        # if no corrector is supplied, try to get one
        if corrector is None:
            corrector = self.get_spatial_corrector()
        if corrector is not None:
            cf = corrector(cf)
        else:
            print('No spatial correction files supplied. Will return uncorrected file.')
    
        return cf

    def update_colfile_pars(self, cf, phase_name=None):
        """Load parameters and update geometry for colfile"""
        if (not hasattr(self, 'parfile')) or (self.parfile is None):
            raise AttributeError("You must supply a parameter file first with ds.parfile = '/path/to/pars.json'")
        cf.parameters.loadparameters(self.parfile, phase_name=phase_name)
        cf.updateGeometry()

    def get_cf_2d(self, ignore_existing=False):
        if os.path.exists(self.col2dfile) and not ignore_existing:
            print("Loading existing colfile from", self.col2dfile)
            return self.get_cf_2d_from_disk()
        return self.get_colfile_from_peaks_dict()

    def get_cf_4d(self, ignore_existing=False):
        if os.path.exists(self.col4dfile) and not ignore_existing:
            print("Loading existing colfile from", self.col4dfile)
            return self.get_cf_4d_from_disk()
        return self.get_colfile_from_peaks_dict(peaks_dict=self.pk4d)

    def get_cf_2d_from_disk(self):
        cf_2d = ImageD11.columnfile.columnfile(self.col2dfile)
        return cf_2d

    def get_cf_3d_from_disk(self):
        cf_3d = ImageD11.columnfile.columnfile(self.col3dfile)
        return cf_3d

    def get_cf_4d_from_disk(self):
        cf_4d = ImageD11.columnfile.columnfile(self.col4dfile)
        return cf_4d

    def get_grains_from_disk(self, phase_name=None):
        group_name = "grains"
        if phase_name is not None:
            group_name = phase_name
        grains = ImageD11.grain.read_grain_file_h5(
            self.grainsfile, group_name=group_name
        )
        if phase_name is not None and hasattr(self, "phases"):
            print("Adding reference unitcells from self.phases")
            for g in grains:
                g.ref_unitcell = self.phases.unitcells[phase_name]
        return grains

    def save_grains_to_disk(self, grains, phase_name=None):
        group_name = "grains"
        if phase_name is not None:
            group_name = phase_name
        ImageD11.grain.write_grain_file_h5(
            self.grainsfile, grains, group_name=group_name
        )

    def import_nnz(self):
        """Read the nnz arrays from the sparsefiles"""
        nnz = []
        for spname in self.sparsefiles:
            with h5py.File(os.path.join(self.analysispath, spname), "r") as hin:
                nnz.append(hin[self.limapath]["nnz"][:])
        self.nnz_raw = np.concatenate(nnz).astype(np.int32)
        # spread onto the grid; an empty cell has no frame, hence zero pixels
        if getattr(self, "bins_to_frames", None) is not None:
            cf = self.bins_to_frames.reshape(self.shape)
            self.nnz = np.zeros(self.shape, np.int32)
            real = cf >= 0
            self.nnz[real] = self.nnz_raw[cf[real]]
        else:
            self.nnz = self.nnz_raw.reshape(self.shape)
        logging.info(
            "imported nnz, average %f" % (self.nnz.mean())
        )  # expensive if you are not logging it.

    def import_nnz_from_sparse(self):
        """Read the nnz arrays from the sparsefiles"""
        with h5py.File(self.sparsefile, "r") as hin:
            nnz = [hin[scan]["nnz"][:] for scan in self.scans]
        self.nnz_raw = np.concatenate(nnz).astype(np.int32)
        if getattr(self, "bins_to_frames", None) is not None:
            cf = self.bins_to_frames.reshape(self.shape)
            self.nnz = np.zeros(self.shape, np.int32)
            real = cf >= 0
            self.nnz[real] = self.nnz_raw[cf[real]]
        else:
            self.nnz = np.array([n for n in nnz], np.int32)
        logging.info(
            "imported nnz, average %f" % (self.nnz.mean())
        )  # expensive if you are not logging it.
        self.frames_per_scan = [len(n) for n in nnz]

    #    def compute_pixel_labels(self):
    # this should instead from from the pk2d file generated by sinograms/properties.py
    #        nlm = []
    #        for spname in self.sparsefiles:
    #            n, l = peaklabel.add_localmax_peaklabel( os.path.join( self.analysispath, spname ),
    #                                                     self.limapath )
    #            nlm.append(n)
    #        self.nlm = np.concatenate( nlm ).reshape( self.shape )

    #    def import_nlm(self):
    # this should instead from from the pk2d file generated by sinograms/properties.py
    #        """ Read the Nlmlabels
    #        These are the number of localmax peaks per frame
    #        """
    #        nlm = []
    #        for spname in self.sparsefiles:
    #            with h5py.File( os.path.join( self.analysispath, spname ), "r" ) as hin:
    #                nlm.append( hin[self.limapath]['Nlmlabel'][:] )
    #        self.nlm = np.concatenate( nlm ).reshape( self.shape )
    #        logging.info('imported nlm, max %d'%(self.nlm.max()))

    def check_files(self, path, filenames, verbose=0):
        """See whether files are created or not"""
        # images collected
        done = 0
        missing = 0
        for fname in filenames:
            fullname = os.path.join(path, fname)
            if os.path.exists(fullname):
                done += 1
            else:
                missing += 1
                if verbose > 0:
                    print("missing", fullname)
                    verbose -= 1
        return done, missing

    def check_images(self):
        """Is the experiment finished ?"""
        return self.check_files(self.datapath, self.imagefiles)

    def check_sparse(self):
        """Has the segmentation been done ?"""
        return self.check_files(self.analysispath, self.sparsefiles, verbose=2)

    def save(self, h5name=None, h5group="/"):
        if h5name is None:
            if os.path.exists( self.dsfile ):
                h5name = self.dsfile

            # none supplied, so use default path
            h5name = self.dsfile_default
            # make sure parent directories exist
            # ensure that the analysis path exists
            dsfile_folder = os.path.dirname(self.dsfile_default)
            if not os.path.exists(dsfile_folder):
                os.makedirs(dsfile_folder)

        ZIP = {"compression": "lzf", "shuffle": True}

        with h5py.File(h5name, "a") as hout:
            grp = hout[h5group]
            # Simple small objects
            for name in self.ATTRNAMES:
                data = getattr(self, name, None)
                if data is not None:
                    grp.attrs[name] = data
                    # The string lists
            for name in self.STRINGLISTS:
                data = getattr(self, name, None)
                if data is not None and len(data):
                    sdata = np.array(data, "S")
                    ds = grp.require_dataset(
                        name,
                        shape=sdata.shape,
                        chunks=sdata.shape,
                        dtype=h5py.string_dtype(),
                        **ZIP
                    )
                    ds[:] = sdata
            #
            for name in self.NDNAMES:
                data = getattr(self, name, None)
                if data is not None:
                    data = np.asarray(data)
                    try:
                        chunks = guess_chunks(name, data.shape)
                        ds = grp.require_dataset(
                            name,
                            shape=data.shape,
                            chunks=chunks,
                            dtype=data.dtype,
                            **ZIP
                        )
                        ds[:] = data
                    except:
                        print(name)
                        print(len(data))
                        print(data.shape)
                        print(chunks)
                        raise

        # if we got here, we saved the file successfully
        self.dsfile = h5name

    def load(self, h5name=None, h5group="/"):

        if h5name is None:
            if os.path.exists( self.dsfile ):
                h5name = self.dsfile
            elif os.path.exists( self.dsfile_default ):
                # none supplied, so use default path
                h5name = self.dsfile_default
            else:
                raise Exception( "Filename for dataset not found")

        """ Recover this from a hdf5 file """
        with h5py.File(h5name, "r") as hin:
            grp = hin[h5group]
            for name in self.ATTRNAMES:
                if name in grp.attrs:
                    setattr(self, name, grp.attrs.get(name))
            self.shape = tuple(self.shape)  # hum
            for name in self.NDNAMES:
                if name in grp:
                    data = grp[name][()]
                    setattr(self, name, data)
            for name in self.STRINGLISTS:
                if name in grp:
                    stringlist = list(grp[name][()])
                    if hasattr(stringlist[0], "decode") or isinstance(
                        stringlist[0], np.ndarray
                    ):
                        data = [s.decode() for s in stringlist]
                    else:
                        data = stringlist
                    setattr(self, name, data)
        # tolerate files written before the raw/frame-map arrays existed
        if getattr(self, "bins_to_frames", None) is None:
            # identity: each frame in its own cell (frame index = sinogram column)
            self.bins_to_frames = np.arange(
                self.shape[0] * self.shape[1], dtype=np.int64)
        if getattr(self, "omega_raw", None) is None:
            self.omega_raw = np.asarray(self.omega, float).ravel()
        if getattr(self, "dty_raw", None) is None:
            self.dty_raw = np.asarray(self.dty, float).ravel()
        if getattr(self, "scan_frame_offset", None) is None:
            self.scan_frame_offset = (
                np.arange(self.shape[0]) * self.shape[1])
        if getattr(self, "nnz_raw", None) is None and getattr(
                self, "nnz", None) is not None:
            self.nnz_raw = np.asarray(self.nnz).ravel()
        if getattr(self, "monitor_raw", None) is None and getattr(
                self, "monitor", None) is not None:
            self.monitor_raw = np.asarray(self.monitor, float).ravel()
        if getattr(self, "masked_frames", None) is None:
            self.masked_frames = np.array([], np.int64)
        # these fields are not written back out as some are only used in-memory
        if getattr(self, "frame_location", None) is None:
            self.frame_location = np.full(len(self.omega_raw), -1, np.int64)
            flat = self.bins_to_frames
            real = flat >= 0
            self.frame_location[flat[real]] = np.nonzero(real)[0]
        self._unplaced_in_scan = np.array(
            np.nonzero(self.frame_location < 0)[0], np.int64)
        self.unplaced_frames = self._unplaced_in_scan
        self.guessbins()

        # analysis paths can only be calculated once
        self.update_paths()
        # if we got here, we loaded the file successfully
        self.dsfile = h5name

        return self


def load(h5name, h5group="/"):
    ds_obj = DataSet(filename=h5name)
    return ds_obj


# Example
#    s = dataset(
#        dataroot = "/data/visitor/ma5415/id11/20221027",
#        analysisroot = "/data/visitor/ma5415/id11/20221027/analysis/SparsePixels",
#        sample = "NSCOPE_SiMo1000_6",
#        dset = "DT600nm_Z1" )
#    s.import_all()


def check(dataroot, analysisroot, sample, dset, destination, scans=None):
    h5o = DataSet(
        dataroot=dataroot, analysisroot=analysisroot, sample=sample, dset=dset
    )
    h5o.import_all(scans=scans)
    h5o.save(destination)

    print("Checking: Read back from hdf5")
    t = load(destination)
    t.report()
    return t.compare(h5o)


if __name__ == "__main__":
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    dataroot = sys.argv[1]
    analysisroot = sys.argv[2]
    sample = sys.argv[3]
    dset = sys.argv[4]
    destination = sys.argv[5]

    check(dataroot, analysisroot, sample, dset, destination)
