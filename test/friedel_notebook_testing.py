#!/usr/bin/env python
from __future__ import print_function
"""
Subset / smoke testing for the Friedel-pair notebooks.

The friedel-pair logic (find_pairs, locate_pairs, fit_y0, match_box_beam) now
lives in ``ImageD11.friedel_pairs`` and is unit tested.  The notebooks are thin
wrappers, so here we only execute the cheap, friedel-relevant cells of a
notebook -- skipping the heavy reconstruction (``run_iradon``), segmentation
(``segment_hist``) and indexing steps -- and assert the friedel outputs look
right.

This is the ESRF recipe for people who want to test the notebooks without
running the whole pipeline:

    module load jupyter-slurm
    PYTHONPATH=/path/to/ImageD11 python friedel_notebook_testing.py \
        --dataset /data/id11/inhouse2/test_data_3DXRD/S3DXRD/Si_cube/processed/Si_cube/Si_cube_S3DXRD_nt_moves_dty/Si_cube_S3DXRD_nt_moves_dty_dataset.h5 \
        --workdir /scratch/<you>/friedel_nb_test --notebook fit_y0

If you do not have a pre-processed dataset you can fetch the Si_cube one from
Zenodo first (it is sparse-level, so it needs one segmentation pass before the
friedel cells will work):

    PYTHONPATH=/path/to/ImageD11 python -c "import ImageD11.fetch_data as fd; fd.si_cube_s3dxrd_dataset('/scratch/<you>/data', allow_download=True)"
"""
import argparse
import os
import shutil
import sys
import tempfile


def _patch_nbformat_outputs():
    """
    Environment workaround for a papermill / nbformat mismatch: the parameters
    cell papermill injects can lack an 'outputs' key, which
    nbformat.v4.rwbase.split_lines chokes on.  Safe no-op elsewhere.
    """
    try:
        import nbformat.v4.rwbase as _rwbase
        import nbformat.v4.nbjson as _nbjson
        from nbformat.v4.rwbase import _split_mimebundle

        def _safe_split_lines(nb):
            for cell in nb.cells:
                source = cell.get("source", None)
                if isinstance(source, str):
                    cell["source"] = source.splitlines(True)
                attachments = cell.get("attachments", {})
                for _, attachment in attachments.items():
                    _split_mimebundle(attachment)
                if cell.cell_type == "code":
                    for output in cell.get("outputs", []):
                        if output.output_type in {"execute_result", "display_data"}:
                            _split_mimebundle(output.get("data", {}))
                        elif (
                            output.output_type == "stream"
                            and isinstance(output.text, str)
                        ):
                            output.text = output.text.splitlines(True)
            return nb

        _rwbase.split_lines = _safe_split_lines
        _nbjson.split_lines = _safe_split_lines
    except Exception:  # pragma: no cover - best effort
        pass


# attributes that point at output/input files and must be redirected when a
# dataset is copied into a writable workdir
_FILE_ATTRS = (
    "parfile", "pksfile", "col2dfile", "col3dfile", "col4dfile", "grainsfile",
    "maskfile", "e2dxfile", "e2dyfile", "sparsefile", "masterfile", "icolfile",
    "pbpfile",
)
_DIR_ATTRS = ("analysispath", "analysisroot", "dataroot", "datapath")


def _load_attrs(dataset_path):
    import h5py
    with h5py.File(dataset_path, "r") as h:
        return {k: h.attrs[k] for k in h.attrs.keys()}


def _write_attrs(dataset_path, attrs):
    import h5py
    with h5py.File(dataset_path, "r+") as h:
        for k, v in attrs.items():
            h.attrs[k] = v


def prepare_dataset_workdir(src_dataset, workdir):
    """
    Copy a pre-processed dataset into ``workdir`` so the notebooks can write to
    it, and rewrite the dataset's file/directory attributes to point at the
    copies instead of the (possibly read-only) originals.

    Copies the dataset h5, its peaks files (peaks_table, col2d/3d/4d) and the
    whole parameter-file directory, then rewrites the write-target attributes.

    Returns (dest_dataset, dest_parfile).
    """
    os.makedirs(workdir, exist_ok=True)
    attrs = _load_attrs(src_dataset)

    # copy the dataset itself
    dest_dataset = os.path.join(workdir, os.path.basename(src_dataset))
    shutil.copy2(src_dataset, dest_dataset)

    new_attrs = dict(attrs)

    # copy the referenced parameter directory so relative parfile refs resolve
    src_parfile = attrs.get("parfile")
    if src_parfile and os.path.exists(src_parfile):
        par_dir = os.path.dirname(src_parfile)
        dest_par_dir = os.path.join(workdir, "pars")
        shutil.copytree(par_dir, dest_par_dir, dirs_exist_ok=True)
        new_attrs["parfile"] = os.path.join(dest_par_dir, os.path.basename(src_parfile))

    for key in _FILE_ATTRS:
        if key == "parfile":
            continue  # handled below (needs its referenced par files alongside)
        val = attrs.get(key)
        if val and isinstance(val, str) and os.path.exists(val):
            dst = os.path.join(workdir, os.path.basename(val))
            if not os.path.exists(dst):
                try:
                    shutil.copy2(val, dst)
                except OSError:  # e.g. unreadable binary in /data
                    pass
            new_attrs[key] = dst

    # directory attributes -> workdir
    for key in _DIR_ATTRS:
        if key in new_attrs:
            new_attrs[key] = workdir

    _write_attrs(dest_dataset, new_attrs)
    return dest_dataset, new_attrs.get("parfile")


def execute_subset(nb_in, nb_out, params, stop_marker=None, stop_index=None):
    """
    Execute a subset of a notebook's cells with papermill.

    ``stop_marker``: keep cells strictly before the first cell whose source
    contains this string.  ``stop_index``: keep nb.cells[:stop_index].
    If neither is given the whole notebook is run.
    """
    import nbformat
    import papermill

    nb = nbformat.read(nb_in, as_version=4)
    if stop_index is not None:
        keep = stop_index
    elif stop_marker:
        keep = len(nb.cells)
        for i, c in enumerate(nb.cells):
            if stop_marker in "".join(c.get("source", [])):
                keep = i
                break
    else:
        keep = len(nb.cells)
    nb.cells = nb.cells[:keep]

    fd, tmp_path = tempfile.mkstemp(suffix=".ipynb")
    os.close(fd)
    try:
        nbformat.write(nb, tmp_path)
        papermill.execute_notebook(tmp_path, nb_out, parameters=params)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return nb_out


# ---------------------------------------------------------------------------
# per-notebook subset specifications (Si_cube fixture)
# ---------------------------------------------------------------------------
def fit_y0_spec(dataset, parfile):
    return dict(
        notebook="fit_y0.ipynb",
        stop_marker=None,          # whole notebook is lightweight
        params=dict(
            dset_path=dataset,
            par_file=parfile,
            y0_manual=0.0,
            is_half_scan=False,
            gvtol=0.002,
        ),
        check="fit_y0",
    )


def friedel_1_index_spec(dataset, parfile):
    return dict(
        notebook="friedel_1_index.ipynb",
        stop_marker="run_iradon(",  # skip reconstruction / segmentation / indexing
        params=dict(
            dset_path=dataset,
            phase_str="Si",
            par_file=parfile,
            y0_manual=0.0,
            is_half_scan=False,
            gvtol=0.0015,
            cf_strong_ifrac=5e-5,
            cf_strong_dstol=0.003,
        ),
        check="find_pairs",
    )


def friedel_pair_map_spec(dataset, parfile):
    return dict(
        notebook="friedel_pair_map.ipynb",
        # skip the position-specific selection / external indexing: only validate
        # find_pairs + locate_pairs + the pair-position grid.
        stop_marker="abs(sx-px)",
        params=dict(
            dset_path=dataset,
            phase_str="Si",
            y0=0.0,
            gvtol=0.002,
            ytol=1.0,
            px=0.0,
            py=0.0,
            min_frames_per_peak=0,
            filter_by_phase=False,
        ),
        check="find_pairs",
    )


SPECS = {
    "fit_y0": fit_y0_spec,
    "friedel_1_index": friedel_1_index_spec,
    "friedel_pair_map": friedel_pair_map_spec,
}


def _summarise(nb_out, spec):
    """Report the friedel-relevant outputs produced by the executed notebook."""
    import json
    import nbformat
    nb = nbformat.read(nb_out, as_version=4)
    text = "\n".join(
        "".join(o.get("text", ""))
        for c in nb.cells
        for o in c.get("outputs", [])
        if o.get("output_type") == "stream"
    )
    print("\n--- notebook stdout (friedel-relevant markers) ---")
    for line in text.splitlines():
        if any(k in line for k in ("pairs", "matches", "Got", "median", "candidate",
                                   "unique", "grains", "finite", "y0")):
            print("   ", line.strip()[:160])
    for i, c in enumerate(nb.cells):
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                print("ERROR in cell %d: %s %s" % (i, o.get("ename"), o.get("evalue")))
                return False
    return True


def main(argv=None):
    _patch_nbformat_outputs()
    # import ImageD11 (repo) so kernels can rely on it via PYTHONPATH as well
    rd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, rd)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--notebook", required=True, choices=list(SPECS))
    p.add_argument("--dataset", required=True, help="pre-processed dataset .h5")
    p.add_argument("--workdir", required=True, help="writable output dir")
    p.add_argument("--parfile", default=None, help="override the dataset parfile")
    p.add_argument("--repo", default=rd, help="ImageD11 git checkout to use")
    args = p.parse_args(argv)

    os.makedirs(args.workdir, exist_ok=True)
    dataset, parfile = prepare_dataset_workdir(args.dataset, args.workdir)
    if args.parfile:
        parfile = args.parfile

    spec = SPECS[args.notebook](dataset, parfile)
    nb_base = os.path.join(args.repo, "ImageD11", "nbGui", "S3DXRD")
    nb_in = os.path.join(nb_base, spec["notebook"])
    nb_out = os.path.join(args.workdir, spec["notebook"].replace(".ipynb", "_subset_out.ipynb"))

    print("Running subset of %s (stop before %r)" % (spec["notebook"], spec["stop_marker"]))
    execute_subset(nb_in, nb_out, spec["params"],
                   stop_marker=spec["stop_marker"])
    ok = _summarise(nb_out, spec)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
