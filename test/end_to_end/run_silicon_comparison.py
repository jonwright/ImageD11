"""Run label_silicon.py under the old and new ImageD11 and compare the result.

The old ImageD11 is the one in the jupyter-slurm site-packages; the new ImageD11
is this repository. Run from inside `module load jupyter-slurm`:

    python run_silicon_comparison.py [label]

What it does:
- runs label_silicon.py twice (old then new), redirecting each run's stdout and
  stderr into <output>/<old|new>/run.log, and appending the timing and the
  ImageD11 import path (as reported by the worker) to a compare.log;
- compares the omega/dty bins and the pks2d peaks/merges from the two runs;
- exit code 0 if old and new agree, otherwise 1.

Runs only under Python 3.
"""
import os
import sys
import json
import time
import subprocess
import datetime

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # .../ImageD11_llm
DEST = os.path.join(REPO, "test", "test_fetch_data")
WORKER = os.path.join(HERE, "label_silicon.py")
PY = sys.executable
OUTROOT = os.path.join(HERE, "output")

LABEL_KEYS = ("ImageD11 imported from:", "python executable:",
              "cpu affinity pinned to:", "sparse file:",
              "import_from_sparse:", "properties.main:", "peaks:",
              "pairs_ii:", "nlabel:", "glabel saved:")


def run_one(code, outdir, wait=False):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if code == "new":
        env["PYTHONPATH"] = REPO
    os.makedirs(outdir, exist_ok=True)
    logpath = os.path.join(outdir, "run.log")
    start = time.time()
    with open(logpath, "w") as fh:
        rc = subprocess.run([PY, WORKER, DEST, outdir],
                            env=env, cwd="/tmp", stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - start
    return rc, elapsed, logpath


def parse_log(logpath):
    info = {}
    with open(logpath) as fh:
        for line in fh:
            for key in LABEL_KEYS:
                if line.startswith(key):
                    info[key.rstrip(":")] = line[len(key):].strip()
    return info


def read_bins(path):
    with open(path) as fh:
        return json.load(fh)


def read_pks(path):
    with h5py.File(path, "r") as h:
        g = h["pks2d"]
        return {
            "pk_props": g["pk_props"][:],
            "npk": g["npk"][:],
            "glabel": g["glabel"][:] if "glabel" in g else None,
            "nlabel": int(g.attrs.get("nlabel", 0)),
        }


def compare_bins(a, b):
    ok = True
    report = []
    for k in ("shape", "nscans", "omega_wraps"):
        same = a.get(k) == b.get(k)
        ok = ok and same
        report.append("bin  %-12s match=%-5s old=%s new=%s"
                      % (k, same, a.get(k), b.get(k)))
    for k in ("obinedges", "obincens", "ybinedges", "ybincens"):
        va = None if a.get(k) is None else np.asarray(a[k], float)
        vb = None if b.get(k) is None else np.asarray(b[k], float)
        if va is None or vb is None:
            same = va is None and vb is None
            diff = float("nan")
        else:
            same = va.shape == vb.shape and np.array_equal(va, vb)
            diff = float(np.abs(va - vb).max()) if va.shape == vb.shape else float("nan")
        ok = ok and same
        report.append("bin  %-12s match=%-5s max|diff|=%.3g" % (k, same, diff))
    for k in ("omin", "omax", "ostep", "ymin", "ymax", "ystep"):
        va, vb = a.get(k), b.get(k)
        if va is None and vb is None:
            same = True
        elif va is not None and vb is not None:
            same = abs(va - vb) <= 1e-9 * max(1.0, abs(va))
        else:
            same = False
        ok = ok and same
        report.append("bin  %-12s match=%-5s old=%s new=%s"
                      % (k, same, va, vb))
    return ok, report


def compare_pks(a, b):
    ok = True
    report = []
    for k in ("npk",):
        va, vb = np.asarray(a[k]), np.asarray(b[k])
        same = va.shape == vb.shape and np.array_equal(va, vb)
        ok = ok and same
        report.append("peak %-8s identical=%-5s old=%s new=%s"
                      % (k, same, va.shape, vb.shape))
    # row 4 of pk_props is the frame id. It is expected to differ: old stores a
    # frame-index grid address, new stores the omega-aligned sinogram address.
    # The measured peak values (rows 0-3) must be identical.
    va, vb = np.asarray(a["pk_props"]), np.asarray(b["pk_props"])
    same = va.shape == vb.shape and np.array_equal(va[0:4], vb[0:4])
    ok = ok and same
    report.append("peak %-8s identical=%-5s old=%s new=%s"
                  % ("pk_props[:4]", same, va.shape, vb.shape))
    if a["glabel"] is None or b["glabel"] is None:
        same = a["glabel"] is None and b["glabel"] is None
        ok = ok and same
        report.append("peak glabel match=%-5s old=%s new=%s"
                      % (same, a["glabel"] is not None, b["glabel"] is not None))
    else:
        ok = ok and (a["nlabel"] == b["nlabel"])
        report.append("peak nlabel   old=%s new=%s" % (a["nlabel"], b["nlabel"]))
        same = np.array_equal(np.asarray(a["glabel"]), np.asarray(b["glabel"]))
        ok = ok and same
        report.append("peak glabel identical=%-5s (n mismatches=%d)"
                      % (same, int((np.asarray(a["glabel"]) != np.asarray(b["glabel"])).sum())))
    return ok, report


def main():
    label = sys.argv[1] if len(sys.argv) > 1 \
        else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outroot = os.path.join(OUTROOT, label)
    old_dir = os.path.join(outroot, "old")
    new_dir = os.path.join(outroot, "new")

    rc_old, t_old, old_log = run_one("old", old_dir)
    rc_new, t_new, new_log = run_one("new", new_dir)
    old_info = parse_log(old_log)
    new_info = parse_log(new_log)

    ok_bins, rep_bins = compare_bins(
        read_bins(os.path.join(old_dir, "bins.json")),
        read_bins(os.path.join(new_dir, "bins.json")))
    ok_pks, rep_pks = compare_pks(
        read_pks(os.path.join(old_dir, "peaks.h5")),
        read_pks(os.path.join(new_dir, "peaks.h5")))

    compare_log = os.path.join(outroot, "compare.log")
    with open(compare_log, "w") as fh:
        fh.write("ImageD11 (old) imported from: %s\n" % old_info.get("ImageD11 imported from"))
        fh.write("ImageD11 (new) imported from: %s\n" % new_info.get("ImageD11 imported from"))
        fh.write("old rc=%s wall=%.3fs import=%s label=%s  (peaks=%s nlabel=%s)\n"
                 % (rc_old, t_old, old_info.get("import_from_sparse"),
                    old_info.get("properties.main"), old_info.get("peaks"),
                    old_info.get("nlabel")))
        fh.write("new rc=%s wall=%.3fs import=%s label=%s  (peaks=%s nlabel=%s)\n"
                 % (rc_new, t_new, new_info.get("import_from_sparse"),
                    new_info.get("properties.main"), new_info.get("peaks"),
                    new_info.get("nlabel")))
        fh.write("\n== bins ==\n")
        fh.write("\n".join(rep_bins) + "\n")
        fh.write("\n== peaks ==\n")
        fh.write("\n".join(rep_pks) + "\n")
        fh.write("\nRESULT: %s\n" % ("MATCH" if (ok_bins and ok_pks) else "MISMATCH"))

    ok = rc_old == 0 and rc_new == 0 and ok_bins and ok_pks
    print("output folder: %s" % outroot)
    print("old rc=%d wall=%.3fs  new rc=%d wall=%.3fs" % (rc_old, t_old, rc_new, t_new))
    print("bins match: %s   peaks match: %s" % (ok_bins, ok_pks))
    print("RESULT: %s" % ("MATCH" if ok else "MISMATCH"))
    with open(compare_log) as fh:
        sys.stdout.write(fh.read())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
