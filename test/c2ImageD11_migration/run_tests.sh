#!/bin/bash
set -euo pipefail
# Run ImageD11 test suite in three configurations and compare results.
#
# Prerequisites: run setup_venvs.sh first.
#
# Configurations:
#   A - Old f2py backend  (IMAGED11_USE_C2 unset, default)
#   B - New c2py23 backend (IMAGED11_USE_C2=1, expected to have failures)
#   C - Equivalence tests   (c2ImageD11's own test_equivalence.py)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

# Log versions at the top
{
    echo "=== Versions used in this test run ==="
    echo "Timestamp: $TIMESTAMP"
    echo ""
    echo "ImageD11:"
    echo "  branch: $(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD)"
    echo "  commit: $(cd "$REPO_ROOT" && git rev-parse HEAD)"
    echo "  cImageD11.py last change: $(cd "$REPO_ROOT" && git log -1 --format='%h %s' -- ImageD11/cImageD11.py)"
    echo ""
    C2PY23_COMMIT=$(source "$REPO_ROOT/venv-new/bin/activate" 2>/dev/null && \
        python -c 'import c2py23, os; print(os.popen("cd "+os.path.dirname(c2py23.__file__)+" && git rev-parse HEAD").read().strip())' 2>/dev/null || echo unknown)
    echo "c2py23:"
    echo "  version: $(source "$REPO_ROOT/venv-new/bin/activate" 2>/dev/null && python -c 'import c2py23; print(c2py23.__version__)' 2>/dev/null || echo unknown)"
    echo "  commit:  $C2PY23_COMMIT"
    echo ""
    C2_DIR="$REPO_ROOT/c2ImageD11_clone"
    if [ -d "$C2_DIR" ]; then
        echo "c2ImageD11:"
        echo "  commit:  $(cd "$C2_DIR" && git rev-parse HEAD)"
        echo "  branch:  $(cd "$C2_DIR" && git rev-parse --abbrev-ref HEAD)"
    fi
    echo ""
    echo "============================================"
    echo ""
} | tee "$RESULTS_DIR/VERSIONS.txt"

echo "=== ImageD11 c2ImageD11 migration test run ==="
echo "Timestamp: $TIMESTAMP"
echo "Results:   $RESULTS_DIR"
echo ""

# ---------------------------------------------------------------------------
# Configuration A: Old f2py backend (default)
# ---------------------------------------------------------------------------
echo "--- [A] Old backend: IMAGED11_USE_C2 unset ---"
source "$REPO_ROOT/venv-old/bin/activate"
cd "$REPO_ROOT/test"

set +e
python -m pytest -v 2>&1 | tee "$RESULTS_DIR/A_old_backend.log"
RC_A=$?
set -e

if [ $RC_A -eq 0 ]; then
    echo "[A] PASS (all tests passed)"
    echo "PASS" > "$RESULTS_DIR/A_status.txt"
else
    echo "[A] FAIL (some tests failed)"
    echo "FAIL (rc=$RC_A)" > "$RESULTS_DIR/A_status.txt"
fi
deactivate
echo ""

# ---------------------------------------------------------------------------
# Configuration B: New c2py23 backend
# ---------------------------------------------------------------------------
echo "--- [B] New backend: IMAGED11_USE_C2=1 ---"
source "$REPO_ROOT/venv-new/bin/activate"
cd "$REPO_ROOT/test"

set +e
IMAGED11_USE_C2=1 python -m pytest -v 2>&1 | tee "$RESULTS_DIR/B_new_backend.log"
RC_B=$?
set -e

if [ $RC_B -eq 0 ]; then
    echo "[B] PASS (all tests passed with new backend - API is compatible!)"
    echo "PASS" > "$RESULTS_DIR/B_status.txt"
else
    echo "[B] FAIL (some tests failed - expected due to API differences)"
    echo "FAIL (rc=$RC_B) - expected, API not yet compatible" > "$RESULTS_DIR/B_status.txt"
fi
deactivate
echo ""

# ---------------------------------------------------------------------------
# Configuration C: c2ImageD11 equivalence tests
# ---------------------------------------------------------------------------
echo "--- [C] Equivalence tests (c2ImageD11 vs ImageD11._cImageD11) ---"
source "$REPO_ROOT/venv-new/bin/activate"
C2_DIR="$REPO_ROOT/c2ImageD11_clone"

if [ -d "$C2_DIR/tests" ]; then
    cd "$C2_DIR/tests"

    set +e
    python -m pytest -v test_equivalence.py 2>&1 | tee "$RESULTS_DIR/C_equivalence.log"
    RC_C=$?
    set -e

    if [ $RC_C -eq 0 ]; then
        echo "[C] PASS (all equivalence tests pass)"
        echo "PASS" > "$RESULTS_DIR/C_status.txt"
    else
        echo "[C] FAIL (some equivalence tests failed)"
        echo "FAIL (rc=$RC_C)" > "$RESULTS_DIR/C_status.txt"
    fi

    # Also run the full c2ImageD11 test suite
    set +e
    python -m pytest -v 2>&1 | tee "$RESULTS_DIR/C_full_c2.log"
    RC_C2=$?
    set -e
    if [ $RC_C2 -eq 0 ]; then
        echo "[C-full] PASS"
        echo "PASS" > "$RESULTS_DIR/C_full_status.txt"
    else
        echo "[C-full] FAIL (rc=$RC_C2)"
        echo "FAIL (rc=$RC_C2)" > "$RESULTS_DIR/C_full_status.txt"
    fi
else
    echo "[C] SKIP - c2ImageD11_clone not found (run setup_venvs.sh first)"
    echo "SKIP" > "$RESULTS_DIR/C_status.txt"
fi
deactivate
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== Summary ==="
echo ""
echo "Config A (old backend):   $(cat "$RESULTS_DIR/A_status.txt")"
echo "Config B (new backend):   $(cat "$RESULTS_DIR/B_status.txt")"
echo "Config C (equivalence):   $(cat "$RESULTS_DIR/C_status.txt")"
if [ -f "$RESULTS_DIR/C_full_status.txt" ]; then
    echo "Config C (full c2 tests): $(cat "$RESULTS_DIR/C_full_status.txt")"
fi
echo ""
echo "Full logs: $RESULTS_DIR/"
echo ""
echo "Diff old vs new test output:"
echo "  diff $RESULTS_DIR/A_old_backend.log $RESULTS_DIR/B_new_backend.log"
