#!/bin/bash
set -euo pipefail
# Build pydoc and Sphinx documentation for both C backends.
#
# Outputs:
#   results_*/pydoc-old/    - pydoc HTML for old f2py backend
#   results_*/pydoc-new/    - pydoc HTML for new c2py23 backend
#   results_*/sphinx-old/   - Sphinx HTML for old f2py backend
#   results_*/sphinx-new/   - Sphinx HTML for new c2py23 backend
#
# Prerequisites: run setup_venvs.sh first.

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/results_docs_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "=== Building docs for both backends ==="
echo "Timestamp: $TIMESTAMP"
echo "Output:    $RESULTS_DIR"
echo ""

# ---------------------------------------------------------------------------
# pydoc generation
# ---------------------------------------------------------------------------

echo "--- [pydoc] Old backend ---"
source "$REPO_ROOT/venv-old/bin/activate"
mkdir -p "$RESULTS_DIR/pydoc-old"
cd "$RESULTS_DIR/pydoc-old"

set +e
python -m pydoc -w ImageD11.cImageD11 2>&1 | tee "$RESULTS_DIR/pydoc-old.log"
set -e
deactivate
echo ""

echo "--- [pydoc] New backend ---"
source "$REPO_ROOT/venv-new/bin/activate"
mkdir -p "$RESULTS_DIR/pydoc-new"
cd "$RESULTS_DIR/pydoc-new"

set +e
IMAGED11_USE_C2=1 python -m pydoc -w ImageD11.cImageD11 2>&1 | tee "$RESULTS_DIR/pydoc-new.log"
# Also try pydoc on c2ImageD11 directly
python -m pydoc -w c2ImageD11 2>&1 | tee -a "$RESULTS_DIR/pydoc-new.log"
set -e
deactivate
echo ""

# ---------------------------------------------------------------------------
# Sphinx documentation
# ---------------------------------------------------------------------------

SPHINX_DIR="$REPO_ROOT/docs/sphx"

# Old backend Sphinx
echo "--- [Sphinx] Old backend ---"
source "$REPO_ROOT/venv-old/bin/activate"
mkdir -p "$RESULTS_DIR/sphinx-old"

# Build C extension in-place so sphinx autodoc can import it
cd "$REPO_ROOT"
set +e
python setup.py build_ext --inplace 2>&1 | tee "$RESULTS_DIR/sphinx-old_build.log"

# Build sphinx docs
cd "$SPHINX_DIR"
make clean 2>&1 | tee -a "$RESULTS_DIR/sphinx-old.log" || true
make html 2>&1 | tee -a "$RESULTS_DIR/sphinx-old.log"

if [ -d "_build/html" ]; then
    cp -r _build/html "$RESULTS_DIR/sphinx-old/html"
    echo "[Sphinx-old] HTML docs copied to $RESULTS_DIR/sphinx-old/html"
else
    echo "[Sphinx-old] Build may have failed — check logs"
fi
set -e
deactivate
echo ""

# New backend Sphinx
echo "--- [Sphinx] New backend ---"
source "$REPO_ROOT/venv-new/bin/activate"
mkdir -p "$RESULTS_DIR/sphinx-new"

# Rebuild C extension (not strictly needed since we use c2ImageD11, but
# sphinx autodoc imports ImageD11 which may try to import _cImageD11)
cd "$REPO_ROOT"
set +e
IMAGED11_USE_C2=1 python setup.py build_ext --inplace 2>&1 | tee "$RESULTS_DIR/sphinx-new_build.log"

cd "$SPHINX_DIR"
make clean 2>&1 | tee -a "$RESULTS_DIR/sphinx-new.log" || true
IMAGED11_USE_C2=1 make html 2>&1 | tee -a "$RESULTS_DIR/sphinx-new.log"

if [ -d "_build/html" ]; then
    cp -r _build/html "$RESULTS_DIR/sphinx-new/html"
    echo "[Sphinx-new] HTML docs copied to $RESULTS_DIR/sphinx-new/html"
else
    echo "[Sphinx-new] Build may have failed — check logs"
fi
set -e
deactivate
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== Done ==="
echo ""
echo "pydoc (old): $RESULTS_DIR/pydoc-old/"
echo "pydoc (new): $RESULTS_DIR/pydoc-new/"
echo "sphinx (old): $RESULTS_DIR/sphinx-old/"
echo "sphinx (new): $RESULTS_DIR/sphinx-new/"
echo ""
echo "Compare pydoc:"
echo "  diff -u $RESULTS_DIR/pydoc-old/ $RESULTS_DIR/pydoc-new/"
echo ""
echo "Open sphinx:"
echo "  firefox $RESULTS_DIR/sphinx-old/html/index.html"
echo "  firefox $RESULTS_DIR/sphinx-new/html/index.html"
