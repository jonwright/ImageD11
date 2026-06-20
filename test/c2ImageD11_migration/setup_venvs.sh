#!/bin/bash
set -euo pipefail
# Set up venvs for testing the old (f2py) and new (c2py23) C backends.
#
# Creates:
#   venv-old/  - ImageD11 only (old f2py backend, default)
#   venv-new/  - ImageD11 + c2py23 + c2ImageD11 (new c2py23 backend)
#
# Also clones c2ImageD11 with submodules into ./c2ImageD11_clone/
# Uses git HEAD/main for c2py23 and c2ImageD11 (commits logged during build).

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Setting up c2ImageD11 migration test venvs ==="
echo "Repo root: $REPO_ROOT"
echo ""

# --- venv-old: ImageD11 only (old f2py backend) ---
if [ -d "$REPO_ROOT/venv-old" ]; then
    echo "[old] venv-old already exists, skipping creation"
else
    echo "[old] Creating venv-old..."
    python3 -m venv "$REPO_ROOT/venv-old"
    source "$REPO_ROOT/venv-old/bin/activate"
    pip install --upgrade pip
    pip install -e "$REPO_ROOT[full]"
    deactivate
    echo "[old] Done."
fi

# --- venv-new: ImageD11 + c2py23 + c2ImageD11 ---
if [ -d "$REPO_ROOT/venv-new" ]; then
    echo "[new] venv-new already exists, skipping creation"
else
    echo "[new] Creating venv-new..."
    python3 -m venv "$REPO_ROOT/venv-new"
    source "$REPO_ROOT/venv-new/bin/activate"
    pip install --upgrade pip

    # Install ImageD11 first (from local source, with full extras)
    echo "[new] Installing ImageD11 (local editable, [full])..."
    pip install -e "$REPO_ROOT[full]"

    # Install c2py23 from HEAD of main (not on PyPI)
    echo "[new] Installing c2py23 from git HEAD (main)..."
    pip install "git+https://github.com/jonwright/c2py23.git"

    # Clone c2ImageD11 with submodules at HEAD of master, then install
    echo "[new] Cloning c2ImageD11 (master) with submodules..."
    C2_DIR="$REPO_ROOT/c2ImageD11_clone"
    if [ ! -d "$C2_DIR" ]; then
        git clone --recurse-submodules https://github.com/jonwright/c2ImageD11.git "$C2_DIR"
    else
        echo "[new] c2ImageD11_clone already exists, updating..."
        (cd "$C2_DIR" && git checkout master && git pull --recurse-submodules)
    fi

    echo "[new] Installing c2ImageD11..."
    C2PY23_REBUILD=1 pip install --no-build-isolation -e "$C2_DIR"

    deactivate
    echo "[new] Done."
fi

# Log the commits actually used
echo ""
echo "=== Resolved versions ==="
echo "  ImageD11:      local editable install at $REPO_ROOT"
echo "    branch:       $(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD)"
echo "    commit:       $(cd "$REPO_ROOT" && git rev-parse HEAD)"
echo "    cImageD11.py: $(cd "$REPO_ROOT" && git log -1 --format=%h -- ImageD11/cImageD11.py) (last change)"
echo ""
echo "  c2py23:"
echo "    version: $(source "$REPO_ROOT/venv-new/bin/activate" && python -c 'import c2py23; print(c2py23.__version__)' 2>/dev/null || echo unknown)"
echo "    commit:  $(source "$REPO_ROOT/venv-new/bin/activate" && python -c 'import c2py23; print(getattr(c2py23, "__commit__", "not set"))' 2>/dev/null || echo 'check pip show')"
echo "  c2ImageD11:"
C2_DIR="$REPO_ROOT/c2ImageD11_clone"
if [ -d "$C2_DIR" ]; then
    echo "    commit:  $(cd "$C2_DIR" && git rev-parse HEAD)"
fi
echo ""
echo "=== Setup complete ==="
echo "  venv-old: source $REPO_ROOT/venv-old/bin/activate"
echo "  venv-new: source $REPO_ROOT/venv-new/bin/activate"
