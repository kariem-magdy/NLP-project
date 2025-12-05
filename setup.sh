#!/usr/bin/env bash
set -euo pipefail

# ============================================
# setup.sh (Windows Git Bash + Kaggle)
# ============================================

VENV_DIR="${VENV_DIR:-venv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Detect Kaggle
if [[ -d "/kaggle" ]]; then
    echo "[INFO] Kaggle environment detected → system installs disabled"
    INSTALL_SYSTEM=false
else
    INSTALL_SYSTEM=true
fi

# Parse optional args
for arg in "$@"; do
  case $arg in
    --no-system) INSTALL_SYSTEM=false ;;
    --venv=*) VENV_DIR="${arg#*=}" ;;
    --python=*) PYTHON_BIN="${arg#*=}" ;;
    *) ;;
  esac
done

echo "=== Setup Started ==="
echo "Platform: $(uname -a || echo Windows)"
echo "Venv Dir: $VENV_DIR"
echo "Python: $PYTHON_BIN"
echo "Install System Packages: $INSTALL_SYSTEM"

# --------------------------------------
# STEP 1 — System packages (Skip on Kaggle & Windows)
# --------------------------------------
if [ "$INSTALL_SYSTEM" = true ]; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "[INFO] Installing minimal apt packages..."
        sudo apt-get update -y
        sudo apt-get install -y build-essential git curl ffmpeg libsndfile1
    else
        echo "[INFO] No apt-get found (Windows environment). Skipping system packages."
    fi
fi

# --------------------------------------
# STEP 2 — Create virtualenv
# --------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtualenv..."
    $PYTHON_BIN -m venv $VENV_DIR
else
    echo "[INFO] Virtualenv already exists."
fi

# --------------------------------------
# STEP 3 — Install Python packages
# --------------------------------------
echo "[INFO] Upgrading pip..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools

echo "[INFO] Installing project dependencies..."
"$VENV_DIR/bin/pip" install -r requirements.txt

# --------------------------------------
# STEP 4 — Create folders
# --------------------------------------
mkdir -p outputs/models
mkdir -p outputs/logs
mkdir -p data

# --------------------------------------
# STEP 5 — Fix permissions (Git Bash only)
# --------------------------------------
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x setup.sh 2>/dev/null || true

echo "=== Setup Completed ==="
echo "Activate venv with:"
echo "  source $VENV_DIR/bin/activate"
