#!/usr/bin/env bash
# Scalp Analyzer installer — AMD ROCm / CUDA / CPU auto-detection.
set -euo pipefail
# This script lives in <project>/linux/ ; work from the project root one level up.
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
INSTALL_MODE="${INSTALL_MODE:-auto}"   # auto | rocm | cpu

# --- 1. Python check ---------------------------------------------------------
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON' not found. Install Python 3.10+ and retry." >&2
  exit 1
fi
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    sys.stderr.write(f"ERROR: Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}\n")
    sys.exit(1)
PY

# --- 2. Virtual environment --------------------------------------------------
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate

echo "Upgrading pip ..."
python -m pip install --upgrade pip --quiet

# --- 3. Base dependencies ----------------------------------------------------
echo "Installing numpy + opencv-python ..."
python -m pip install -r requirements-base.txt

# --- 4. ONNX Runtime selection ----------------------------------------------
ROCM_DETECTED=0
if [ -d /opt/rocm ] || command -v rocm-smi >/dev/null 2>&1 || command -v amd-smi >/dev/null 2>&1; then
  ROCM_DETECTED=1
fi

ROCM_VERSION=""
if [ -f /opt/rocm/.info/version ]; then
  ROCM_VERSION=$(head -1 /opt/rocm/.info/version 2>/dev/null | tr -d '\r\n' || true)
fi
if [ -z "$ROCM_VERSION" ] && command -v hipconfig >/dev/null 2>&1; then
  ROCM_VERSION=$(hipconfig --version 2>/dev/null | head -1 || true)
fi
if [ -n "$ROCM_VERSION" ]; then
  echo "Detected ROCm version: $ROCM_VERSION"
elif [ "$ROCM_DETECTED" = "1" ]; then
  echo "Detected ROCm install (version string not found)."
fi

install_cpu() {
  echo "Installing onnxruntime (CPU) ..."
  python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-rocm >/dev/null 2>&1 || true
  python -m pip install onnxruntime
}

install_rocm() {
  echo "Installing onnxruntime-rocm ..."
  python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-rocm >/dev/null 2>&1 || true

  # Optional AMD-hosted wheel index, e.g.
  #   ONNXRUNTIME_ROCM_INDEX=https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/
  if [ -n "${ONNXRUNTIME_ROCM_INDEX:-}" ]; then
    echo "Using AMD wheel index: $ONNXRUNTIME_ROCM_INDEX"
    if python -m pip install onnxruntime-rocm -f "$ONNXRUNTIME_ROCM_INDEX"; then
      return 0
    fi
    echo "AMD index install failed; trying PyPI ..."
  fi

  if ! python -m pip install onnxruntime-rocm; then
    echo "onnxruntime-rocm unavailable; falling back to CPU wheel."
    install_cpu
    return 0
  fi

  # Sanity check that a GPU provider is actually exposed
  if ! python - <<'PY'
import sys, onnxruntime as ort
p = ort.get_available_providers()
sys.exit(0 if any(x in p for x in
    ("MIGraphXExecutionProvider", "ROCMExecutionProvider")) else 1)
PY
  then
    echo "onnxruntime-rocm installed but no GPU provider exposed."
    echo "  Check that LD_LIBRARY_PATH includes /opt/rocm/lib."
    echo "  Keeping the ROCm wheel anyway; it falls back to CPU at runtime."
  fi
}

case "$INSTALL_MODE" in
  cpu)
    install_cpu
    ;;
  rocm)
    install_rocm
    ;;
  auto)
    if [ "$ROCM_DETECTED" = "1" ]; then
      install_rocm
    else
      install_cpu
    fi
    ;;
  *)
    echo "ERROR: INSTALL_MODE must be one of auto|rocm|cpu (got '$INSTALL_MODE')" >&2
    exit 1
    ;;
esac

# --- 5. Directory scaffolding ------------------------------------------------
mkdir -p data models samples outputs

if [ ! -f "data/calibration_reference.json" ]; then
  cat > data/calibration_reference.json <<'JSON'
{
  "x20":  {"px_per_mm": 1180.0},
  "x50":  {"px_per_mm": 2950.0},
  "x100": {"px_per_mm": 5900.0}
}
JSON
  echo "Wrote default calibration to data/calibration_reference.json"
fi

# --- 6. Smoke test -----------------------------------------------------------
echo
echo "Verifying environment ..."
python check_env.py || true

echo
echo "======================================================"
echo " Installation complete."
echo "======================================================"
echo " Activate:      source .venv/bin/activate"
echo " Run analysis:  ./linux/run_analysis.sh /path/to/scalp.jpg   (or python main.py ...)"
echo " Verify GPU:    python check_env.py"
echo
echo " GPU inference only activates once .onnx models exist"
echo " under models/. See models/README.md."
