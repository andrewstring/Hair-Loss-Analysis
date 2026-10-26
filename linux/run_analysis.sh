#!/usr/bin/env bash
# Scalp Analyzer - run helper (Linux).
#   ./linux/run_analysis.sh /path/to/scalp.jpg [x20|x50|x100]
# A JSON report is saved to outputs/<image name>.json
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x ".venv/bin/python" ]; then
  echo "The environment is not set up yet. Run ./linux/install.sh first." >&2
  exit 1
fi
if [ $# -lt 1 ]; then
  echo "Usage: ./linux/run_analysis.sh IMAGE [x20|x50|x100]" >&2
  exit 1
fi

IMG="$1"
MAG="${2:-x20}"
NAME="$(basename "${IMG%.*}")"
mkdir -p outputs
.venv/bin/python main.py "$IMG" --mag "$MAG" --json "outputs/${NAME}.json"
echo "Report saved to outputs/${NAME}.json"
