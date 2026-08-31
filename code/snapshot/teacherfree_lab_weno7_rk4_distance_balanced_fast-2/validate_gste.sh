#!/usr/bin/env bash
# Usage:
#   bash validate_gste.sh RUN_DIR "3500 4000 6250"
set -euo pipefail

cd "$(dirname "$0")"

if [[ "$#" -lt 1 ]]; then
  echo "usage: bash validate_gste.sh RUN_DIR [\"STEP STEP ...\"]" >&2
  exit 2
fi

RUN_DIR="$1"
STEP_TEXT="${2:-250 500 1000 1750 2500 3500 4000}"
read -r -a STEPS <<< "$STEP_TEXT"
read -r -a CFLS <<< "${CFLS:-0.2 0.4 0.6 0.8}"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODULE_NAME="${MODULE_NAME-miniconda3/24.3.0-quc3pyu}"
CONDA_ENV="${CONDA_ENV:-base}"
NX="${NX:-200}"
T_END="${T_END:-10}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/gste_validation}"

if [[ -n "$MODULE_NAME" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
    [[ -f /usr/share/Modules/init/bash ]] && source /usr/share/Modules/init/bash
  fi
  if command -v module >/dev/null 2>&1; then
    module load "$MODULE_NAME"
  fi
fi
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u evaluate_gste.py \
  --run-dir "$RUN_DIR" \
  --steps "${STEPS[@]}" \
  --nx "$NX" \
  --t-end "$T_END" \
  --cfls "${CFLS[@]}" \
  --quadrature 15 \
  --device cuda \
  --out-dir "$OUT_DIR"

echo "figures and ranking: $OUT_DIR"
