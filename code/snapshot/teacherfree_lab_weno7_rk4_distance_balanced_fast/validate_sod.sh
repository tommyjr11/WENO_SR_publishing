#!/usr/bin/env bash
# Usage:
#   bash validate_sod.sh RUN_DIR "3500 4000 6250"
set -euo pipefail

cd "$(dirname "$0")"

if [[ "$#" -lt 1 ]]; then
  echo "usage: bash validate_sod.sh RUN_DIR [\"STEP STEP ...\"]" >&2
  exit 2
fi

RUN_DIR="$1"
STEP_TEXT="${2:-250 500 1000 1750 2500 3500 4000}"
read -r -a STEPS <<< "$STEP_TEXT"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODULE_NAME="${MODULE_NAME-miniconda3/24.3.0-quc3pyu}"
CONDA_ENV="${CONDA_ENV:-base}"
SOLVER="${SOLVER:-hllc}"
NX="${NX:-100}"
NY="${NY:-10}"
T_END="${T_END:-0.25}"
CFL="${CFL:-0.4}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/sod_validation}"

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

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u validate_sod.py \
  --run-dir "$RUN_DIR" \
  --steps "${STEPS[@]}" \
  --nx "$NX" --ny "$NY" \
  --t-end "$T_END" --cfl "$CFL" \
  --solver "$SOLVER" \
  --device cuda \
  --out-dir "$OUT_DIR"

echo "Sod figures and metrics: $OUT_DIR"
