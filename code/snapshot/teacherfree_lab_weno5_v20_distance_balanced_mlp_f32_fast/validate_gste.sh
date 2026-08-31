#!/usr/bin/env bash
# Usage: bash validate_gste.sh RUN_DIR "3500 4000 10000 12250"
set -euo pipefail

cd "$(dirname "$0")"

if [[ "$#" -lt 1 ]]; then
  echo "usage: bash validate_gste.sh RUN_DIR [\"STEP STEP ...\"]" >&2
  exit 2
fi

RUN_DIR="$1"
STEP_TEXT="${2:-2500 3500 4000 10000 12250}"
read -r -a STEPS <<< "$STEP_TEXT"
read -r -a CFLS <<< "${CFLS:-0.2 0.4 0.6 0.8}"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODULE_NAME="${MODULE_NAME:-}"
CONDA_ENV="${CONDA_ENV:-}"
NX="${NX:-200}"
T_END="${T_END:-10}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/gste_validation}"
mkdir -p "$OUT_DIR"

if [[ -n "$MODULE_NAME" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
    [[ -f /usr/share/Modules/init/bash ]] && source /usr/share/Modules/init/bash
  fi
  command -v module >/dev/null 2>&1 && module load "$MODULE_NAME"
fi
if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

for step in "${STEPS[@]}"; do
  step_number=$((10#$step))
  step_tag="$(printf '%06d' "$step_number")"
  checkpoint="$RUN_DIR/checkpoints/model_step_${step_tag}.npz"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 2
  fi
  step_dir="$OUT_DIR/step_${step_tag}"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u evaluate_gste.py \
    --model "$checkpoint" \
    --out-dir "$step_dir" \
    --nx "$NX" --t-end "$T_END" --quadrature 15 \
    --cfls "${CFLS[@]}" \
    --device cuda \
    | tee "$step_dir.log"
done

echo "GSTE figures and metrics: $OUT_DIR"
