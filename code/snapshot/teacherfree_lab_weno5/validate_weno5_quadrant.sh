#!/usr/bin/env bash
# Validate one WENO5 teacher-free checkpoint with the trusted existing 2D
# WENO5 Warp-RK3 runner. Defaults reproduce the old validated run:
#   plots/WENO5_MLP/weno5_quadrant_400_t05_5_10_6_6_3_step137000_mlp_only_evilin_cuda
# First CUDA run may spend ~10 minutes compiling Warp kernels.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODULE_NAME="${MODULE_NAME:-}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -n "$MODULE_NAME" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
    [[ -f /usr/share/Modules/init/bash ]] && source /usr/share/Modules/init/bash
  fi
  module load "$MODULE_NAME"
fi

if [[ -n "$CONDA_ENV" ]]; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

MODEL="${1:-${MODEL:-}}"
if [[ -z "$MODEL" ]]; then
  echo "usage: bash teacherfree_lab_weno5/validate_weno5_quadrant.sh path/to/model_step_XXXXXX.npz" >&2
  exit 2
fi

NX="${NX:-400}"
NY="${NY:-400}"
CFL="${CFL:-0.4}"
T_END="${T_END:-0.5}"
CASE="${CASE:-case12}"
SOLVER="${SOLVER:-evilin}"
GPU_ID="${GPU_ID:-0}"
TAG="$(basename "$(dirname "$(dirname "$MODEL")")")_$(basename "$MODEL" .npz)"
OUT_ROOT="${OUT_ROOT:-plots/WENO5_MLP/teacherfree_weno5_v1}"
T_TAG="$(printf '%g' "$T_END" | tr -d '.')"
CFL_TAG="$(printf '%g' "$CFL" | tr -d '.')"
OUT="$OUT_ROOT/${TAG}_${CASE}_${NX}_t${T_TAG}_${SOLVER}_cfl${CFL_TAG}"

mkdir -p "$OUT"
echo "trusted_warp_rk3_runner=run_weno5_quadrant_mlp_only.py"
echo "model=$MODEL"
echo "out=$OUT"
echo "note=first CUDA run may compile Warp kernels for several minutes"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u run_weno5_quadrant_mlp_only.py \
  --model "$MODEL" \
  --nx "$NX" --ny "$NY" \
  --cfl "$CFL" --t-end "$T_END" \
  --quadrant-case "$CASE" \
  --riemann-solver "$SOLVER" \
  --weno-space characteristic \
  --no-eno-cutoff \
  --device cuda \
  --report-interval 50 \
  --out-dir "$OUT" | tee "$OUT/run.log"
