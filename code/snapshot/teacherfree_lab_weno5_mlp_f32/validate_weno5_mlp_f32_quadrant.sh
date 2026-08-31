#!/usr/bin/env bash
# Validate one mixed WENO5 checkpoint with the isolated mixed Warp-RK3 copy.
# Defaults are the case6 screen the current branch is meant to pass.
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
  echo "usage: bash teacherfree_lab_weno5_mlp_f32/validate_weno5_mlp_f32_quadrant.sh path/to/model_step_XXXXXX.npz" >&2
  exit 2
fi

NX="${NX:-400}"
NY="${NY:-400}"
CFL="${CFL:-0.4}"
T_END="${T_END:-0.25}"
CASE="${CASE:-case6}"
SOLVER="${SOLVER:-evilin}"
GPU_ID="${GPU_ID:-0}"
TAG="$(basename "$(dirname "$(dirname "$MODEL")")")_$(basename "$MODEL" .npz)"
OUT_ROOT="${OUT_ROOT:-plots/WENO5_MLP/teacherfree_weno5_mlp_f32}"
T_TAG="$(printf '%g' "$T_END" | tr -d '.')"
CFL_TAG="$(printf '%g' "$CFL" | tr -d '.')"
OUT="$OUT_ROOT/${TAG}_${CASE}_${NX}_t${T_TAG}_${SOLVER}_cfl${CFL_TAG}"

mkdir -p "$OUT"
echo "mixed_warp_rk3_runner=teacherfree_lab_weno5_mlp_f32.warp_mlp_f32.run_weno5_quadrant_mlp_only_mlp_f32"
echo "precision=mlp_float32_state_float64"
echo "model=$MODEL"
echo "out=$OUT"
echo "note=first CUDA run may compile Warp kernels for several minutes"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u -m teacherfree_lab_weno5_mlp_f32.warp_mlp_f32.run_weno5_quadrant_mlp_only_mlp_f32 \
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
