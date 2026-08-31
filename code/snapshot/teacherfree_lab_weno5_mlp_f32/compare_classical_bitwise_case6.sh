#!/usr/bin/env bash
# Full acceptance check: trusted double classical path must match the mixed
# classical branch bitwise. Defaults are case6 400x400 t=0.25 CFL=0.4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODULE_NAME="${MODULE_NAME:-}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"

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

NX="${NX:-400}"
NY="${NY:-400}"
CFL="${CFL:-0.4}"
T_END="${T_END:-0.25}"
CASE="${CASE:-case6}"
OUT_DIR="${OUT_DIR:-plots/WENO5_MLP/teacherfree_weno5_mlp_f32/bitwise_classical_${CASE}_${NX}}"

echo "bitwise classical check: case=$CASE ${NX}x${NY} t=$T_END cfl=$CFL solver=evilin weno=characteristic"
echo "note=first CUDA run may compile Warp kernels for several minutes"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u teacherfree_lab_weno5_mlp_f32/compare_classical_bitwise_case6.py \
  --nx "$NX" --ny "$NY" \
  --cfl "$CFL" --t-end "$T_END" \
  --case "$CASE" \
  --device cuda \
  --out-dir "$OUT_DIR"
