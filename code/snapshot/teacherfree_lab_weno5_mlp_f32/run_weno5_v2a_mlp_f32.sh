#!/usr/bin/env bash
# WENO5 teacher-free v2a, mixed precision:
#   state/flux/RK3/Riemann/loss/reference stay float64;
#   only MLP parameters and MLP forward path are float32.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODULE_NAME="${MODULE_NAME:-}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_A="${GPU_A:-0}"
SOD_NX="${SOD_NX:-100}"
SOD_NY="${SOD_NY:-10}"
SOD_T_END="${SOD_T_END:-0.25}"
SOD_CFL="${SOD_CFL:-0.4}"
SOD_DEVICE="${SOD_DEVICE:-}"

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

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print("python=", sys.executable)
print("torch=", torch.__version__, "cuda_build=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available(), "device_count=", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda.is_available() is false; load the CUDA torch environment first.")
PY

SOD_DEVICE_ARGS=()
if [[ -n "$SOD_DEVICE" ]]; then
  SOD_DEVICE_ARGS=(--sod-device "$SOD_DEVICE")
fi

OUT="teacherfree_lab_weno5_mlp_f32/runs/apost_weno5_v2a_mlp_f32_gate3e3_200k"
mkdir -p "$OUT"

nohup setsid env CUDA_VISIBLE_DEVICES="$GPU_A" "$PYTHON_BIN" -u -m teacherfree_lab_weno5_mlp_f32.train_apost_weno5 \
  --steps 200000 --batch 32 --grid 96 --cfl 0.2 \
  --horizons 20 40 80 120 \
  --lr 3e-4 --lr-final 1e-5 \
  --err-power 4 \
  --tv-lambda 0 --tv-bg-lambda 0.03 \
  --smooth-anchor-lambda 1.0 --anchor-floor 1e-3 \
  --bound-lambda 3.0 --bound-floor 2e-4 --bound-tol 1e-3 \
  --ampgate-lambda 1.0 --ampgate-amp-min 1e-7 --ampgate-amp-max 3e-3 --ampgate-floor 1e-3 \
  --grad-clip 1.0 --grad-skip 10.0 \
  --checkpoint-interval 200 --eval-interval 200 \
  --sod-eval --sod-nx "$SOD_NX" --sod-ny "$SOD_NY" --sod-t-end "$SOD_T_END" --sod-cfl "$SOD_CFL" \
  --sod-axis x --sod-eno-cutoff --sod-weno-space characteristic --sod-riemann-solver evilin \
  "${SOD_DEVICE_ARGS[@]}" \
  --out-dir "$OUT" > "$OUT/nohup.out" 2>&1 < /dev/null &

echo $! > "$OUT/pid.txt"
echo "launched apost_weno5_v2a_mlp_f32_gate3e3_200k on GPU $GPU_A pid=$(cat "$OUT/pid.txt")"
echo
echo "monitor:  grep EVAL $OUT/nohup.out | tail"
