#!/usr/bin/env bash
# WENO5 v3: keep the proven v2a gate and sweep only always-on TV strength.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODULE_NAME="${MODULE_NAME:-miniconda3/24.3.0-quc3pyu}"
CONDA_ENV="${CONDA_ENV:-base}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
SOD_DEVICE="${SOD_DEVICE:-}"
WAIT_FOR_JOBS="${WAIT_FOR_JOBS:-0}"
PIDS=()

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
import teacherfree_lab_weno5.train_apost_weno5
import teacherfree_lab_weno5.warp_sod_validation

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

launch () {  # $1=gpu $2=tag $3=tv_bg
  local OUT="teacherfree_lab_weno5/runs/$2"
  mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES="$1" nohup "$PYTHON_BIN" -u -m teacherfree_lab_weno5.train_apost_weno5 \
    --steps 200000 --batch 32 --grid 96 --cfl 0.2 \
    --horizons 20 40 80 120 \
    --lr 3e-4 --lr-final 1e-5 \
    --err-power 4 \
    --tv-lambda 0 --tv-bg-lambda "$3" \
    --smooth-anchor-lambda 1.0 --anchor-floor 1e-3 \
    --bound-lambda 3.0 --bound-floor 2e-4 --bound-tol 1e-3 \
    --ampgate-lambda 1.0 --ampgate-amp-min 1e-7 --ampgate-amp-max 3e-3 --ampgate-floor 1e-3 \
    --grad-clip 1.0 --grad-skip 10.0 \
    --checkpoint-interval 200 --eval-interval 200 \
    --sod-eval --sod-nx 100 --sod-ny 10 --sod-t-end 0.25 --sod-cfl 0.4 \
    --sod-axis x --sod-eno-cutoff --sod-weno-space characteristic --sod-riemann-solver evilin \
    "${SOD_DEVICE_ARGS[@]}" \
    --out-dir "$OUT" > "$OUT/nohup.out" 2>&1 &
  local PID="$!"
  PIDS+=("$PID")
  echo "$PID" > "$OUT/pid.txt"
  echo "launched $2 on GPU $1 tv_bg=$3 pid=$PID"
  sleep 2
}

launch "$GPU_A" apost_weno5_v3_tv001_gate3e3_200k 0.01
launch "$GPU_B" apost_weno5_v3_tv002_gate3e3_200k 0.02

echo
echo "monitor:"
echo "  grep EVAL teacherfree_lab_weno5/runs/apost_weno5_v3_*/nohup.out | tail -20"
echo "  tail -f teacherfree_lab_weno5/runs/apost_weno5_v3_tv001_gate3e3_200k/nohup.out"

if [[ "$WAIT_FOR_JOBS" == "1" ]]; then
  terminate_children() {
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  }
  trap terminate_children INT TERM

  failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      echo "training process failed: pid=$pid" >&2
      failed=1
    fi
  done
  exit "$failed"
fi
