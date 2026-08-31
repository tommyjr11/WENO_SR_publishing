#!/usr/bin/env bash
# WENO5 mixed precision v3: v2a recipe with only tv_bg reduced to 0.01.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
SOD_DEVICE="${SOD_DEVICE:-}"
TV_BG="${TV_BG:-0.01}"
RUN_NAME="${RUN_NAME:-apost_weno5_v3_tv001_mlp_f32_gate3e3_200k}"
OUT="teacherfree_lab_weno5_mlp_f32/runs/$RUN_NAME"

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print("python=", sys.executable)
print("torch=", torch.__version__, "cuda_build=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available(), "device_count=", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda.is_available() is false")
PY

SOD_DEVICE_ARGS=()
if [[ -n "$SOD_DEVICE" ]]; then
  SOD_DEVICE_ARGS=(--sod-device "$SOD_DEVICE")
fi

mkdir -p "$OUT"
nohup setsid env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u \
  -m teacherfree_lab_weno5_mlp_f32.train_apost_weno5 \
  --steps 200000 --batch 32 --grid 96 --cfl 0.2 \
  --horizons 20 40 80 120 \
  --lr 3e-4 --lr-final 1e-5 \
  --err-power 4 \
  --tv-lambda 0 --tv-bg-lambda "$TV_BG" \
  --smooth-anchor-lambda 1.0 --anchor-floor 1e-3 \
  --bound-lambda 3.0 --bound-floor 2e-4 --bound-tol 1e-3 \
  --ampgate-lambda 1.0 --ampgate-amp-min 1e-7 --ampgate-amp-max 3e-3 --ampgate-floor 1e-3 \
  --grad-clip 1.0 --grad-skip 10.0 \
  --checkpoint-interval 200 --eval-interval 200 \
  --sod-eval --sod-nx 100 --sod-ny 10 --sod-t-end 0.25 --sod-cfl 0.4 \
  --sod-axis x --sod-eno-cutoff --sod-weno-space characteristic --sod-riemann-solver evilin \
  "${SOD_DEVICE_ARGS[@]}" \
  --out-dir "$OUT" > "$OUT/nohup.out" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "launched $RUN_NAME on GPU $GPU_ID tv_bg=$TV_BG pid=$!"
echo "monitor: grep EVAL $OUT/nohup.out | tail -20"
