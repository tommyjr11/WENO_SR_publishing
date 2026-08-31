#!/usr/bin/env bash
# V20: V19 recipe with equal-probability propagation distances at CFL 0.5.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
DETACH="${DETACH:-1}"
OUT="${OUT:-teacherfree_lab_weno5_v20_distance_balanced/runs/apost_weno5_v20_distance_balanced_cfl05_200k}"
LATEST="$OUT/training_state/latest.pt"

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print("python=", sys.executable)
print("torch=", torch.__version__, "cuda_build=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available(), "device_count=", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA PyTorch is required")
PY

mkdir -p "$OUT"
RESUME_ARGS=()
if [[ -f "$LATEST" ]]; then
  RESUME_ARGS=(--resume "$LATEST")
  echo "auto-resume: $LATEST"
elif [[ -s "$OUT/history.csv" ]]; then
  echo "ERROR: $OUT has history but no resumable state; choose a new OUT." >&2
  exit 2
fi

COMMAND=(
  "$PYTHON_BIN" -u -m teacherfree_lab_weno5_v20_distance_balanced.train_weno5_v20
  --steps 200000 --grid 96
  --distances 2 8 64 128 256 512 1024
  --distance-batches 32 32 16 8 4 2 2
  --profile-probs 0.25 0.15 0.15 0.15 0.10 0.10 0.10
  --primary-cfl 0.5 --edge-cfls 0.5
  --edge-max-steps 40 --edge-lambda 0.25
  --lr 1e-4 --lr-final 2e-5
  --face-path-lambda 0.04 --exact-recon-lambda 0.15
  --flat-d2-lambda 0.05 --flat-tolerance 2e-3 --tv-lambda 0.03
  --global-guard-lambda 1.0 --local-guard-lambda 1.0
  --guard-tolerance 0 --local-window 8 --cvar-fraction 0.25
  --grad-clip 1.0 --grad-skip 10.0
  --log-interval 20 --checkpoint-interval 250 --eval-interval 250
  --state-interval 2500
  --out-dir "$OUT"
  "${RESUME_ARGS[@]}"
)

launch_monitor () {
  local old_pid=""
  [[ -f "$OUT/sod_monitor.pid" ]] && old_pid="$(cat "$OUT/sod_monitor.pid")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Sod monitor already running pid=$old_pid"
    return
  fi
  nohup setsid env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u \
    -m teacherfree_lab_weno5_v12_reflection_sym.sod_checkpoint_monitor_v12 \
    --run-dir "$OUT" --interval 250 --max-step 200000 --poll-seconds 10 \
    --nx 100 --ny 8 --cfl 0.4 --t-end 0.25 --axis x --device cuda \
    </dev/null >> "$OUT/sod_monitor.log" 2>&1 &
  echo "$!" > "$OUT/sod_monitor.pid"
  echo "launched trusted symmetric Warp Sod monitor pid=$!"
}

if [[ "$DETACH" == "1" ]]; then
  nohup setsid env CUDA_VISIBLE_DEVICES="$GPU_ID" "${COMMAND[@]}" \
    </dev/null >> "$OUT/nohup.out" 2>&1 &
  PID="$!"
  echo "$PID" > "$OUT/pid.txt"
  echo "launched WENO5 V20 distance-balanced training on GPU $GPU_ID pid=$PID"
  launch_monitor
else
  echo "$$" > "$OUT/pid.txt"
  exec env CUDA_VISIBLE_DEVICES="$GPU_ID" "${COMMAND[@]}" >> "$OUT/nohup.out" 2>&1
fi

echo "log: $OUT/nohup.out"
echo "Sod: $OUT/sod_monitor.log"
