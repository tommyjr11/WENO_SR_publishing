#!/usr/bin/env bash
# Launch the isolated, execution-optimized WENO5 V20 mixed-FP32 run.
set -euo pipefail

cd "$(dirname "$0")"

MODULE_NAME="${MODULE_NAME:-}"
CONDA_ENV="${CONDA_ENV:-}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-runs/apost_weno5_v20_mlp_f32_fast_a800_200k}"
CHUNK_STEPS="${CHUNK_STEPS:-16}"
TARGET_CHUNK="${TARGET_CHUNK:-128}"
COMPILE_MODE="${COMPILE_MODE:-jit}"

if [[ -n "$MODULE_NAME" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
    [[ -f /usr/share/Modules/init/bash ]] && source /usr/share/Modules/init/bash
  fi
  if command -v module >/dev/null 2>&1; then
    module load "$MODULE_NAME"
  else
    echo "warning: environment modules are unavailable; using current PATH" >&2
  fi
fi

if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print("python=", sys.executable)
print("torch=", torch.__version__, "cuda_build=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available(), "device_count=", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this Python environment")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
print("tf32_matmul=", torch.backends.cuda.matmul.allow_tf32)
print("tf32_cudnn=", torch.backends.cudnn.allow_tf32)
PY

if [[ -f "$OUT_DIR/pid.txt" ]]; then
  old_pid="$(tr -d '[:space:]' < "$OUT_DIR/pid.txt")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "training is already running: pid=$old_pid"
    if [[ "${START_SOD_MONITOR:-1}" == "1" ]]; then
      RUN_DIR="$OUT_DIR" GPU_ID="$GPU_ID" bash run_sod_monitor.sh
    fi
    exit 0
  fi
fi

resume_args=()
log_mode="overwrite"
resume_setting="${RESUME:-auto}"
if [[ "$resume_setting" == "auto" ]]; then
  if [[ -f "$OUT_DIR/training_state/latest.pt" ]]; then
    resume_args=(--resume "$OUT_DIR/training_state/latest.pt")
    log_mode="append"
  elif [[ -e "$OUT_DIR/history.csv" || -d "$OUT_DIR/checkpoints" ]]; then
    echo "auto-resume found output but no latest.pt: $OUT_DIR" >&2
    echo "use a new OUT_DIR or provide RESUME=/path/to/state.pt" >&2
    exit 2
  fi
elif [[ "$resume_setting" == "fresh" || -z "$resume_setting" ]]; then
  if [[ -e "$OUT_DIR/history.csv" \
        || -e "$OUT_DIR/training_state/latest.pt" \
        || -d "$OUT_DIR/checkpoints" ]]; then
    echo "refusing to overwrite an existing run in fresh mode: $OUT_DIR" >&2
    echo "choose a new OUT_DIR, or use RESUME=auto" >&2
    exit 2
  fi
elif [[ -n "$resume_setting" ]]; then
  if [[ ! -f "$resume_setting" ]]; then
    echo "resume state does not exist: $resume_setting" >&2
    exit 2
  fi
  resume_args=(--resume "$resume_setting")
  log_mode="append"
fi

if [[ "$log_mode" == "append" ]]; then
  {
    echo
    echo "===== resume $(date --iso-8601=seconds) ====="
  } >> "$OUT_DIR/nohup.out"
fi

command=(
  "$PYTHON_BIN" -u train.py
  --steps 200000
  --grid 96
  --distances 2 8 64 128 256 512 1024
  --distance-batches 32 32 16 8 4 2 2
  --profile-probs 0.25 0.15 0.15 0.15 0.10 0.10 0.10
  --primary-cfl 0.5 --edge-cfls 0.5
  --edge-max-steps 40 --edge-lambda 0.25
  --lr 1e-4 --lr-final 2e-5
  --face-path-lambda 0.04
  --exact-recon-lambda 0.15
  --flat-d2-lambda 0.05 --flat-tolerance 2e-3
  --tv-lambda 0.03
  --global-guard-lambda 1.0
  --local-guard-lambda 1.0
  --guard-tolerance 0
  --local-window 8 --cvar-fraction 0.25
  --grad-clip 1.0 --grad-skip 10.0
  --chunk-steps "$CHUNK_STEPS"
  --target-chunk "$TARGET_CHUNK"
  --compile-mode "$COMPILE_MODE"
  --log-interval 20
  --checkpoint-interval 250
  --eval-interval 250
  --state-interval 2500
  --out-dir "$OUT_DIR"
  "${resume_args[@]}"
)

if [[ "$log_mode" == "append" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" nohup "${command[@]}" \
    >> "$OUT_DIR/nohup.out" 2>&1 &
else
  CUDA_VISIBLE_DEVICES="$GPU_ID" nohup "${command[@]}" \
    > "$OUT_DIR/nohup.out" 2>&1 &
fi

pid=$!
printf '%s\n' "$pid" > "$OUT_DIR/pid.txt"
echo "launched WENO5 V20 MLP-FP32 fast training on GPU $GPU_ID pid=$pid"
echo "output: $OUT_DIR"
echo "monitor: tail -f $OUT_DIR/nohup.out"

sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "training exited during startup; last log lines:" >&2
  tail -80 "$OUT_DIR/nohup.out" >&2 || true
  exit 1
fi

monitor_pid=""
if [[ "${START_SOD_MONITOR:-1}" == "1" ]]; then
  RUN_DIR="$OUT_DIR" GPU_ID="$GPU_ID" bash run_sod_monitor.sh
  monitor_pid="$(tr -d '[:space:]' < "$OUT_DIR/sod_monitor.pid")"
fi

if [[ "${WAIT_FOR_TRAIN:-0}" == "1" ]]; then
  wait "$pid"
  if [[ -n "$monitor_pid" ]]; then
    while kill -0 "$monitor_pid" 2>/dev/null; do
      sleep 10
    done
  fi
fi
