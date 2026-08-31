#!/usr/bin/env bash
# Launch the trusted characteristic WENO7/Shu-RK4 Sod checkpoint monitor.
set -euo pipefail

cd "$(dirname "$0")"

MODULE_NAME="${MODULE_NAME-miniconda3/24.3.0-quc3pyu}"
CONDA_ENV="${CONDA_ENV:-base}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DIR="${RUN_DIR:-runs/apost_weno7_rk4_distance_balanced_fast_200k}"
INTERVAL="${INTERVAL:-250}"
MAX_STEP="${MAX_STEP:-200000}"
POLL_SECONDS="${POLL_SECONDS:-10}"
NX="${NX:-100}"
NY="${NY:-10}"
CFL="${CFL:-0.4}"
T_END="${T_END:-0.25}"
SOLVER="${SOLVER:-hllc}"

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

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

mkdir -p "$RUN_DIR"
if [[ -f "$RUN_DIR/sod_monitor.pid" ]]; then
  old_pid="$(tr -d '[:space:]' < "$RUN_DIR/sod_monitor.pid")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Sod monitor is already running: pid=$old_pid"
    exit 0
  fi
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" nohup "$PYTHON_BIN" -u sod_checkpoint_monitor.py \
  --run-dir "$RUN_DIR" \
  --interval "$INTERVAL" \
  --max-step "$MAX_STEP" \
  --poll-seconds "$POLL_SECONDS" \
  --nx "$NX" --ny "$NY" \
  --cfl "$CFL" --t-end "$T_END" \
  --solver "$SOLVER" \
  --device cuda \
  >> "$RUN_DIR/sod_monitor.log" 2>&1 &

pid=$!
printf '%s\n' "$pid" > "$RUN_DIR/sod_monitor.pid"
echo "launched trusted WENO7 Sod monitor on GPU $GPU_ID pid=$pid"
echo "monitor: tail -f $RUN_DIR/sod_monitor.log"

if [[ "${WAIT_FOR_MONITOR:-0}" == "1" ]]; then
  wait "$pid"
fi
