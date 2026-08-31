#!/usr/bin/env bash
# WENO5 v2: widen the amplitude gate to cover the MEASURED case6 sawtooth.
#
# Measurement (case6 @ v1 step 4000, density second differences on the weak
# lower-left arc, cuts y=0.25/0.30/0.35): oscillation relative amplitude
# = 1.4~1.5e-3, i.e. just OUTSIDE the v1 gate band (amp-max 1e-3). The gate
# must therefore be WIDENED (3e-3), not narrowed.
#
# Two decoupled variants on two GPUs:
#   GPU 0 v2a: amp-max 1e-3 -> 3e-3 only          [single-variable fix]
#   GPU 1 v2b: amp-max 3e-3 + floor 1e-4 + tv 0.038 [stronger polish combo]
# TV note: Sod gain already decays 34%->13% over 2k->12k under tv_bg=0.03
# (WENO5 is more dissipative than WENO7), so v2a keeps tv at 0.03 to
# preserve gain; only v2b tries the heavier setting.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODULE_NAME="${MODULE_NAME:-}"
CONDA_ENV="${CONDA_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
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

launch () {  # $1=gpu $2=tag $3=tv_bg $4=amp_max $5=gate_floor
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
    --ampgate-lambda 1.0 --ampgate-amp-min 1e-7 --ampgate-amp-max "$4" --ampgate-floor "$5" \
    --grad-clip 1.0 --grad-skip 10.0 \
    --checkpoint-interval 200 --eval-interval 200 \
    --sod-eval --sod-nx "$SOD_NX" --sod-ny "$SOD_NY" --sod-t-end "$SOD_T_END" --sod-cfl "$SOD_CFL" \
    --sod-axis x --sod-eno-cutoff --sod-weno-space characteristic --sod-riemann-solver evilin \
    "${SOD_DEVICE_ARGS[@]}" \
    --out-dir "$OUT" > "$OUT/nohup.out" 2>&1 &
  echo $! > "$OUT/pid.txt"
  echo "launched $2 on GPU $1 pid=$(cat "$OUT/pid.txt")"
  sleep 2
}

# VARIANT=a (default) | b | both.  Single GPU: bash run_weno5_v2.sh  (runs v2a on GPU_A)
VARIANT="${VARIANT:-a}"

if [[ "$VARIANT" == "a" || "$VARIANT" == "both" ]]; then
  launch "$GPU_A" apost_weno5_v2a_gate3e3_200k             0.03  3e-3 1e-3
fi
if [[ "$VARIANT" == "b" ]]; then
  launch "$GPU_A" apost_weno5_v2b_gate3e3_f1e4_tv0038_200k 0.038 3e-3 1e-4
elif [[ "$VARIANT" == "both" ]]; then
  launch "$GPU_B" apost_weno5_v2b_gate3e3_f1e4_tv0038_200k 0.038 3e-3 1e-4
fi

echo
echo "monitor:  grep EVAL teacherfree_lab_weno5/runs/apost_weno5_v2*/nohup.out | tail"
