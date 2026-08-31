#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd "$HERE/../.." && pwd)"
SNAPSHOT="$REPOSITORY/code/snapshot"
DEVICE="${DEVICE:-cuda}"
cd "$SNAPSHOT"
export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
RUN_ROOT="$HERE/runs"
mkdir -p "$RUN_ROOT"

redirect_output() {
  local source="$1"
  local target="$2"
  mkdir -p "$(dirname "$source")" "$target"
  if [[ -e "$source" && ! -L "$source" ]]; then
    printf "refusing to replace existing generated directory: %s\n" "$source" >&2
    exit 2
  fi
  ln -sfn "$target" "$source"
}

redirect_output "$SNAPSHOT/for_paper_results/raw" "$RUN_ROOT/for_paper_results/raw"
redirect_output "$SNAPSHOT/for_paper_results/figures" "$RUN_ROOT/for_paper_results/figures"
redirect_output "$SNAPSHOT/for_paper_results/tables" "$RUN_ROOT/for_paper_results/tables"
redirect_output "$SNAPSHOT/weno_z_borges_p2_results/raw" "$RUN_ROOT/weno_z_borges_p2_results/raw"
redirect_output "$SNAPSHOT/weno_z_borges_p2_results/figures" "$RUN_ROOT/weno_z_borges_p2_results/figures"
redirect_output "$SNAPSHOT/weno_z_borges_p2_results/tables" "$RUN_ROOT/weno_z_borges_p2_results/tables"
redirect_output "$SNAPSHOT/shockbubble_t0006_cfl04_server/results" "$RUN_ROOT/shockbubble_t0006_cfl04_server/results"
redirect_output "$SNAPSHOT/shockbubble_ma3_t0001_cfl04_server/results" "$RUN_ROOT/shockbubble_ma3_t0001_cfl04_server/results"
redirect_output "$SNAPSHOT/plots" "$RUN_ROOT/plots"
redirect_output "$SNAPSHOT/runs" "$RUN_ROOT/solver_runs"

python3 -u run_double_mach_compare.py --model teacherfree_lab_weno5_v20_distance_balanced/runs/apost_weno5_v20_distance_balanced_cfl05_200k/checkpoints/model_step_012250.npz --nx 1200 --ny 300 --cfl 0.4 --t-end 0.2 --init-quadrature 15 --weno-space characteristic --riemann-solver hllc --no-eno-cutoff --run-weno5 --no-run-weno7 --out-dir plots/WENO5_MLP/weno_double_reflective_1200 --device $DEVICE
python3 -u -m for_paper_results.run_double_mach --nx 1200 --ny 300 --cfl 0.4 --t-end 0.2 --init-quadrature 15 --methods weno5_sr_f64,weno5_sr_f32,weno7_js,weno7_sr_f64 --device "$DEVICE"
python3 -u -m weno_z_borges_p2_results.run_double_mach --methods weno5_z_p2,weno7_z_p3 --nx 1200 --ny 300 --cfl 0.4 --t-end 0.2 --init-quadrature 15 --device "$DEVICE"
