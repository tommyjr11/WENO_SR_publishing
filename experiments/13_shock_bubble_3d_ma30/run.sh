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

python3 -u -m warp_weno5_3d_rk3.run_shockbubble_ma3 --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device "$DEVICE" --out-dir runs/weno5_js
python3 -u -m warp_weno5_3d_rk3.run_shockbubble_ma3_mlp --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device "$DEVICE" --out-dir runs/weno5_sr_f64
python3 -u -m warp_weno5_3d_rk3.run_shockbubble_ma3_mlp_f32 --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device "$DEVICE" --out-dir runs/weno5_sr_f32
python3 -u -m warp_weno7_3d_rk4.run_shockbubble_ma3 --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device "$DEVICE" --out-dir runs/weno7_js
python3 -u -m warp_weno7_3d_rk4.run_shockbubble_ma3 --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device "$DEVICE" --model teacherfree_lab_weno7_rk4_distance_balanced_fast/runs/apost_weno7_rk4_distance_balanced_fast_4090_200k/checkpoints/model_step_016750.npz --out-dir runs/weno7_sr_f64
python3 -u -m weno_z_borges_p2_results.run_shockbubble_3d --method weno5_z_p2 --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device "$DEVICE" --out-dir runs/weno5_z
python3 -u -m weno_z_borges_p2_results.run_shockbubble_3d --method weno7_z_p3 --nx 224 --ny 88 --nz 88 --cfl 0.25 --t-end 0.0001 --device "$DEVICE" --out-dir runs/weno7_z
