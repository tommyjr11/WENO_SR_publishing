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

MAIN_DIR=shockbubble_ma3_t0001_cfl04_server/results/raw/shockbubble_ma3_t0001_cfl04/N1000x396
REFERENCE_DIR=shockbubble_ma3_t0001_cfl04_server/results/raw/shockbubble_ma3_t0001_cfl04/reference_weno7_N2000x791
mkdir -p "$MAIN_DIR" "$REFERENCE_DIR"
for method in weno5_js weno5_sr_f64 weno5_sr_f32 weno7_js weno7_sr_f64; do python3 -u shockbubble_ma3_t0001_cfl04_server/for_paper_results/run_weno5_shockbubble.py --method "$method" --nx 1000 --ny 396 --cfl 0.4 --t-end 0.0001 --report-interval 200 --out-dir "$MAIN_DIR" --device "$DEVICE"; done
python3 -u shockbubble_ma3_t0001_cfl04_server/for_paper_results/run_weno5_shockbubble.py --method weno7_js --nx 2000 --ny 791 --cfl 0.4 --t-end 0.0001 --report-interval 200 --out-dir "$REFERENCE_DIR" --device "$DEVICE"
python3 -u -m weno_z_borges_p2_results.run_shockbubble_2d --case ma30 --method weno5_z_p2 --nx 1000 --ny 396 --cfl 0.4 --device "$DEVICE"
python3 -u -m weno_z_borges_p2_results.run_shockbubble_2d --case ma30 --method weno7_z_p3 --nx 1000 --ny 396 --cfl 0.4 --device "$DEVICE"
