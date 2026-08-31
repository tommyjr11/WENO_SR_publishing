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

python3 -u -m weno_z_borges_p2_results.plot_gste
