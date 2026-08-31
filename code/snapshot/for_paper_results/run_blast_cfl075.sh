#!/usr/bin/env bash
# Test 3 at CFL 0.75, using the exact Riemann solution for plotting/errors.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
NX=200
NY=10
CFL=0.75
BENCHMARK=woodward_colella_half
ROOT="for_paper_results/raw/shock_tubes_cfl075/$BENCHMARK/N${NX}x${NY}"
FIG_ROOT="for_paper_results/figures/shock_tubes_cfl075/$BENCHMARK"
TABLE_ROOT="for_paper_results/tables/shock_tubes_cfl075/$BENCHMARK"
LOG_ROOT="for_paper_results/logs/shock_tubes_cfl075"
METHODS=(weno5_js weno5_sr_f64 weno5_sr_f32 weno7_js weno7_sr_f64)

mkdir -p "$ROOT" "$LOG_ROOT"
for method in "${METHODS[@]}"; do
  if ! "$PYTHON_BIN" -u -m for_paper_results.run_shock_tube_benchmark \
    --benchmark "$BENCHMARK" \
    --method "$method" \
    --nx "$NX" --ny "$NY" --cfl "$CFL" \
    --device "$DEVICE" --report-interval 100 \
    --out-dir "$ROOT" \
    > "$LOG_ROOT/${BENCHMARK}_${method}_N${NX}.log" 2>&1; then
    echo "FAILED benchmark=$BENCHMARK method=$method; see log" | tee \
      "$LOG_ROOT/${BENCHMARK}_${method}_N${NX}.failed"
  fi
done

"$PYTHON_BIN" -u -m for_paper_results.make_shock_tube_figures \
  --benchmark "$BENCHMARK" \
  --main-dir "$ROOT" \
  --figure-dir "$FIG_ROOT" \
  --table-dir "$TABLE_ROOT" \
  --allow-missing \
  | tee "$LOG_ROOT/${BENCHMARK}_figures.log"

echo "completed Test 3 at CFL $CFL"
echo "figure: $FIG_ROOT"
echo "table:  $TABLE_ROOT"
