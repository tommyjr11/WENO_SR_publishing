#!/usr/bin/env bash
# Formal Test 2 (Lax) and Test 3 (left Woodward--Colella blast wave).
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
MAIN_NX=200
NY=10
REFERENCE_NX=4000
CFL=0.8
ROOT="for_paper_results/raw/shock_tubes_cfl08"
FIG_ROOT="for_paper_results/figures/shock_tubes_cfl08"
TABLE_ROOT="for_paper_results/tables/shock_tubes_cfl08"
LOG_ROOT="for_paper_results/logs/shock_tubes_cfl08"
METHODS=(weno5_js weno5_sr_f64 weno5_sr_f32 weno7_js weno7_sr_f64)
BENCHMARKS=(lax woodward_colella_half)

mkdir -p "$LOG_ROOT"

for benchmark in "${BENCHMARKS[@]}"; do
  main_dir="$ROOT/$benchmark/N${MAIN_NX}x${NY}"
  ref_dir="$ROOT/$benchmark/reference_N${REFERENCE_NX}x${NY}"
  mkdir -p "$main_dir" "$ref_dir"

  for method in "${METHODS[@]}"; do
    if ! "$PYTHON_BIN" -u -m for_paper_results.run_shock_tube_benchmark \
      --benchmark "$benchmark" \
      --method "$method" \
      --nx "$MAIN_NX" --ny "$NY" --cfl "$CFL" \
      --device "$DEVICE" --report-interval 100 \
      --out-dir "$main_dir" \
      > "$LOG_ROOT/${benchmark}_${method}_N${MAIN_NX}.log" 2>&1; then
      echo "FAILED benchmark=$benchmark method=$method; see log" | tee \
        "$LOG_ROOT/${benchmark}_${method}_N${MAIN_NX}.failed"
    fi
  done

  "$PYTHON_BIN" -u -m for_paper_results.run_shock_tube_benchmark \
    --benchmark "$benchmark" \
    --method weno7_js \
    --nx "$REFERENCE_NX" --ny "$NY" --cfl "$CFL" \
    --device "$DEVICE" --report-interval 500 \
    --out-dir "$ref_dir" \
    > "$LOG_ROOT/${benchmark}_weno7_js_N${REFERENCE_NX}.log" 2>&1

  "$PYTHON_BIN" -u -m for_paper_results.make_shock_tube_figures \
    --benchmark "$benchmark" \
    --main-dir "$main_dir" \
    --reference "$ref_dir/weno7_js.npz" \
    --figure-dir "$FIG_ROOT/$benchmark" \
    --table-dir "$TABLE_ROOT/$benchmark" \
    --allow-missing \
    | tee "$LOG_ROOT/${benchmark}_figures.log"
done

echo "completed Lax and Woodward--Colella half blast-wave tests"
echo "figures: $FIG_ROOT"
echo "tables:  $TABLE_ROOT"
