#!/usr/bin/env bash
# Restartable five-method Titarev--Toro suite at CFL=0.8 on a 1001x10 strip.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

NX=1001
NY=10
CFL=0.8
T_END=5.0
RAW_DIR="$ROOT/for_paper_results/raw/titarev_toro_cfl08/N${NX}x${NY}"
FIGURE_DIR="$ROOT/for_paper_results/figures/titarev_toro_cfl08"
TABLE_DIR="$ROOT/for_paper_results/tables/titarev_toro_cfl08"
LOG_DIR="$ROOT/for_paper_results/logs"
STATUS_FILE="$LOG_DIR/titarev_toro_cfl08_ny10.status"
mkdir -p "$RAW_DIR" "$FIGURE_DIR" "$TABLE_DIR" "$LOG_DIR"
: > "$STATUS_FILE"

failures=0

run_phase() {
  local label="$1"
  shift
  printf '[%s] START %s\n' "$(date '+%F %T')" "$label" | tee -a "$STATUS_FILE"
  if "$@"; then
    printf '[%s] DONE  %s\n' "$(date '+%F %T')" "$label" | tee -a "$STATUS_FILE"
  else
    local code=$?
    printf '[%s] FAIL  %s exit=%s\n' "$(date '+%F %T')" "$label" "$code" | tee -a "$STATUS_FILE"
    failures=$((failures + 1))
  fi
}

is_complete() {
  local npz="$1"
  local metadata="${npz%.npz}.json"
  [[ -f "$npz" && -f "$metadata" ]] \
    && grep -q '"complete": true' "$metadata" \
    && grep -q '"cfl": 0.8' "$metadata" \
    && grep -q '"ny": 10' "$metadata"
}

refresh_figures() {
  run_phase \
    "refresh Titarev--Toro CFL=0.8 figures" \
    "$PYTHON_BIN" -u -m for_paper_results.make_titarev_toro_figures \
      --main-dir "$RAW_DIR" --figure-dir "$FIGURE_DIR" \
      --table-dir "$TABLE_DIR"
}

for method in weno5_js weno5_sr_f64 weno5_sr_f32 weno7_js weno7_sr_f64; do
  output="$RAW_DIR/$method.npz"
  if is_complete "$output"; then
    printf '[%s] SKIP  %s (matching complete result exists)\n' \
      "$(date '+%F %T')" "$method" | tee -a "$STATUS_FILE"
  else
    run_phase \
      "Titarev--Toro ${NX}x${NY} CFL=0.8 $method" \
      "$PYTHON_BIN" -u -m for_paper_results.run_weno5_titarev_toro \
        --method "$method" --nx "$NX" --ny "$NY" --cfl "$CFL" \
        --t-end "$T_END" --device cuda --report-interval 200 \
        --out-dir "$RAW_DIR"
  fi
  refresh_figures
done

printf '[%s] FINISH failures=%s\n' "$(date '+%F %T')" "$failures" | tee -a "$STATUS_FILE"
exit "$failures"
