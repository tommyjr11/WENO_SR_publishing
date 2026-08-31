#!/usr/bin/env bash
# Sequential five-method paper suite. Extended shock-bubble data are written
# to a new t0006 tree, leaving the completed t=2e-4 fields untouched.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

LOG_DIR="$ROOT/for_paper_results/logs"
STATUS_FILE="$LOG_DIR/shockbubble_t0006_titarev_all.status"
mkdir -p "$LOG_DIR"
: > "$STATUS_FILE"

SHOCK_NX="${SHOCK_NX:-1000}"
SHOCK_NY="${SHOCK_NY:-396}"
SHOCK_REF_NX="${SHOCK_REF_NX:-2000}"
SHOCK_REF_NY="${SHOCK_REF_NY:-791}"
SHOCK_T_END="${SHOCK_T_END:-0.0006}"
SHOCK_CFL="${SHOCK_CFL:-0.228}"
TITAREV_NX="${TITAREV_NX:-1001}"
TITAREV_NY="${TITAREV_NY:-8}"

SHOCK_MAIN="$ROOT/for_paper_results/raw/shockbubble_t0006/N${SHOCK_NX}x${SHOCK_NY}"
SHOCK_REF="$ROOT/for_paper_results/raw/shockbubble_t0006/reference_N${SHOCK_REF_NX}x${SHOCK_REF_NY}"
SHOCK_FIGURES="$ROOT/for_paper_results/figures/shockbubble_t0006"
SHOCK_TABLE="$ROOT/for_paper_results/tables/shockbubble_t0006_linecut_errors.csv"
TITAREV_MAIN="$ROOT/for_paper_results/raw/titarev_toro/N${TITAREV_NX}x${TITAREV_NY}"
mkdir -p "$SHOCK_MAIN" "$SHOCK_REF" "$SHOCK_FIGURES" "$TITAREV_MAIN"

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
  [[ -f "$npz" && -f "$metadata" ]] && grep -q '"complete": true' "$metadata"
}

run_if_incomplete() {
  local output="$1"
  local label="$2"
  shift 2
  if is_complete "$output"; then
    printf '[%s] SKIP  %s (complete result exists)\n' "$(date '+%F %T')" "$label" | tee -a "$STATUS_FILE"
  else
    run_phase "$label" "$@"
  fi
}

refresh_shock_figure() {
  run_phase \
    "refresh t=0.0006 shock-bubble figures" \
    "$PYTHON_BIN" -u -m for_paper_results.make_shockbubble_figures \
      --main-dir "$SHOCK_MAIN" --reference-dir "$SHOCK_REF" \
      --figure-dir "$SHOCK_FIGURES" --table-path "$SHOCK_TABLE"
}

for method in weno5_js weno5_sr_f64 weno5_sr_f32 weno7_js weno7_sr_f64; do
  run_if_incomplete \
    "$SHOCK_MAIN/$method.npz" \
    "shock-bubble t=0.0006 ${SHOCK_NX}x${SHOCK_NY} $method" \
    "$PYTHON_BIN" -u -m for_paper_results.run_weno5_shockbubble \
      --method "$method" --nx "$SHOCK_NX" --ny "$SHOCK_NY" \
      --cfl "$SHOCK_CFL" --t-end "$SHOCK_T_END" --device cuda \
      --report-interval 200 --out-dir "$SHOCK_MAIN"
  refresh_shock_figure
done

for method in weno5_js weno5_sr_f64 weno5_sr_f32 weno7_js weno7_sr_f64; do
  run_if_incomplete \
    "$TITAREV_MAIN/$method.npz" \
    "Titarev-Toro ${TITAREV_NX}x${TITAREV_NY} $method" \
    "$PYTHON_BIN" -u -m for_paper_results.run_weno5_titarev_toro \
      --method "$method" --nx "$TITAREV_NX" --ny "$TITAREV_NY" \
      --device cuda --report-interval 200 --out-dir "$TITAREV_MAIN"
  run_phase \
    "refresh Titarev-Toro figures" \
    "$PYTHON_BIN" -u -m for_paper_results.make_titarev_toro_figures \
      --main-dir "$TITAREV_MAIN"
done

run_if_incomplete \
  "$SHOCK_REF/weno5_js.npz" \
  "shock-bubble t=0.0006 reference ${SHOCK_REF_NX}x${SHOCK_REF_NY} weno5_js" \
  "$PYTHON_BIN" -u -m for_paper_results.run_weno5_shockbubble \
    --method weno5_js --nx "$SHOCK_REF_NX" --ny "$SHOCK_REF_NY" \
    --cfl "$SHOCK_CFL" --t-end "$SHOCK_T_END" --device cuda \
    --report-interval 200 --out-dir "$SHOCK_REF"

refresh_shock_figure
printf '[%s] FINISH failures=%s\n' "$(date '+%F %T')" "$failures" | tee -a "$STATUS_FILE"
exit "$failures"
