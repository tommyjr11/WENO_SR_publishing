#!/usr/bin/env bash
# Sequential overnight WENO5 paper suite. One GPU is used deliberately so the
# large shock-bubble jobs never compete for memory.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

LOG_DIR="$ROOT/for_paper_results/logs"
STATUS_FILE="$LOG_DIR/shockbubble_titarev_overnight.status"
mkdir -p "$LOG_DIR"
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

run_if_missing() {
  local output="$1"
  local label="$2"
  shift 2
  if [[ -f "$output" ]]; then
    printf '[%s] SKIP  %s (%s exists)\n' "$(date '+%F %T')" "$label" "$output" | tee -a "$STATUS_FILE"
  else
    run_phase "$label" "$@"
  fi
}

SHOCK_MAIN="$ROOT/for_paper_results/raw/shockbubble/N1000x396"
SHOCK_REF="$ROOT/for_paper_results/raw/shockbubble/reference_N2000x791"
TITAREV_MAIN="$ROOT/for_paper_results/raw/titarev_toro/N1001x8"
mkdir -p "$SHOCK_MAIN" "$SHOCK_REF" "$TITAREV_MAIN"

for method in weno5_js weno5_sr_f64 weno5_sr_f32; do
  run_if_missing \
    "$SHOCK_MAIN/$method.npz" \
    "shockbubble 1000x396 $method" \
    "$PYTHON_BIN" -u -m for_paper_results.run_weno5_shockbubble \
      --method "$method" --nx 1000 --ny 396 --device cuda \
      --out-dir "$SHOCK_MAIN"
  run_phase \
    "refresh shockbubble figure after $method" \
    "$PYTHON_BIN" -u -m for_paper_results.make_shockbubble_figures \
      --main-dir "$SHOCK_MAIN" --reference-dir "$SHOCK_REF"
done

for method in weno5_js weno5_sr_f64 weno5_sr_f32; do
  run_if_missing \
    "$TITAREV_MAIN/$method.npz" \
    "Titarev-Toro 1001x8 $method" \
    "$PYTHON_BIN" -u -m for_paper_results.run_weno5_titarev_toro \
      --method "$method" --nx 1001 --ny 8 --device cuda \
      --out-dir "$TITAREV_MAIN"
  run_phase \
    "refresh Titarev-Toro figure after $method" \
    "$PYTHON_BIN" -u -m for_paper_results.make_titarev_toro_figures \
      --main-dir "$TITAREV_MAIN"
done

run_if_missing \
  "$SHOCK_REF/weno5_js.npz" \
  "shockbubble reference 2000x791 weno5_js" \
  "$PYTHON_BIN" -u -m for_paper_results.run_weno5_shockbubble \
    --method weno5_js --nx 2000 --ny 791 --device cuda \
    --out-dir "$SHOCK_REF"

run_phase \
  "final shockbubble figures and line cuts" \
  "$PYTHON_BIN" -u -m for_paper_results.make_shockbubble_figures \
    --main-dir "$SHOCK_MAIN" --reference-dir "$SHOCK_REF"

printf '[%s] FINISH failures=%s\n' "$(date '+%F %T')" "$failures" | tee -a "$STATUS_FILE"
exit "$failures"
