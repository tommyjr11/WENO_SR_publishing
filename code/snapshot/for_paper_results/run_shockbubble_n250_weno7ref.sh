#!/usr/bin/env bash
# Two shock--bubble cases at 250x99 against a 2000x791 WENO7-JS reference.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

NX=250
NY=99
REF_NX=2000
REF_NY=791
CFL=0.4
LOG_DIR="$ROOT/for_paper_results/logs"
STATUS_FILE="$LOG_DIR/shockbubble_n250_weno7ref.status"
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

is_complete() {
  local json_path="$1"
  [[ -f "$json_path" ]] \
    && grep -q '"complete": true' "$json_path" \
    && grep -q '"cfl": 0.4' "$json_path"
}

run_solver() {
  local package="$1"
  local method="$2"
  local nx="$3"
  local ny="$4"
  local t_end="$5"
  local out_dir="$6"
  (
    cd "$package"
    "$PYTHON_BIN" -u -m for_paper_results.run_weno5_shockbubble \
      --method "$method" --nx "$nx" --ny "$ny" --cfl "$CFL" \
      --t-end "$t_end" --device cuda --report-interval 200 \
      --out-dir "$out_dir"
  )
}

run_case() {
  local package="$1"
  local case_name="$2"
  local t_end="$3"
  local figure_name="$4"
  local main_dir="$package/results/raw/$case_name/N${NX}x${NY}"
  local ref_dir="$package/results/raw/$case_name/reference_weno7_N${REF_NX}x${REF_NY}"
  local figure_dir="$package/results/figures/$figure_name"
  local table_path="$package/results/tables/${figure_name}_linecut_errors.csv"
  mkdir -p "$main_dir" "$ref_dir" "$figure_dir" "$(dirname "$table_path")"

  for method in weno5_js weno5_sr_f64 weno5_sr_f32 weno7_js weno7_sr_f64; do
    if is_complete "$main_dir/$method.json"; then
      printf '[%s] SKIP  %s %s\n' "$(date '+%F %T')" "$case_name" "$method" | tee -a "$STATUS_FILE"
    else
      run_phase \
        "$case_name ${NX}x${NY} $method" \
        run_solver "$package" "$method" "$NX" "$NY" "$t_end" "$ROOT/$main_dir"
    fi
  done

  if is_complete "$ref_dir/weno7_js.json"; then
    printf '[%s] SKIP  %s WENO7-JS reference\n' "$(date '+%F %T')" "$case_name" | tee -a "$STATUS_FILE"
  else
    run_phase \
      "$case_name reference ${REF_NX}x${REF_NY} weno7_js" \
      run_solver "$package" weno7_js "$REF_NX" "$REF_NY" "$t_end" "$ROOT/$ref_dir"
  fi

  run_phase \
    "$case_name N250 figures against WENO7-JS reference" \
    "$PYTHON_BIN" -u -m for_paper_results.make_shockbubble_figures \
      --main-dir "$main_dir" --reference-dir "$ref_dir" \
      --reference-method weno7_js --figure-dir "$figure_dir" \
      --table-path "$table_path"
}

run_case \
  shockbubble_t0006_cfl04_server \
  shockbubble_t0006_cfl04 0.0006 \
  shockbubble_t0006_cfl04_N250_weno7ref

run_case \
  shockbubble_ma3_t0001_cfl04_server \
  shockbubble_ma3_t0001_cfl04 0.0001 \
  shockbubble_ma3_t0001_cfl04_N250_weno7ref

printf '[%s] FINISH failures=%s\n' "$(date '+%F %T')" "$failures" | tee -a "$STATUS_FILE"
exit "$failures"
