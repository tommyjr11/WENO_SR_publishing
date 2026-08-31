#!/usr/bin/env bash
# Reproduce the isolated WENO-SR paper experiments in validation-gate order.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p for_paper_results/logs

run_logged() {
  local name="$1"
  shift
  echo "[$(date -Is)] start $name"
  "$@" 2>&1 | tee "for_paper_results/logs/${name}.log"
  echo "[$(date -Is)] complete $name"
}

run_logged verify_hllc python3 -m for_paper_results.verify_hllc
run_logged smoke_euler python3 -m for_paper_results.smoke_euler
run_logged gste python3 -m for_paper_results.run_gste \
  --t-end 10 --init-quadrature 15 --weno5-cfl 0.6 --weno7-cfl 0.6 \
  --methods weno5_js,weno5_sr_f64,weno5_sr_f32,weno7_js,weno7_sr_f64
run_logged sod_points bash for_paper_results/run_sod_point_study.sh
run_logged vortex python3 -m for_paper_results.run_vortex
run_logged riemann_c4_c6 bash for_paper_results/run_riemann_suite.sh
run_logged weno5_timing python3 -m for_paper_results.run_weno5_timing
run_logged double_mach python3 -m for_paper_results.run_double_mach
run_logged figures python3 -m for_paper_results.make_figures
run_logged riemann_figures python3 -m for_paper_results.make_riemann_figures
run_logged double_mach_figures python3 -m for_paper_results.make_double_mach_figures
run_logged manifest python3 -m for_paper_results.make_manifest
run_logged paper bash for_paper_results/compile_paper.sh
