#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
TAG="vortex_cfl04"

"$PYTHON_BIN" -u -m for_paper_results.run_vortex \
  --grids 25,50,100,200 \
  --cfl 0.4 --t-end 2.0 --quadrature 15 \
  --methods weno5_js,weno5_sr_f64,weno5_sr_f32,weno7_js,weno7_sr_f64 \
  --device "$DEVICE" --report-interval 100 \
  --out-tag "$TAG" \
  | tee "for_paper_results/logs/${TAG}.log"

"$PYTHON_BIN" -u -m for_paper_results.make_vortex_convergence \
  --raw-dir "for_paper_results/raw/$TAG" \
  --figure-dir "for_paper_results/figures/$TAG" \
  --table-dir "for_paper_results/tables/$TAG" \
  --expected-cfl 0.4
