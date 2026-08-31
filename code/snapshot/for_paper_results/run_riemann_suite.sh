#!/usr/bin/env bash
# Compute C.4--C.6; the retained C.3/q400 arrays are already validated.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p for_paper_results/logs

for case_name in c4 c5 c6; do
  echo "[$(date -Is)] start ${case_name}_N400"
  python3 -m for_paper_results.run_quadrant \
    --case "$case_name" --nx 400 --ny 400 --cfl 0.4 \
    --init-quadrature 15 --report-interval 50 \
    2>&1 | tee "for_paper_results/logs/${case_name}_N400.log"
  python3 -m for_paper_results.make_riemann_figures \
    --case "$case_name" \
    2>&1 | tee "for_paper_results/logs/${case_name}_figure.log"
  echo "[$(date -Is)] complete ${case_name}_N400"
done
