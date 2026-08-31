#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p for_paper_results/logs

python3 -u -m for_paper_results.run_sod \
  --nx 51 --ny 11 --t-end 0.2 --cfl 0.4 --init-quadrature 15 \
  --out-tag N51_t020 2>&1 | tee for_paper_results/logs/sod_N51_t020.log

python3 -m for_paper_results.make_sod_point_figures
