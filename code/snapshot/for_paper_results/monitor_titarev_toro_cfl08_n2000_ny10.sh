#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATUS_FILE="$ROOT/for_paper_results/logs/titarev_toro_cfl08_n2000_ny10.status"

echo "--- phases ---"
tail -30 "$STATUS_FILE" 2>/dev/null || true
echo "--- raw results ---"
find "$ROOT/for_paper_results/raw/titarev_toro_cfl08/N2000x10" \
  -maxdepth 1 -type f -print 2>/dev/null | sort
echo "--- figures ---"
find "$ROOT/for_paper_results/figures/titarev_toro_cfl08_N2000" \
  -maxdepth 1 -type f -print 2>/dev/null | sort
