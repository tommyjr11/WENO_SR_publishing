#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$ROOT/for_paper_results/logs/titarev_toro_cfl08_ny10.pid"
LOG_FILE="$ROOT/for_paper_results/logs/titarev_toro_cfl08_ny10.log"
STATUS_FILE="$ROOT/for_paper_results/logs/titarev_toro_cfl08_ny10.status"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "pid=$pid running=yes"
  else
    echo "pid=$pid running=no"
  fi
fi

echo "--- phases ---"
tail -30 "$STATUS_FILE" 2>/dev/null || true
echo "--- solver output ---"
tail -40 "$LOG_FILE" 2>/dev/null || true
echo "--- figures ---"
find "$ROOT/for_paper_results/figures/titarev_toro_cfl08" \
  -maxdepth 1 -type f -print 2>/dev/null | sort
