#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="$ROOT/for_paper_results/logs/shockbubble_t0006_titarev_all.log"
STATUS_FILE="$ROOT/for_paper_results/logs/shockbubble_t0006_titarev_all.status"
PID_FILE="$ROOT/for_paper_results/logs/shockbubble_t0006_titarev_all.pid"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "queue_pid=$pid running=yes"
  else
    echo "queue_pid=$pid running=no"
  fi
else
  echo "queue_pid=missing"
fi

echo "--- status ---"
tail -30 "$STATUS_FILE" 2>/dev/null || true
echo "--- solver log ---"
tail -40 "$LOG_FILE" 2>/dev/null || true
echo "--- completed fields ---"
find \
  "$ROOT/for_paper_results/raw/shockbubble_t0006" \
  "$ROOT/for_paper_results/raw/titarev_toro/N1001x8" \
  -maxdepth 2 -name '*.json' -type f -print 2>/dev/null | sort
echo "--- figures ---"
find \
  "$ROOT/for_paper_results/figures/shockbubble_t0006" \
  "$ROOT/for_paper_results/figures/titarev_toro" \
  -maxdepth 1 -type f \( -name '*.png' -o -name '*.pdf' \) -print 2>/dev/null | sort
