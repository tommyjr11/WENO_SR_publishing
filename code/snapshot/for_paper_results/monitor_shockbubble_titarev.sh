#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$ROOT/for_paper_results/logs/shockbubble_titarev_overnight.pid"
LOG_FILE="$ROOT/for_paper_results/logs/shockbubble_titarev_overnight.log"
STATUS_FILE="$ROOT/for_paper_results/logs/shockbubble_titarev_overnight.status"

if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "overnight job running pid=$pid"
  else
    echo "overnight job not running (recorded pid=$pid)"
  fi
else
  echo "no PID file: $PID_FILE"
fi

if [[ -f "$STATUS_FILE" ]]; then
  echo "status:"
  tail -20 "$STATUS_FILE"
fi

if [[ -f "$LOG_FILE" ]]; then
  echo "log tail:"
  tail -40 "$LOG_FILE"
fi

echo "figures:"
find "$ROOT/for_paper_results/figures/shockbubble" "$ROOT/for_paper_results/figures/titarev_toro" \
  -maxdepth 1 -type f 2>/dev/null | sort
