#!/usr/bin/env bash
# WENO5 mixed precision v3: v2a recipe with only tv_bg reduced to 0.02.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TV_BG=0.02 \
RUN_NAME=apost_weno5_v3_tv002_mlp_f32_gate3e3_200k \
  exec "$SCRIPT_DIR/run_weno5_v3_tv001_mlp_f32.sh"
