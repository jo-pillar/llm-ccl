#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CASE CONFIG RESULT LOG" >&2
  exit 2
fi

CASE_ID="$1"
CONFIG_PATH="$2"
RESULT_PATH="$3"
LOG_PATH="$4"
SYNTHESIZE_BIN="/root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/bin/origin-syccl-synthesize"
ORIGIN_SOLVE_TIMEOUT="${ORIGIN_SOLVE_TIMEOUT:-10h}"
export SYNTHESIZE_PARALLEL_THREAD_NUM="${SYNTHESIZE_PARALLEL_THREAD_NUM:-72}"

mkdir -p "$(dirname "$RESULT_PATH")" "$(dirname "$LOG_PATH")"
echo "[v100-compare:origin] case=$CASE_ID config=$CONFIG_PATH"
timeout "$ORIGIN_SOLVE_TIMEOUT" "$SYNTHESIZE_BIN" -f "$CONFIG_PATH" solve 2>&1 | tee "$LOG_PATH"
