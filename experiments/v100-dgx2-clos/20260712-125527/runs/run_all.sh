#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS="$SCRIPT_DIR/tasks.tsv"

xargs -P "${LLM_MAX_PARALLEL:-2}" -n 9 "$SCRIPT_DIR/run_one.sh" < "$TASKS"
