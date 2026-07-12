#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 CASE STRATEGY CONFIG INSTRUCTION INIT_PROGRAM OUTPUT_PATH ARTIFACT_DIR LOG_PATH TOPODSL" >&2
  exit 2
fi

CASE_NAME="$1"
STRATEGY="$2"
CONFIG_PATH="$3"
INSTRUCTION_PATH="$4"
INIT_PROGRAM_PATH="$5"
OUTPUT_PATH="$6"
ARTIFACT_DIR="$7"
LOG_PATH="$8"
TOPODSL_PATH="$9"
STATUS_PATH="${LOG_PATH%.log}.status"

AGENT_ROOT="/root/llm-ccl/agent"
MAX_GENERATIONS="${MAX_GENERATIONS:-10000}"
K_CANDIDATES="${K_CANDIDATES:-4}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-1}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-1}"
LLM_POLICY_POOL_SIZE="${LLM_POLICY_POOL_SIZE:-100}"
SIMPLETES_TIMEOUT="${SIMPLETES_TIMEOUT:-2h}"
EXTRA_SIMPLETES_ARGS="${EXTRA_SIMPLETES_ARGS:-}"
MODEL_NAME="${MODEL_NAME:-hosted_vllm/deepseek-ai/DeepSeek-V4-Flash}"
API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-sk-nokey}}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
FLOW_SIM_BIN="${FLOW_SIM_BIN:-/root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/bin/flow-sim-rs}"

mkdir -p "$OUTPUT_PATH" "$ARTIFACT_DIR" "$(dirname "$LOG_PATH")"
cd "$AGENT_ROOT"

export PYTHONPATH="$AGENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SYCCL_BASE_CONFIG="$CONFIG_PATH"
export SYCCL_EVAL_ARTIFACT_DIR="$ARTIFACT_DIR"
export FLOW_SIM_BIN="$FLOW_SIM_BIN"
export SYCCL_EVALUATOR_TIMEOUT_SECONDS="${SYCCL_EVALUATOR_TIMEOUT_SECONDS:-3600}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-nokey}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
for host in 127.0.0.1 localhost; do
  case ",${NO_PROXY:-}," in
    *",$host,"*) ;;
    *) export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$host" ;;
  esac
  case ",${no_proxy:-}," in
    *",$host,"*) ;;
    *) export no_proxy="${no_proxy:+$no_proxy,}$host" ;;
  esac
done

echo "[syccl-v100-dgx2-clos] case=$CASE_NAME strategy=$STRATEGY config=$CONFIG_PATH topodsl=$TOPODSL_PATH"
START_EPOCH="$(date +%s)"
START_TIME="$(date -Iseconds)"
{
  echo "case=$CASE_NAME"
  echo "strategy=$STRATEGY"
  echo "start_epoch=$START_EPOCH"
  echo "start_time=$START_TIME"
  echo "per_run_timeout=$SIMPLETES_TIMEOUT"
  echo "config=$CONFIG_PATH"
  echo "topodsl=$TOPODSL_PATH"
  echo "flow_sim_bin=$FLOW_SIM_BIN"
} > "$STATUS_PATH"

set +e
timeout "$SIMPLETES_TIMEOUT" uv run python main.py \
  --init-program "$INIT_PROGRAM_PATH" \
  --evaluator "$AGENT_ROOT/datasets/syccl/scheme1_direct_events/evaluator.py" \
  --instruction "$INSTRUCTION_PATH" \
  --selector llm_elite \
  --elite-selection-strategy "$STRATEGY" \
  --num-chains 1 \
  --k-candidates "$K_CANDIDATES" \
  --stream-k-candidates \
  --max-generations "$MAX_GENERATIONS" \
  --eval-concurrency "$EVAL_CONCURRENCY" \
  --gen-concurrency "$GEN_CONCURRENCY" \
  --init-eval-repeats 1 \
  --llm-policy-pool-size "$LLM_POLICY_POOL_SIZE" \
  --model "$MODEL_NAME" \
  --api-base "$API_BASE" \
  --api-key "$API_KEY" \
  --max-tokens "$MAX_TOKENS" \
  --output-path "$OUTPUT_PATH" \
  --save-llm-io \
  --skip-preflight \
  $EXTRA_SIMPLETES_ARGS 2>&1 | tee "$LOG_PATH"
status="${PIPESTATUS[0]}"
set -e
END_EPOCH="$(date +%s)"
END_TIME="$(date -Iseconds)"
{
  echo "end_epoch=$END_EPOCH"
  echo "end_time=$END_TIME"
  echo "elapsed_s=$((END_EPOCH - START_EPOCH))"
  echo "exit_status=$status"
} >> "$STATUS_PATH"
exit "$status"
