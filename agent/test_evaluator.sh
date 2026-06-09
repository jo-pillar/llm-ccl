#!/bin/bash
cd /tmp/syccl-llm-scheme1-direct-events/agent
PROGRAM_PATH=$1 \
SYCCL_BASE_CONFIG=config/a100-8gpu-4nic-clos-ag-4k.json \
SYCCL_TASK_HOST_NUM=4 \
SYCCL_EVALUATOR_TIMEOUT_SECONDS=1800 \
SYCCL_EVAL_ARTIFACT_DIR=./test_result  \
.venv/bin/python - <<'PY'
import importlib.util, json, os
from pathlib import Path

evaluator_path = Path("datasets/syccl/scheme1_direct_events/evaluator.py")
program_path = Path(os.environ["PROGRAM_PATH"])  # 改成你要测的 program.py

spec = importlib.util.spec_from_file_location("syccl_eval", evaluator_path)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

metrics = ev.evaluate(str(program_path))
print(json.dumps(metrics, indent=2, ensure_ascii=False))
PY
