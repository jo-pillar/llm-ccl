# V100 DGX-2 Clos Search Launch Commands (llm-ccl vs origin-SyCCL)

This 4-host and 8-host comparison bundle contains 72 llm-ccl tasks and 24 origin-SyCCL tasks. It is prepared but not started.

Bundle root: `/root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527`

## Start llm-ccl search

```bash
cd /root/llm-ccl/agent
setsid nohup env \
  OPENAI_API_KEY=sk-nokey \
  MODEL_NAME=hosted_vllm/deepseek-ai/DeepSeek-V4-Flash \
  API_BASE=http://127.0.0.1:8000/v1 \
  MAX_TOKENS=32768 \
  /root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/runs/run_llm_all.sh \
  > /root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/runs/run_llm_all.nohup.log 2>&1 &
echo $! > /root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/runs/run_llm_all.pid
```

## Start origin-SyCCL solve

```bash
cd /root/llm-ccl/agent
setsid nohup env \
  ORIGIN_MAX_PARALLEL=1 \
  ORIGIN_SOLVE_TIMEOUT=10h \
  SYNTHESIZE_PARALLEL_THREAD_NUM=72 \
  /root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/runs/run_origin_all.sh \
  > /root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/runs/run_origin_all.nohup.log 2>&1 &
echo $! > /root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/runs/run_origin_all.pid
```

## Re-simulate and build the comparison report

Run these after the corresponding llm-ccl and origin-SyCCL stages finish:

```bash
python3 /root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/runs/run_origin_flow_sim_all.py
python3 /root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/runs/summarize_llm_outputs.py
python3 /root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/runs/build_final_report.py
```

Final outputs:

- `/root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/reports/main_case_summary.csv`
- `/root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/reports/main_case_summary.json`

## Monitor llm-ccl

```bash
tail -f /root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/runs/run_llm_all.nohup.log
```

```bash
find /root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527 -path '*/logs/*.status' -print -exec cat {} \;
```

## Monitor origin-SyCCL

```bash
tail -f /root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/runs/run_origin_all.nohup.log
```
