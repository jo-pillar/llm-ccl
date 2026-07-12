#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUNDLE = Path("/root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527")
SYCCL_WORKTREE = Path("/root/origin-syccl-h800-855a184")
TASKS = BUNDLE / "runs" / "origin_tasks.tsv"
OUT_ROOT = BUNDLE / "origin" / "solve-summary"
DEFAULT_TIMEOUT = "10h"
OOM_RE = re.compile(r"(out of memory|oom|cannot allocate memory|killed)", re.IGNORECASE)

def read_json(path: Path):
  return json.loads(path.read_text(encoding="utf-8"))

def alg_times_best(result):
  alg_times = result.get("alg_times")
  if not isinstance(alg_times, list):
    return "", ""
  best_index = ""
  best_value = ""
  for index, raw in enumerate(alg_times):
    if isinstance(raw, bool):
      continue
    if isinstance(raw, (int, float)) and raw > 0 and (best_value == "" or raw < best_value):
      best_index = index
      best_value = raw
  return best_index, best_value

def inspect_result(path: Path):
  if not path.exists():
    return {"result_exists": False, "result_parseable": False, "result_has_algorithms": False}
  try:
    result = read_json(path)
  except Exception as exc:
    return {
        "result_exists": True,
        "result_parseable": False,
        "result_has_algorithms": False,
        "result_error": str(exc),
    }
  algorithms = result.get("algorithms")
  has_algorithms = isinstance(algorithms, list) and len(algorithms) > 0
  best_index, best_time = alg_times_best(result)
  return {
      "result_exists": True,
      "result_parseable": True,
      "result_has_algorithms": has_algorithms,
      "algorithm_count": len(algorithms) if isinstance(algorithms, list) else 0,
      "origin_alg_times_best_index": best_index,
      "origin_alg_times_best_us": best_time,
  }

def classify(returncode: int, oom_seen: bool, result_info: dict):
  if result_info.get("result_parseable") and result_info.get("result_has_algorithms"):
    return "ok", ""
  if oom_seen:
    return "oom", "oom"
  if returncode in (124, 137):
    return "timeout", "outer_timeout"
  return "solve_failed", "missing_or_invalid_result"

def run_task(task):
  case_id, config_path, result_path, log_path = task
  result = Path(result_path)
  log = Path(log_path)
  result.parent.mkdir(parents=True, exist_ok=True)
  log.parent.mkdir(parents=True, exist_ok=True)
  build_dir = Path(os.environ.get("SYCCL_BUILD_DIR", str(SYCCL_WORKTREE / "build")))
  synthesize = Path('/root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527/bin/origin-syccl-synthesize')
  timeout_value = os.environ.get("ORIGIN_SOLVE_TIMEOUT", DEFAULT_TIMEOUT)
  started = time.time()
  if not synthesize.exists():
    log.write_text(f"missing synthesize binary: {synthesize}\n", encoding="utf-8")
    return {
        "case_id": case_id,
        "status": "solve_failed",
        "failure_category": "missing_synthesize",
        "returncode": "",
        "origin_solve_wall_time_s": 0.0,
        "config_path": config_path,
        "origin_result_path": str(result),
        "origin_log_path": str(log),
    }
  cmd = ["timeout", timeout_value, str(synthesize), "-f", config_path, "solve"]
  env = os.environ.copy()
  env.setdefault("SYNTHESIZE_PARALLEL_THREAD_NUM", "72")
  oom_seen = False
  with log.open("w", encoding="utf-8", buffering=1) as handle:
    handle.write("$ " + " ".join(cmd) + "\n")
    handle.write(f"SYNTHESIZE_PARALLEL_THREAD_NUM={env['SYNTHESIZE_PARALLEL_THREAD_NUM']}\n")
    handle.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(build_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
      print(line, end="")
      handle.write(line)
      if OOM_RE.search(line):
        oom_seen = True
    returncode = proc.wait()
  elapsed = time.time() - started
  result_info = inspect_result(result)
  status, failure_category = classify(returncode, oom_seen, result_info)
  row = {
      "case_id": case_id,
      "status": status,
      "failure_category": failure_category,
      "returncode": returncode,
      "origin_solve_wall_time_s": elapsed,
      "config_path": config_path,
      "origin_result_path": str(result),
      "origin_log_path": str(log),
  }
  row.update(result_info)
  return row

def read_tasks():
  with TASKS.open("r", encoding="utf-8") as handle:
    for raw in handle:
      raw = raw.rstrip("\n")
      if raw:
        yield raw.split("\t")

def main():
  tasks = list(read_tasks())
  max_parallel = max(1, int(os.environ.get("ORIGIN_MAX_PARALLEL", "1")))
  if max_parallel == 1:
    rows = [run_task(task) for task in tasks]
  else:
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
      rows = list(pool.map(run_task, tasks))
  OUT_ROOT.mkdir(parents=True, exist_ok=True)
  summary_json = OUT_ROOT / "summary.json"
  summary_csv = OUT_ROOT / "summary.csv"
  summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
  fields = sorted({key for row in rows for key in row})
  with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)
  print(f"wrote {summary_json}")

if __name__ == "__main__":
  main()
