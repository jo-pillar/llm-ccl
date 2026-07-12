#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

BUNDLE = Path('/root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527')
FLOW_SIM = BUNDLE / "bin" / "flow-sim-rs"
TASKS = BUNDLE / "runs" / "origin_tasks.tsv"
OUT_ROOT = BUNDLE / "origin" / "flow-sim"

def read_json(path: Path):
  return json.loads(path.read_text(encoding="utf-8"))

def best_solution(path: Path):
  data = read_json(path)
  solutions = data.get("solutions", [])
  best = None
  for index, solution in enumerate(solutions):
    value = solution.get("rust_time_us")
    if isinstance(value, (int, float)) and value > 0:
      if best is None or value < best["rust_time_us"]:
        best = dict(solution)
        best.setdefault("solution_index", index)
  return best

def alg_times_best(path: Path):
  try:
    data = read_json(path)
  except Exception:
    return "", ""
  alg_times = data.get("alg_times") if isinstance(data, dict) else None
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

rows = []
with TASKS.open("r", encoding="utf-8") as handle:
  for raw in handle:
    case_id, config_path, result_path, _log_path = raw.rstrip("\n").split("\t")
    result = Path(result_path)
    alg_best_index, alg_best_us = alg_times_best(result)
    output = OUT_ROOT / case_id / "origin-all-flow-sim.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not result.exists():
      rows.append({"case_id": case_id, "status": "missing_origin_result", "origin_result": str(result)})
      continue
    proc = subprocess.run(
        [str(FLOW_SIM), "simulate", "--config", config_path, "--translated", str(result), "--output", str(output)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = time.time() - started
    if proc.returncode != 0:
      rows.append({"case_id": case_id, "status": "flow_sim_failed", "elapsed_s": elapsed, "error": proc.stdout[-1000:]})
      continue
    best = best_solution(output)
    if best is None:
      rows.append({"case_id": case_id, "status": "no_positive_solution", "elapsed_s": elapsed, "output": str(output)})
      continue
    rows.append({
        "case_id": case_id,
        "status": "ok",
        "elapsed_s": elapsed,
        "origin_flow_sim_us": best["rust_time_us"],
        "origin_flow_sim_solution_index": best.get("solution_index"),
        "origin_alg_times_best_us": alg_best_us,
        "origin_alg_times_best_index": alg_best_index,
        "origin_flow_sim_output": str(output),
    })

summary_json = OUT_ROOT / "summary.json"
summary_csv = OUT_ROOT / "summary.csv"
summary_json.parent.mkdir(parents=True, exist_ok=True)
summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
fields = sorted({key for row in rows for key in row})
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
  writer = csv.DictWriter(handle, fieldnames=fields)
  writer.writeheader()
  for row in rows:
    writer.writerow(row)
print(f"wrote {summary_json}")
