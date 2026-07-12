#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

BUNDLE = Path("/root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527")

def read_json(path: Path, default):
  if not path.exists():
    return default
  return json.loads(path.read_text(encoding="utf-8"))

def as_float(value):
  if isinstance(value, bool):
    return None
  if isinstance(value, (int, float)):
    return float(value)
  if isinstance(value, str) and value:
    try:
      return float(value)
    except ValueError:
      return None
  return None

def best_llm(rows):
  best = None
  for row in rows:
    value = as_float(row.get("best_flow_sim_us"))
    if value is None or value <= 0:
      continue
    if best is None or value < best["value"]:
      best = {"value": value, "row": row}
  return best

def by_case(rows):
  grouped = {}
  for row in rows:
    grouped.setdefault(row.get("case_id"), []).append(row)
  return grouped

def single_by_case(rows):
  return {row.get("case_id"): row for row in rows if row.get("case_id")}

manifest = read_json(BUNDLE / "configs" / "manifest.json", {"cases": []})
launch_manifest = read_json(BUNDLE / "manifest.json", {})
flow_info = launch_manifest.get("flow_sim", {}) if isinstance(launch_manifest, dict) else {}
llm_rows = read_json(BUNDLE / "reports" / "llm_strategy_summary.json", [])
origin_solve = single_by_case(read_json(BUNDLE / "origin" / "solve-summary" / "summary.json", []))
origin_flow = single_by_case(read_json(BUNDLE / "origin" / "flow-sim" / "summary.json", []))
llm_by_case = by_case(llm_rows)

rows = []
for case in manifest.get("cases", []):
  case_id = case["case_id"]
  strategy_rows = llm_by_case.get(case_id, [])
  best = best_llm(strategy_rows)
  best_row = best["row"] if best else {}
  solve_row = origin_solve.get(case_id, {})
  flow_row = origin_flow.get(case_id, {})
  llm_best_us = best["value"] if best else ""
  origin_flow_us = as_float(flow_row.get("origin_flow_sim_us"))
  speedup = ""
  if origin_flow_us is not None and isinstance(llm_best_us, float) and llm_best_us > 0:
    speedup = origin_flow_us / llm_best_us
  token_totals = {
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "total_tokens": 0,
      "reasoning_tokens": 0,
  }
  for row in strategy_rows:
    for key in token_totals:
      value = row.get(key)
      if isinstance(value, int):
        token_totals[key] += value
      elif isinstance(value, str) and value.isdigit():
        token_totals[key] += int(value)
  joined = {
      "case_id": case_id,
      "gpu_count": case.get("gpu_count", ""),
      "host_count": case.get("host_count", ""),
      "gpus_per_host": case.get("gpus_per_host", ""),
      "nics_per_host": case.get("nics_per_host", ""),
      "rails": case.get("rails", ""),
      "collective": case.get("collective", ""),
      "total_message_size": case.get("total_message_size", ""),
      "coll_byte": case.get("coll_byte", ""),
      "config_sha256": case.get("config_sha256", ""),
      "config_semantic_sha256": case.get("config_semantic_sha256", ""),
      "flow_sim_sha256": flow_info.get("sha256", ""),
      "llm_status": best_row.get("status", "missing"),
      "llm_best_flow_sim_us": llm_best_us,
      "llm_best_strategy": best_row.get("strategy", ""),
      "llm_best_eval_id": best_row.get("best_eval_id", ""),
      "llm_linear_rank_best_us": "",
      "llm_balance_best_us": "",
      "llm_all_best_us": "",
      "llm_total_candidates": sum(int(row.get("total_candidates") or 0) for row in strategy_rows),
      "llm_valid_candidates": sum(int(row.get("valid_candidates") or 0) for row in strategy_rows),
      "llm_prompt_tokens": token_totals["prompt_tokens"],
      "llm_completion_tokens": token_totals["completion_tokens"],
      "llm_total_tokens": token_totals["total_tokens"],
      "llm_reasoning_tokens": token_totals["reasoning_tokens"],
      "origin_status": solve_row.get("status", "missing"),
      "origin_solve_wall_time_s": solve_row.get("origin_solve_wall_time_s", ""),
      "origin_flow_sim_us": flow_row.get("origin_flow_sim_us", ""),
      "origin_flow_sim_solution_index": flow_row.get("origin_flow_sim_solution_index", ""),
      "origin_alg_times_best_us": solve_row.get("origin_alg_times_best_us", flow_row.get("origin_alg_times_best_us", "")),
      "origin_alg_times_best_index": solve_row.get("origin_alg_times_best_index", flow_row.get("origin_alg_times_best_index", "")),
      "origin_failure_category": solve_row.get("failure_category", ""),
      "origin_log_path": solve_row.get("origin_log_path", ""),
      "speedup_vs_syccl": speedup,
  }
  for row in strategy_rows:
    strategy = row.get("strategy")
    if strategy in ("linear_rank", "balance", "all"):
      joined[f"llm_{strategy}_best_us"] = row.get("best_flow_sim_us", "")
  rows.append(joined)

out_dir = BUNDLE / "reports"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "main_case_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
fields = list(rows[0].keys()) if rows else []
with (out_dir / "main_case_summary.csv").open("w", encoding="utf-8", newline="") as handle:
  writer = csv.DictWriter(handle, fieldnames=fields)
  writer.writeheader()
  for row in rows:
    writer.writerow(row)
(out_dir / "provenance_summary.json").write_text(json.dumps(launch_manifest, indent=2), encoding="utf-8")
print(f"wrote {out_dir / 'main_case_summary.json'}")
