#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

BUNDLE = Path('/root/llm-ccl/experiments/v100-dgx2-clos/20260712-125527')
CONFIG_MANIFEST = BUNDLE / "configs" / "manifest.json"

def read_json(path: Path):
  if path.suffix == ".gz":
    with gzip.open(path, "rt", encoding="utf-8") as handle:
      return json.load(handle)
  return json.loads(path.read_text(encoding="utf-8"))

def sha256_file(path: Path):
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()

def config_semantic_sha256(path: Path):
  data = read_json(path)
  if isinstance(data, dict):
    data.pop("sketch", None)
  payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()

def flow_time(output_path: Path):
  try:
    data = read_json(output_path)
  except Exception:
    return None
  value = data.get("time_us") if isinstance(data, dict) else None
  return float(value) if isinstance(value, (int, float)) and value > 0 else None

def iter_eval_dirs(case_id: str, strategy: str):
  root = BUNDLE / "llm" / "outputs" / strategy / case_id / "eval_artifacts" / "scheme1_direct_events"
  if not root.exists():
    return
  for child in sorted(root.iterdir()):
    if child.is_dir():
      yield child

def checkpoint_nodes(case_id: str, strategy: str):
  root = BUNDLE / "llm" / "outputs" / strategy / case_id / "checkpoints"
  for nodes_path in sorted(root.rglob("nodes.json")) + sorted(root.rglob("nodes.json.gz")):
    try:
      yield from read_json(nodes_path)
    except Exception:
      continue

def token_totals(case_id: str, strategy: str):
  totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
  for node in checkpoint_nodes(case_id, strategy):
    usage = node.get("token_usage") if isinstance(node, dict) else None
    if not isinstance(usage, dict):
      continue
    for key in totals:
      value = usage.get(key)
      if isinstance(value, int) and not isinstance(value, bool):
        totals[key] += value
  return totals

cases = read_json(CONFIG_MANIFEST)["cases"]
flow_sim_info_path = BUNDLE / "provenance" / "flow-sim-rs.json"
flow_sim_info = read_json(flow_sim_info_path) if flow_sim_info_path.exists() else {}
rows = []
for case in cases:
  case_id = case["case_id"]
  for strategy in ("linear_rank", "balance", "all"):
    total_candidates = 0
    valid_candidates = 0
    invalid_provenance = 0
    provenance_error = ""
    candidate_config_sha256 = ""
    candidate_config_semantic_sha256 = ""
    best = None
    for eval_dir in iter_eval_dirs(case_id, strategy) or []:
      manifest_path = eval_dir / "flow-sim-manifest.json"
      if not manifest_path.exists():
        continue
      candidate_config = eval_dir / "candidate-config.json"
      provenance_ok = True
      if not candidate_config.exists():
        provenance_ok = False
        provenance_error = f"missing candidate-config.json in {eval_dir}"
      else:
        try:
          candidate_config_sha256 = sha256_file(candidate_config)
          candidate_config_semantic_sha256 = config_semantic_sha256(candidate_config)
        except Exception as exc:
          provenance_ok = False
          provenance_error = f"could not hash candidate-config.json in {eval_dir}: {exc}"
        if provenance_ok and case.get("config_semantic_sha256") and candidate_config_semantic_sha256 != case.get("config_semantic_sha256"):
          provenance_ok = False
          provenance_error = f"candidate semantic config hash mismatch in {eval_dir}"
      try:
        manifest = read_json(manifest_path)
      except Exception:
        continue
      for item in manifest.get("cases", []):
        total_candidates += 1
        if not provenance_ok:
          invalid_provenance += 1
          continue
        output = Path(item.get("rust_output", ""))
        time_us = flow_time(output)
        if time_us is None:
          continue
        valid_candidates += 1
        if best is None or time_us < best["best_flow_sim_us"]:
          best = {"best_flow_sim_us": time_us, "best_eval_id": eval_dir.name, "best_output_path": str(output)}
    tokens = token_totals(case_id, strategy)
    rows.append({
        "case_id": case_id,
        "strategy": strategy,
        "status": "ok" if best else ("invalid_provenance" if invalid_provenance else "missing"),
        "provenance_error": provenance_error,
        "best_flow_sim_us": best["best_flow_sim_us"] if best else "",
        "best_eval_id": best["best_eval_id"] if best else "",
        "best_output_path": best["best_output_path"] if best else "",
        "valid_candidates": valid_candidates,
        "total_candidates": total_candidates,
        "invalid_provenance": invalid_provenance,
        "expected_config_sha256": case.get("config_sha256", ""),
        "candidate_config_sha256": candidate_config_sha256,
        "config_full_hash_match": candidate_config_sha256 == case.get("config_sha256", ""),
        "expected_config_semantic_sha256": case.get("config_semantic_sha256", ""),
        "candidate_config_semantic_sha256": candidate_config_semantic_sha256,
        "config_semantic_hash_match": candidate_config_semantic_sha256 == case.get("config_semantic_sha256", ""),
        "flow_sim_sha256": flow_sim_info.get("sha256", ""),
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
        "reasoning_tokens": tokens["reasoning_tokens"],
    })

out_dir = BUNDLE / "reports"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "llm_strategy_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
fields = list(rows[0].keys()) if rows else []
with (out_dir / "llm_strategy_summary.csv").open("w", encoding="utf-8", newline="") as handle:
  writer = csv.DictWriter(handle, fieldnames=fields)
  writer.writeheader()
  for row in rows:
    writer.writerow(row)
print(f"wrote {out_dir / 'llm_strategy_summary.json'}")
