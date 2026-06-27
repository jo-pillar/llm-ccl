#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


AGENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = AGENT_ROOT / "result" / "config" / "h800-64hosts-8gpu-8nic-rail" / "ag"
DEFAULT_ORIGIN_ROOT = Path(
    "/home/antl/wzd/origin-syccl/build/syn_res/results-0.5/h800-64hosts-8gpu-8nic-rail/ag"
)
DEFAULT_FLOW_SIM_BIN = Path(
    os.environ.get(
        "FLOW_SIM_BIN",
        "/home/antl/wzd/llm-ccl/Flow-Simulator/flow-sim-rs/target/release/flow-sim-rs",
    )
)
DEFAULT_SYNTHESIZE_BIN = Path(os.environ.get("SYNTHESIZE_BIN", "/home/antl/wzd/syccl/build/synthesize"))
STRATEGIES = ("linear_rank", "balance", "all")
CASE_NAMES = (
    "65536B-prune=small",
    "262144B-prune=small",
    "1048576B-prune=small",
    "4194304B-prune=small",
)


@dataclass(frozen=True)
class LlmArtifact:
  case_name: str
  strategy: str
  case_dir: Path
  eval_dir: Path
  config_path: Path
  sketch_path: Path
  flow_output_path: Path
  flow_time_us: float


@dataclass(frozen=True)
class CommandResult:
  ok: bool
  returncode: int
  wall_time_s: float
  output: str
  error: str = ""


def read_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def as_float(value: Any) -> float | None:
  if isinstance(value, bool):
    return None
  if isinstance(value, (int, float)):
    return float(value)
  if isinstance(value, str):
    try:
      return float(value)
    except ValueError:
      return None
  return None


def read_flow_time(output_path: Path) -> float:
  output = read_json(output_path)
  time_us = as_float(output.get("time_us")) if isinstance(output, dict) else None
  if (time_us is None or time_us <= 0) and isinstance(output, dict):
    solutions = output.get("solutions")
    if isinstance(solutions, list):
      candidates = []
      for solution in solutions:
        if not isinstance(solution, dict):
          continue
        solution_time = as_float(solution.get("rust_time_us"))
        result = solution.get("result")
        if solution_time is None and isinstance(result, dict):
          solution_time = as_float(result.get("time_us"))
        if solution_time is not None and solution_time > 0:
          candidates.append(solution_time)
      if candidates:
        time_us = min(candidates)
  if time_us is None or time_us <= 0:
    raise ValueError(f"flow-sim output has no positive time_us: {output_path}")
  return time_us


def iter_llm_eval_dirs(experiment_root: Path, case_name: str, strategies: Sequence[str]) -> Iterable[tuple[str, Path]]:
  for strategy in strategies:
    case_dir = experiment_root / "outputs" / strategy / case_name
    eval_root = case_dir / "eval_artifacts" / "scheme1_direct_events"
    if not eval_root.exists():
      continue
    for eval_dir in sorted(eval_root.iterdir()):
      if eval_dir.is_dir():
        yield strategy, eval_dir


def select_best_llm_artifact(
    experiment_root: Path,
    case_name: str,
    *,
    strategies: Sequence[str] = STRATEGIES,
) -> LlmArtifact | None:
  best: LlmArtifact | None = None
  for strategy, eval_dir in iter_llm_eval_dirs(experiment_root, case_name, strategies):
    manifest_path = eval_dir / "flow-sim-manifest.json"
    config_path = eval_dir / "candidate-config.json"
    sketch_path = eval_dir / "flow-sim-inputs" / "candidate-000" / "candidate-sketch.json"
    if not manifest_path.exists() or not config_path.exists() or not sketch_path.exists():
      continue
    manifest = read_json(manifest_path)
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(cases, list):
      continue
    for case in cases:
      if not isinstance(case, dict):
        continue
      raw_output = case.get("rust_output")
      if not isinstance(raw_output, str) or not raw_output:
        continue
      output_path = Path(raw_output)
      if not output_path.exists():
        continue
      try:
        time_us = read_flow_time(output_path)
      except (ValueError, json.JSONDecodeError):
        continue
      artifact = LlmArtifact(
          case_name=case_name,
          strategy=strategy,
          case_dir=experiment_root / "outputs" / strategy / case_name,
          eval_dir=eval_dir,
          config_path=config_path,
          sketch_path=sketch_path,
          flow_output_path=output_path,
          flow_time_us=time_us,
      )
      if best is None or artifact.flow_time_us < best.flow_time_us:
        best = artifact
  return best


def select_best_origin_algorithm(result: dict[str, Any]) -> tuple[int, float]:
  alg_times = result.get("alg_times")
  algorithms = result.get("algorithms")
  if not isinstance(alg_times, list) or not isinstance(algorithms, list):
    raise ValueError("origin result must contain alg_times and algorithms lists")
  if not alg_times or not algorithms:
    raise ValueError("origin result has no algorithms")
  best_index = -1
  best_time = float("inf")
  for index, raw_time in enumerate(alg_times):
    if index >= len(algorithms):
      break
    time_us = as_float(raw_time)
    if time_us is not None and time_us > 0 and time_us < best_time:
      best_index = index
      best_time = time_us
  if best_index < 0:
    raise ValueError("origin result has no positive algorithm time")
  return best_index, best_time


def origin_algorithm_as_translated(result: dict[str, Any], algorithm_index: int) -> dict[str, Any]:
  algorithms = result.get("algorithms")
  if not isinstance(algorithms, list):
    raise ValueError("origin result algorithms must be a list")
  try:
    algorithm = algorithms[algorithm_index]
  except IndexError as exc:
    raise ValueError(f"origin algorithm index out of range: {algorithm_index}") from exc
  if not isinstance(algorithm, dict):
    raise ValueError(f"origin algorithm {algorithm_index} is not an object")
  return {
      "coll_name": result.get("coll_name"),
      "ngpus": result.get("ngpus"),
      "chunk_size_byte": result.get("chunk_size_byte"),
      "algorithms": [algorithm],
  }


def compact_sketch_to_native(compact: Any, config_path: Path) -> list[dict[str, Any]]:
  config = read_json(config_path)
  ngpus = int(config["hosts"]["host_num"]) * int(config["hosts"]["host_gpu_num"])
  candidates = compact if isinstance(compact, list) else [compact]
  if not candidates:
    raise ValueError("compact sketch is empty")
  if all(isinstance(item, list) and len(item) == 5 for item in candidates):
    candidates = [candidates]

  native_candidates: list[dict[str, Any]] = []
  for candidate_index, candidate in enumerate(candidates):
    if isinstance(candidate, dict) and {"ngpus", "src_gpu", "nodes"}.issubset(candidate):
      native_candidates.append(candidate)
      continue
    if not isinstance(candidate, list):
      raise ValueError(f"sketch candidate {candidate_index} must be a list or native sketch")
    nodes = []
    for node_id, raw in enumerate(candidate):
      if isinstance(raw, list) and len(raw) == 5:
        step, layer, group, srcs, dsts = raw
      elif isinstance(raw, dict):
        step = raw.get("step")
        layer = raw.get("layer")
        group = raw.get("group")
        srcs = raw.get("srcs", raw.get("src"))
        dsts = raw.get("dsts", raw.get("dst"))
      else:
        raise ValueError(f"sketch node {node_id} must be a list or object")
      nodes.append(
          {
              "id": node_id,
              "step": step,
              "layer": layer,
              "group": group,
              "src_dest_pair": {
                  "srcs": normalize_gpu_list(srcs),
                  "dsts": normalize_gpu_list(dsts),
              },
              "deps": [],
              "next": [],
          }
      )
    rebuild_native_edges(nodes)
    native_candidates.append({"ngpus": ngpus, "src_gpu": 0, "nodes": nodes})
  return native_candidates


def normalize_gpu_list(value: Any) -> list[int]:
  if isinstance(value, list):
    return [int(v) for v in value]
  return [int(value)]


def rebuild_native_edges(nodes: list[dict[str, Any]]) -> None:
  nodes.sort(
      key=lambda node: (
          int(node["step"]),
          int(node["layer"]),
          int(node["group"]),
          node["src_dest_pair"]["srcs"],
          node["src_dest_pair"]["dsts"],
          int(node["id"]),
      )
  )
  reached_by_node: dict[int, int] = {0: -1}
  reached_step: dict[int, int] = {0: 0}
  for new_id, node in enumerate(nodes):
    node["id"] = new_id
    node["deps"] = []
    node["next"] = []

  for node in nodes:
    step = int(node["step"])
    deps: set[int] = set()
    srcs = [int(src) for src in node["src_dest_pair"]["srcs"]]
    dsts = [int(dst) for dst in node["src_dest_pair"]["dsts"]]
    for src in srcs:
      if src not in reached_by_node:
        raise ValueError(f"source GPU {src} has not been reached before step {step}")
      dep = reached_by_node[src]
      if dep >= 0:
        if reached_step[src] >= step:
          raise ValueError(
              f"source GPU {src} is reached at step {reached_step[src]} "
              f"but reused at step {step}"
          )
        deps.add(dep)
    for dst in dsts:
      if dst in reached_by_node:
        raise ValueError(f"destination GPU {dst} is reached more than once")
    node["deps"] = sorted(deps)
    for dst in dsts:
      reached_by_node[dst] = int(node["id"])
      reached_step[dst] = step

  for node in nodes:
    for dep in node["deps"]:
      nodes[dep]["next"].append(int(node["id"]))
  for node in nodes:
    node["next"] = sorted(set(node["next"]))


def run_command(cmd: Sequence[str], *, timeout_s: int, cwd: Path | None = None) -> CommandResult:
  start = time.time()
  try:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    wall = time.time() - start
    return CommandResult(
        ok=proc.returncode == 0,
        returncode=proc.returncode,
        wall_time_s=wall,
        output=proc.stdout,
    )
  except subprocess.TimeoutExpired as exc:
    return CommandResult(
        ok=False,
        returncode=124,
        wall_time_s=time.time() - start,
        output=exc.stdout or "",
        error=f"timeout after {timeout_s}s",
    )


def flow_sim_simulate(
    flow_sim_bin: Path,
    config_path: Path,
    translated_path: Path,
    output_path: Path,
    *,
    timeout_s: int,
) -> CommandResult:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  return run_command(
      [
          str(flow_sim_bin),
          "simulate",
          "--config",
          str(config_path),
          "--translated",
          str(translated_path),
          "--output",
          str(output_path),
      ],
      timeout_s=timeout_s,
      cwd=AGENT_ROOT,
  )


def flow_sim_simulate_sketch(
    flow_sim_bin: Path,
    config_path: Path,
    sketch_path: Path,
    output_path: Path,
    *,
    timeout_s: int,
) -> CommandResult:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  return run_command(
      [
          str(flow_sim_bin),
          "simulate-sketch",
          "--config",
          str(config_path),
          "--sketch",
          str(sketch_path),
          "--output",
          str(output_path),
      ],
      timeout_s=timeout_s,
      cwd=AGENT_ROOT,
  )


def syccl_resim(
    synthesize_bin: Path,
    config_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    timeout_s: int,
) -> CommandResult:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  return run_command(
      [
          str(synthesize_bin),
          "-f",
          str(config_path),
          "resim",
          "-i",
          str(input_path),
          "-o",
          str(output_path),
      ],
      timeout_s=timeout_s,
      cwd=synthesize_bin.parent.parent,
  )


def syccl_resim_sketch(
    synthesize_bin: Path,
    config_path: Path,
    sketch_path: Path,
    output_path: Path,
    translated_path: Path,
    *,
    timeout_s: int,
) -> CommandResult:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  translated_path.parent.mkdir(parents=True, exist_ok=True)
  return run_command(
      [
          str(synthesize_bin),
          "-f",
          str(config_path),
          "resim",
          "--sketch",
          "-i",
          str(sketch_path),
          "-o",
          str(output_path),
          "--dump-translated",
          str(translated_path),
      ],
      timeout_s=timeout_s,
      cwd=synthesize_bin.parent.parent,
  )


def read_resim_time(path: Path) -> float | None:
  if not path.exists():
    return None
  # SyCCL resim outputs can be hundreds of MB due to LinkTrace. Avoid loading the
  # full JSON just to read the top-level Time field.
  pattern = re.compile(r'"Time"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
  with path.open("r", encoding="utf-8", errors="replace") as f:
    for line in f:
      match = pattern.search(line)
      if match:
        value = as_float(match.group(1))
        if value is not None and value > 0:
          return value
  return None


def write_summary(rows: list[dict[str, Any]], output_root: Path) -> None:
  output_root.mkdir(parents=True, exist_ok=True)
  json_path = output_root / "summary.json"
  csv_path = output_root / "summary.csv"
  write_json(json_path, rows)
  fieldnames = [
      "case_name",
      "llm_strategy",
      "origin_algorithm_index",
      "llm_flow_sim_us",
      "origin_flow_sim_us",
      "flow_sim_origin_over_llm",
      "llm_resim_us",
      "origin_resim_us",
      "resim_origin_over_llm",
      "status",
      "notes",
  ]
  with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({key: row.get(key, "") for key in fieldnames})


def ratio(numerator: float | None, denominator: float | None) -> float | None:
  if numerator is None or denominator is None or denominator <= 0:
    return None
  return numerator / denominator


def prepare_case_artifacts(
    experiment_root: Path,
    origin_root: Path,
    output_root: Path,
    case_name: str,
) -> tuple[dict[str, Any], LlmArtifact | None, Path | None, Path | None, int | None]:
  row: dict[str, Any] = {"case_name": case_name, "status": "pending", "notes": ""}
  llm = select_best_llm_artifact(experiment_root, case_name)
  if llm is None:
    row["status"] = "missing_llm_artifact"
    return row, None, None, None, None
  row["llm_strategy"] = llm.strategy
  row["llm_flow_sim_us"] = llm.flow_time_us

  origin_result_path = origin_root / f"{case_name}-result.json"
  origin_config_path = origin_root / "configs" / f"{case_name}-config.json"
  if not origin_result_path.exists() or not origin_config_path.exists():
    row["status"] = "missing_origin_artifact"
    return row, llm, None, None, None

  origin_result = read_json(origin_result_path)
  origin_algorithm_index, origin_best_time = select_best_origin_algorithm(origin_result)
  row["origin_algorithm_index"] = origin_algorithm_index
  row["origin_original_syccl_us"] = origin_best_time

  artifact_dir = output_root / "artifacts" / case_name
  llm_artifact_dir = artifact_dir / "llm"
  origin_artifact_dir = artifact_dir / "origin"
  llm_artifact_dir.mkdir(parents=True, exist_ok=True)
  origin_artifact_dir.mkdir(parents=True, exist_ok=True)

  llm_config_path = llm_artifact_dir / "candidate-config.json"
  llm_compact_sketch_path = llm_artifact_dir / "candidate-sketch-compact.json"
  llm_native_sketch_path = llm_artifact_dir / "candidate-sketch-native.json"
  shutil.copy2(llm.config_path, llm_config_path)
  shutil.copy2(llm.sketch_path, llm_compact_sketch_path)
  write_json(
      llm_native_sketch_path,
      compact_sketch_to_native(read_json(llm.sketch_path), llm.config_path),
  )

  origin_config_copy = origin_artifact_dir / "origin-config.json"
  origin_translated_path = origin_artifact_dir / "origin-best-translated.json"
  shutil.copy2(origin_config_path, origin_config_copy)
  write_json(origin_translated_path, origin_algorithm_as_translated(origin_result, origin_algorithm_index))

  row["status"] = "artifacts_ready"
  return row, llm, origin_config_copy, origin_translated_path, origin_algorithm_index


def run_cross_eval(args: argparse.Namespace) -> int:
  experiment_root = args.experiment_root.resolve()
  origin_root = args.origin_root.resolve()
  output_root = args.output_root.resolve()
  flow_sim_bin = args.flow_sim_bin.resolve()
  synthesize_bin = args.synthesize_bin.resolve()
  cases = args.cases or list(CASE_NAMES)
  rows: list[dict[str, Any]] = []

  for case_name in cases:
    row, llm, origin_config_path, origin_translated_path, _origin_algorithm_index = prepare_case_artifacts(
        experiment_root,
        origin_root,
        output_root,
        case_name,
    )
    if llm is None or origin_config_path is None or origin_translated_path is None:
      rows.append(row)
      continue

    case_artifact_dir = output_root / "artifacts" / case_name
    llm_config_path = case_artifact_dir / "llm" / "candidate-config.json"
    llm_compact_sketch_path = case_artifact_dir / "llm" / "candidate-sketch-compact.json"
    llm_native_sketch_path = case_artifact_dir / "llm" / "candidate-sketch-native.json"

    origin_flow_out = output_root / "flow_sim_rs" / "origin" / case_name / "candidate-000.json"
    if not args.skip_flow_sim:
      if origin_flow_out.exists() and not args.rerun_existing:
        row["origin_flow_sim_us"] = read_flow_time(origin_flow_out)
      else:
        res = flow_sim_simulate(
            flow_sim_bin,
            origin_config_path,
            origin_translated_path,
            origin_flow_out,
            timeout_s=args.timeout_s,
        )
        if res.ok:
          row["origin_flow_sim_us"] = read_flow_time(origin_flow_out)
        else:
          row["notes"] += f" origin_flow_sim_failed={res.error or res.output[-300:]};"

      llm_flow_out = output_root / "flow_sim_rs" / "llm" / case_name / "candidate-000.json"
      if llm_flow_out.exists() and not args.rerun_existing:
        row["llm_flow_sim_us"] = read_flow_time(llm_flow_out)
      else:
        res = flow_sim_simulate_sketch(
            flow_sim_bin,
            llm_config_path,
            llm_compact_sketch_path,
            llm_flow_out,
            timeout_s=args.timeout_s,
        )
        if res.ok:
          row["llm_flow_sim_us"] = read_flow_time(llm_flow_out)
        else:
          row["notes"] += f" llm_flow_sim_failed={res.error or res.output[-300:]};"

    if not args.skip_resim:
      origin_resim_out = output_root / "syccl_resim" / "origin" / case_name / "resim.json"
      if origin_resim_out.exists() and not args.rerun_existing:
        row["origin_resim_us"] = read_resim_time(origin_resim_out)
      else:
        res = syccl_resim(
            synthesize_bin,
            origin_config_path,
            origin_translated_path,
            origin_resim_out,
            timeout_s=args.timeout_s,
        )
        if res.ok:
          row["origin_resim_us"] = read_resim_time(origin_resim_out)
        else:
          row["notes"] += f" origin_resim_failed={res.error or res.output[-300:]};"

      llm_resim_out = output_root / "syccl_resim" / "llm" / case_name / "resim.json"
      llm_translated_out = output_root / "syccl_resim" / "llm" / case_name / "translated.json"
      if llm_resim_out.exists() and not args.rerun_existing:
        row["llm_resim_us"] = read_resim_time(llm_resim_out)
      else:
        res = syccl_resim_sketch(
            synthesize_bin,
            llm_config_path,
            llm_native_sketch_path,
            llm_resim_out,
            llm_translated_out,
            timeout_s=args.timeout_s,
        )
        if res.ok:
          row["llm_resim_us"] = read_resim_time(llm_resim_out)
        else:
          row["notes"] += f" llm_resim_failed={res.error or res.output[-300:]};"

    row["flow_sim_origin_over_llm"] = ratio(row.get("origin_flow_sim_us"), row.get("llm_flow_sim_us"))
    row["resim_origin_over_llm"] = ratio(row.get("origin_resim_us"), row.get("llm_resim_us"))
    row["status"] = "ok" if not row.get("notes") else "partial"
    rows.append(row)

  write_summary(rows, output_root)
  return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Cross-evaluate LLM and origin SyCCL artifacts under flow-sim-rs and SyCCL resim."
  )
  parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
  parser.add_argument("--origin-root", type=Path, default=DEFAULT_ORIGIN_ROOT)
  parser.add_argument(
      "--output-root",
      type=Path,
      default=DEFAULT_EXPERIMENT_ROOT / "cross_eval",
  )
  parser.add_argument("--flow-sim-bin", type=Path, default=DEFAULT_FLOW_SIM_BIN)
  parser.add_argument("--synthesize-bin", type=Path, default=DEFAULT_SYNTHESIZE_BIN)
  parser.add_argument("--case", dest="cases", action="append", help="Case name to evaluate; may be repeated")
  parser.add_argument("--timeout-s", type=int, default=3600)
  parser.add_argument("--skip-flow-sim", action="store_true")
  parser.add_argument("--skip-resim", action="store_true")
  parser.add_argument("--rerun-existing", action="store_true")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  return run_cross_eval(args)


if __name__ == "__main__":
  raise SystemExit(main())
