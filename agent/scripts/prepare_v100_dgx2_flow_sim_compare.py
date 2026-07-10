#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, NamedTuple, Sequence


AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_ROOT.parent
DATASET_ROOT = AGENT_ROOT / "datasets" / "syccl" / "scheme1_direct_events"
V100_TEMPLATE_DIR = DATASET_ROOT / "templates" / "v100_clos"
V100_INIT_TEMPLATE = V100_TEMPLATE_DIR / "clos_program.py"
INSTRUCTION_TEMPLATE = DATASET_ROOT / "prompt_templete.txt"
FLOW_SIM_ROOT = DATASET_ROOT / "flow-sim-rs"

DEFAULT_ORIGIN_SYCCL = Path("/home/antl/wzd/origin-syccl")
DEFAULT_BUNDLE_ROOT = REPO_ROOT / "experiments" / "v100-dgx2-flow-sim-compare"
DEFAULT_MODEL = "gemini/gemini-2.0-flash"

STRATEGIES = ("linear_rank", "balance", "all")
PRUNE_TYPE = "small"
GPUS_PER_HOST = 16
NICS_PER_HOST = 1
SPINE_COUNT = 1


class CaseSpec(NamedTuple):
  case_id: str
  group: str
  gpu_count: int
  host_count: int
  gpus_per_host: int
  nics_per_host: int
  leaf_count: int
  spine_count: int
  collective: str
  total_message_size: int
  coll_byte: int
  prune_type: str
  a2a: bool


def runexp_v100_total_sizes() -> list[int]:
  sizes = []
  size = 2 ** 10
  while size <= 4 * (2 ** 32):
    sizes.append(size)
    size *= 4
  return sizes


def build_case_specs() -> list[CaseSpec]:
  cases: list[CaseSpec] = []
  for gpu_count, host_count, leaf_count in ((64, 4, 2), (128, 8, 4)):
    for total_size in runexp_v100_total_sizes():
      cases.append(
          CaseSpec(
              case_id=f"v100-dgx2-{gpu_count}gpu-allgather-{total_size}B-prune={PRUNE_TYPE}",
              group=f"v100-dgx2-{host_count}hosts-{GPUS_PER_HOST}gpu-{NICS_PER_HOST}nic-clos/ag",
              gpu_count=gpu_count,
              host_count=host_count,
              gpus_per_host=GPUS_PER_HOST,
              nics_per_host=NICS_PER_HOST,
              leaf_count=leaf_count,
              spine_count=SPINE_COUNT,
              collective="allgather",
              total_message_size=total_size,
              coll_byte=total_size // gpu_count,
              prune_type=PRUNE_TYPE,
              a2a=False,
          )
      )
  return cases


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def config_semantic_sha256(config: dict[str, Any]) -> str:
  normalized = json.loads(json.dumps(config, sort_keys=True))
  normalized.pop("sketch", None)
  payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()


def run_capture(cmd: Sequence[str], *, cwd: Path | None = None, check: bool = False) -> str:
  proc = subprocess.run(
      list(cmd),
      cwd=str(cwd) if cwd else None,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      check=False,
  )
  if check and proc.returncode != 0:
    raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
  return proc.stdout.strip()


def load_config_gen(origin_syccl: Path):
  script_path = origin_syccl / "scripts" / "config_gen.py"
  if not script_path.is_file():
    raise FileNotFoundError(f"missing origin SyCCL config_gen.py: {script_path}")
  spec = importlib.util.spec_from_file_location("origin_syccl_config_gen", script_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"could not import {script_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def render_syccl_config(config_gen: Any, case: CaseSpec, bundle: Path) -> dict[str, Any]:
  cfg = config_gen.ConfigGen()
  result_path = bundle / "origin" / "results" / case.case_id / "result.json"
  sketch_path = bundle / "origin" / "sketches" / f"{case.case_id}-sketch.json"
  config = cfg.v100conf(
      case.coll_byte,
      case.host_count,
      case.gpus_per_host,
      case.nics_per_host,
      case.leaf_count,
      case.spine_count,
      case.prune_type,
      str(result_path),
      str(sketch_path),
      case.a2a,
  )
  if not isinstance(config, dict):
    raise TypeError("ConfigGen.v100conf() did not return a config dict")
  return config


def bw_mbpus_to_gbps(value: float) -> str:
  # Origin SyCCL records this field as MiB per microsecond.
  gbps = value / 1024.0 * 1_000_000.0
  if abs(gbps - round(gbps)) < 1e-9:
    return f"{int(round(gbps))}GB/s"
  return f"{gbps:g}GB/s"


def us(value: float) -> str:
  if abs(value - round(value)) < 1e-9:
    return f"{int(round(value))}us"
  return f"{value:g}us"


def topo_layer(config: dict[str, Any], layer_type: str, *, index: int = 0) -> dict[str, Any]:
  matches = [layer for layer in config.get("topo", []) if layer.get("type") == layer_type]
  if index >= len(matches):
    raise ValueError(f"config missing topo layer type {layer_type!r} at index {index}")
  return matches[index]


def render_topodsl(case: CaseSpec, config: dict[str, Any]) -> str:
  link_spec = config["link_spec"]
  host_layer = topo_layer(config, "host")
  nic_layer = topo_layer(config, "nic")
  switch_layers = [layer for layer in config.get("topo", []) if layer.get("type") == "switch"]
  if len(switch_layers) != 2:
    raise ValueError("V100 DGX-2 Clos config must have leaf and spine switch layers")
  host_link = link_spec[host_layer.get("link_spec", "nvlink")]
  nic_link = link_spec[nic_layer.get("link_spec", "link_nic")]
  leaf_link = link_spec[switch_layers[0].get("link_spec", "netlink_leaf")]
  spine_link = link_spec[switch_layers[1].get("link_spec", "netlink_spine")]
  hosts_per_leaf = case.host_count // case.leaf_count
  return f'''from syccl_agents.base_topology import BaseTopology, LinkSpec, NodeType, Node, CollectiveType, LayerSpec

###TopoBegin
class ClosTopology(BaseTopology):
    """Prompt-facing DGX-2 V100 Clos topology."""
    def __init__(self, gpu_num, coll_bytes, collective: CollectiveType, **kwargs: LayerSpec):
        self.connections = {{}}
        self.gpu_num = gpu_num
        for key, value in kwargs.items():
            if isinstance(value, LayerSpec):
                setattr(self, f"layer_spec_{{value.layer_id}}", value)
        self.coll_bytes = coll_bytes
        self.collective = collective
        self.build_topology()

    def build_topology(self):
        self.f_gpu2gpu()
        self.f_gpu2hostnic()
        self.f_host_leaf()
        self.f_leaf_spine()
        return self.connections

    def f_gpu2gpu(self):
        host_local_spec = getattr(self, "layer_spec_1", None)
        gpu_per_host = self.gpu_num // host_local_spec.group_num
        for host_id in range(host_local_spec.group_num):
            for src in range(gpu_per_host):
                for dst in range(gpu_per_host):
                    if src == dst:
                        continue
                    self.connect(
                        Node(node_id=f"gpu[{{host_id}}][{{src}}]", node_type=NodeType.GPU),
                        Node(node_id=f"gpu[{{host_id}}][{{dst}}]", node_type=NodeType.GPU),
                        host_local_spec.link_spec,
                    )

    def f_gpu2hostnic(self):
        host_local_spec = getattr(self, "layer_spec_1", None)
        host_nic_spec = getattr(self, "layer_spec_2", None)
        gpu_per_host = self.gpu_num // host_local_spec.group_num
        for host_id in range(host_local_spec.group_num):
            for gpu_id in range(gpu_per_host):
                self.connect(
                    Node(node_id=f"gpu[{{host_id}}][{{gpu_id}}]", node_type=NodeType.GPU),
                    Node(node_id=f"host[{{host_id}}]", node_type=host_nic_spec.node_type),
                    host_nic_spec.link_spec,
                )

    def f_host_leaf(self):
        host_nic_spec = getattr(self, "layer_spec_2", None)
        host_leaf_spec = getattr(self, "layer_spec_3", None)
        for leaf_id in range(host_leaf_spec.group_num):
            for host_id in range(leaf_id * host_leaf_spec.node_num, (leaf_id + 1) * host_leaf_spec.node_num):
                self.connect(
                    Node(node_id=f"host[{{host_id}}]", node_type=host_nic_spec.node_type),
                    Node(node_id=f"leaf[{{leaf_id}}]", node_type=NodeType.SWITCH),
                    host_leaf_spec.link_spec,
                )

    def f_leaf_spine(self):
        host_leaf_spec = getattr(self, "layer_spec_3", None)
        leaf_spine_spec = getattr(self, "layer_spec_4", None)
        for leaf_id in range(host_leaf_spec.group_num):
            for spine_id in range(leaf_spine_spec.group_num):
                self.connect(
                    Node(node_id=f"leaf[{{leaf_id}}]", node_type=NodeType.SWITCH),
                    Node(node_id=f"spine[{{spine_id}}]", node_type=NodeType.SWITCH),
                    leaf_spine_spec.link_spec,
                )


topology = ClosTopology(
    {case.gpu_count},
    {case.total_message_size},
    CollectiveType.ALLGATHER,
    layer0=LayerSpec(1, LinkSpec("{bw_mbpus_to_gbps(float(host_link["bw_mbpus"]))}", "{us(float(host_link["lat_us"]))}"), group_num={case.host_count}, node_num={case.gpus_per_host}, node_type=NodeType.GPU),
    layer1=LayerSpec(2, LinkSpec("{bw_mbpus_to_gbps(float(nic_link["bw_mbpus"]))}", "{us(float(nic_link["lat_us"]))}"), group_num={case.host_count}, node_num={case.nics_per_host}, node_type=NodeType.NIC),
    layer2=LayerSpec(3, LinkSpec("{bw_mbpus_to_gbps(float(leaf_link["bw_mbpus"]))}", "{us(float(leaf_link["lat_us"]))}"), group_num={case.leaf_count}, node_num={hosts_per_leaf}, node_type=NodeType.SWITCH),
    layer3=LayerSpec(4, LinkSpec("{bw_mbpus_to_gbps(float(spine_link["bw_mbpus"]))}", "{us(float(spine_link["lat_us"]))}"), group_num={case.spine_count}, node_num={case.leaf_count}, node_type=NodeType.SWITCH),
)
###TopoEND
'''


def prompt_source(topodsl_text: str) -> str:
  begin = topodsl_text.find("###TopoBegin")
  end = topodsl_text.find("###TopoEND")
  if begin < 0 or end < begin:
    return topodsl_text
  return topodsl_text[begin:end + len("###TopoEND")].strip()


def render_instruction(case: CaseSpec, topodsl_text: str, instruction_template: str) -> str:
  values = {
      "GPU_NUM": str(case.gpu_count),
      "Collective": case.collective,
      "COLLECTIVE": case.collective,
      "TOPOLOGY": prompt_source(topodsl_text),
      "TOPOLOGY_FAMILY": "clos",
      "MESSAGE_SIZE": str(case.coll_byte),
      "HOST_NUM": str(case.host_count),
      "HOST_GPU_NUM": str(case.gpus_per_host),
      "NIC_NUM": str(case.nics_per_host),
      "TOPODSL": prompt_source(topodsl_text),
  }
  rendered = Template(instruction_template).safe_substitute(values)
  rendered = rendered.replace(
      "InterHost link via switch like Layer 3",
      "InterHost links via leaf/spine switch layers like Layers 3 and 4",
  )
  rendered = rendered.replace("Use only layers 1，3.", "Use only layers 1, 3, and 4.")
  rendered = rendered.replace("Use only layers 1,3.", "Use only layers 1, 3, and 4.")
  return rendered


def extract_evolve_block(source_text: str, source: Path) -> str:
  lines = source_text.splitlines()
  start_idx = -1
  end_idx = -1
  for idx, line in enumerate(lines):
    if "EVOLVE-BLOCK-START" in line:
      start_idx = idx
    elif "EVOLVE-BLOCK-END" in line:
      end_idx = idx
      break
  if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
    raise ValueError(f"invalid EVOLVE-BLOCK markers in init program: {source}")
  block = "\n".join(lines[start_idx + 1:end_idx]).strip("\n")
  if "def construct_sketches" not in block:
    raise ValueError(f"init program evolve block must define construct_sketches(...): {source}")
  return block


def render_init_program(template_text: str, template_path: Path, case: CaseSpec) -> str:
  evolve_block = extract_evolve_block(template_text, template_path)
  hosts_per_leaf = case.host_count // case.leaf_count
  return f'''"""Generated V100 DGX-2 flow-sim comparison initial program."""


GPU_NUM = {case.gpu_count}
GPUS_PER_HOST = {case.gpus_per_host}
HOSTS_PER_LEAF = {hosts_per_leaf}

# EVOLVE-BLOCK-START
{evolve_block.rstrip()}


# EVOLVE-BLOCK-END


def run_code():
  return construct_sketches(GPU_NUM, gpus_per_host=16, hosts_per_leaf={hosts_per_leaf})
'''


def write_case_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = [
      "case_id",
      "group",
      "gpu_count",
      "host_count",
      "gpus_per_host",
      "nics_per_host",
      "leaf_count",
      "spine_count",
      "collective",
      "total_message_size",
      "coll_byte",
      "config_sha256",
      "config_semantic_sha256",
      "config_path",
      "topodsl_path",
      "instruction_path",
      "init_program_path",
      "origin_result_path",
      "origin_log_path",
      "origin_flow_sim_output_path",
  ]
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_case_inputs(bundle: Path, origin_syccl: Path) -> list[dict[str, Any]]:
  config_gen = load_config_gen(origin_syccl)
  instruction_template = INSTRUCTION_TEMPLATE.read_text(encoding="utf-8")
  init_template = V100_INIT_TEMPLATE.read_text(encoding="utf-8")
  rows: list[dict[str, Any]] = []
  for case in build_case_specs():
    case_dir = bundle / "configs" / case.case_id
    config = render_syccl_config(config_gen, case, bundle)
    config_path = case_dir / "config.json"
    write_json(config_path, config)
    config_sha = sha256_file(config_path)

    topodsl_text = render_topodsl(case, config)
    topodsl_path = case_dir / "topodsl.py"
    topodsl_path.write_text(topodsl_text, encoding="utf-8")

    instruction_path = case_dir / "instruction.txt"
    instruction_path.write_text(render_instruction(case, topodsl_text, instruction_template), encoding="utf-8")

    init_path = case_dir / "init_program.py"
    init_path.write_text(render_init_program(init_template, V100_INIT_TEMPLATE, case), encoding="utf-8")

    origin_log_path = bundle / "origin" / "logs" / f"{case.case_id}.log"
    origin_flow_sim_output = bundle / "origin" / "flow-sim" / case.case_id / "origin-all-flow-sim.json"
    row = {
        **case._asdict(),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "config_semantic_sha256": config_semantic_sha256(config),
        "topodsl_path": str(topodsl_path),
        "instruction_path": str(instruction_path),
        "init_program_path": str(init_path),
        "origin_result_path": config["algo_solve"]["solve_output"],
        "origin_sketch_path": config["sketch"].get("sketch_path", ""),
        "origin_log_path": str(origin_log_path),
        "origin_flow_sim_output_path": str(origin_flow_sim_output),
    }
    rows.append(row)

  write_json(bundle / "configs" / "manifest.json", {"cases": rows})
  write_case_csv(bundle / "configs" / "manifest.csv", rows)
  return rows


def write_executable(path: Path, text: str) -> None:
  path.write_text(text, encoding="utf-8")
  path.chmod(0o755)


def freeze_flow_sim(bundle: Path, source_bin: Path | None) -> dict[str, Any]:
  flow_bin_dir = bundle / "bin"
  flow_bin_dir.mkdir(parents=True, exist_ok=True)
  dest = flow_bin_dir / "flow-sim-rs"
  source = source_bin or FLOW_SIM_ROOT / "target" / "release" / "flow-sim-rs"
  status: dict[str, Any] = {
      "source_path": str(source),
      "bundle_path": str(dest),
      "status": "missing",
  }
  if source.is_file():
    shutil.copy2(source, dest)
    dest.chmod(dest.stat().st_mode | 0o111)
    status.update(
        {
            "status": "ok",
            "sha256": sha256_file(dest),
            "git_commit": run_capture(["git", "-C", str(FLOW_SIM_ROOT), "rev-parse", "HEAD"]),
            "git_diff_stat": run_capture(["git", "-C", str(FLOW_SIM_ROOT), "diff", "--stat", "--", "."]),
        }
    )
  else:
    (bundle / "provenance" / "FLOW_SIM_BUILD_REQUIRED.md").write_text(
        "flow-sim-rs binary was not found during preparation.\n\n"
        f"Expected source: `{source}`\n\n"
        "Build it or rerun preparation with `--flow-sim-bin /path/to/flow-sim-rs`.\n",
        encoding="utf-8",
    )
  write_json(bundle / "provenance" / "flow-sim-rs.json", status)
  return status


def syccl_provenance(origin_syccl: Path) -> dict[str, Any]:
  synthesize = origin_syccl / "build" / "synthesize"
  return {
      "root": str(origin_syccl),
      "git_commit": run_capture(["git", "-C", str(origin_syccl), "rev-parse", "HEAD"]),
      "git_status_short": run_capture(["git", "-C", str(origin_syccl), "status", "--short"]),
      "config_gen_path": str(origin_syccl / "scripts" / "config_gen.py"),
      "runexp_path": str(origin_syccl / "scripts" / "runexp.py"),
      "runexp_reverse_path": str(origin_syccl / "scripts" / "runexp_reverse.py"),
      "runexp_scale_path": str(origin_syccl / "scripts" / "runexp_scale.py"),
      "synthesize_path": str(synthesize),
      "synthesize_exists": synthesize.is_file(),
      "synthesize_sha256": sha256_file(synthesize) if synthesize.is_file() else None,
  }


def write_run_scripts(
    bundle: Path,
    cases: list[dict[str, Any]],
    *,
    origin_syccl: Path,
    max_generations: int,
    k_candidates: int,
    max_parallel: int,
    simpletes_timeout: str,
    origin_timeout: str,
) -> None:
  runs = bundle / "runs"
  runs.mkdir(parents=True, exist_ok=True)

  with (runs / "llm_tasks.tsv").open("w", encoding="utf-8") as handle:
    for row in cases:
      for strategy in STRATEGIES:
        output_path = bundle / "llm" / "outputs" / strategy / row["case_id"] / "checkpoints"
        artifact_dir = bundle / "llm" / "outputs" / strategy / row["case_id"] / "eval_artifacts"
        log_path = bundle / "llm" / "logs" / strategy / f"{row['case_id']}.log"
        handle.write(
            "\t".join(
                [
                    row["case_id"],
                    strategy,
                    row["config_path"],
                    row["instruction_path"],
                    row["init_program_path"],
                    str(output_path),
                    str(artifact_dir),
                    str(log_path),
                ]
            )
            + "\n"
        )

  with (runs / "origin_tasks.tsv").open("w", encoding="utf-8") as handle:
    for row in cases:
      handle.write(
          "\t".join(
              [
                  row["case_id"],
                  row["config_path"],
                  row["origin_result_path"],
                  row["origin_log_path"],
                  row["origin_flow_sim_output_path"],
              ]
          )
          + "\n"
      )

  write_executable(
      runs / "run_llm_one.sh",
      f'''#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 CASE STRATEGY CONFIG INSTRUCTION INIT_PROGRAM OUTPUT_PATH ARTIFACT_DIR LOG_PATH" >&2
  exit 2
fi

CASE_ID="$1"
STRATEGY="$2"
CONFIG_PATH="$3"
INSTRUCTION_PATH="$4"
INIT_PROGRAM_PATH="$5"
OUTPUT_PATH="$6"
ARTIFACT_DIR="$7"
LOG_PATH="$8"

AGENT_ROOT="{AGENT_ROOT}"
FLOW_SIM_BIN="{bundle / 'bin' / 'flow-sim-rs'}"
MODEL_NAME="${{MODEL_NAME:-{DEFAULT_MODEL}}}"
MAX_GENERATIONS="${{MAX_GENERATIONS:-{max_generations}}}"
K_CANDIDATES="${{K_CANDIDATES:-{k_candidates}}}"
SIMPLETES_TIMEOUT="${{SIMPLETES_TIMEOUT:-{simpletes_timeout}}}"
EVAL_CONCURRENCY="${{EVAL_CONCURRENCY:-1}}"
GEN_CONCURRENCY="${{GEN_CONCURRENCY:-1}}"
LLM_POLICY_POOL_SIZE="${{LLM_POLICY_POOL_SIZE:-100}}"
EXTRA_SIMPLETES_ARGS="${{EXTRA_SIMPLETES_ARGS:-}}"

mkdir -p "$OUTPUT_PATH" "$ARTIFACT_DIR" "$(dirname "$LOG_PATH")"
cd "$AGENT_ROOT"

export PYTHONPATH="$AGENT_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
export SYCCL_BASE_CONFIG="$CONFIG_PATH"
export SYCCL_EVAL_ARTIFACT_DIR="$ARTIFACT_DIR"
export FLOW_SIM_BIN="$FLOW_SIM_BIN"
export SYCCL_EVALUATOR_TIMEOUT_SECONDS="${{SYCCL_EVALUATOR_TIMEOUT_SECONDS:-3600}}"

echo "[v100-dgx2:llm] case=$CASE_ID strategy=$STRATEGY config=$CONFIG_PATH"

timeout "$SIMPLETES_TIMEOUT" uv run python main.py \\
  --init-program "$INIT_PROGRAM_PATH" \\
  --evaluator "$AGENT_ROOT/datasets/syccl/scheme1_direct_events/evaluator.py" \\
  --instruction "$INSTRUCTION_PATH" \\
  --selector llm_elite \\
  --elite-selection-strategy "$STRATEGY" \\
  --num-chains 1 \\
  --k-candidates "$K_CANDIDATES" \\
  --stream-k-candidates \\
  --max-generations "$MAX_GENERATIONS" \\
  --eval-concurrency "$EVAL_CONCURRENCY" \\
  --gen-concurrency "$GEN_CONCURRENCY" \\
  --init-eval-repeats 1 \\
  --llm-policy-pool-size "$LLM_POLICY_POOL_SIZE" \\
  --model "$MODEL_NAME" \\
  --output-path "$OUTPUT_PATH" \\
  --save-llm-io \\
  --disable-reflection \\
  --skip-preflight \\
  $EXTRA_SIMPLETES_ARGS 2>&1 | tee "$LOG_PATH"
''',
  )

  write_executable(
      runs / "run_llm_all.sh",
      f'''#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
xargs -P "${{LLM_MAX_PARALLEL:-{max_parallel}}}" -n 8 "$SCRIPT_DIR/run_llm_one.sh" < "$SCRIPT_DIR/llm_tasks.tsv"
''',
  )

  write_executable(
      runs / "run_origin_one.sh",
      f'''#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 CASE CONFIG RESULT LOG FLOW_SIM_OUTPUT" >&2
  exit 2
fi

CASE_ID="$1"
CONFIG_PATH="$2"
RESULT_PATH="$3"
LOG_PATH="$4"
FLOW_SIM_OUTPUT="$5"
ORIGIN_SYCCL="{origin_syccl}"
SYNTHESIZE_BIN="${{SYNTHESIZE_BIN:-$ORIGIN_SYCCL/build/synthesize}}"
ORIGIN_SOLVE_TIMEOUT="${{ORIGIN_SOLVE_TIMEOUT:-{origin_timeout}}}"
export SYNTHESIZE_PARALLEL_THREAD_NUM="${{SYNTHESIZE_PARALLEL_THREAD_NUM:-72}}"

mkdir -p "$(dirname "$RESULT_PATH")" "$(dirname "$LOG_PATH")" "$(dirname "$FLOW_SIM_OUTPUT")"
cd "$ORIGIN_SYCCL"

echo "[v100-dgx2:origin] case=$CASE_ID config=$CONFIG_PATH"
timeout "$ORIGIN_SOLVE_TIMEOUT" "$SYNTHESIZE_BIN" -f "$CONFIG_PATH" solve 2>&1 | tee "$LOG_PATH"
''',
  )

  write_executable(
      runs / "run_origin_all.sh",
      f'''#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
xargs -P "${{ORIGIN_MAX_PARALLEL:-1}}" -n 5 "$SCRIPT_DIR/run_origin_one.sh" < "$SCRIPT_DIR/origin_tasks.tsv"
''',
  )

  write_executable(
      runs / "run_origin_flow_sim_all.py",
      f'''#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

BUNDLE = Path({str(bundle)!r})
FLOW_SIM = BUNDLE / "bin" / "flow-sim-rs"
TASKS = BUNDLE / "runs" / "origin_tasks.tsv"
OUT_ROOT = BUNDLE / "origin" / "flow-sim"

rows = []
with TASKS.open("r", encoding="utf-8") as handle:
  for raw in handle:
    case_id, config_path, result_path, _log_path, output_path = raw.rstrip("\\n").split("\\t")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = Path(result_path)
    if not result.exists():
      rows.append({{"case_id": case_id, "status": "missing_origin_result", "origin_result": str(result)}})
      continue
    started = time.time()
    proc = subprocess.run(
        [str(FLOW_SIM), "simulate", "--config", config_path, "--translated", str(result), "--output", str(output)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = time.time() - started
    if proc.returncode != 0:
      rows.append({{"case_id": case_id, "status": "flow_sim_failed", "elapsed_s": elapsed, "error": proc.stdout[-1000:]}})
      continue
    rows.append({{"case_id": case_id, "status": "ok", "elapsed_s": elapsed, "origin_flow_sim_output": str(output)}})

summary_json = OUT_ROOT / "summary.json"
summary_csv = OUT_ROOT / "summary.csv"
summary_json.parent.mkdir(parents=True, exist_ok=True)
summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
fields = sorted({{key for row in rows for key in row}})
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
  writer = csv.DictWriter(handle, fieldnames=fields)
  writer.writeheader()
  for row in rows:
    writer.writerow(row)
print(f"wrote {{summary_json}}")
''',
  )


def write_launch_manifest(
    bundle: Path,
    *,
    cases: list[dict[str, Any]],
    syccl: dict[str, Any],
    flow_sim: dict[str, Any],
    max_generations: int,
    k_candidates: int,
    max_parallel: int,
    simpletes_timeout: str,
    origin_timeout: str,
) -> None:
  manifest = {
      "experiment": "v100-dgx2-flow-sim-compare",
      "launch_id": bundle.name,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "case_count": len(cases),
      "strategies": list(STRATEGIES),
      "max_generations": max_generations,
      "k_candidates": k_candidates,
      "max_parallel": max_parallel,
      "simpletes_timeout": simpletes_timeout,
      "origin_timeout": origin_timeout,
      "syccl": syccl,
      "flow_sim": flow_sim,
      "llm_ccl": {
          "repo": str(REPO_ROOT),
          "git_commit": run_capture(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
          "dirty_diffstat": run_capture(["git", "-C", str(REPO_ROOT), "diff", "--stat"]),
      },
      "notes": [
          "Preparation only. This bundle does not run origin SyCCL or llm-ccl.",
          "V100 uses DGX-2-style 16 GPUs per host and one NIC per host.",
          "64 GPU case is 4 hosts; 128 GPU case is 8 hosts.",
          "Canonical configs are generated by origin-syccl ConfigGen.v100conf().",
      ],
  }
  write_json(bundle / "manifest.json", manifest)


def prepare_bundle(
    *,
    bundle_root: Path = DEFAULT_BUNDLE_ROOT,
    launch_id: str | None = None,
    origin_syccl: Path = DEFAULT_ORIGIN_SYCCL,
    flow_sim_bin: Path | None = None,
    max_generations: int = 10000,
    k_candidates: int = 4,
    max_parallel: int = 3,
    simpletes_timeout: str = "2h",
    origin_timeout: str = "10h",
) -> Path:
  launch = launch_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
  bundle = (bundle_root / launch).expanduser().resolve()
  bundle.mkdir(parents=True, exist_ok=True)
  for directory in (
      bundle / "provenance",
      bundle / "llm" / "outputs",
      bundle / "llm" / "logs",
      bundle / "origin" / "results",
      bundle / "origin" / "logs",
      bundle / "origin" / "flow-sim",
      bundle / "reports",
  ):
    directory.mkdir(parents=True, exist_ok=True)

  origin = origin_syccl.expanduser().resolve()
  if not origin.exists():
    raise FileNotFoundError(f"origin SyCCL checkout not found: {origin}")
  flow_sim = freeze_flow_sim(bundle, flow_sim_bin.expanduser().resolve() if flow_sim_bin else None)
  syccl = syccl_provenance(origin)
  write_json(bundle / "provenance" / "origin-syccl.json", syccl)
  cases = write_case_inputs(bundle, origin)
  write_run_scripts(
      bundle,
      cases,
      origin_syccl=origin,
      max_generations=max_generations,
      k_candidates=k_candidates,
      max_parallel=max_parallel,
      simpletes_timeout=simpletes_timeout,
      origin_timeout=origin_timeout,
  )
  write_launch_manifest(
      bundle,
      cases=cases,
      syccl=syccl,
      flow_sim=flow_sim,
      max_generations=max_generations,
      k_candidates=k_candidates,
      max_parallel=max_parallel,
      simpletes_timeout=simpletes_timeout,
      origin_timeout=origin_timeout,
  )
  return bundle


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Prepare V100 DGX-2 llm-ccl vs SyCCL flow-sim comparison bundle.")
  parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
  parser.add_argument("--launch-id", default=None)
  parser.add_argument("--origin-syccl", type=Path, default=DEFAULT_ORIGIN_SYCCL)
  parser.add_argument("--flow-sim-bin", type=Path, default=None)
  parser.add_argument("--max-generations", type=int, default=10000)
  parser.add_argument("--k-candidates", type=int, default=4)
  parser.add_argument("--max-parallel", type=int, default=3)
  parser.add_argument("--simpletes-timeout", default="2h")
  parser.add_argument("--origin-timeout", default="10h")
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  bundle = prepare_bundle(
      bundle_root=args.bundle_root,
      launch_id=args.launch_id,
      origin_syccl=args.origin_syccl,
      flow_sim_bin=args.flow_sim_bin,
      max_generations=args.max_generations,
      k_candidates=args.k_candidates,
      max_parallel=args.max_parallel,
      simpletes_timeout=args.simpletes_timeout,
      origin_timeout=args.origin_timeout,
  )
  print(f"Wrote V100 DGX-2 comparison preparation bundle: {bundle}")
  print(f"Inspect manifest: {bundle / 'manifest.json'}")
  print(f"Config manifest: {bundle / 'configs' / 'manifest.json'}")
  print(f"LLM tasks: {bundle / 'runs' / 'llm_tasks.tsv'}")
  print(f"Origin tasks: {bundle / 'runs' / 'origin_tasks.tsv'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
