#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Iterable, NamedTuple


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
  sys.path.insert(0, str(AGENT_ROOT))

from syccl_agents.topodsl import TopoDSLSpec, load_topodsl


DATASET_ROOT = AGENT_ROOT / "datasets" / "syccl" / "scheme1_direct_events"
DEFAULT_TEMPLATE_DIR = DATASET_ROOT / "templates" / "v100_dgx2_clos"
DEFAULT_INSTRUCTION_TEMPLATE = DEFAULT_TEMPLATE_DIR / "prompt_template.txt"
DEFAULT_OUTPUT_ROOT = (
    AGENT_ROOT
    / "result"
    / "search"
    / f"v100-dgx2-clos-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
)
DEFAULT_FLOW_SIM_BIN = DATASET_ROOT / "flow-sim-rs" / "target" / "release" / "flow-sim-rs"
TOPOLOGY_TEMPLATE_NAME = "clos_topo.py"
CONFIG_TEMPLATE_NAME = "clos_v100_config.json"
INIT_PROGRAM_TEMPLATE_NAME = "clos_program.py"

TOTAL_MESSAGE_SIZES = tuple(4 ** power for power in range(8, 20))

STRATEGIES = ("linear_rank", "balance", "all")
MODEL_NAME = "hosted_vllm/deepseek-ai/DeepSeek-V4-Flash"
API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_MAX_TOKENS = 32768
DEFAULT_K_CANDIDATES = 4


class CaseSpec(NamedTuple):
  name: str
  total_message_size: int
  config_coll_byte: int


class ScaleSpec(NamedTuple):
  hosts: int

  @property
  def gpus_per_host(self) -> int:
    return 16

  @property
  def nics_per_host(self) -> int:
    return 1

  @property
  def spine_switches(self) -> int:
    return 1

  @property
  def total_gpus(self) -> int:
    return self.hosts * self.gpus_per_host

  @property
  def leaf_switches(self) -> int:
    return self.hosts

  @property
  def name(self) -> str:
    return f"{self.hosts}hosts-{self.total_gpus}gpu"

  @property
  def experiment_rel(self) -> Path:
    return Path(
        f"v100-dgx2-{self.hosts}hosts-{self.gpus_per_host}gpu-{self.nics_per_host}nic-clos"
    ) / "ag"


SCALE_SPECS = (ScaleSpec(4), ScaleSpec(8))


def scale_summary(scales: Iterable[ScaleSpec] = SCALE_SPECS) -> str:
  return " and ".join(f"{scale.hosts}-host" for scale in scales)


def bundle_experiment_id(scales: Iterable[ScaleSpec] = SCALE_SPECS) -> str:
  scale_slug = "-".join(f"{scale.hosts}host" for scale in scales)
  return f"v100-dgx2-clos-{scale_slug}/ag"


def size_label(size: int) -> str:
  for suffix, multiplier in (("g", 1024 ** 3), ("m", 1024 ** 2), ("k", 1024)):
    if size >= multiplier and size % multiplier == 0:
      return f"{size // multiplier}{suffix}"
  return f"{size}b"


def parse_message_size(value: str) -> int:
  normalized = value.strip().lower()
  multipliers = {
      "b": 1,
      "k": 1024,
      "kb": 1024,
      "m": 1024 ** 2,
      "mb": 1024 ** 2,
      "g": 1024 ** 3,
      "gb": 1024 ** 3,
  }
  for suffix, multiplier in sorted(multipliers.items(), key=lambda item: len(item[0]), reverse=True):
    if normalized.endswith(suffix):
      number = normalized[: -len(suffix)]
      return int(number) * multiplier
  return int(normalized)


def parse_message_sizes(value: str) -> tuple[int, ...]:
  return tuple(parse_message_size(part) for part in value.split(",") if part.strip())


def build_case_specs(
    scale: ScaleSpec,
    message_sizes: Iterable[int] = TOTAL_MESSAGE_SIZES,
) -> list[CaseSpec]:
  sizes = tuple(int(size) for size in message_sizes)
  if not sizes:
    raise ValueError("message size sweep must not be empty")
  for size in sizes:
    if size <= 0:
      raise ValueError(f"message size must be positive: {size}")
    if size % scale.total_gpus != 0:
      raise ValueError(f"message size must be divisible by total GPU count {scale.total_gpus}: {size}")
  return [
      CaseSpec(
          name=f"{size_label(total_message_size)}-total",
          total_message_size=total_message_size,
          config_coll_byte=total_message_size // scale.total_gpus,
      )
      for total_message_size in sizes
  ]


def resolve_template_dir(template_dir: Path | None = None) -> Path:
  resolved = (template_dir if template_dir is not None else DEFAULT_TEMPLATE_DIR).expanduser().resolve()
  if not resolved.is_dir():
    raise FileNotFoundError(f"V100 DGX-2 Topology Template directory not found: {resolved}")
  return resolved


def default_init_program_for_template_dir(template_dir: Path) -> Path:
  candidate = template_dir / INIT_PROGRAM_TEMPLATE_NAME
  if not candidate.is_file():
    raise FileNotFoundError(f"init program template not found: {candidate}")
  return candidate


def render_topodsl(case: CaseSpec, template_text: str, scale: ScaleSpec) -> str:
  if scale.hosts % scale.leaf_switches != 0:
    raise ValueError("host count must be divisible by leaf switch count for V100 Clos template generation")
  instantiation = f'''topology = DGX2ClosTopology(
    {scale.total_gpus},
    {case.total_message_size},
    CollectiveType.ALLGATHER,
    layer1=LayerSpec(1, LinkSpec("125GB/s", "3us"), group_num={scale.hosts}, node_num={scale.gpus_per_host}, node_type=NodeType.GPU),
    layer2=LayerSpec(2, LinkSpec("12.5GB/s", "0us"), group_num={scale.hosts}, node_num={scale.nics_per_host}, node_type=NodeType.NIC),
    layer3=LayerSpec(3, LinkSpec("12.5GB/s", "25us"), group_num={scale.leaf_switches}, node_num={scale.hosts // scale.leaf_switches}, node_type=NodeType.SWITCH),
    layer4=LayerSpec(4, LinkSpec("400GB/s", "25us"), group_num={scale.spine_switches}, node_num={scale.leaf_switches}, node_type=NodeType.SWITCH),
)
'''
  rendered, count = re.subn(
      r"(?ms)^topology\s*=\s*DGX2ClosTopology\(\n.*?\n\)\s*(?=###TopoEND)",
      instantiation,
      template_text,
      count=1,
  )
  if count != 1:
    raise ValueError("V100 DGX-2 Clos topology template must contain one topology = DGX2ClosTopology(...) block")
  return rendered


def total_gpus(topo: TopoDSLSpec) -> int:
  return topo.params.hosts * topo.params.gpus_per_host


def build_resimulation_config(case: CaseSpec, template_config: dict, scale: ScaleSpec) -> dict:
  config = copy.deepcopy(template_config)
  config["coll"] = {
      "name": "allgather",
      "byte": case.config_coll_byte,
      "root_sender": -1,
      "root_receiver": -1,
  }
  config["hosts"] = {
      "host_num": scale.hosts,
      "host_gpu_num": scale.gpus_per_host,
      "host_nic_num": scale.nics_per_host,
      "host_links": "nvswitch",
  }
  config.pop("host_links", None)
  _set_layer_link_spec(config, layer_id=1, link_spec="nvswitch")
  _set_switch_num(config, layer_id=3, switch_num=scale.leaf_switches)
  _set_switch_num(config, layer_id=4, switch_num=scale.spine_switches)
  config["link_spec"]["nvswitch"] = {"bw_mbpus": 0.125, "lat_us": 3}
  config["link_spec"]["link_nic"] = {"bw_mbpus": 0.0125, "lat_us": 0}
  config["link_spec"]["netlink_leaf"] = {"bw_mbpus": 0.0125, "lat_us": 25}
  config["link_spec"]["netlink_spine"] = {"bw_mbpus": 0.4, "lat_us": 25}
  config["sketch"] = {
      "customize_sketch": False,
      "use_sketch_input": False,
      "save_sketch": False,
      "sketch_path": "",
  }
  return config


def _set_switch_num(config: dict, *, layer_id: int, switch_num: int) -> None:
  for layer in config.get("topo", []):
    if isinstance(layer, dict) and layer.get("layer_id") == layer_id:
      layer["switch_num"] = switch_num
      return
  raise ValueError(f"Resimulation Config Template missing topo layer_id={layer_id}")


def _set_layer_link_spec(config: dict, *, layer_id: int, link_spec: str) -> None:
  for layer in config.get("topo", []):
    if isinstance(layer, dict) and layer.get("layer_id") == layer_id:
      layer["link_spec"] = link_spec
      return
  raise ValueError(f"Resimulation Config Template missing topo layer_id={layer_id}")


def render_instruction(topo: TopoDSLSpec, template_path: Path) -> str:
  params = topo.params
  values = {
      "GPU_NUM": str(total_gpus(topo)),
      "Collective": params.collective,
      "COLLECTIVE": params.collective,
      "TOPOLOGY": topo.prompt_source,
      "TOPOLOGY_FAMILY": params.family,
      "MESSAGE_SIZE": str(params.message_size),
      "HOST_NUM": str(params.hosts),
      "HOST_GPU_NUM": str(params.gpus_per_host),
      "NIC_NUM": str(params.nics_per_host),
      "TOPODSL": topo.prompt_source,
  }
  return Template(template_path.read_text(encoding="utf-8")).safe_substitute(values)


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
    raise ValueError(f"init program evolve block must define construct_sketches: {source}")
  return block


def write_init_program(source_path: Path, output_path: Path, *, gpu_num: int) -> Path:
  evolve_block = extract_evolve_block(source_path.read_text(encoding="utf-8"), source_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
      f'''"""Generated SyCCL V100 DGX-2 Clos initial program."""


GPU_NUM = {int(gpu_num)}

# EVOLVE-BLOCK-START
{evolve_block.rstrip()}


# EVOLVE-BLOCK-END


def run_code():
  return construct_sketches(GPU_NUM)
''',
      encoding="utf-8",
  )
  return output_path


def _append_no_proxy_shell() -> str:
  return '''for host in 127.0.0.1 localhost; do
  case ",${NO_PROXY:-}," in
    *",$host,"*) ;;
    *) export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$host" ;;
  esac
  case ",${no_proxy:-}," in
    *",$host,"*) ;;
    *) export no_proxy="${no_proxy:+$no_proxy,}$host" ;;
  esac
done'''


def write_run_scripts(
    base_dir: Path,
    task_rows: list[dict[str, str]],
    *,
    max_parallel: int,
    max_generations: int,
    k_candidates: int,
    per_run_timeout: str,
    model: str,
    api_base: str,
    max_tokens: int,
    flow_sim_bin: Path,
) -> None:
  runs_dir = base_dir / "runs"
  runs_dir.mkdir(parents=True, exist_ok=True)
  tasks_path = runs_dir / "tasks.tsv"
  with tasks_path.open("w", encoding="utf-8") as f:
    for row in task_rows:
      f.write(
          "\t".join(
              [
                  row["case_id"],
                  row["strategy"],
                  row["config_path"],
                  row["instruction_path"],
                  row["init_program_path"],
                  row["output_path"],
                  row["artifact_dir"],
                  row["log_path"],
                  row["topodsl_path"],
              ]
          )
          + "\n"
      )

  run_one = runs_dir / "run_one.sh"
  run_one.write_text(
      f'''#!/usr/bin/env bash
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
STATUS_PATH="${{LOG_PATH%.log}}.status"

AGENT_ROOT="{AGENT_ROOT}"
MAX_GENERATIONS="${{MAX_GENERATIONS:-{max_generations}}}"
K_CANDIDATES="${{K_CANDIDATES:-{k_candidates}}}"
EVAL_CONCURRENCY="${{EVAL_CONCURRENCY:-1}}"
GEN_CONCURRENCY="${{GEN_CONCURRENCY:-1}}"
LLM_POLICY_POOL_SIZE="${{LLM_POLICY_POOL_SIZE:-100}}"
SIMPLETES_TIMEOUT="${{SIMPLETES_TIMEOUT:-{per_run_timeout}}}"
EXTRA_SIMPLETES_ARGS="${{EXTRA_SIMPLETES_ARGS:-}}"
MODEL_NAME="${{MODEL_NAME:-{model}}}"
API_BASE="${{API_BASE:-{api_base}}}"
API_KEY="${{API_KEY:-${{OPENAI_API_KEY:-sk-nokey}}}}"
MAX_TOKENS="${{MAX_TOKENS:-{max_tokens}}}"
FLOW_SIM_BIN="${{FLOW_SIM_BIN:-{flow_sim_bin}}}"

mkdir -p "$OUTPUT_PATH" "$ARTIFACT_DIR" "$(dirname "$LOG_PATH")"
cd "$AGENT_ROOT"

export PYTHONPATH="$AGENT_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
export SYCCL_BASE_CONFIG="$CONFIG_PATH"
export SYCCL_EVAL_ARTIFACT_DIR="$ARTIFACT_DIR"
export FLOW_SIM_BIN="$FLOW_SIM_BIN"
export SYCCL_EVALUATOR_TIMEOUT_SECONDS="${{SYCCL_EVALUATOR_TIMEOUT_SECONDS:-3600}}"
export OPENAI_API_KEY="${{OPENAI_API_KEY:-sk-nokey}}"
export UV_CACHE_DIR="${{UV_CACHE_DIR:-/tmp/uv-cache}}"
{_append_no_proxy_shell()}

echo "[syccl-v100-dgx2-clos] case=$CASE_NAME strategy=$STRATEGY config=$CONFIG_PATH topodsl=$TOPODSL_PATH"
START_EPOCH="$(date +%s)"
START_TIME="$(date -Iseconds)"
{{
  echo "case=$CASE_NAME"
  echo "strategy=$STRATEGY"
  echo "start_epoch=$START_EPOCH"
  echo "start_time=$START_TIME"
  echo "per_run_timeout=$SIMPLETES_TIMEOUT"
  echo "config=$CONFIG_PATH"
  echo "topodsl=$TOPODSL_PATH"
  echo "flow_sim_bin=$FLOW_SIM_BIN"
}} > "$STATUS_PATH"

set +e
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
  --api-base "$API_BASE" \\
  --api-key "$API_KEY" \\
  --max-tokens "$MAX_TOKENS" \\
  --output-path "$OUTPUT_PATH" \\
  --save-llm-io \\
  --skip-preflight \\
  $EXTRA_SIMPLETES_ARGS 2>&1 | tee "$LOG_PATH"
status="${{PIPESTATUS[0]}}"
set -e
END_EPOCH="$(date +%s)"
END_TIME="$(date -Iseconds)"
{{
  echo "end_epoch=$END_EPOCH"
  echo "end_time=$END_TIME"
  echo "elapsed_s=$((END_EPOCH - START_EPOCH))"
  echo "exit_status=$status"
}} >> "$STATUS_PATH"
exit "$status"
''',
      encoding="utf-8",
  )

  run_all = runs_dir / "run_all.sh"
  run_all.write_text(
      f'''#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
TASKS="$SCRIPT_DIR/tasks.tsv"

xargs -P "${{LLM_MAX_PARALLEL:-{max_parallel}}}" -n 9 "$SCRIPT_DIR/run_one.sh" < "$TASKS"
''',
      encoding="utf-8",
  )
  run_one.chmod(0o755)
  run_all.chmod(0o755)


def write_start_commands(
    bundle_root: Path,
    *,
    model: str,
    api_base: str,
    max_tokens: int,
    task_count: int,
    scales: Iterable[ScaleSpec] = SCALE_SPECS,
) -> Path:
  start_commands = bundle_root / "START_COMMANDS.md"
  run_all = bundle_root / "runs" / "run_all.sh"
  nohup_log = bundle_root / "runs" / "run_all.nohup.log"
  pid_file = bundle_root / "runs" / "run_all.pid"
  start_commands.write_text(
      f'''# V100 DGX-2 Clos Search Launch Commands

This {scale_summary(scales)} search launch bundle contains {task_count} tasks and is prepared but not started.

Bundle root: `{bundle_root}`

## Start

```bash
cd {AGENT_ROOT}
setsid nohup env \\
  OPENAI_API_KEY=sk-nokey \\
  MODEL_NAME={model} \\
  API_BASE={api_base} \\
  MAX_TOKENS={max_tokens} \\
  {run_all} \\
  > {nohup_log} 2>&1 &
echo $! > {pid_file}
```

## Monitor

```bash
tail -f {nohup_log}
```

```bash
find {bundle_root} -path '*/logs/*.status' -print -exec cat {{}} \\;
```
''',
      encoding="utf-8",
  )
  return start_commands


def prepare_scale_inputs(
    *,
    scale: ScaleSpec,
    bundle_root: Path,
    topology_template: str,
    resimulation_config_template: dict,
    resolved_instruction_template: Path,
    init_program_source: Path,
    message_sizes: Iterable[int],
) -> tuple[dict, list[dict[str, str]]]:
  base_dir = bundle_root / scale.experiment_rel
  topodsl_dir = base_dir / "topodsl"
  config_dir = base_dir / "flow-sim-configs"
  instruction_dir = base_dir / "instructions"
  init_dir = base_dir / "init_programs"
  output_dir = base_dir / "outputs"
  log_dir = base_dir / "logs"
  for directory in (topodsl_dir, config_dir, instruction_dir, init_dir, output_dir, log_dir):
    directory.mkdir(parents=True, exist_ok=True)

  init_program_path = write_init_program(
      init_program_source,
      init_dir / "init_program.py",
      gpu_num=scale.total_gpus,
  )
  case_records = []
  run_records = []
  for case in build_case_specs(scale, message_sizes):
    topodsl_path = topodsl_dir / f"{case.name}-topodsl.py"
    topodsl_path.write_text(render_topodsl(case, topology_template, scale), encoding="utf-8")
    topo = load_topodsl(topodsl_path)
    if total_gpus(topo) != scale.total_gpus:
      raise ValueError(f"TopoDSL {topodsl_path} did not resolve to {scale.total_gpus} GPUs")
    if topo.params.message_size != case.config_coll_byte:
      raise ValueError(
          f"TopoDSL {topodsl_path} resolved coll.byte={topo.params.message_size}, "
          f"expected {case.config_coll_byte}"
      )

    config_path = config_dir / f"{case.name}-config.json"
    config_path.write_text(
        json.dumps(build_resimulation_config(case, resimulation_config_template, scale), indent=2),
        encoding="utf-8",
    )
    case_instruction_dir = instruction_dir / case.name
    case_instruction_dir.mkdir(parents=True, exist_ok=True)
    instruction_path = case_instruction_dir / "syccl_instruction.txt"
    instruction_path.write_text(render_instruction(topo, resolved_instruction_template), encoding="utf-8")

    case_record = {
        "case": case.name,
        "total_message_size": case.total_message_size,
        "config_coll_byte": case.config_coll_byte,
        "topodsl_path": str(topodsl_path),
        "config_path": str(config_path),
        "instruction_path": str(instruction_path),
        "init_program_path": str(init_program_path),
    }
    case_records.append(case_record)
    for strategy in STRATEGIES:
      run_record = {
          "scale": scale.name,
          "case_id": f"{scale.name}/{case.name}",
          "case": case.name,
          "strategy": strategy,
          "config_path": str(config_path),
          "instruction_path": str(instruction_path),
          "init_program_path": str(init_program_path),
          "output_path": str(output_dir / strategy / case.name / "checkpoints"),
          "artifact_dir": str(output_dir / strategy / case.name / "eval_artifacts"),
          "log_path": str(log_dir / strategy / f"{case.name}.log"),
          "topodsl_path": str(topodsl_path),
      }
      run_records.append(run_record)

  scale_manifest_path = base_dir / "manifest.json"
  scale_record = {
      "name": scale.name,
      "experiment": str(scale.experiment_rel),
      "hosts": scale.hosts,
      "gpus_per_host": scale.gpus_per_host,
      "nics_per_host": scale.nics_per_host,
      "leaf_switches": scale.leaf_switches,
      "spine_switches": scale.spine_switches,
      "total_gpus": scale.total_gpus,
      "collective": "allgather",
      "manifest_path": str(scale_manifest_path),
      "cases": case_records,
      "runs": run_records,
  }
  scale_manifest_path.write_text(json.dumps(scale_record, indent=2), encoding="utf-8")
  return scale_record, run_records


def prepare_experiment(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    template_dir: Path | None = None,
    init_program: Path | None = None,
    instruction_template: Path = DEFAULT_INSTRUCTION_TEMPLATE,
    max_parallel: int = 2,
    max_generations: int = 10000,
    k_candidates: int = DEFAULT_K_CANDIDATES,
    per_run_timeout: str = "2h",
    model: str = MODEL_NAME,
    api_base: str = API_BASE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    flow_sim_bin: Path | None = None,
    message_sizes: Iterable[int] = TOTAL_MESSAGE_SIZES,
) -> Path:
  bundle_root = output_root.expanduser().resolve()
  bundle_root.mkdir(parents=True, exist_ok=True)
  message_sizes = tuple(int(size) for size in message_sizes)
  resolved_template_dir = resolve_template_dir(template_dir)
  resolved_instruction_template = instruction_template.expanduser().resolve()
  if not resolved_instruction_template.is_file():
    raise FileNotFoundError(f"instruction template not found: {resolved_instruction_template}")
  init_program_source = (
      init_program.expanduser().resolve()
      if init_program is not None
      else default_init_program_for_template_dir(resolved_template_dir).resolve()
  )
  topology_template_path = resolved_template_dir / TOPOLOGY_TEMPLATE_NAME
  resimulation_config_template_path = resolved_template_dir / CONFIG_TEMPLATE_NAME
  topology_template = topology_template_path.read_text(encoding="utf-8")
  with resimulation_config_template_path.open("r", encoding="utf-8") as f:
    resimulation_config_template = json.load(f)

  source_flow_sim_bin = (
      flow_sim_bin.expanduser().resolve()
      if flow_sim_bin is not None
      else DEFAULT_FLOW_SIM_BIN.resolve()
  )
  if not source_flow_sim_bin.is_file():
    raise FileNotFoundError(f"flow-sim-rs binary not found: {source_flow_sim_bin}")
  resolved_flow_sim_bin = bundle_root / "bin" / "flow-sim-rs"
  resolved_flow_sim_bin.parent.mkdir(parents=True, exist_ok=True)
  if source_flow_sim_bin != resolved_flow_sim_bin:
    shutil.copy2(source_flow_sim_bin, resolved_flow_sim_bin)
  resolved_flow_sim_bin.chmod(0o755)

  scale_records = []
  run_records = []
  for scale in SCALE_SPECS:
    scale_record, scale_runs = prepare_scale_inputs(
        scale=scale,
        bundle_root=bundle_root,
        topology_template=topology_template,
        resimulation_config_template=resimulation_config_template,
        resolved_instruction_template=resolved_instruction_template,
        init_program_source=init_program_source,
        message_sizes=message_sizes,
    )
    scale_records.append(scale_record)
    run_records.extend(scale_runs)

  write_run_scripts(
      bundle_root,
      run_records,
      max_parallel=max_parallel,
      max_generations=max_generations,
      k_candidates=k_candidates,
      per_run_timeout=per_run_timeout,
      model=model,
      api_base=api_base,
      max_tokens=max_tokens,
      flow_sim_bin=resolved_flow_sim_bin,
  )

  manifest = {
      "experiment": bundle_experiment_id(),
      "collective": "allgather",
      "strategies": list(STRATEGIES),
      "max_parallel": max_parallel,
      "per_run_timeout": per_run_timeout,
      "max_generations": max_generations,
      "k_candidates": k_candidates,
      "model": model,
      "api_base": api_base,
      "max_tokens": max_tokens,
      "flow_sim_bin": str(resolved_flow_sim_bin),
      "flow_sim_source": str(source_flow_sim_bin),
      "openai_api_key_default": "sk-nokey",
      "total_message_sizes": list(message_sizes),
      "template_dir": str(resolved_template_dir),
      "topology_template": str(topology_template_path),
      "resimulation_config_template": str(resimulation_config_template_path),
      "init_program_source": str(init_program_source),
      "instruction_template": str(resolved_instruction_template),
      "scales": scale_records,
      "runs": run_records,
  }
  manifest_path = bundle_root / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  write_start_commands(
      bundle_root,
      model=model,
      api_base=api_base,
      max_tokens=max_tokens,
      task_count=len(run_records),
      scales=SCALE_SPECS,
  )
  return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Prepare SyCCL V100 DGX-2 Clos llm_elite sweep inputs")
  parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
  parser.add_argument("--template-dir", type=Path, default=None)
  parser.add_argument("--init-program", type=Path, default=None)
  parser.add_argument("--instruction-template", type=Path, default=DEFAULT_INSTRUCTION_TEMPLATE)
  parser.add_argument("--max-parallel", type=int, default=2)
  parser.add_argument("--max-generations", type=int, default=10000)
  parser.add_argument("--k-candidates", type=int, default=DEFAULT_K_CANDIDATES)
  parser.add_argument("--per-run-timeout", default="2h")
  parser.add_argument("--model", default=MODEL_NAME)
  parser.add_argument("--api-base", default=API_BASE)
  parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
  parser.add_argument("--flow-sim-bin", type=Path, default=None)
  parser.add_argument(
      "--message-sizes",
      default=None,
      help="Comma-separated total message sizes, for example: 1k,2k,4k,1m,4g",
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  manifest_path = prepare_experiment(
      output_root=args.output_root,
      template_dir=args.template_dir,
      init_program=args.init_program,
      instruction_template=args.instruction_template,
      max_parallel=args.max_parallel,
      max_generations=args.max_generations,
      k_candidates=args.k_candidates,
      per_run_timeout=args.per_run_timeout,
      model=args.model,
      api_base=args.api_base,
      max_tokens=args.max_tokens,
      flow_sim_bin=args.flow_sim_bin,
      message_sizes=parse_message_sizes(args.message_sizes) if args.message_sizes else TOTAL_MESSAGE_SIZES,
  )
  print(f"Wrote SyCCL V100 DGX-2 Clos experiment manifest: {manifest_path}")
  print(f"Run all experiments with: {manifest_path.parent / 'runs' / 'run_all.sh'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
