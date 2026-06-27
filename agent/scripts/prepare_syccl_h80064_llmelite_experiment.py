#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from string import Template
from typing import NamedTuple


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
  sys.path.insert(0, str(AGENT_ROOT))

from syccl_agents.config_render import total_gpus
from syccl_agents.topodsl import TopoDSLSpec, load_topodsl


DATASET_ROOT = AGENT_ROOT / "datasets" / "syccl" / "scheme1_direct_events"
DEFAULT_INIT_PROGRAM = DATASET_ROOT / "multirail_program.py"
DEFAULT_INSTRUCTION_TEMPLATE = DATASET_ROOT / "prompt_templete.txt"
DEFAULT_OUTPUT_ROOT = AGENT_ROOT / "result" / "config"
EXPERIMENT_REL = Path("h800-64hosts-8gpu-8nic-rail") / "ag"

HOSTS = 64
GPUS_PER_HOST = 8
NICS_PER_HOST = 8
RAILS = 8
TOTAL_GPUS = HOSTS * GPUS_PER_HOST
SMALL_CHUNK_THRESHOLD_B = 1024 * 1024
STRATEGIES = ("linear_rank", "balance", "all")
TOTAL_MESSAGE_SIZES = (
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
    268435456,
    1073741824,
)


class CaseSpec(NamedTuple):
  name: str
  total_message_size: int
  config_coll_byte: int
  host_lat_us: float
  net_lat_us: float


def build_case_specs() -> list[CaseSpec]:
  cases = []
  for total_message_size in TOTAL_MESSAGE_SIZES:
    config_coll_byte = total_message_size // TOTAL_GPUS
    is_small = config_coll_byte <= SMALL_CHUNK_THRESHOLD_B
    cases.append(
        CaseSpec(
            name=f"{total_message_size}B-prune=small",
            total_message_size=total_message_size,
            config_coll_byte=config_coll_byte,
            host_lat_us=3.0 if is_small else 10.5,
            net_lat_us=10.0 if is_small else 21.5,
        )
    )
  return cases


def render_topodsl(case: CaseSpec) -> str:
  return f'''###TopoBegin
def topology():
  return {{
    "family": "multirail",
    "hosts": {HOSTS},
    "gpus_per_host": {GPUS_PER_HOST},
    "nics_per_host": {NICS_PER_HOST},
    "rails": {RAILS},
    "message_size": {case.total_message_size},
    "collective": "allgather",
    "host_links": "nvswitch",
    "host_bw_mbpus": 0.15,
    "host_lat_us": {case.host_lat_us:g},
    "nic_bw_mbpus": 0.0455,
    "nic_lat_us": 0,
    "net_bw_mbpus": 0.0455,
    "net_lat_us": {case.net_lat_us:g},
  }}
###TopoEND
'''


def render_flow_sim_config(case: CaseSpec, *, solve_output: Path, sketch_path: Path) -> dict:
  return {
      "coll": {
          "name": "allgather",
          "byte": case.config_coll_byte,
          "root_sender": -1,
          "root_receiver": -1,
      },
      "hosts": {
          "host_num": HOSTS,
          "host_gpu_num": GPUS_PER_HOST,
          "host_nic_num": NICS_PER_HOST,
          "host_links": "nvswitch",
      },
      "topo": [
          {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
          {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
          {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
          {
              "layer_id": 3,
              "type": "switch",
              "switch_topo": "multirail",
              "switch_num": RAILS,
              "link_spec": "netlink",
          },
      ],
      "host_links": {
          "torus": {"dim_num": 2, "per_dim_num": [4, 4]},
          "nvlink": [
              [1, 2, 3, 5],
              [0, 2, 3, 4],
              [0, 1, 3, 7],
              [0, 1, 2, 6],
              [5, 6, 7, 1],
              [4, 6, 7, 0],
              [4, 5, 7, 3],
              [4, 5, 6, 2],
          ],
      },
      "link_spec": {
          "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
          "nvlink": {"bw_mbpus": 0.15, "lat_us": case.host_lat_us},
          "link_nic": {"bw_mbpus": 0.0455, "lat_us": 0},
          "netlink": {"bw_mbpus": 0.0455, "lat_us": case.net_lat_us},
      },
  }


def render_instruction(topo: TopoDSLSpec, template_path: Path) -> str:
  params = topo.params
  gpu_num = total_gpus(params)
  values = {
      "GPU_NUM": str(gpu_num),
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
    raise ValueError(f"init program evolve block must define construct_sketches(GPU_NUM): {source}")
  return block


def write_init_program(source_path: Path, output_path: Path, *, gpu_num: int) -> Path:
  evolve_block = extract_evolve_block(source_path.read_text(encoding="utf-8"), source_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
      f'''"""Generated SyCCL H800-64 initial program."""


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


def write_run_scripts(base_dir: Path, task_rows: list[dict[str, str]], *, max_parallel: int, max_generations: int) -> None:
  runs_dir = base_dir / "runs"
  runs_dir.mkdir(parents=True, exist_ok=True)
  tasks_path = runs_dir / "tasks.tsv"
  with tasks_path.open("w", encoding="utf-8") as f:
    for row in task_rows:
      f.write(
          "\t".join(
              [
                  row["case"],
                  row["strategy"],
                  row["config_path"],
                  row["instruction_path"],
                  row["init_program_path"],
                  row["output_path"],
                  row["artifact_dir"],
                  row["log_path"],
              ]
          )
          + "\n"
      )

  run_one = runs_dir / "run_one.sh"
  run_one.write_text(
      f'''#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 CASE STRATEGY CONFIG INSTRUCTION INIT_PROGRAM OUTPUT_PATH ARTIFACT_DIR LOG_PATH" >&2
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

AGENT_ROOT="{AGENT_ROOT}"
MAX_GENERATIONS="${{MAX_GENERATIONS:-{max_generations}}}"
EVAL_CONCURRENCY="${{EVAL_CONCURRENCY:-1}}"
GEN_CONCURRENCY="${{GEN_CONCURRENCY:-1}}"
LLM_POLICY_POOL_SIZE="${{LLM_POLICY_POOL_SIZE:-100}}"
EXTRA_SIMPLETES_ARGS="${{EXTRA_SIMPLETES_ARGS:-}}"

mkdir -p "$OUTPUT_PATH" "$ARTIFACT_DIR" "$(dirname "$LOG_PATH")"
cd "$AGENT_ROOT"

export PYTHONPATH="$AGENT_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
export SYCCL_BASE_CONFIG="$CONFIG_PATH"
export SYCCL_EVAL_ARTIFACT_DIR="$ARTIFACT_DIR"
export SYCCL_EVALUATOR_TIMEOUT_SECONDS="${{SYCCL_EVALUATOR_TIMEOUT_SECONDS:-3600}}"

echo "[syccl-h80064] case=$CASE_NAME strategy=$STRATEGY config=$CONFIG_PATH"

timeout 1h uv run python main.py \\
  --init-program "$INIT_PROGRAM_PATH" \\
  --evaluator "$AGENT_ROOT/datasets/syccl/scheme1_direct_events/evaluator.py" \\
  --instruction "$INSTRUCTION_PATH" \\
  --selector llm_elite \\
  --elite-selection-strategy "$STRATEGY" \\
  --num-chains 1 \\
  --k-candidates 1 \\
  --max-generations "$MAX_GENERATIONS" \\
  --eval-concurrency "$EVAL_CONCURRENCY" \\
  --gen-concurrency "$GEN_CONCURRENCY" \\
  --init-eval-repeats 1 \\
  --llm-policy-pool-size "$LLM_POLICY_POOL_SIZE" \\
  --output-path "$OUTPUT_PATH" \\
  --skip-preflight \\
  $EXTRA_SIMPLETES_ARGS 2>&1 | tee "$LOG_PATH"
''',
      encoding="utf-8",
  )

  run_all = runs_dir / "run_all.sh"
  run_all.write_text(
      f'''#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
TASKS="$SCRIPT_DIR/tasks.tsv"

xargs -P {max_parallel} -n 8 "$SCRIPT_DIR/run_one.sh" < "$TASKS"
''',
      encoding="utf-8",
  )
  run_one.chmod(0o755)
  run_all.chmod(0o755)


def prepare_experiment(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    init_program: Path = DEFAULT_INIT_PROGRAM,
    instruction_template: Path = DEFAULT_INSTRUCTION_TEMPLATE,
    max_parallel: int = 5,
    max_generations: int = 10000,
) -> Path:
  base_dir = output_root.expanduser().resolve() / EXPERIMENT_REL
  topodsl_dir = base_dir / "topodsl"
  config_dir = base_dir / "flow-sim-configs"
  instruction_dir = base_dir / "instructions"
  init_dir = base_dir / "init_programs"
  output_dir = base_dir / "outputs"
  log_dir = base_dir / "logs"
  synth_dir = base_dir / "synth_outputs"
  sketch_path = config_dir / "sketch.json"

  for directory in (topodsl_dir, config_dir, instruction_dir, init_dir, output_dir, log_dir, synth_dir):
    directory.mkdir(parents=True, exist_ok=True)

  init_program_path = write_init_program(init_program.expanduser().resolve(), init_dir / "init_program.py", gpu_num=TOTAL_GPUS)

  case_records = []
  run_records = []
  task_rows = []

  for case in build_case_specs():
    topodsl_path = topodsl_dir / f"{case.name}-topodsl.py"
    topodsl_path.write_text(render_topodsl(case), encoding="utf-8")
    topo = load_topodsl(topodsl_path)
    if total_gpus(topo.params) != TOTAL_GPUS:
      raise ValueError(f"TopoDSL {topodsl_path} did not resolve to {TOTAL_GPUS} GPUs")
    if topo.params.message_size != case.config_coll_byte:
      raise ValueError(
          f"TopoDSL {topodsl_path} resolved coll.byte={topo.params.message_size}, "
          f"expected {case.config_coll_byte}"
      )

    config_path = config_dir / f"{case.name}-config.json"
    solve_output = synth_dir / f"{case.name}-result.json"
    config = render_flow_sim_config(case, solve_output=solve_output, sketch_path=sketch_path)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    case_instruction_dir = instruction_dir / case.name
    case_instruction_dir.mkdir(parents=True, exist_ok=True)
    instruction_path = case_instruction_dir / "syccl_instruction.txt"
    instruction_path.write_text(render_instruction(topo, instruction_template), encoding="utf-8")

    case_record = {
        "case": case.name,
        "total_message_size": case.total_message_size,
        "config_coll_byte": case.config_coll_byte,
        "topodsl_path": str(topodsl_path),
        "config_path": str(config_path),
        "instruction_path": str(instruction_path),
        "init_program_path": str(init_program_path),
        "host_lat_us": case.host_lat_us,
        "net_lat_us": case.net_lat_us,
    }
    case_records.append(case_record)

    for strategy in STRATEGIES:
      run_output = output_dir / strategy / case.name / "checkpoints"
      artifact_dir = output_dir / strategy / case.name / "eval_artifacts"
      log_path = log_dir / strategy / f"{case.name}.log"
      run_record = {
          "case": case.name,
          "strategy": strategy,
          "config_path": str(config_path),
          "instruction_path": str(instruction_path),
          "init_program_path": str(init_program_path),
          "output_path": str(run_output),
          "artifact_dir": str(artifact_dir),
          "log_path": str(log_path),
      }
      run_records.append(run_record)
      task_rows.append(run_record)

  write_run_scripts(base_dir, task_rows, max_parallel=max_parallel, max_generations=max_generations)

  manifest = {
      "experiment": "h800-64hosts-8gpu-8nic-rail/ag",
      "origin_reference": "/home/antl/wzd/origin-syccl/build/syn_res/results-0.5/h800-64hosts-8gpu-8nic-rail/ag",
      "hosts": HOSTS,
      "gpus_per_host": GPUS_PER_HOST,
      "nics_per_host": NICS_PER_HOST,
      "rails": RAILS,
      "total_gpus": TOTAL_GPUS,
      "collective": "allgather",
      "strategies": list(STRATEGIES),
      "max_parallel": max_parallel,
      "per_run_timeout": "1h",
      "max_generations": max_generations,
      "cases": case_records,
      "runs": run_records,
  }
  manifest_path = base_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Prepare SyCCL H800-64 llm_elite experiment inputs")
  parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
  parser.add_argument("--init-program", type=Path, default=DEFAULT_INIT_PROGRAM)
  parser.add_argument("--instruction-template", type=Path, default=DEFAULT_INSTRUCTION_TEMPLATE)
  parser.add_argument("--max-parallel", type=int, default=5)
  parser.add_argument("--max-generations", type=int, default=10000)
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  manifest_path = prepare_experiment(
      output_root=args.output_root,
      init_program=args.init_program,
      instruction_template=args.instruction_template,
      max_parallel=args.max_parallel,
      max_generations=args.max_generations,
  )
  print(f"Wrote SyCCL H800-64 experiment manifest: {manifest_path}")
  print(f"Run all experiments with: {manifest_path.parent / 'runs' / 'run_all.sh'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
