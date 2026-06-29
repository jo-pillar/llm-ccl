#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from string import Template
from typing import NamedTuple
from urllib.parse import urlparse

try:
  import tomllib
except ModuleNotFoundError:  # pragma: no cover
  tomllib = None

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
  sys.path.insert(0, str(AGENT_ROOT))

class _CurrentStdoutHandler(logging.StreamHandler):
  def emit(self, record: logging.LogRecord) -> None:
    self.stream = sys.stdout
    super().emit(record)


if not logging.getLogger().handlers:
  handler = _CurrentStdoutHandler()
  handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
  logging.basicConfig(level=logging.INFO, handlers=[handler])
LOGGER = logging.getLogger("run_syccl_simpletes")

from syccl_agents.config_render import (
    total_gpus,
    write_syccl_config,
)
from syccl_agents.topodsl import TopoDSLSpec, load_topodsl

DATASET_ROOT = AGENT_ROOT / "datasets" / "syccl" / "scheme1_direct_events"
DEFAULT_ENV_TOML = AGENT_ROOT / "env.toml"
DEFAULT_OUTPUT_ROOT = Path("/home/antl/mntdisk/syccl-llm-scheme1-direct-events/simpletes")


class RunSpec(NamedTuple):
  command: list[str]
  env: dict[str, str]
  cwd: Path
  instruction_path: Path
  output_path: Path
  config_path: Path
  init_program_path: Path


def size_label(size: int) -> str:
  for suffix, multiplier in (("g", 1024 ** 3), ("m", 1024 ** 2), ("k", 1024)):
    if size >= multiplier and size % multiplier == 0:
      return f"{size // multiplier}{suffix}"
  return f"{size}b"


def load_env_toml(path: str | Path | None = DEFAULT_ENV_TOML) -> dict[str, str]:
  if path is None:
    return {}
  env_path = Path(path).expanduser()
  if not env_path.exists() or tomllib is None:
    return {}
  data = tomllib.loads(env_path.read_text(encoding="utf-8"))
  return {
      key: str(value)
      for key, value in data.items()
      if key in {"model", "api_base", "api_key"} and value
  }


def load_required_topodsl_from_env() -> TopoDSLSpec:
  raw_path = os.environ.get("TOPODSL", "").strip()
  if not raw_path:
    raise SystemExit("fatal: TOPODSL environment variable must point to a TopoDSL file")
  topo_path = Path(raw_path).expanduser()
  if not topo_path.exists():
    raise SystemExit(f"fatal: TOPODSL file does not exist: {topo_path}")
  try:
    return load_topodsl(topo_path)
  except Exception as exc:
    raise SystemExit(f"fatal: could not parse TOPODSL {topo_path}: {exc}") from exc


def _is_private_api_host(host: str | None) -> bool:
  if not host:
    return False
  if host in {"localhost"}:
    return True
  try:
    ip = ipaddress.ip_address(host)
  except ValueError:
    return False
  return ip.is_private or ip.is_loopback or ip.is_link_local


def _append_no_proxy(env: dict[str, str], host: str) -> None:
  for key in ("NO_PROXY", "no_proxy"):
    existing = [
        item.strip()
        for item in env.get(key, "").split(",")
        if item.strip()
    ]
    if host not in existing:
      existing.append(host)
    env[key] = ",".join(existing)


def _restart_every_n(max_generations: int, num_chains: int, k: int, preferred: int = 125) -> int:
  if max_generations <= 0:
    return 1
  base = max_generations // num_chains
  remainder = max_generations % num_chains
  prompt_budgets = []
  for chain_idx in range(num_chains):
    chain_budget = base + (1 if chain_idx < remainder else 0)
    if chain_budget > 0:
      prompt_budgets.append((chain_budget + k - 1) // k)
  if prompt_budgets and all(budget % preferred == 0 for budget in prompt_budgets):
    return preferred
  return 1


def _output_path(topo: TopoDSLSpec, root: str | Path, condition: str) -> Path:
  params = topo.params
  return (
      Path(root).expanduser().resolve()
      / topo.config_id
      / params.family
      / params.collective
      / size_label(params.message_size)
      / condition
  )


def _template_values(topo: TopoDSLSpec) -> dict[str, str]:
  params = topo.params
  gpu_num = total_gpus(params)
  return {
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


def write_instruction(topo: TopoDSLSpec, template_path: str | Path, output_path: Path) -> Path:
  source = Path(template_path).expanduser()
  if not source.exists():
    raise FileNotFoundError(f"instruction template not found: {source}")
  instruction_dir = output_path / "instructions"
  instruction_dir.mkdir(parents=True, exist_ok=True)
  rendered = Template(source.read_text(encoding="utf-8")).safe_substitute(_template_values(topo))
  instruction_path = instruction_dir / "syccl_instruction.txt"
  instruction_path.write_text(rendered, encoding="utf-8")
  return instruction_path


def write_init_program(source_path: str | Path, output_path: str | Path, *, gpu_num: int) -> Path:
  source = Path(source_path).expanduser().resolve()
  if not source.exists():
    raise FileNotFoundError(f"init program not found: {source}")
  evolve_block = _extract_user_evolve_block(source.read_text(encoding="utf-8"), source)
  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(
      f'''"""Generated SyCCL initial program."""


GPU_NUM = {int(gpu_num)}

# EVOLVE-BLOCK-START
{evolve_block.rstrip()}


# EVOLVE-BLOCK-END


def run_code():
  return construct_sketches(GPU_NUM)
''',
      encoding="utf-8",
  )
  return output


def _extract_user_evolve_block(source_text: str, source: Path) -> str:
  lines = source_text.splitlines()
  start_idx = -1
  end_idx = -1
  for idx, line in enumerate(lines):
    if "EVOLVE-BLOCK-START" in line:
      start_idx = idx
    elif "EVOLVE-BLOCK-END" in line:
      end_idx = idx
      break
  if start_idx >= 0 or end_idx >= 0:
    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
      raise ValueError(f"invalid EVOLVE-BLOCK markers in init program: {source}")
    block = "\n".join(lines[start_idx + 1:end_idx]).strip("\n")
  else:
    block = source_text.strip("\n")
  if "def construct_sketches" not in block:
    raise ValueError(f"init program evolve block must define construct_sketches(GPU_NUM): {source}")
  return block


def build_command(
    topo: TopoDSLSpec,
    *,
    init_program: str | Path,
    instruction_template: str | Path,
    max_generations: int,
    output_root: str | Path | None = None,
    env_toml: str | Path | None = DEFAULT_ENV_TOML,
    skip_preflight: bool = False,
    save_llm_io: bool = False,
) -> RunSpec:

  root = output_root if output_root is not None else DEFAULT_OUTPUT_ROOT
  output_path = _output_path(topo, root,"FULL")
  generated_dir = output_path / "generated"
  generated_dir.mkdir(parents=True, exist_ok=True)

  params = topo.params
  gpu_num = total_gpus(params)
  config_path = write_syccl_config(params, generated_dir / "flow-sim-config.json")
  instruction_path = write_instruction(topo, instruction_template, output_path)
  init_program_path = write_init_program(init_program, generated_dir / "init_program.py", gpu_num=gpu_num)

  model_config = load_env_toml(env_toml)
  env = os.environ.copy()
  for key in list(env):
    if key.startswith("SYCCL_TASK_"):
      env.pop(key, None)
  env["SYCCL_BASE_CONFIG"] = str(config_path)
  env["SYCCL_EVAL_ARTIFACT_DIR"] = str(output_path / "eval_artifacts")
  api_host = urlparse(model_config.get("api_base", "")).hostname
  if _is_private_api_host(api_host):
    _append_no_proxy(env, api_host)


# 如果你要的是第二个“dry startup”配置，或者第三个 pytest 配置，我也可以继续帮你还原成对应的终端命令。
  command = [
      "uv",
      "run",
      "python",
      "main.py",
      "--init-program",
      str(init_program_path),
      "--evaluator",
      str(DATASET_ROOT / "evaluator.py"),
      "--instruction",
      str(instruction_path),
      "--max-generations",
      str(max_generations),
      "--output-path",
      str(output_path / "checkpoints"),
      "--init-eval-repeats",
      "1",
      "--num-chains",
      "1",
      "--k-candidates",
      "1",
      "--selector",
      "llm_elite",
      "--llm-policy-pool-size",
      "100",
      "--elite-selection-strategy",
      "all",   
  ]
  if skip_preflight:
    command.append("--skip-preflight")
  if save_llm_io:
    command.append("--save-llm-io")

  for key, flag in (("model", "--model"), ("api_base", "--api-base"), ("api_key", "--api-key")):
    if key in model_config:
      command.extend([flag, model_config[key]])


  return RunSpec(
      command=command,
      env=env,
      cwd=AGENT_ROOT,
      instruction_path=instruction_path,
      output_path=output_path,
      config_path=config_path,
      init_program_path=init_program_path,
  )


def _redacted_env(env: dict[str, str]) -> dict[str, str]:
  return {
      key: ("***" if "KEY" in key.upper() else value)
      for key, value in env.items()
      if key.startswith("SYCCL_")
  }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run SimpleTES on a SyCCL TopoDSL task")
  parser.add_argument("--init-program", required=True, type=Path)
  parser.add_argument("--instruction", required=True, type=Path)

  parser.add_argument("--max-generations", type=int, default=10000)
  parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
  parser.add_argument("--env-toml", default=str(DEFAULT_ENV_TOML))
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--skip-preflight", action="store_true")
  parser.add_argument("--save-llm-io", "--save_llm_io", dest="save_llm_io", action="store_true")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = _parse_args(argv)
  topo = load_required_topodsl_from_env()
  spec = build_command(
      topo,
      init_program=args.init_program,
      instruction_template=args.instruction,
      max_generations=args.max_generations,
      output_root=args.output_root,
      env_toml=args.env_toml,
      skip_preflight=args.skip_preflight,
      save_llm_io=args.save_llm_io,
  )
  
  LOGGER.info("SimpleTES command: %s", " ".join(spec.command))
  exit(0)
  if args.dry_run:
    print("Command:")
    print(" ".join(spec.command))
    print("Environment:")
    print(json.dumps(_redacted_env(spec.env), indent=2))
    print(f"TopoDSL: {topo.path}")
    print(f"Config: {spec.config_path}")
    print(f"Init program: {spec.init_program_path}")
    print(f"Instruction: {spec.instruction_path}")
    print(f"Output: {spec.output_path}")
    return 0
  subprocess.run(spec.command, cwd=spec.cwd, env=spec.env, check=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
