#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

try:
  import tomllib
except ModuleNotFoundError:  # pragma: no cover
  tomllib = None


AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_ROOT.parent
DATASET_ROOT = AGENT_ROOT / "datasets" / "syccl" / "scheme1_direct_events"
DEFAULT_ENV_TOML = AGENT_ROOT / "env.toml"
DEFAULT_OUTPUT_ROOT = Path("/home/antl/mntdisk/syccl-llm-scheme1-direct-events")


class RunSpec(NamedTuple):
  command: list[str]
  env: dict[str, str]
  cwd: Path
  instruction_path: Path
  output_path: Path


def parse_size(value: str | int) -> int:
  if isinstance(value, int):
    return value
  text = value.strip().lower()
  suffixes = {
      "k": 1024,
      "kb": 1024,
      "m": 1024 ** 2,
      "mb": 1024 ** 2,
      "g": 1024 ** 3,
      "gb": 1024 ** 3,
  }
  for suffix, multiplier in suffixes.items():
    if text.endswith(suffix):
      return int(text[:-len(suffix)]) * multiplier
  return int(text)


def size_label(size: int) -> str:
  for suffix, multiplier in (("g", 1024 ** 3), ("m", 1024 ** 2), ("k", 1024)):
    if size >= multiplier and size % multiplier == 0:
      return f"{size // multiplier}{suffix}"
  return f"{size}b"


def load_manifest(path: str | Path) -> dict[str, Any]:
  return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def select_entry(
    manifest: dict[str, Any],
    case_id: str,
    collective: str,
    coll_byte: int,
) -> dict[str, Any]:
  matches = [
      entry for entry in manifest.get("entries", [])
      if entry.get("case_id") == case_id
      and entry.get("collective") == collective
      and int(entry.get("coll_byte_B", -1)) == int(coll_byte)
  ]
  if not matches:
    raise ValueError(
        f"No manifest entry for case={case_id}, collective={collective}, coll_byte={coll_byte}"
    )
  if len(matches) > 1:
    raise ValueError(
        f"Multiple manifest entries for case={case_id}, collective={collective}, coll_byte={coll_byte}"
    )
  return matches[0]


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


def _load_config(path: str | Path) -> dict[str, Any]:
  return json.loads(Path(path).read_text(encoding="utf-8"))


def _topology_name(config: dict[str, Any]) -> str:
  switch_topos = [
      layer.get("switch_topo")
      for layer in config.get("topo", [])
      if layer.get("type") == "switch"
  ]
  if "multirail" in switch_topos:
    return "multirail"
  if "pod" in switch_topos:
    return "clos"
  return "host"


def _cross_layer_env(config: dict[str, Any]) -> tuple[str, str]:
  switches = [
      layer for layer in config.get("topo", [])
      if layer.get("type") == "switch"
  ]
  if not switches:
    return "1", "0"
  topology = _topology_name(config)
  if topology == "multirail":
    layer = next(layer for layer in switches if layer.get("switch_topo") == "multirail")
  else:
    layer = switches[-1]
  return str(layer["layer_id"]), "0"


def _task_env(entry: dict[str, Any]) -> dict[str, str]:
  config_path = Path(entry["config_path"]).expanduser().resolve()
  config = _load_config(config_path)
  hosts = config.get("hosts", {})
  host_num = int(hosts.get("host_num", 1))
  host_gpu_num = int(hosts.get("host_gpu_num", 1))
  cross_layer, cross_group = _cross_layer_env(config)
  return {
      "SYCCL_BASE_CONFIG": str(config_path),
      "SYCCL_TASK_HOST_NUM": str(host_num),
      "SYCCL_TASK_HOST_GPU_NUM": str(host_gpu_num),
      "SYCCL_TASK_NGPUS": str(host_num * host_gpu_num),
      "SYCCL_TASK_TOPOLOGY": _topology_name(config),
      "SYCCL_TASK_CROSS_LAYER": cross_layer,
      "SYCCL_TASK_CROSS_GROUP": cross_group,
  }


def _load_evaluator_module(base_config: str):
  evaluator_path = DATASET_ROOT / "evaluator.py"
  spec = importlib.util.spec_from_file_location("syccl_scheme1_prompt_renderer", evaluator_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load evaluator: {evaluator_path}")
  module = importlib.util.module_from_spec(spec)
  old = os.environ.get("SYCCL_BASE_CONFIG")
  os.environ["SYCCL_BASE_CONFIG"] = base_config
  try:
    spec.loader.exec_module(module)
  finally:
    if old is None:
      os.environ.pop("SYCCL_BASE_CONFIG", None)
    else:
      os.environ["SYCCL_BASE_CONFIG"] = old
  return module


def write_instruction(entry: dict[str, Any], output_path: Path) -> Path:
  instruction_dir = output_path / "instructions"
  instruction_dir.mkdir(parents=True, exist_ok=True)
  instruction_path = instruction_dir / "syccl_instruction.txt"
  module = _load_evaluator_module(str(Path(entry["config_path"]).expanduser().resolve()))
  instruction_path.write_text(
      module.render_instruction_for_config(entry["config_path"]),
      encoding="utf-8",
  )
  return instruction_path


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


def build_command(
    entry: dict[str, Any],
    *,
    condition: str,
    max_generations: int,
    output_root: str | Path | None = None,
    env_toml: str | Path | None = DEFAULT_ENV_TOML,
    skip_preflight: bool = False,
    save_llm_io: bool = False,
) -> RunSpec:
  if condition not in {"full", "ablation"}:
    raise ValueError("condition must be full or ablation")

  config_path = Path(entry["config_path"]).expanduser().resolve()
  if output_root:
    root = Path(output_root).expanduser().resolve()
  else:
    root = (config_path.parents[3] / "simpletes") if len(config_path.parents) > 3 else (config_path.parent / "simpletes")
  output_path = (
      root
      / entry["case_id"]
      / entry["collective"]
      / size_label(int(entry["coll_byte_B"]))
      / condition
  )
  output_path.mkdir(parents=True, exist_ok=True)
  instruction_path = write_instruction(entry, output_path)

  model_config = load_env_toml(env_toml)
  env = os.environ.copy()
  env.update(_task_env(entry))
  env["SYCCL_EVAL_ARTIFACT_DIR"] = str(output_path / "eval_artifacts")
  api_host = urlparse(model_config.get("api_base", "")).hostname
  if _is_private_api_host(api_host):
    _append_no_proxy(env, api_host)

  command = [
      "uv",
      "run",
      "python",
      "main.py",
      "--init-program",
      str(DATASET_ROOT / "init_program.py"),
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
      "--selector",
      "balance",
  ]
  if skip_preflight:
    command.append("--skip-preflight")
  if save_llm_io:
    command.append("--save-llm-io")

  for key, flag in (("model", "--model"), ("api_base", "--api-base"), ("api_key", "--api-key")):
    if key in model_config:
      command.extend([flag, model_config[key]])

  if condition == "full":
    num_chains, k = 4, 4
    command.extend([
        "--num-chains",
        str(num_chains),
        "--k-candidates",
        str(k),
        "--restart-every-n",
        str(_restart_every_n(max_generations, num_chains, k)),
        "--include-construction",
    ])
  else:
    num_chains, k = 1, 1
    command.extend([
        "--num-inspirations",
        "0",
        "--num-chains",
        str(num_chains),
        "--k-candidates",
        str(k),
        "--restart-every-n",
        str(_restart_every_n(max_generations, num_chains, k)),
        "--disable-reflection",
        "--disable-failure-patterns",
    ])

  return RunSpec(
      command=command,
      env=env,
      cwd=AGENT_ROOT,
      instruction_path=instruction_path,
      output_path=output_path,
  )


def _redacted_env(env: dict[str, str]) -> dict[str, str]:
  return {
      key: ("***" if "KEY" in key.upper() else value)
      for key, value in env.items()
      if key.startswith("SYCCL_")
  }


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run SimpleTES on a SyCCL manifest entry")
  parser.add_argument("--manifest", required=True)
  parser.add_argument("--case", required=True)
  parser.add_argument("--collective", default="allgather", choices=["allgather", "alltoall"])
  parser.add_argument("--coll-byte", required=True)
  parser.add_argument("--condition", required=True, choices=["full", "ablation"])
  parser.add_argument("--max-generations", type=int, default=10000)
  parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT / "simpletes"))
  parser.add_argument("--env-toml", default=str(DEFAULT_ENV_TOML))
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--skip-preflight", action="store_true")
  parser.add_argument("--save-llm-io", "--save_llm_io", dest="save_llm_io", action="store_true")
  return parser.parse_args()


def main() -> int:
  args = _parse_args()
  manifest = load_manifest(args.manifest)
  entry = select_entry(
      manifest,
      case_id=args.case,
      collective=args.collective,
      coll_byte=parse_size(args.coll_byte),
  )
  spec = build_command(
      entry,
      condition=args.condition,
      max_generations=args.max_generations,
      output_root=args.output_root,
      env_toml=args.env_toml,
      skip_preflight=args.skip_preflight,
      save_llm_io=args.save_llm_io,
  )
  if args.dry_run:
    print("Command:")
    print(" ".join(spec.command))
    print("Environment:")
    print(json.dumps(_redacted_env(spec.env), indent=2))
    print(f"Instruction: {spec.instruction_path}")
    print(f"Output: {spec.output_path}")
    return 0
  subprocess.run(spec.command, cwd=spec.cwd, env=spec.env, check=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
