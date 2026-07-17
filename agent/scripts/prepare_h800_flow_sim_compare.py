#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Sequence

try:
  import tomllib
except ModuleNotFoundError:  # pragma: no cover
  tomllib = None


AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_ROOT.parent
DATASET_ROOT = AGENT_ROOT / "datasets" / "syccl" / "scheme1_direct_events"
H800_TEMPLATE_DIR = DATASET_ROOT / "templates" / "H800_multirail"
H800_TOPO_TEMPLATE = H800_TEMPLATE_DIR / "multirail_topo.py"
H800_INIT_TEMPLATE = H800_TEMPLATE_DIR / "multirail_program.py"
INSTRUCTION_TEMPLATE = DATASET_ROOT / "prompt_templete.txt"
FLOW_SIM_ROOT = DATASET_ROOT / "flow-sim-rs"

TARGET_SYCCL_COMMIT = "855a184ed0746b46595763661345d76812c9a7dd"
DEFAULT_ORIGIN_REPO = Path("/root/origin-syccl")
DEFAULT_SYCCL_WORKTREE = Path("/root/origin-syccl-h800-855a184")
DEFAULT_BUNDLE_ROOT = REPO_ROOT / "experiments" / "h800-flow-sim-compare"
DEFAULT_ENV_TOML = AGENT_ROOT / "env.toml"
DEFAULT_MODEL = "gemini/gemini-2.0-flash"
DEFAULT_CMAKE_BIN = Path("/root/nanoGPT/.venv/bin/cmake") if Path("/root/nanoGPT/.venv/bin/cmake").is_file() else Path("cmake")
DEFAULT_SCIP_SUITE_DIR = (
    Path("/root/syccl/deps/scipoptsuite-9.2.2/install")
    if Path("/root/syccl/deps/scipoptsuite-9.2.2/install").is_dir()
    else Path("/usr/local/scipoptsuite-9.2.2/install")
)
DEFAULT_SCIP_PP_DIR = (
    Path("/root/syccl/deps/scippp-install")
    if Path("/root/syccl/deps/scippp-install").is_dir()
    else Path("/usr/local/install")
)

STRATEGIES = ("linear_rank", "balance", "all")
GPU_PER_HOST = 8
NICS_PER_HOST = 8
RAILS = 8
PRUNE_TYPE = "small"


@dataclass(frozen=True)
class CaseSpec:
  case_id: str
  group: str
  gpu_count: int
  host_count: int
  gpus_per_host: int
  nics_per_host: int
  rails: int
  collective: str
  total_message_size: int
  coll_byte: int
  prune_type: str
  a2a: bool


def runexp_h800_total_sizes(hosts: int) -> list[int]:
  sizes: list[int] = []
  size = 2 ** 10
  while size <= 4 * (2 ** 32):
    sizes.append(size)
    size *= 4
  if hosts >= 32:
    sizes = sizes[3:]
    sizes.append(sizes[-1] * 4)
    sizes.append(sizes[-1] * 4)
  return sizes


def runexp_scale_h800_total_sizes(hosts: int) -> list[int]:
  sizes: list[int] = []
  size = 2 ** 10
  while size <= 2 ** 18:
    sizes.append(size)
    size *= 16
  if hosts >= 32:
    sizes = sizes[2:]
    sizes.append(sizes[-1] * 4)
    sizes.append(sizes[-1] * 4)
  return sizes


def build_case_specs() -> list[CaseSpec]:
  cases: list[CaseSpec] = []
  for collective, a2a in (("allgather", False), ("alltoall", True)):
    hosts = 32
    gpu_count = hosts * GPU_PER_HOST
    for total_size in runexp_h800_total_sizes(hosts):
      cases.append(
          CaseSpec(
              case_id=f"h800-256gpu-{collective}-{total_size}B-prune={PRUNE_TYPE}",
              group=f"h800-32hosts-8gpu-8nic-rail/{'a2a' if a2a else 'ag'}",
              gpu_count=gpu_count,
              host_count=hosts,
              gpus_per_host=GPU_PER_HOST,
              nics_per_host=NICS_PER_HOST,
              rails=RAILS,
              collective=collective,
              total_message_size=total_size,
              coll_byte=total_size // gpu_count,
              prune_type=PRUNE_TYPE,
              a2a=a2a,
          )
      )

  hosts = 512
  gpu_count = hosts * GPU_PER_HOST
  for total_size in runexp_scale_h800_total_sizes(hosts):
    cases.append(
        CaseSpec(
            case_id=f"h800-4096gpu-allgather-{total_size}B-prune={PRUNE_TYPE}",
            group="h800-512hosts-8gpu-8nic-rail-scale/ag",
            gpu_count=gpu_count,
            host_count=hosts,
            gpus_per_host=GPU_PER_HOST,
            nics_per_host=NICS_PER_HOST,
            rails=RAILS,
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


def config_semantic_sha256(config: dict[str, Any]) -> str:
  normalized = json.loads(json.dumps(config, sort_keys=True))
  normalized.pop("sketch", None)
  payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_env_toml(path: Path | None) -> dict[str, str]:
  if path is None:
    return {}
  env_path = path.expanduser()
  if not env_path.exists():
    return {}
  text = env_path.read_text(encoding="utf-8")
  if tomllib is not None:
    data = tomllib.loads(text)
  else:
    data = {}
    for line in text.splitlines():
      stripped = line.strip()
      if not stripped or stripped.startswith("#") or "=" not in stripped:
        continue
      key, raw_value = stripped.split("=", 1)
      key = key.strip()
      if key not in {"model", "api_base", "api_key"}:
        continue
      raw_value = raw_value.split("#", 1)[0].strip()
      try:
        data[key] = ast.literal_eval(raw_value)
      except (SyntaxError, ValueError):
        data[key] = raw_value.strip("\"'")
  return {
      key: str(value)
      for key, value in data.items()
      if key in {"model", "api_base", "api_key"} and value
  }


def redact_secret(value: str | None) -> str:
  if not value:
    return ""
  if len(value) <= 12:
    return "***"
  return f"{value[:6]}...{value[-4:]}"


def shell_single_quote(value: str) -> str:
  return "'" + value.replace("'", "'\"'\"'") + "'"


def run_capture(cmd: Sequence[str], *, cwd: Path | None = None, check: bool = True) -> str:
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


def symlink_or_copy_dir(source: Path, dest: Path) -> None:
  if dest.exists():
    return
  dest.parent.mkdir(parents=True, exist_ok=True)
  try:
    dest.symlink_to(source, target_is_directory=True)
  except OSError:
    shutil.copytree(source, dest)


def prepare_syccl_dependency_compat(bundle: Path, scip_suite_dir: Path, scip_pp_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
  compat_root = bundle / "provenance" / "syccl-deps-compat"
  status: dict[str, Any] = {
      "source_scip_suite_dir": str(scip_suite_dir),
      "source_scip_pp_dir": str(scip_pp_dir),
      "used_compat_dirs": False,
  }

  scip_has_lib64 = (scip_suite_dir / "lib64" / "cmake" / "scip").is_dir()
  scippp_has_lib64 = (scip_pp_dir / "lib64").is_dir()
  if scip_has_lib64 and scippp_has_lib64:
    status.update({"scip_suite_dir": str(scip_suite_dir), "scip_pp_dir": str(scip_pp_dir)})
    return scip_suite_dir, scip_pp_dir, status

  compat_scip = compat_root / "scip"
  compat_scippp = compat_root / "scippp"
  for name in ("lib", "include", "bin"):
    source = scip_suite_dir / name
    if source.exists():
      symlink_or_copy_dir(source, compat_scip / name)
  if (scip_suite_dir / "lib64").exists():
    symlink_or_copy_dir(scip_suite_dir / "lib64", compat_scip / "lib64")
  elif (scip_suite_dir / "lib").exists():
    symlink_or_copy_dir(scip_suite_dir / "lib", compat_scip / "lib64")

  for name in ("lib", "include", "bin"):
    source = scip_pp_dir / name
    if source.exists():
      symlink_or_copy_dir(source, compat_scippp / name)
  if (scip_pp_dir / "lib64").exists():
    symlink_or_copy_dir(scip_pp_dir / "lib64", compat_scippp / "lib64")
  elif (scip_pp_dir / "lib").exists():
    symlink_or_copy_dir(scip_pp_dir / "lib", compat_scippp / "lib64")

  status.update(
      {
          "used_compat_dirs": True,
          "scip_suite_dir": str(compat_scip),
          "scip_pp_dir": str(compat_scippp),
      }
  )
  write_json(bundle / "provenance" / "syccl-deps-compat.json", status)
  return compat_scip, compat_scippp, status


def run_logged(cmd: Sequence[str], *, cwd: Path, log_path: Path) -> dict[str, Any]:
  rendered = [str(part) for part in cmd]
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with log_path.open("a", encoding="utf-8") as log:
    log.write(f"$ {' '.join(rendered)}\n")
    proc = subprocess.run(
        rendered,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log.write(proc.stdout)
    if proc.stdout and not proc.stdout.endswith("\n"):
      log.write("\n")
  return {
      "cmd": rendered,
      "cwd": str(cwd),
      "returncode": proc.returncode,
      "output_tail": proc.stdout[-4000:],
  }


def build_syccl_synthesize(
    bundle: Path,
    worktree: Path,
    *,
    cmake_bin: Path,
    scip_suite_dir: Path,
    scip_pp_dir: Path,
    parallel: int,
    require_success: bool,
) -> dict[str, Any]:
  build_dir = worktree / "build"
  log_path = bundle / "provenance" / "syccl-build.log"
  resolved_scip, resolved_scippp, compat_status = prepare_syccl_dependency_compat(bundle, scip_suite_dir, scip_pp_dir)
  status: dict[str, Any] = {
      "requested": True,
      "status": "started",
      "build_dir": str(build_dir),
      "cmake_bin": str(cmake_bin),
      "scip_suite_dir": str(resolved_scip),
      "scip_pp_dir": str(resolved_scippp),
      "parallel": parallel,
      "log_path": str(log_path),
      "dependency_compat": compat_status,
      "commands": [],
  }
  build_dir.mkdir(parents=True, exist_ok=True)
  commands = [
      [
          str(cmake_bin),
          "-S",
          str(worktree),
          "-B",
          str(build_dir),
          f"-DSCIP_SUITE_DIR={resolved_scip}",
          f"-DSCIP_PP_DIR={resolved_scippp}",
      ],
      [str(cmake_bin), "--build", str(build_dir), "--parallel", str(parallel)],
  ]
  for cmd in commands:
    result = run_logged(cmd, cwd=worktree, log_path=log_path)
    status["commands"].append(result)
    if result["returncode"] != 0:
      status["status"] = "failed"
      status["failure"] = result["output_tail"]
      break
  synthesize = build_dir / "synthesize"
  status["synthesize_path"] = str(synthesize)
  status["synthesize_exists"] = synthesize.is_file()
  if status["status"] != "failed":
    status["status"] = "ok" if synthesize.is_file() else "missing_synthesize"
  if synthesize.is_file():
    status["synthesize_sha256"] = sha256_file(synthesize)
  write_json(bundle / "provenance" / "syccl-build.json", status)
  if require_success and status["status"] != "ok":
    raise RuntimeError(f"SyCCL build did not produce synthesize; see {log_path}")
  return status


def load_config_gen(syccl_worktree: Path):
  script_path = syccl_worktree / "scripts" / "config_gen.py"
  if not script_path.is_file():
    raise FileNotFoundError(f"missing SyCCL config_gen.py: {script_path}")
  spec = importlib.util.spec_from_file_location("syccl_fixed_config_gen", script_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"could not import {script_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def ensure_syccl_worktree(origin_repo: Path, worktree: Path, commit: str) -> None:
  if not origin_repo.is_dir():
    raise FileNotFoundError(f"origin SyCCL repo not found: {origin_repo}")
  if worktree.exists():
    head = run_capture(["git", "-C", str(worktree), "rev-parse", "HEAD"])
    if head != commit:
      raise RuntimeError(f"existing worktree {worktree} is at {head}, expected {commit}")
  else:
    worktree.parent.mkdir(parents=True, exist_ok=True)
    run_capture(["git", "-C", str(origin_repo), "worktree", "add", str(worktree), commit])
  patch_syccl_runexp_logging(worktree)


def patch_syccl_runexp_logging(worktree: Path) -> None:
  replacement = '''def solve(config_path, solver_path, log_path, thrs=None):
  cmd = [solver_path, "-f", config_path, "solve"]
  env = os.environ.copy()
  env["SYNTHESIZE_PARALLEL_THREAD_NUM"] = str(thrs if thrs is not None else 72)
  os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
  with open("./exp_progress.txt", "a") as f:
    f.write(" ".join(cmd) + f" > {log_path} 2>&1\\n")
  print(" ".join(cmd))
  with open(log_path, "w") as log:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
      print(line, end="")
      log.write(line)
    returncode = proc.wait()
  if returncode != 0:
    raise subprocess.CalledProcessError(returncode, cmd)
'''
  for rel in ("scripts/runexp.py", "scripts/runexp_scale.py", "scripts/runexp_reverse.py"):
    path = worktree / rel
    text = path.read_text(encoding="utf-8")
    patched, count = re.subn(
        r"(?ms)^def solve\(config_path, solver_path, log_path, thrs=None\):\n.*?(?=^def |\Z)",
        lambda _match: replacement + "\n",
        text,
        count=1,
    )
    if count != 1:
      raise ValueError(f"could not patch solve() in {path}")
    path.write_text(patched, encoding="utf-8")


def render_syccl_config(config_gen: Any, case: CaseSpec, bundle: Path) -> dict[str, Any]:
  cfg = config_gen.ConfigGen()
  result_path = bundle / "origin" / "results" / case.case_id / "result.json"
  sketch_path = bundle / "origin" / "sketches" / f"{case.case_id}-sketch.json"
  config = cfg.h800conf(
      case.coll_byte,
      case.host_count,
      case.gpus_per_host,
      case.nics_per_host,
      case.rails,
      case.prune_type,
      str(result_path),
      str(sketch_path),
      case.a2a,
  )
  return config


def bw_mbpus_to_gbps(value: float) -> str:
  gbps = value * 1000.0
  if abs(gbps - round(gbps)) < 1e-9:
    return f"{int(round(gbps))}GB/s"
  return f"{gbps:g}GB/s"


def us(value: float) -> str:
  if abs(value - round(value)) < 1e-9:
    return f"{int(round(value))}us"
  return f"{value:g}us"


def topo_layer(config: dict[str, Any], layer_type: str) -> dict[str, Any]:
  for layer in config.get("topo", []):
    if layer.get("type") == layer_type:
      return layer
  raise ValueError(f"config missing topo layer type {layer_type!r}")


def render_topodsl(case: CaseSpec, config: dict[str, Any], template_text: str) -> str:
  link_spec = config["link_spec"]
  host_layer = topo_layer(config, "host")
  nic_layer = topo_layer(config, "nic")
  switch_layer = topo_layer(config, "switch")
  host_link = link_spec[host_layer.get("link_spec", "nvlink")]
  nic_link = link_spec[nic_layer.get("link_spec", "link_nic")]
  net_link = link_spec[switch_layer.get("link_spec", "netlink")]
  collective_enum = {
      "allgather": "CollectiveType.ALLGATHER",
      "alltoall": "CollectiveType.ALLTOALL",
  }[case.collective]
  total_nics = case.host_count * case.nics_per_host
  instantiation = f'''topology = MultiRailTopology(
    {case.gpu_count},
    {case.total_message_size},
    {collective_enum},
    layer1=LayerSpec(1, LinkSpec("{bw_mbpus_to_gbps(float(host_link["bw_mbpus"]))}", "{us(float(host_link["lat_us"]))}"), group_num={case.host_count}, node_num={case.gpus_per_host}, node_type=NodeType.GPU),
    layer2=LayerSpec(2, LinkSpec("{bw_mbpus_to_gbps(float(nic_link["bw_mbpus"]))}", "{us(float(nic_link["lat_us"]))}"), group_num={total_nics}, node_num={max(1, total_nics // case.gpu_count)}, node_type=NodeType.NIC),
    layer3=LayerSpec(3, LinkSpec("{bw_mbpus_to_gbps(float(net_link["bw_mbpus"]))}", "{us(float(net_link["lat_us"]))}"), group_num={case.rails}, node_num={case.host_count}, node_type=NodeType.SWITCH),
)
'''
  rendered, count = re.subn(
      r"(?ms)^topology\s*=\s*MultiRailTopology\(\n.*?\n\)\s*(?=###TopoEND)",
      instantiation,
      template_text,
      count=1,
  )
  if count != 1:
    raise ValueError("H800 multirail topology template must contain one topology instantiation")
  return rendered


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
      "TOPOLOGY_FAMILY": "multirail",
      "MESSAGE_SIZE": str(case.coll_byte),
      "HOST_NUM": str(case.host_count),
      "HOST_GPU_NUM": str(case.gpus_per_host),
      "NIC_NUM": str(case.nics_per_host),
      "TOPODSL": prompt_source(topodsl_text),
  }
  return Template(instruction_template).safe_substitute(values)


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


def render_init_program(template_text: str, template_path: Path, gpu_num: int) -> str:
  evolve_block = extract_evolve_block(template_text, template_path)
  return f'''"""Generated H800 flow-sim comparison initial program."""


GPU_NUM = {gpu_num}

# EVOLVE-BLOCK-START
{evolve_block.rstrip()}


# EVOLVE-BLOCK-END


def run_code():
  return construct_sketches(GPU_NUM)
'''


def write_case_inputs(bundle: Path, syccl_worktree: Path) -> list[dict[str, Any]]:
  config_gen = load_config_gen(syccl_worktree)
  topodsl_template = H800_TOPO_TEMPLATE.read_text(encoding="utf-8")
  init_template = H800_INIT_TEMPLATE.read_text(encoding="utf-8")
  instruction_template = INSTRUCTION_TEMPLATE.read_text(encoding="utf-8")
  rows: list[dict[str, Any]] = []
  for case in build_case_specs():
    case_dir = bundle / "configs" / case.case_id
    config = render_syccl_config(config_gen, case, bundle)
    config_path = case_dir / "config.json"
    write_json(config_path, config)
    config_sha = sha256_file(config_path)

    topodsl_text = render_topodsl(case, config, topodsl_template)
    topodsl_path = case_dir / "topodsl.py"
    topodsl_path.write_text(topodsl_text, encoding="utf-8")

    instruction_path = case_dir / "instruction.txt"
    instruction_path.write_text(
        render_instruction(case, topodsl_text, instruction_template),
        encoding="utf-8",
    )

    init_path = case_dir / "init_program.py"
    init_path.write_text(render_init_program(init_template, H800_INIT_TEMPLATE, case.gpu_count), encoding="utf-8")

    row = {
        **asdict(case),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "config_semantic_sha256": config_semantic_sha256(config),
        "topodsl_path": str(topodsl_path),
        "instruction_path": str(instruction_path),
        "init_program_path": str(init_path),
        "origin_result_path": config["algo_solve"]["solve_output"],
        "origin_sketch_path": config["sketch"]["sketch_path"],
    }
    rows.append(row)

  write_json(bundle / "configs" / "manifest.json", {"cases": rows})
  write_case_csv(bundle / "configs" / "manifest.csv", rows)
  return rows


def write_case_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = [
      "case_id",
      "group",
      "gpu_count",
      "host_count",
      "gpus_per_host",
      "nics_per_host",
      "rails",
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
  ]
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row.get(field, "") for field in fieldnames})


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
            "git_commit": run_capture(["git", "-C", str(FLOW_SIM_ROOT), "rev-parse", "HEAD"], check=False),
            "git_diff_stat": run_capture(["git", "-C", str(FLOW_SIM_ROOT), "diff", "--stat", "--", "."], check=False),
        }
    )
    (bundle / "provenance" / "flow-sim-rs.sha256").write_text(
        f"{status['sha256']}  {dest}\n",
        encoding="utf-8",
    )
    (bundle / "provenance" / "flow-sim-rs.git.txt").write_text(
        f"commit={status['git_commit']}\n\n{status['git_diff_stat']}\n",
        encoding="utf-8",
    )
  else:
    (bundle / "provenance" / "FLOW_SIM_BUILD_REQUIRED.md").write_text(
        "flow-sim-rs binary was not found during preparation.\n\n"
        f"Expected source: `{source}`\n\n"
        "Build or provide a release binary, then rerun preparation with "
        "`--flow-sim-bin /path/to/flow-sim-rs`.\n",
        encoding="utf-8",
    )
  write_json(bundle / "provenance" / "flow-sim-rs.json", status)
  return status


def write_syccl_provenance(bundle: Path, worktree: Path, commit: str, build: dict[str, Any] | None = None) -> dict[str, Any]:
  head = run_capture(["git", "-C", str(worktree), "rev-parse", "HEAD"])
  diff = run_capture(["git", "-C", str(worktree), "diff", "--", "scripts/runexp.py", "scripts/runexp_scale.py", "scripts/runexp_reverse.py"], check=False)
  patch_sha = hashlib.sha256(diff.encode("utf-8")).hexdigest()
  synthesize = worktree / "build" / "synthesize"
  payload = {
      "worktree": str(worktree),
      "target_commit": commit,
      "head": head,
      "logging_patch_sha256": patch_sha,
      "logging_patch_diff": diff,
      "synthesize_path": str(synthesize),
      "synthesize_exists": synthesize.is_file(),
      "synthesize_sha256": sha256_file(synthesize) if synthesize.is_file() else None,
      "build": build or {"requested": False, "status": "skipped"},
      "run_script_sha256": {
          rel: sha256_file(worktree / rel)
          for rel in ("scripts/runexp.py", "scripts/runexp_scale.py", "scripts/runexp_reverse.py")
      },
  }
  write_json(bundle / "provenance" / "syccl-worktree.json", payload)
  (bundle / "provenance" / "syccl-logging.patch").write_text(diff, encoding="utf-8")
  return payload


def write_run_scripts(
    bundle: Path,
    cases: list[dict[str, Any]],
    *,
    syccl_worktree: Path,
    model_config: dict[str, str],
    max_generations: int,
    k_candidates: int,
    max_parallel: int,
    simpletes_timeout: str,
    origin_timeout: str,
) -> None:
  runs = bundle / "runs"
  runs.mkdir(parents=True, exist_ok=True)
  llm_tasks = runs / "llm_tasks.tsv"
  with llm_tasks.open("w", encoding="utf-8") as handle:
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

  origin_tasks = runs / "origin_tasks.tsv"
  with origin_tasks.open("w", encoding="utf-8") as handle:
    for row in cases:
      log_path = bundle / "origin" / "logs" / f"{row['case_id']}.log"
      handle.write("\t".join([row["case_id"], row["config_path"], row["origin_result_path"], str(log_path)]) + "\n")

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
MODEL_NAME=${{MODEL_NAME:-{shell_single_quote(model_config.get('model', DEFAULT_MODEL))}}}
API_BASE=${{API_BASE:-{shell_single_quote(model_config.get('api_base', ''))}}}
API_KEY=${{API_KEY:-{shell_single_quote(model_config.get('api_key', ''))}}}
MAX_GENERATIONS="${{MAX_GENERATIONS:-{max_generations}}}"
K_CANDIDATES="${{K_CANDIDATES:-{k_candidates}}}"
EVAL_CONCURRENCY="${{EVAL_CONCURRENCY:-1}}"
GEN_CONCURRENCY="${{GEN_CONCURRENCY:-1}}"
LLM_POLICY_POOL_SIZE="${{LLM_POLICY_POOL_SIZE:-100}}"
SIMPLETES_TIMEOUT="${{SIMPLETES_TIMEOUT:-{simpletes_timeout}}}"
EXTRA_SIMPLETES_ARGS="${{EXTRA_SIMPLETES_ARGS:-}}"

mkdir -p "$OUTPUT_PATH" "$ARTIFACT_DIR" "$(dirname "$LOG_PATH")"
cd "$AGENT_ROOT"

export PYTHONPATH="$AGENT_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
export SYCCL_BASE_CONFIG="$CONFIG_PATH"
export SYCCL_EVAL_ARTIFACT_DIR="$ARTIFACT_DIR"
export FLOW_SIM_BIN="$FLOW_SIM_BIN"
export SYCCL_EVALUATOR_TIMEOUT_SECONDS="${{SYCCL_EVALUATOR_TIMEOUT_SECONDS:-3600}}"

echo "[h800-compare:llm] case=$CASE_ID strategy=$STRATEGY config=$CONFIG_PATH"

LLM_ARGS=(--model "$MODEL_NAME")
if [[ -n "$API_BASE" ]]; then
  LLM_ARGS+=(--api-base "$API_BASE")
fi
if [[ -n "$API_KEY" ]]; then
  LLM_ARGS+=(--api-key "$API_KEY")
fi

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
  --output-path "$OUTPUT_PATH" \\
  "${{LLM_ARGS[@]}}" \\
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

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CASE CONFIG RESULT LOG" >&2
  exit 2
fi

CASE_ID="$1"
CONFIG_PATH="$2"
RESULT_PATH="$3"
LOG_PATH="$4"
SYCCL_WORKTREE="{syccl_worktree}"
SYCCL_BUILD_DIR="${{SYCCL_BUILD_DIR:-$SYCCL_WORKTREE/build}}"
SYNTHESIZE_BIN="${{SYNTHESIZE_BIN:-$SYCCL_BUILD_DIR/synthesize}}"
ORIGIN_SOLVE_TIMEOUT="${{ORIGIN_SOLVE_TIMEOUT:-{origin_timeout}}}"
export SYNTHESIZE_PARALLEL_THREAD_NUM="${{SYNTHESIZE_PARALLEL_THREAD_NUM:-72}}"

mkdir -p "$(dirname "$RESULT_PATH")" "$(dirname "$LOG_PATH")"
cd "$SYCCL_BUILD_DIR"

echo "[h800-compare:origin] case=$CASE_ID config=$CONFIG_PATH"
timeout "$ORIGIN_SOLVE_TIMEOUT" "$SYNTHESIZE_BIN" -f "$CONFIG_PATH" solve 2>&1 | tee "$LOG_PATH"
''',
  )

  write_executable(
      runs / "run_origin_all.sh",
      f'''#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
python3 "$SCRIPT_DIR/run_origin_solve_all.py"
''',
  )

  write_origin_solve_script(runs / "run_origin_solve_all.py", bundle, syccl_worktree, origin_timeout)
  write_origin_flow_sim_script(runs / "run_origin_flow_sim_all.py", bundle)
  write_summarizer_script(runs / "summarize_llm_outputs.py", bundle)
  write_final_report_script(runs / "build_final_report.py", bundle)


def write_executable(path: Path, text: str) -> None:
  path.write_text(text, encoding="utf-8")
  path.chmod(0o755)


def write_origin_solve_script(path: Path, bundle: Path, syccl_worktree: Path, origin_timeout: str) -> None:
  script = r'''#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUNDLE = Path("__BUNDLE__")
SYCCL_WORKTREE = Path("__SYCCL_WORKTREE__")
TASKS = BUNDLE / "runs" / "origin_tasks.tsv"
OUT_ROOT = BUNDLE / "origin" / "solve-summary"
DEFAULT_TIMEOUT = "__ORIGIN_TIMEOUT__"
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
  synthesize = Path(os.environ.get("SYNTHESIZE_BIN", str(build_dir / "synthesize")))
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
'''
  write_executable(
      path,
      script.replace("__BUNDLE__", str(bundle))
      .replace("__SYCCL_WORKTREE__", str(syccl_worktree))
      .replace("__ORIGIN_TIMEOUT__", origin_timeout),
  )


def write_origin_flow_sim_script(path: Path, bundle: Path) -> None:
  write_executable(
      path,
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
    case_id, config_path, result_path, _log_path = raw.rstrip("\\n").split("\\t")
    result = Path(result_path)
    alg_best_index, alg_best_us = alg_times_best(result)
    output = OUT_ROOT / case_id / "origin-all-flow-sim.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not result.exists():
      rows.append({{"case_id": case_id, "status": "missing_origin_result", "origin_result": str(result)}})
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
      rows.append({{"case_id": case_id, "status": "flow_sim_failed", "elapsed_s": elapsed, "error": proc.stdout[-1000:]}})
      continue
    best = best_solution(output)
    if best is None:
      rows.append({{"case_id": case_id, "status": "no_positive_solution", "elapsed_s": elapsed, "output": str(output)}})
      continue
    rows.append({{
        "case_id": case_id,
        "status": "ok",
        "elapsed_s": elapsed,
        "origin_flow_sim_us": best["rust_time_us"],
        "origin_flow_sim_solution_index": best.get("solution_index"),
        "origin_alg_times_best_us": alg_best_us,
        "origin_alg_times_best_index": alg_best_index,
        "origin_flow_sim_output": str(output),
    }})

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


def write_summarizer_script(path: Path, bundle: Path) -> None:
  write_executable(
      path,
      f'''#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

BUNDLE = Path({str(bundle)!r})
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
  totals = {{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}}
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
flow_sim_info = read_json(flow_sim_info_path) if flow_sim_info_path.exists() else {{}}
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
        provenance_error = f"missing candidate-config.json in {{eval_dir}}"
      else:
        try:
          candidate_config_sha256 = sha256_file(candidate_config)
          candidate_config_semantic_sha256 = config_semantic_sha256(candidate_config)
        except Exception as exc:
          provenance_ok = False
          provenance_error = f"could not hash candidate-config.json in {{eval_dir}}: {{exc}}"
        if provenance_ok and case.get("config_semantic_sha256") and candidate_config_semantic_sha256 != case.get("config_semantic_sha256"):
          provenance_ok = False
          provenance_error = f"candidate semantic config hash mismatch in {{eval_dir}}"
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
          best = {{"best_flow_sim_us": time_us, "best_eval_id": eval_dir.name, "best_output_path": str(output)}}
    tokens = token_totals(case_id, strategy)
    rows.append({{
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
    }})

out_dir = BUNDLE / "reports"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "llm_strategy_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
fields = list(rows[0].keys()) if rows else []
with (out_dir / "llm_strategy_summary.csv").open("w", encoding="utf-8", newline="") as handle:
  writer = csv.DictWriter(handle, fieldnames=fields)
  writer.writeheader()
  for row in rows:
    writer.writerow(row)
print(f"wrote {{out_dir / 'llm_strategy_summary.json'}}")
''',
  )


def write_final_report_script(path: Path, bundle: Path) -> None:
  script = r'''#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

BUNDLE = Path("__BUNDLE__")

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
'''
  write_executable(path, script.replace("__BUNDLE__", str(bundle)))


def write_launch_manifest(
    bundle: Path,
    *,
    cases: list[dict[str, Any]],
    syccl: dict[str, Any],
    flow_sim: dict[str, Any],
    model_config: dict[str, str],
    args: argparse.Namespace,
) -> None:
  llm_diffstat = run_capture(["git", "-C", str(REPO_ROOT), "diff", "--stat"], check=False)
  manifest = {
      "launch_id": bundle.name,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "case_count": len(cases),
      "strategies": list(STRATEGIES),
      "max_generations": args.max_generations,
      "k_candidates": args.k_candidates,
      "simpletes_timeout": args.simpletes_timeout,
      "origin_timeout": args.origin_timeout,
      "syccl": syccl,
      "flow_sim": flow_sim,
      "llm_ccl": {
          "repo": str(REPO_ROOT),
          "git_commit": run_capture(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=False),
          "dirty_diffstat": llm_diffstat,
      },
      "llm_api": {
          "env_toml": str(args.env_toml.expanduser().resolve()) if args.env_toml else None,
          "model": model_config.get("model", DEFAULT_MODEL),
          "api_base": model_config.get("api_base", ""),
          "api_key_redacted": redact_secret(model_config.get("api_key")),
          "api_key_present": bool(model_config.get("api_key")),
      },
      "notes": [
          "Preparation only. Do not start long solve/search runs until this bundle is inspected.",
          "llm-ccl run scripts use the SimpleTES llm_elite path.",
      ],
  }
  write_json(bundle / "manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Prepare H800 llm-ccl vs SyCCL flow-sim comparison bundle.")
  parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
  parser.add_argument("--launch-id", default=None)
  parser.add_argument("--origin-repo", type=Path, default=DEFAULT_ORIGIN_REPO)
  parser.add_argument("--syccl-worktree", type=Path, default=DEFAULT_SYCCL_WORKTREE)
  parser.add_argument("--syccl-commit", default=TARGET_SYCCL_COMMIT)
  parser.add_argument("--setup-syccl-worktree", action="store_true")
  parser.add_argument("--build-syccl", action="store_true")
  parser.add_argument("--require-syccl-build", action="store_true")
  parser.add_argument("--cmake-bin", type=Path, default=DEFAULT_CMAKE_BIN)
  parser.add_argument("--scip-suite-dir", type=Path, default=DEFAULT_SCIP_SUITE_DIR)
  parser.add_argument("--scip-pp-dir", type=Path, default=DEFAULT_SCIP_PP_DIR)
  parser.add_argument("--syccl-build-parallel", type=int, default=max(1, min(16, os.cpu_count() or 1)))
  parser.add_argument("--env-toml", type=Path, default=DEFAULT_ENV_TOML)
  parser.add_argument("--flow-sim-bin", type=Path, default=None)
  parser.add_argument("--max-generations", type=int, default=10000)
  parser.add_argument("--k-candidates", type=int, default=4)
  parser.add_argument("--max-parallel", type=int, default=3)
  parser.add_argument("--simpletes-timeout", default="2h")
  parser.add_argument("--origin-timeout", default="10h")
  return parser


def prepare(args: argparse.Namespace) -> Path:
  launch_id = args.launch_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
  bundle = (args.bundle_root / launch_id).expanduser().resolve()
  bundle.mkdir(parents=True, exist_ok=True)
  (bundle / "provenance").mkdir(exist_ok=True)

  if args.setup_syccl_worktree:
    ensure_syccl_worktree(args.origin_repo.expanduser().resolve(), args.syccl_worktree.expanduser().resolve(), args.syccl_commit)
  if not args.syccl_worktree.exists():
    raise FileNotFoundError(
        f"SyCCL worktree not found: {args.syccl_worktree}. "
        "Run with --setup-syccl-worktree after reviewing the target path."
    )

  syccl_build = None
  if args.build_syccl:
    syccl_build = build_syccl_synthesize(
        bundle,
        args.syccl_worktree.expanduser().resolve(),
        cmake_bin=args.cmake_bin.expanduser(),
        scip_suite_dir=args.scip_suite_dir.expanduser().resolve(),
        scip_pp_dir=args.scip_pp_dir.expanduser().resolve(),
        parallel=args.syccl_build_parallel,
        require_success=args.require_syccl_build,
    )

  syccl = write_syccl_provenance(bundle, args.syccl_worktree.expanduser().resolve(), args.syccl_commit, syccl_build)
  model_config = load_env_toml(args.env_toml)
  flow_sim = freeze_flow_sim(bundle, args.flow_sim_bin.expanduser().resolve() if args.flow_sim_bin else None)
  cases = write_case_inputs(bundle, args.syccl_worktree.expanduser().resolve())
  write_run_scripts(
      bundle,
      cases,
      syccl_worktree=args.syccl_worktree.expanduser().resolve(),
      model_config=model_config,
      max_generations=args.max_generations,
      k_candidates=args.k_candidates,
      max_parallel=args.max_parallel,
      simpletes_timeout=args.simpletes_timeout,
      origin_timeout=args.origin_timeout,
  )
  write_launch_manifest(bundle, cases=cases, syccl=syccl, flow_sim=flow_sim, model_config=model_config, args=args)
  return bundle


def main(argv: Sequence[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  bundle = prepare(args)
  print(f"Wrote H800 comparison preparation bundle: {bundle}")
  print(f"Inspect manifest: {bundle / 'manifest.json'}")
  print(f"LLM runs: {bundle / 'runs' / 'run_llm_all.sh'}")
  print(f"Origin runs: {bundle / 'runs' / 'run_origin_all.sh'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
