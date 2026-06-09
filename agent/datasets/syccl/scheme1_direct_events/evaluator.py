"""SimpleTES evaluator for SyCCL scheme 1: compact sketch -> direct events."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from datasets.syccl.common.config import (
  SycclConfigContext,
  derive_layer_groups,
  load_syccl_config,
)
from datasets.syccl.common.sketch import (
  SketchValidationError,
  normalize_sketches as normalize_common_sketches,
)

try:
  from simpletes.construction import capture_construction_if_requested
except Exception:  # pragma: no cover - evaluator also works outside SimpleTES
  def capture_construction_if_requested(value: Any) -> bool:
    return False


AGENT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = AGENT_ROOT.parent


def _resolve_syccl_root() -> Path:
  raw_path = os.environ.get("SYCCL_REPO_ROOT")
  if raw_path:
    return Path(raw_path).expanduser().resolve()
  if (WORKSPACE_ROOT / "syccl" / "config").exists():
    return WORKSPACE_ROOT / "syccl"
  return WORKSPACE_ROOT


REPO_ROOT = _resolve_syccl_root()
DEFAULT_BASE_CONFIG = REPO_ROOT / "config" / "a100-8gpu-4nic-clos-ag-4k.json"
FALLBACK_BASE_CONFIG = REPO_ROOT / "config" / "a100-8gpu-4nic-clos-ag.json"
SYNTHESIZE_BIN = REPO_ROOT / "build" / "synthesize"
SCHEME_NAME = "scheme1_direct_events"
DEFAULT_ARTIFACT_ROOT = AGENT_ROOT / "eval_artifacts"


def _resolve_flow_sim_root() -> Path:
  raw_path = os.environ.get("SYCCL_FLOW_SIM_ROOT")
  if raw_path:
    return Path(raw_path).expanduser().resolve()
  if (WORKSPACE_ROOT / "Flow-Simulator" / "flow-sim-rs").exists():
    return WORKSPACE_ROOT / "Flow-Simulator" / "flow-sim-rs"
  return WORKSPACE_ROOT / "flow-sim-rs"


FLOW_SIM_ROOT = _resolve_flow_sim_root()
FLOW_SIM_BIN = Path(os.environ.get(
    "SYCCL_FLOW_SIM_BIN",
    str(FLOW_SIM_ROOT / "target" / "release" / "flow-sim-rs"),
)).expanduser()
EVAL_ARTIFACT_ROOT = Path(os.environ.get(
    "SYCCL_EVAL_ARTIFACT_DIR",
    str(DEFAULT_ARTIFACT_ROOT),
)).expanduser()

MAX_SKETCHES = int(os.environ.get("SYCCL_MAX_CANDIDATE_SKETCHES", "6"))
MAX_STEP = int(os.environ.get("SYCCL_MAX_SKETCH_STEP", "31"))
TIMEOUT_SECONDS = int(os.environ.get("SYCCL_EVALUATOR_TIMEOUT_SECONDS", "120"))
LOG_PREVIEW_CHARS = 4000
FAILURE_SCORE = -1_000_000_000_000.0
FAILURE_BEST_TIME_US = 1_000_000_000_000.0


def _resolve_base_config() -> Path:
  raw_path = os.environ.get("SYCCL_BASE_CONFIG")
  if not raw_path:
    return DEFAULT_BASE_CONFIG if DEFAULT_BASE_CONFIG.exists() else FALLBACK_BASE_CONFIG
  path = Path(raw_path)
  if not path.is_absolute():
    path = REPO_ROOT / path
  return path


BASE_CONFIG = _resolve_base_config()

ROOT_GPU = 0


def _load_config(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def _host_range(host: int, host_gpu_num: int) -> set[int]:
  start = host * host_gpu_num
  return set(range(start, start + host_gpu_num))


def _derive_multirail_groups(
    *,
    host_num: int,
    host_gpu_num: int,
    host_nic_num: int,
    switch_num: int,
) -> dict[int, set[int]]:
  if host_gpu_num % host_nic_num != 0:
    raise ValueError("host_gpu_num must be divisible by host_nic_num for multirail groups")
  if host_nic_num % switch_num != 0:
    raise ValueError("host_nic_num must be divisible by multirail switch_num")
  gpu_per_nic = host_gpu_num // host_nic_num
  nics_per_switch = host_nic_num // switch_num
  groups: dict[int, set[int]] = {}
  for switch_id in range(switch_num):
    members: set[int] = set()
    nic_begin = switch_id * nics_per_switch
    nic_end = (switch_id + 1) * nics_per_switch
    for host in range(host_num):
      host_base = host * host_gpu_num
      for nic in range(nic_begin, nic_end):
        local_begin = nic * gpu_per_nic
        local_end = local_begin + gpu_per_nic
        members.update(range(host_base + local_begin, host_base + local_end))
    groups[switch_id] = members
  return groups


def _derive_first_pod_switch_groups(
    *,
    host_num: int,
    host_gpu_num: int,
    switch_num: int,
) -> dict[int, set[int]]:
  if host_num % switch_num != 0:
    raise ValueError("host_num must be divisible by pod switch_num")
  hosts_per_switch = host_num // switch_num
  groups: dict[int, set[int]] = {}
  for switch_id in range(switch_num):
    members: set[int] = set()
    for host in range(switch_id * hosts_per_switch, (switch_id + 1) * hosts_per_switch):
      members.update(_host_range(host, host_gpu_num))
    groups[switch_id] = members
  return groups


def _derive_upper_pod_switch_groups(
    lower_groups: dict[int, set[int]],
    switch_num: int,
) -> dict[int, set[int]]:
  if len(lower_groups) % switch_num != 0:
    raise ValueError("lower pod switch count must be divisible by upper switch_num")
  lower_per_switch = len(lower_groups) // switch_num
  groups: dict[int, set[int]] = {}
  ordered_lower = [lower_groups[i] for i in sorted(lower_groups)]
  for switch_id in range(switch_num):
    members: set[int] = set()
    for group in ordered_lower[switch_id * lower_per_switch:(switch_id + 1) * lower_per_switch]:
      members.update(group)
    groups[switch_id] = members
  return groups


def _topology_name(config: dict[str, Any]) -> str:
  switch_topos = [
      str(layer.get("switch_topo"))
      for layer in config.get("topo", [])
      if layer.get("type") == "switch"
  ]
  if "multirail" in switch_topos:
    return "multirail"
  if "pod" in switch_topos:
    return "Clos"
  return "host-local"


def _format_gpu_members(members: set[int], *, host_gpu_num: int) -> str:
  ordered = sorted(members)
  if not ordered:
    return "empty"
  if len(ordered) <= 16:
    return ", ".join(str(gpu) for gpu in ordered)
  host_ids = sorted({gpu // host_gpu_num for gpu in ordered})
  if len(host_ids) <= 8:
    host_text = ", ".join(str(host) for host in host_ids)
  else:
    host_text = f"{host_ids[0]}..{host_ids[-1]} ({len(host_ids)} hosts)"
  local_ids = sorted({gpu % host_gpu_num for gpu in ordered})
  if len(local_ids) <= 8:
    local_text = ", ".join(str(local) for local in local_ids)
  else:
    local_text = f"{local_ids[0]}..{local_ids[-1]}"
  return f"hosts {host_text}, local GPUs {local_text}"


def _layer_group_text(config: dict[str, Any], layer_groups: dict[int, dict[int, set[int]]]) -> str:
  hosts = config["hosts"]
  host_gpu_num = int(hosts["host_gpu_num"])
  lines = []
  topo_by_layer = {
      int(layer["layer_id"]): layer
      for layer in config.get("topo", [])
  }
  for layer_id in sorted(layer_groups):
    layer = topo_by_layer.get(layer_id, {})
    label = layer.get("switch_topo") or layer.get("type") or "layer"
    group_map = layer_groups[layer_id]
    lines.append(f"- Layer {layer_id} ({label}) has {len(group_map)} groups:")
    if len(group_map) > 8:
      first = group_map[min(group_map)]
      lines.append(
          f"  - group g follows the same formula; group 0 contains "
          f"{_format_gpu_members(first, host_gpu_num=host_gpu_num)}"
      )
    else:
      for group_id in sorted(group_map):
        lines.append(
            f"  - group {group_id}: "
            f"{_format_gpu_members(group_map[group_id], host_gpu_num=host_gpu_num)}"
        )
  return "\n".join(lines)


def render_instruction_for_config(config_path: str | Path) -> str:
  config_path = Path(config_path)
  config = _load_config(config_path)
  hosts = config["hosts"]
  coll = config["coll"]
  host_num = int(hosts["host_num"])
  host_gpu_num = int(hosts["host_gpu_num"])
  ngpus = host_num * host_gpu_num
  collective = str(coll["name"])
  coll_byte = int(coll["byte"])
  topology = _topology_name(config)
  layer_groups = derive_layer_groups(config)
  allowed_layers = ", ".join(str(layer_id) for layer_id in sorted(layer_groups))

  if collective == "alltoall":
    expansion_text = (
      "The evaluator interprets the compact tree as routing information for "
      "alltoall direct events. Good sketches should provide balanced parent "
      "paths from root 0 to every destination because the expanded alltoall "
      "uses those parent relationships for every source/destination pair."
    )
  else:
    expansion_text = (
      "The evaluator expands the single-root broadcast tree into one analogous "
      "tree per source chunk/root GPU using topology-preserving rotations."
    )

  return f"""Optimize SyCCL direct-event sketch generation for a {host_num}-host {topology} {collective} target.

You are evolving Python code that returns candidate single-root broadcast sketches
in a compact DSL. The active config is `{config_path}` with {ngpus} GPUs total
and coll.byte={coll_byte}. {expansion_text}

Optimization objective:

    combined_score = -best_time_us

Higher `combined_score` is better, so minimize the simulated completion time.

Topology and allowed groups:
{_layer_group_text(config, layer_groups)}

Required return format:
- Implement `construct_sketches()`.
- Only Return on Sketch at one time
- A sketch is a list of compact transmissions.
- Preferred compact transmission form: `(step, layer, group, srcs, dsts)`.
- `srcs` and `dsts` may each be a single GPU id or a flat list of GPU ids.

Validity constraints:
- Use only layers {allowed_layers}.
- A transmission's srcs and dsts must all be inside the specified layer/group.
- GPU ids must be integers in 0..{ngpus - 1}.
- GPU 0 starts with the root chunk.
- Every GPU 1..{ngpus - 1} must be reached exactly once; GPU 0 must not appear as a dst.
- A GPU may be used as a source only after it has received the chunk in an earlier step, except GPU 0.
- Dependent steps must be strictly increasing. Same-step forwarding through a newly reached GPU is invalid.
- No transmission may have duplicate dst ids, and no sketch may deliver to the same dst twice.

Evaluator metrics:
- `combined_score`: the only score to maximize.
- `validity`: 1.0 for a valid sketch that simulates successfully, 0.0 otherwise.
- `best_time_us`: primary performance value to minimize.
- Critical-link fields are diagnostic clues for contention and latency.
"""


try:
  BASE_CONFIG_DATA = _load_config(BASE_CONFIG)
except OSError:
  BASE_CONFIG_DATA = {
    "hosts": {
      "host_num": int(os.environ.get("SYCCL_TASK_HOST_NUM", "4")),
      "host_gpu_num": int(os.environ.get("SYCCL_TASK_HOST_GPU_NUM", "8")),
      "host_nic_num": int(os.environ.get("SYCCL_TASK_HOST_NIC_NUM", "4")),
    },
    "coll": {"name": "allgather", "byte": 4096},
    "topo": [
      {"layer_id": 1, "type": "host"},
      {"layer_id": 3, "type": "switch", "switch_topo": "pod", "switch_num": 2},
      {"layer_id": 4, "type": "switch", "switch_topo": "pod", "switch_num": 1},
    ],
  }

TASK_HOST_NUM = int(BASE_CONFIG_DATA.get("hosts", {}).get("host_num", 4))
TASK_HOST_GPU_NUM = int(BASE_CONFIG_DATA.get("hosts", {}).get("host_gpu_num", 8))
TASK_HOST_NIC_NUM = int(BASE_CONFIG_DATA.get("hosts", {}).get("host_nic_num", 4))
NGPUS = TASK_HOST_NUM * TASK_HOST_GPU_NUM
LAYER_GROUPS = derive_layer_groups(BASE_CONFIG_DATA)
CONFIG_CONTEXT = load_syccl_config(BASE_CONFIG) if BASE_CONFIG.exists() else SycclConfigContext(
    path=BASE_CONFIG,
    data=BASE_CONFIG_DATA,
    host_num=TASK_HOST_NUM,
    host_gpu_num=TASK_HOST_GPU_NUM,
    host_nic_num=TASK_HOST_NIC_NUM,
    ngpus=NGPUS,
    collective=str(BASE_CONFIG_DATA.get("coll", {}).get("name", "allgather")),
    coll_byte=int(BASE_CONFIG_DATA.get("coll", {}).get("byte", 4096)),
    layer_groups=LAYER_GROUPS,
)


def _resolve_artifact_root() -> Path:
  path = EVAL_ARTIFACT_ROOT
  if not path.is_absolute():
    path = AGENT_ROOT / path
  return path


def _make_eval_workdir() -> Path:
  base = _resolve_artifact_root() / SCHEME_NAME
  base.mkdir(parents=True, exist_ok=True)
  stamp = time.strftime("%Y%m%d-%H%M%S")
  return Path(tempfile.mkdtemp(prefix=f"eval_{stamp}_{os.getpid()}_", dir=str(base)))


def _copy_program(program_path: str, workdir: Path) -> None:
  try:
    source = Path(program_path)
    (workdir / "program.py").write_text(
        source.read_text(encoding="utf-8", errors="replace"),
        encoding="utf-8",
    )
  except OSError:
    pass


def _write_metrics(workdir: Path | None, metrics: dict[str, Any]) -> dict[str, Any]:
  if workdir is None:
    return metrics
  try:
    (workdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
  except (OSError, TypeError, ValueError):
    pass
  return metrics


def _empty_bottleneck_metrics() -> dict[str, Any]:
  return {
    "critical_link_src_chunk": "",
    "critical_link_path": "",
    "critical_link_hop_count": 0,
    "critical_link_queue_wait_ns": 0,
    "critical_link_latency_ns": 0,
    "critical_link_beta_cost_ns": 0,
    "critical_link_total_ns": 0,
    "critical_link_queue_wait_pct": 0.0,
    "critical_link_latency_pct": 0.0,
    "critical_link_beta_cost_pct": 0.0,
  }


def _failure_details(message: str) -> dict[str, str]:
  lower = message.lower()
  details = {
    "failure_category": "unknown_failure",
    "failure_feedback": _preview(message, 1200),
  }

  if (
      ("failed to map node" in lower and "size mismatch" in lower)
      or (
          "failed to expand sketch node" in lower
          and "mapped src/dst set sizes changed" in lower
      )
  ):
    details["failure_category"] = "syccl_expand_incompatible"
    details["failure_feedback"] = (
      "SyCCL could not complete topology symmetry expansion for this compact "
      "sketch. Choose the proper layer/group for each transmission instead "
      "of assigning all traffic to a complete/top layer; same-host sends "
      "should use the host layer, and split cross-host delivery into "
      "representative sends plus local fanout steps."
    )
    return details

  match = re.search(r"source GPU (\d+).*step (\d+)", message)
  if match:
    gpu, step = match.groups()
    details["failure_category"] = "dependency_error"
    details["failure_location"] = f"GPU {gpu}, step {step}"
    reached_match = re.search(r"first reached at step (\d+)", message)
    if reached_match:
      reached_step = reached_match.group(1)
      details["failure_feedback"] = (
        f"GPU {gpu} is used as a source at step {step}, but it is first "
        f"reached at step {reached_step}. A GPU may fan out only in a "
        "strictly later step than the step that first delivers it."
      )
    else:
      details["failure_feedback"] = (
        f"GPU {gpu} is used as a source at step {step} before it is reachable. "
        "Deliver this GPU in an earlier step, or move its fanout to a later step."
      )
    return details

  match = re.search(r"destination GPU (\d+) is reached more than once", message)
  if match:
    gpu = match.group(1)
    details["failure_category"] = "duplicate_destination"
    details["failure_location"] = f"GPU {gpu}"
    details["failure_feedback"] = (
      f"GPU {gpu} appears as a destination in more than one transmission. "
      "Each GPU should be reached exactly once in the single-root broadcast tree."
    )
    return details

  match = re.search(r"does not cover all GPUs; missing (.+)$", message)
  if match:
    missing = match.group(1)
    details["failure_category"] = "incomplete_coverage"
    details["failure_feedback"] = (
      f"The sketch leaves these GPUs unreachable: {missing}. Add transmissions "
      "from already-reached sources so every GPU 0..31 is covered exactly once."
    )
    return details

  match = re.search(r"layer (\d+) group (\d+) cannot connect GPUs (.+)$", message)
  if match:
    layer, group, gpus = match.groups()
    details["failure_category"] = "topology_group_error"
    details["failure_location"] = f"layer {layer}, group {group}"
    details["failure_feedback"] = (
      f"Layer {layer} group {group} cannot contain GPUs {gpus}. Use only "
      "the layer/group memberships described in the task prompt generated "
      "from the active SyCCL config."
    )
    return details

  if "must not be empty" in lower:
    details["failure_category"] = "dsl_schema_error"
    details["failure_feedback"] = (
      "A transmission has an empty source or destination set. Remove that "
      "transmission or provide at least one integer GPU id."
    )
    return details

  if "flat list" in lower or "must be a gpu id" in lower or "must be an integer" in lower:
    details["failure_category"] = "dsl_schema_error"
    details["failure_feedback"] = (
      "Use either one integer GPU id or a flat list of integer GPU ids for "
      "srcs/dsts. Nested lists such as [[10, 11], [12, 13]] are not valid."
    )
    return details

  if "timed out" in lower:
    details["failure_category"] = "syccl_timeout"
    details["failure_feedback"] = (
      "SyCCL timed out evaluating this sketch. Prefer fewer transmissions, "
      "fewer candidate sketches, and simpler dependency structure."
    )
    return details

  if "syccl direct resim exited" in lower:
    details["failure_category"] = "syccl_runtime_error"
    details["failure_feedback"] = (
      "SyCCL rejected or crashed while evaluating this sketch. Check that the "
      "compact tree can be expanded under the active topology and avoids invalid "
      "layer/group/source/destination combinations."
    )
    return details

  return details


def _error_result(message: str, **extra: Any) -> dict[str, Any]:
  result = {
    "combined_score": FAILURE_SCORE,
    "validity": 0.0,
    "best_time_us": FAILURE_BEST_TIME_US,
    "error": message,
  }
  result.update(_empty_bottleneck_metrics())
  result.update(_failure_details(message))
  result.update(extra)
  return result


def _preview(text: str, limit: int = LOG_PREVIEW_CHARS) -> str:
  if len(text) <= limit:
    return text
  return text[-limit:]


def _load_program(program_path: str):
  spec = importlib.util.spec_from_file_location("candidate_syccl_sketch", program_path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load program: {program_path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _as_int(value: Any, field: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise SketchValidationError(f"{field} must be an integer")
  return value


def _as_gpu_set(value: Any, field: str) -> set[int]:
  if isinstance(value, bool):
    raise SketchValidationError(f"{field} must be a GPU id or a list of GPU ids")
  if isinstance(value, int):
    raw_gpus = [value]
  elif isinstance(value, (list, tuple, set)):
    raw_gpus = list(value)
  else:
    raise SketchValidationError(f"{field} must be a GPU id or a list of GPU ids")

  gpus: set[int] = set()
  for gpu in raw_gpus:
    if isinstance(gpu, (list, tuple, set)):
      raise SketchValidationError(
          f"{field} must be a GPU id or a flat list of GPU ids; "
          f"nested {type(gpu).__name__} values are not valid"
      )
    gpu_i = _as_int(gpu, field)
    if gpu_i < 0 or gpu_i >= NGPUS:
      raise SketchValidationError(f"{field} contains out-of-range GPU {gpu_i}")
    if gpu_i in gpus:
      raise SketchValidationError(f"{field} contains duplicate GPU {gpu_i}")
    gpus.add(gpu_i)
  if not gpus:
    raise SketchValidationError(f"{field} must not be empty")
  return gpus


def _check_layer_group(layer: int, group: int, srcs: set[int], dsts: set[int]) -> None:
  if layer not in LAYER_GROUPS:
    allowed = ", ".join(str(layer_id) for layer_id in sorted(LAYER_GROUPS))
    raise SketchValidationError(f"unsupported layer {layer}; allowed layers are {allowed}")
  if group not in LAYER_GROUPS[layer]:
    allowed = ", ".join(str(group_id) for group_id in sorted(LAYER_GROUPS[layer]))
    raise SketchValidationError(
        f"unsupported group {group} for layer {layer}; allowed groups are {allowed}"
    )
  members = LAYER_GROUPS[layer][group]
  outside = (srcs | dsts) - members
  if outside:
    raise SketchValidationError(
        f"layer {layer} group {group} cannot connect GPUs {sorted(outside)}"
    )


def _looks_like_int(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool)


def _looks_like_gpu_operand(value: Any) -> bool:
  if _looks_like_int(value):
    return True
  if isinstance(value, (list, tuple, set)):
    return all(_looks_like_int(gpu) for gpu in value)
  return False


def _looks_like_native_sketch(value: Any) -> bool:
  return isinstance(value, dict) and "nodes" in value


def _looks_like_transmission(value: Any) -> bool:
  if isinstance(value, dict):
    return (
      {"step", "layer", "group"}.issubset(value)
      and (
        {"srcs", "dsts"}.issubset(value)
        or {"src", "dst"}.issubset(value)
      )
    )
  if not isinstance(value, (list, tuple)) or len(value) != 5:
    return False
  step, layer, group, srcs, dsts = value
  return (
      _looks_like_int(step)
      and _looks_like_int(layer)
      and _looks_like_int(group)
      and _looks_like_gpu_operand(srcs)
      and _looks_like_gpu_operand(dsts)
  )


def _looks_like_transmission_list(value: Any) -> bool:
  return (
      isinstance(value, (list, tuple))
      and bool(value)
      and all(_looks_like_transmission(item) for item in value)
  )


def _normalize_transmission(raw: Any, index: int) -> dict[str, Any]:
  if isinstance(raw, dict):
    step_raw = raw.get("step")
    layer_raw = raw.get("layer")
    group_raw = raw.get("group")
    srcs_raw = raw["srcs"] if "srcs" in raw else raw.get("src")
    dsts_raw = raw["dsts"] if "dsts" in raw else raw.get("dst")
  elif isinstance(raw, (list, tuple)) and len(raw) == 5:
    step_raw, layer_raw, group_raw, srcs_raw, dsts_raw = raw
  else:
    raise SketchValidationError(
        f"transmission {index} must be a dict or (step, layer, group, srcs, dsts)"
    )

  step = _as_int(step_raw, f"transmission {index}.step")
  layer = _as_int(layer_raw, f"transmission {index}.layer")
  group = _as_int(group_raw, f"transmission {index}.group")
  if step < 0 or step > MAX_STEP:
    raise SketchValidationError(
        f"transmission {index}.step must be in [0, {MAX_STEP}]"
    )
  srcs = _as_gpu_set(srcs_raw, f"transmission {index}.srcs")
  dsts = _as_gpu_set(dsts_raw, f"transmission {index}.dsts")
  if srcs & dsts:
    raise SketchValidationError(f"transmission {index} has overlapping srcs/dsts")
  _check_layer_group(layer, group, srcs, dsts)
  return {
    "step": step,
    "layer": layer,
    "group": group,
    "srcs": sorted(srcs),
    "dsts": sorted(dsts),
  }


def _native_to_transmissions(sketch: dict[str, Any]) -> list[dict[str, Any]]:
  if sketch.get("ngpus") != NGPUS:
    raise SketchValidationError(f"native sketch ngpus must be {NGPUS}")
  if sketch.get("src_gpu") != ROOT_GPU:
    raise SketchValidationError(f"native sketch src_gpu must be {ROOT_GPU}")
  nodes = sketch.get("nodes")
  if not isinstance(nodes, list):
    raise SketchValidationError("native sketch nodes must be a list")
  transmissions = []
  for node in nodes:
    pair = node.get("src_dest_pair", {})
    transmissions.append({
      "step": node.get("step"),
      "layer": node.get("layer"),
      "group": node.get("group"),
      "srcs": pair.get("srcs"),
      "dsts": pair.get("dsts"),
    })
  return transmissions


def _normalize_candidate(raw_candidate: Any) -> list[dict[str, Any]]:
  if isinstance(raw_candidate, dict) and "nodes" in raw_candidate:
    raw_candidate = _native_to_transmissions(raw_candidate)
  if not isinstance(raw_candidate, (list, tuple)) or not raw_candidate:
    raise SketchValidationError("each sketch must be a non-empty list of transmissions")

  transmissions = [
    _normalize_transmission(tx, i)
    for i, tx in enumerate(raw_candidate)
  ]
  return sorted(
      transmissions,
      key=lambda tx: (
        tx["step"],
        tx["layer"],
        tx["group"],
        tx["srcs"],
        tx["dsts"],
      ),
  )


def _build_graph(transmissions: list[dict[str, Any]]) -> dict[str, Any]:
  reached_by_node: dict[int, int] = {ROOT_GPU: -1}
  reached_step: dict[int, int] = {ROOT_GPU: -1}
  first_delivery_step: dict[int, int] = {}
  for tx in transmissions:
    for dst in tx["dsts"]:
      first_delivery_step[dst] = min(
          tx["step"],
          first_delivery_step.get(dst, tx["step"]),
      )
  nodes: list[dict[str, Any]] = []

  for tx in transmissions:
    node_id = len(nodes)
    step = tx["step"]
    deps: set[int] = set()
    for src in tx["srcs"]:
      if src not in reached_by_node:
        if src in first_delivery_step:
          delivery_step = first_delivery_step[src]
          raise SketchValidationError(
              f"source GPU {src} has not been reached before step {step}; "
              f"GPU {src} is first reached at step {delivery_step}, so it can "
              "only be used as a source in a strictly later step"
          )
        raise SketchValidationError(
            f"source GPU {src} has not been reached before step {step}"
        )
      dep = reached_by_node[src]
      if dep >= 0:
        if reached_step[src] >= step:
          raise SketchValidationError(
              f"source GPU {src} is reached at step {reached_step[src]} "
              f"but reused at step {step}; dependent steps must be strictly later"
          )
        deps.add(dep)

    for dst in tx["dsts"]:
      if dst in reached_by_node:
        raise SketchValidationError(f"destination GPU {dst} is reached more than once")

    nodes.append({
      "id": node_id,
      "step": step,
      "layer": tx["layer"],
      "group": tx["group"],
      "src_dest_pair": {
        "srcs": tx["srcs"],
        "dsts": tx["dsts"],
      },
      "deps": sorted(deps),
      "next": [],
    })
    for dst in tx["dsts"]:
      reached_by_node[dst] = node_id
      reached_step[dst] = step

  missing = sorted(set(range(NGPUS)) - set(reached_by_node))
  if missing:
    raise SketchValidationError(f"sketch does not cover all GPUs; missing {missing}")

  for node in nodes:
    for dep in node["deps"]:
      nodes[dep]["next"].append(node["id"])
  for node in nodes:
    node["next"] = sorted(node["next"])

  return {
    "ngpus": NGPUS,
    "src_gpu": ROOT_GPU,
    "nodes": nodes,
  }


def _normalize_sketches(raw: Any) -> list[dict[str, Any]]:
  return normalize_common_sketches(
      raw,
      CONFIG_CONTEXT,
      max_sketches=MAX_SKETCHES,
      max_step=MAX_STEP,
  )


def _write_eval_files(
    raw_sketches: Any,
    workdir: Path,
) -> tuple[Path, Path, Path]:
  sketch_path = workdir / "candidate-sketch.json"
  result_path = workdir / "candidate-flow-sim.json"
  log_path = workdir / "candidate-flow-sim.log"

  sketch_path.write_text(json.dumps(raw_sketches, indent=2), encoding="utf-8")
  return sketch_path, result_path, log_path


def _run_syccl(
    config_path: Path,
    sketch_path: Path,
    result_path: Path,
    translated_path: Path,
    log_path: Path,
) -> tuple[int, float, str]:
  env = os.environ.copy()
  env.setdefault("SYNTHESIZE_PARALLEL_THREAD_NUM", "4")
  start = time.time()
  proc = subprocess.run(
      [
        str(SYNTHESIZE_BIN),
        "-f",
        str(config_path),
        "resim",
        "--sketch",
        "-i",
        str(sketch_path),
        "-o",
        str(result_path),
        "--dump-translated",
        str(translated_path),
      ],
      cwd=str(REPO_ROOT),
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      timeout=TIMEOUT_SECONDS,
      check=False,
  )
  wall = time.time() - start
  log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
  return proc.returncode, wall, proc.stdout


def _run_flow_sim(
    config_path: Path,
    translated_path: Path,
    result_path: Path,
    log_path: Path,
) -> tuple[int, float, str]:
  start = time.time()
  proc = subprocess.run(
      [
        str(FLOW_SIM_BIN),
        "simulate",
        "--config",
        str(config_path),
        "--translated",
        str(translated_path),
        "--output",
        str(result_path),
      ],
      cwd=str(FLOW_SIM_ROOT),
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      timeout=TIMEOUT_SECONDS,
      check=False,
  )
  wall = time.time() - start
  log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
  return proc.returncode, wall, proc.stdout


def _run_flow_sim_sketch(
    config_path: Path,
    sketch_path: Path,
    result_path: Path,
    log_path: Path,
) -> tuple[int, float, str]:
  start = time.time()
  proc = subprocess.run(
      [
        str(FLOW_SIM_BIN),
        "simulate-sketch",
        "--config",
        str(config_path),
        "--sketch",
        str(sketch_path),
        "--output",
        str(result_path),
      ],
      cwd=str(FLOW_SIM_ROOT),
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      timeout=TIMEOUT_SECONDS,
      check=False,
  )
  wall = time.time() - start
  log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
  return proc.returncode, wall, proc.stdout


def _float_field(value: Any) -> float | None:
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


def _extract_best_time_us(result: dict[str, Any]) -> float | None:
  for key in ("Time", "time_us", "best_time_us", "sketch_solve_best_time"):
    value = _float_field(result.get(key))
    if value is not None and value > 0:
      return value

  alg_times = result.get("alg_times")
  if isinstance(alg_times, list):
    times = [
      value
      for value in (_float_field(item) for item in alg_times)
      if value is not None and value > 0
    ]
    if times:
      return min(times)

  algorithms = result.get("algorithms")
  if isinstance(algorithms, list):
    times = []
    for algo in algorithms:
      if not isinstance(algo, dict):
        continue
      for key in ("Time", "time_us"):
        value = _float_field(algo.get(key))
        if value is not None and value > 0:
          times.append(value)
      final_schedule = algo.get("final_schedule")
      if isinstance(final_schedule, dict):
        value = _float_field(final_schedule.get("Time"))
        if value is not None and value > 0:
          times.append(value)
    if times:
      return min(times)

  return None


def _parse_src_chunk(value: Any) -> tuple[int, int]:
  if isinstance(value, str):
    match = re.fullmatch(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", value)
    if not match:
      raise ValueError(f"invalid src_chunk string: {value}")
    return int(match.group(1)), int(match.group(2))
  if isinstance(value, (list, tuple)) and len(value) == 2:
    return int(value[0]), int(value[1])
  raise ValueError(f"unsupported src_chunk value: {value}")


def _send_sort_key(send: dict[str, Any]) -> tuple[int, int, int, int]:
  return (
    int(send.get("epoch", 0)),
    int(send.get("src_gpu", -1)),
    int(send.get("dst_gpu", -1)),
    int(send.get("layer_used", -1)),
  )


def _build_predecessor_chain(
    sorted_sends: list[dict[str, Any]],
    target_send: dict[str, Any],
) -> list[dict[str, Any]]:
  first_arrival = {}
  for send in sorted_sends:
    dst_gpu = int(send["dst_gpu"])
    first_arrival.setdefault(dst_gpu, send)

  chain = [target_send]
  current_gpu = int(target_send["src_gpu"])
  visited = set()
  while current_gpu not in visited:
    visited.add(current_gpu)
    predecessor = first_arrival.get(current_gpu)
    if predecessor is None:
      break
    chain.append(predecessor)
    current_gpu = int(predecessor["src_gpu"])

  chain.reverse()
  return chain


def _select_last_send_chain(result: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
  schedule = result.get("Schedule")
  if not isinstance(schedule, dict):
    raise ValueError("resim result has no Schedule object")
  events = schedule.get("Events")
  if not isinstance(events, list):
    raise ValueError("resim result Schedule.Events is not a list")

  latest: tuple[int, dict[str, Any], dict[str, Any]] | None = None
  for event_group in events:
    if not isinstance(event_group, dict):
      continue
    sends = event_group.get("sends")
    if not isinstance(sends, list):
      continue
    for send in sends:
      if not isinstance(send, dict):
        continue
      epoch = int(send.get("epoch", 0))
      if latest is None or epoch >= latest[0]:
        latest = (epoch, event_group, send)

  if latest is None:
    raise ValueError("resim result has no sends")

  _, event_group, target_send = latest
  sends = [
    send
    for send in event_group.get("sends", [])
    if isinstance(send, dict)
  ]
  sorted_sends = sorted(sends, key=_send_sort_key)
  return event_group.get("src_chunk"), _build_predecessor_chain(sorted_sends, target_send)


def _summarize_critical_link_costs(
    link_trace: list[dict[str, Any]],
    src_chunk: tuple[int, int],
    critical_sends: list[dict[str, Any]],
) -> dict[str, Any]:
  critical_edges = {
    (int(send["src_gpu"]), int(send["dst_gpu"]))
    for send in critical_sends
  }
  matched = []
  for trace in link_trace:
    if not isinstance(trace, dict):
      continue
    try:
      trace_src_chunk = _parse_src_chunk(trace.get("src_chunk"))
    except ValueError:
      continue
    if trace_src_chunk != src_chunk:
      continue
    edge = (int(trace["src_gpu"]), int(trace["dst_gpu"]))
    if edge not in critical_edges:
      continue
    if int(trace.get("slice_id", 0)) != int(trace.get("nslices", 1)) - 1:
      continue
    matched.append(trace)

  summary = {
    "critical_link_hop_count": len(matched),
    "critical_link_queue_wait_ns": sum(
      int(trace.get("queue_wait_ns", 0)) for trace in matched
    ),
    "critical_link_latency_ns": sum(
      int(trace.get("latency_ns", 0)) for trace in matched
    ),
    "critical_link_beta_cost_ns": sum(
      int(trace.get("beta_cost_ns", 0)) for trace in matched
    ),
  }
  total = (
    summary["critical_link_queue_wait_ns"]
    + summary["critical_link_latency_ns"]
    + summary["critical_link_beta_cost_ns"]
  )
  summary["critical_link_total_ns"] = total
  if total > 0:
    summary["critical_link_queue_wait_pct"] = (
      summary["critical_link_queue_wait_ns"] * 100.0 / total
    )
    summary["critical_link_latency_pct"] = (
      summary["critical_link_latency_ns"] * 100.0 / total
    )
    summary["critical_link_beta_cost_pct"] = (
      summary["critical_link_beta_cost_ns"] * 100.0 / total
    )
  else:
    summary["critical_link_queue_wait_pct"] = 0.0
    summary["critical_link_latency_pct"] = 0.0
    summary["critical_link_beta_cost_pct"] = 0.0
  return summary


def _critical_link_summary(result: dict[str, Any]) -> dict[str, Any]:
  metrics = _empty_bottleneck_metrics()
  src_chunk_raw, critical_sends = _select_last_send_chain(result)
  src_chunk = _parse_src_chunk(src_chunk_raw)
  path = [str(int(critical_sends[0]["src_gpu"]))] if critical_sends else []
  path.extend(str(int(send["dst_gpu"])) for send in critical_sends)
  metrics["critical_link_src_chunk"] = str(src_chunk_raw)
  metrics["critical_link_path"] = "->".join(path)

  link_trace = result.get("LinkTrace", [])
  if isinstance(link_trace, list):
    metrics.update(_summarize_critical_link_costs(link_trace, src_chunk, critical_sends))
  return metrics


def _flow_sim_summary(result: dict[str, Any]) -> dict[str, Any]:
  metrics = _empty_bottleneck_metrics()
  links = result.get("links")
  if isinstance(links, list) and links:
    critical = max(
        (link for link in links if isinstance(link, dict)),
        key=lambda link: int(link.get("queue_wait_ns", 0)) + int(link.get("busy_ns", 0)),
        default=None,
    )
    if critical is not None:
      metrics["critical_link_path"] = f"{critical.get('src', '')}->{critical.get('dst', '')}"
      metrics["critical_link_hop_count"] = 1
      metrics["critical_link_queue_wait_ns"] = int(critical.get("queue_wait_ns", 0))
      metrics["critical_link_beta_cost_ns"] = int(critical.get("busy_ns", 0))
      total = metrics["critical_link_queue_wait_ns"] + metrics["critical_link_beta_cost_ns"]
      metrics["critical_link_total_ns"] = total
      if total > 0:
        metrics["critical_link_queue_wait_pct"] = metrics["critical_link_queue_wait_ns"] * 100.0 / total
        metrics["critical_link_beta_cost_pct"] = metrics["critical_link_beta_cost_ns"] * 100.0 / total
  for key in (
      "finish_time_ns",
      "flow_count",
      "channel_count",
      "total_queue_wait_ns",
      "max_source_cross_epoch_fanout",
      "source_cross_epoch_pressure_ns",
      "max_epoch",
      "epoch_barrier_ns",
      "critical_flow_id",
  ):
    if key in result:
      metrics[key] = result[key]
  return metrics


def evaluate(program_path: str) -> dict[str, Any]:
  """Evaluate a generated SimpleTES program with the Rust flow simulator."""
  workdir: Path | None = None
  try:
    if not BASE_CONFIG.exists():
      return _error_result(f"missing base config: {BASE_CONFIG}")
    if not FLOW_SIM_BIN.exists():
      return _error_result(f"missing flow-sim binary: {FLOW_SIM_BIN}")

    workdir = _make_eval_workdir()
    _copy_program(program_path, workdir)

    module = _load_program(program_path)
    if not hasattr(module, "run_code"):
      return _write_metrics(workdir, _error_result("program must define run_code()"))

    raw = module.run_code()
    sketch_path, result_path, log_path = _write_eval_files(raw, workdir)
    try:
      returncode, wall, log = _run_flow_sim_sketch(
          BASE_CONFIG,
          sketch_path,
          result_path,
          log_path,
      )
    except subprocess.TimeoutExpired:
      return _write_metrics(workdir, _error_result(
          f"flow-sim timed out after {TIMEOUT_SECONDS}s",
      ))

    if returncode != 0:
      return _write_metrics(workdir, _error_result(
          f"flow-sim exited with code {returncode}: {_preview(log)}",
      ))
    if not result_path.exists():
      return _write_metrics(workdir, _error_result(
          f"flow-sim did not produce {result_path.name}; log={_preview(log)}",
      ))

    try:
      result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
      preview = _preview(result_path.read_text(encoding="utf-8", errors="replace"))
      return _write_metrics(workdir, _error_result(
          f"invalid flow-sim JSON: {exc}; preview={preview}",
      ))

    best_time = _extract_best_time_us(result)
    if best_time is None:
      preview = _preview(json.dumps(result, sort_keys=True)[:LOG_PREVIEW_CHARS])
      return _write_metrics(workdir, _error_result(
          f"flow-sim output has no positive time field; preview={preview}; log={_preview(log)}",
      ))

    capture_construction_if_requested(raw)
    metrics = {
      "combined_score": -float(best_time),
      "validity": 1.0,
      "best_time_us": float(best_time),
    }
    metrics.update(_flow_sim_summary(result))
    return _write_metrics(workdir, metrics)

  except SketchValidationError as exc:
    return _write_metrics(workdir, _error_result(f"invalid sketch: {exc}"))
  except Exception as exc:
    return _write_metrics(
        workdir,
        _error_result(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"),
    )
