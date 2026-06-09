from __future__ import annotations

import json
from typing import Any

from .config import SycclConfigContext


ROOT_GPU = 0
DEFAULT_MAX_SKETCHES = 6
DEFAULT_MAX_STEP = 31


class SketchValidationError(ValueError):
  """Raised when a generated compact sketch is invalid."""


def normalize_sketches(
    raw: Any,
    context: SycclConfigContext,
    *,
    max_sketches: int = DEFAULT_MAX_SKETCHES,
    max_step: int = DEFAULT_MAX_STEP,
) -> list[dict[str, Any]]:
  if looks_like_native_sketch(raw):
    candidates = [raw]
  elif looks_like_transmission(raw):
    raise SketchValidationError(
        "run_code() must return a sketch list, not one bare transmission"
    )
  elif isinstance(raw, (list, tuple)) and raw and all(
      looks_like_native_sketch(item) for item in raw
  ):
    candidates = list(raw)
  elif looks_like_transmission_list(raw):
    candidates = [raw]
  elif isinstance(raw, (list, tuple)):
    candidates = list(raw)
  else:
    raise SketchValidationError(
        "run_code() must return one sketch or a list of sketches"
    )

  if not candidates:
    raise SketchValidationError("no sketches returned")
  if len(candidates) > max_sketches:
    raise SketchValidationError(
        f"too many sketches returned: {len(candidates)} > {max_sketches}"
    )

  graphs = []
  seen = set()
  for candidate in candidates:
    transmissions = normalize_candidate(
        candidate,
        context,
        max_step=max_step,
    )
    graph = build_graph(transmissions, context.ngpus)
    fingerprint = json.dumps(graph["nodes"], sort_keys=True)
    if fingerprint in seen:
      continue
    seen.add(fingerprint)
    graphs.append(graph)

  if not graphs:
    raise SketchValidationError("all sketches were duplicates")
  return graphs


def normalize_candidate(
    raw_candidate: Any,
    context: SycclConfigContext,
    *,
    max_step: int = DEFAULT_MAX_STEP,
) -> list[dict[str, Any]]:
  if isinstance(raw_candidate, dict) and "nodes" in raw_candidate:
    raw_candidate = native_to_transmissions(raw_candidate, context.ngpus)
  if not isinstance(raw_candidate, (list, tuple)) or not raw_candidate:
    raise SketchValidationError("each sketch must be a non-empty list of transmissions")

  transmissions = [
      normalize_transmission(tx, i, context, max_step=max_step)
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


def build_graph(transmissions: list[dict[str, Any]], ngpus: int) -> dict[str, Any]:
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

  missing = sorted(set(range(ngpus)) - set(reached_by_node))
  if missing:
    raise SketchValidationError(f"sketch does not cover all GPUs; missing {missing}")

  for node in nodes:
    for dep in node["deps"]:
      nodes[dep]["next"].append(node["id"])
  for node in nodes:
    node["next"] = sorted(node["next"])

  return {
    "ngpus": ngpus,
    "src_gpu": ROOT_GPU,
    "nodes": nodes,
  }


def normalize_transmission(
    raw: Any,
    index: int,
    context: SycclConfigContext,
    *,
    max_step: int = DEFAULT_MAX_STEP,
) -> dict[str, Any]:
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

  step = as_int(step_raw, f"transmission {index}.step")
  layer = as_int(layer_raw, f"transmission {index}.layer")
  group = as_int(group_raw, f"transmission {index}.group")
  if step < 0 or step > max_step:
    raise SketchValidationError(
        f"transmission {index}.step must be in [0, {max_step}]"
    )
  srcs = as_gpu_set(srcs_raw, f"transmission {index}.srcs", context.ngpus)
  dsts = as_gpu_set(dsts_raw, f"transmission {index}.dsts", context.ngpus)
  if srcs & dsts:
    raise SketchValidationError(f"transmission {index} has overlapping srcs/dsts")
  check_layer_group(context, layer, group, srcs, dsts)
  return {
    "step": step,
    "layer": layer,
    "group": group,
    "srcs": sorted(srcs),
    "dsts": sorted(dsts),
  }


def native_to_transmissions(sketch: dict[str, Any], ngpus: int) -> list[dict[str, Any]]:
  if sketch.get("ngpus") != ngpus:
    raise SketchValidationError(f"native sketch ngpus must be {ngpus}")
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


def check_layer_group(
    context: SycclConfigContext,
    layer: int,
    group: int,
    srcs: set[int],
    dsts: set[int],
) -> None:
  if layer not in context.layer_groups:
    allowed = ", ".join(str(layer_id) for layer_id in sorted(context.layer_groups))
    raise SketchValidationError(f"unsupported layer {layer}; allowed layers are {allowed}")
  if group not in context.layer_groups[layer]:
    allowed = ", ".join(str(group_id) for group_id in sorted(context.layer_groups[layer]))
    raise SketchValidationError(
        f"unsupported group {group} for layer {layer}; allowed groups are {allowed}"
    )
  members = context.layer_groups[layer][group]
  outside = (srcs | dsts) - members
  if outside:
    raise SketchValidationError(
        f"layer {layer} group {group} cannot connect GPUs {sorted(outside)}"
    )


def as_int(value: Any, field: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise SketchValidationError(f"{field} must be an integer")
  return value


def as_gpu_set(value: Any, field: str, ngpus: int) -> set[int]:
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
    gpu_i = as_int(gpu, field)
    if gpu_i < 0 or gpu_i >= ngpus:
      raise SketchValidationError(f"{field} contains out-of-range GPU {gpu_i}")
    if gpu_i in gpus:
      raise SketchValidationError(f"{field} contains duplicate GPU {gpu_i}")
    gpus.add(gpu_i)
  if not gpus:
    raise SketchValidationError(f"{field} must not be empty")
  return gpus


def looks_like_int(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool)


def looks_like_gpu_operand(value: Any) -> bool:
  if looks_like_int(value):
    return True
  if isinstance(value, (list, tuple, set)):
    return all(looks_like_int(gpu) for gpu in value)
  return False


def looks_like_native_sketch(value: Any) -> bool:
  return isinstance(value, dict) and "nodes" in value


def looks_like_transmission(value: Any) -> bool:
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
      looks_like_int(step)
      and looks_like_int(layer)
      and looks_like_int(group)
      and looks_like_gpu_operand(srcs)
      and looks_like_gpu_operand(dsts)
  )


def looks_like_transmission_list(value: Any) -> bool:
  return (
      isinstance(value, (list, tuple))
      and bool(value)
      and all(looks_like_transmission(item) for item in value)
  )
