from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from syccl_agents.topodsl import TopologyParams


TEMPLATE_DIR = Path(__file__).resolve().parent / "config_templates"


def render_syccl_config(params: TopologyParams) -> dict[str, Any]:
  if params.family not in {"clos", "multirail"}:
    raise ValueError(f"unsupported topology family {params.family}")
  config = _load_template(params.family)
  _apply_common_overrides(config, params)
  if params.family == "clos":
    _apply_clos_overrides(config, params)
  else:
    _apply_multirail_overrides(config, params)
  config["topodsl_params"] = asdict(params)
  return config


def write_syccl_config(params: TopologyParams, path: str | Path) -> Path:
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(render_syccl_config(params), indent=2), encoding="utf-8")
  return output


def total_gpus(params: TopologyParams) -> int:
  return params.hosts * params.gpus_per_host


def _load_template(family: str) -> dict[str, Any]:
  template_path = TEMPLATE_DIR / f"{family}.json"
  try:
    data = json.loads(template_path.read_text(encoding="utf-8"))
  except OSError as exc:
    raise FileNotFoundError(f"config template not found: {template_path}") from exc
  if not isinstance(data, dict):
    raise ValueError(f"config template must be a JSON object: {template_path}")
  return copy.deepcopy(data)


def _apply_common_overrides(config: dict[str, Any], params: TopologyParams) -> None:
  coll = config.setdefault("coll", {})
  coll["name"] = params.collective
  coll["byte"] = params.message_size
  coll.setdefault("root_sender", -1)
  coll.setdefault("root_receiver", -1)

  hosts = config.setdefault("hosts", {})
  hosts["host_num"] = params.hosts
  hosts["host_gpu_num"] = params.gpus_per_host
  hosts["host_nic_num"] = params.nics_per_host
  hosts["host_links"] = params.host_links

  _set_link_spec(config, "nvlink", params.host_bw_mbpus, params.host_lat_us)
  _set_link_spec(config, "link_nic", params.nic_bw_mbpus, params.nic_lat_us)

  sketch = config.setdefault("sketch", {})
  sketch["customize_sketch"] = False
  sketch["use_sketch_input"] = False
  sketch["save_sketch"] = False
  sketch["sketch_path"] = ""


def _apply_clos_overrides(config: dict[str, Any], params: TopologyParams) -> None:
  _set_switch_num(config, "pod", 0, params.leaf_switches or 1)
  _set_switch_num(config, "pod", 1, params.spine_switches or 1)
  _set_link_spec(config, "netlink_leaf", params.net_bw_mbpus, params.net_lat_us)
  _set_link_spec(config, "netlink_spine", params.spine_bw_mbpus, params.spine_lat_us)


def _apply_multirail_overrides(config: dict[str, Any], params: TopologyParams) -> None:
  rails = params.rails or params.nics_per_host
  if rails != params.nics_per_host:
    raise ValueError(
        "multirail config requires rails to match nics_per_host because "
        "flow-sim-rs requires switch_num == host_nic_num"
    )
  _set_switch_num(config, "multirail", 0, rails)
  _set_link_spec(config, "netlink", params.net_bw_mbpus, params.net_lat_us)


def _set_link_spec(config: dict[str, Any], name: str, bw_mbpus: float, lat_us: float) -> None:
  link_spec = config.setdefault("link_spec", {})
  entry = link_spec.setdefault(name, {})
  entry["bw_mbpus"] = bw_mbpus
  entry["lat_us"] = lat_us


def _set_switch_num(
    config: dict[str, Any],
    switch_topo: str,
    ordinal: int,
    switch_num: int,
) -> None:
  switches = [
      layer
      for layer in config.get("topo", [])
      if isinstance(layer, dict)
      and layer.get("type") == "switch"
      and layer.get("switch_topo") == switch_topo
  ]
  if ordinal >= len(switches):
    raise ValueError(f"config template missing {switch_topo} switch layer ordinal {ordinal}")
  switches[ordinal]["switch_num"] = switch_num
