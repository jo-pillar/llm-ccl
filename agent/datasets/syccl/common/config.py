from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SycclConfigContext:
  path: Path
  data: dict[str, Any]
  host_num: int
  host_gpu_num: int
  host_nic_num: int
  ngpus: int
  collective: str
  coll_byte: int
  layer_groups: dict[int, dict[int, set[int]]]


def load_json(path: str | Path) -> dict[str, Any]:
  with Path(path).expanduser().open("r", encoding="utf-8") as f:
    return json.load(f)


def load_syccl_config(path: str | Path) -> SycclConfigContext:
  config_path = Path(path).expanduser().resolve()
  data = load_json(config_path)
  hosts = data.get("hosts", {})
  coll = data.get("coll", {})
  host_num = int(hosts.get("host_num", 0))
  host_gpu_num = int(hosts.get("host_gpu_num", 0))
  host_nic_num = int(hosts.get("host_nic_num", 0))
  if host_num <= 0 or host_gpu_num <= 0:
    raise ValueError("config hosts.host_num and hosts.host_gpu_num must be positive")
  return SycclConfigContext(
      path=config_path,
      data=data,
      host_num=host_num,
      host_gpu_num=host_gpu_num,
      host_nic_num=host_nic_num,
      ngpus=host_num * host_gpu_num,
      collective=str(coll.get("name", "allgather")),
      coll_byte=int(coll.get("byte", 0)),
      layer_groups=derive_layer_groups(data),
  )


def host_range(host: int, host_gpu_num: int) -> set[int]:
  start = host * host_gpu_num
  return set(range(start, start + host_gpu_num))


def derive_layer_groups(config: dict[str, Any]) -> dict[int, dict[int, set[int]]]:
  hosts = config.get("hosts", {})
  host_num = int(hosts.get("host_num", 0))
  host_gpu_num = int(hosts.get("host_gpu_num", 0))
  host_nic_num = int(hosts.get("host_nic_num", 0))
  if host_num <= 0 or host_gpu_num <= 0:
    raise ValueError("config hosts.host_num and hosts.host_gpu_num must be positive")

  groups: dict[int, dict[int, set[int]]] = {}
  prev_layer_type = ""
  prev_layer_id: int | None = None
  for layer in config.get("topo", []):
    layer_id = int(layer["layer_id"])
    layer_type = layer.get("type")
    if layer_type == "host":
      groups[layer_id] = {
          host: host_range(host, host_gpu_num)
          for host in range(host_num)
      }
    elif layer_type == "switch":
      switch_num = int(layer["switch_num"])
      switch_topo = layer.get("switch_topo")
      if prev_layer_type == "nic":
        if switch_topo == "multirail":
          groups[layer_id] = derive_multirail_groups(
              host_num=host_num,
              host_gpu_num=host_gpu_num,
              host_nic_num=host_nic_num,
              switch_num=switch_num,
          )
        elif switch_topo == "pod":
          groups[layer_id] = derive_first_pod_switch_groups(
              host_num=host_num,
              host_gpu_num=host_gpu_num,
              switch_num=switch_num,
          )
      elif prev_layer_type == "switch" and prev_layer_id in groups:
        if switch_topo == "pod":
          groups[layer_id] = derive_upper_pod_switch_groups(groups[prev_layer_id], switch_num)
    prev_layer_type = str(layer_type)
    prev_layer_id = layer_id
  return groups


def derive_multirail_groups(
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


def derive_first_pod_switch_groups(
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
      members.update(host_range(host, host_gpu_num))
    groups[switch_id] = members
  return groups


def derive_upper_pod_switch_groups(
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
