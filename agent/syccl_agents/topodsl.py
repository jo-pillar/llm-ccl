from __future__ import annotations

import ast
import hashlib
import json
import runpy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from syccl_agents.base_topology import (
    BaseTopology,
    CollectiveType,
    LayerSpec,
    LinkSpec,
    Node,
    NodeType,
)


@dataclass(frozen=True)
class TopologyParams:
    family: str
    hosts: int
    gpus_per_host: int
    nics_per_host: int
    message_size: int
    collective: str = "allgather"
    leaf_switches: int | None = None
    spine_switches: int | None = None
    rails: int | None = None
    host_links: str = "nvswitch"
    host_bw_mbpus: float = 0.3
    host_lat_us: float = 9.0
    nic_bw_mbpus: float = 0.0225
    nic_lat_us: float = 0.0
    net_bw_mbpus: float = 0.0225
    net_lat_us: float = 25.0
    spine_bw_mbpus: float = 0.36
    spine_lat_us: float = 25.0


@dataclass(frozen=True)
class TopoDSLSpec:
    path: Path
    prompt_source: str
    params: TopologyParams
    config_id: str


def load_topodsl(
    path: str | Path,
    *,
    collective: str | None = None,
    message_size: int | None = None,
) -> TopoDSLSpec:
    topo_path = Path(path).expanduser().resolve()
    source = topo_path.read_text(encoding="utf-8")
    prompt_source = _prompt_source(source) if topo_path.suffix == ".py" else source
    data = (
        _load_python_topology(topo_path, collective=collective, message_size=message_size)
        if topo_path.suffix == ".py"
        else _load_text_topology(source)
    )
    if collective is not None:
        data.setdefault("collective", collective)
    if message_size is not None:
        data.setdefault("message_size", message_size)
    params = _normalize_params(data)
    return TopoDSLSpec(
        path=topo_path,
        prompt_source=prompt_source,
        params=params,
        config_id=_config_id(params),
    )


def _prompt_source(source: str) -> str:
    begin = "###TopoBegin"
    end = "###TopoEND"
    begin_index = source.find(begin)
    end_index = source.find(end)
    if begin_index < 0 or end_index < 0 or end_index <= begin_index:
        return source
    return source[begin_index + len(begin):end_index].strip()


def _load_python_topology(
    path: Path,
    *,
    collective: str | None,
    message_size: int | None,
) -> dict[str, Any]:
    namespace = runpy.run_path(
        str(path),
        init_globals={
            "BaseTopology": BaseTopology,
            "LinkSpec": LinkSpec,
            "LayerSpec": LayerSpec,
            "Node": Node,
            "NodeType": NodeType,
            "CollectiveType": CollectiveType,
        },
    )
    topology = namespace.get("topology")
    if not callable(topology):
        return _extract_instantiated_topology(
            namespace,
            collective=collective,
            message_size=message_size,
        )
    data = topology()
    if not isinstance(data, dict):
        raise ValueError("topology() must return a dict")
    return data


def _extract_instantiated_topology(
    namespace: dict[str, Any],
    *,
    collective: str | None,
    message_size: int | None,
) -> dict[str, Any]:
    candidates = [
        value
        for value in namespace.values()
        if isinstance(value, BaseTopology) and value.__class__ is not BaseTopology
    ]
    if not candidates:
        raise ValueError("TopoDSL Python must define topology() or instantiate a BaseTopology subclass")
    topology = candidates[-1]
    return _topology_object_to_data(topology, collective=collective, message_size=message_size)


def _topology_object_to_data(
    topology: BaseTopology,
    *,
    collective: str | None,
    message_size: int | None,
) -> dict[str, Any]:
    class_name = topology.__class__.__name__.lower()
    if "clos" in class_name:
        return _clos_object_to_data(topology, collective=collective, message_size=message_size)
    if "multirail" in class_name:
        return _multirail_object_to_data(topology, collective=collective, message_size=message_size)
    raise ValueError(f"unsupported BaseTopology subclass {topology.__class__.__name__!r}")


def _clos_object_to_data(
    topology: BaseTopology,
    *,
    collective: str | None,
    message_size: int | None,
) -> dict[str, Any]:
    if hasattr(topology, "layer_spec_0"):
        return _clos_layer_spec_object_to_data(
            topology,
            collective=collective,
            message_size=message_size,
        )
    else:
        raise ValueError("ClosTopology must define layer_spec_0, layer_spec_1, layer_spec_2, and layer_spec_3")
    gpu_num = _positive_attr(topology, "gpu_num")
    host_num = _positive_attr(topology, "host_num")
    leaf_num = _positive_attr(topology, "leaf_num")
    spine_num = _positive_attr(topology, "spine_num")
    if gpu_num % host_num != 0:
        raise ValueError("gpu_num must be divisible by host_num")
    return {
        "family": "clos",
        "hosts": host_num,
        "gpus_per_host": gpu_num // host_num,
        "nics_per_host": int(getattr(topology, "nics_per_host", 1)),
        "leaf_switches": leaf_num,
        "spine_switches": spine_num,
        "message_size": _message_size_from_object(topology, fallback=message_size),
        "collective": _collective_from_object(topology, fallback=collective),
        "host_bw_mbpus": _bandwidth_to_mbpus(topology.link_spec_0.bandwidth),
        "host_lat_us": _latency_to_us(topology.link_spec_0.latency),
        "net_bw_mbpus": _bandwidth_to_mbpus(topology.link_spec_1.bandwidth),
        "net_lat_us": _latency_to_us(topology.link_spec_1.latency),
        "spine_bw_mbpus": _bandwidth_to_mbpus(topology.link_spec_2.bandwidth),
        "spine_lat_us": _latency_to_us(topology.link_spec_2.latency),
    }


def _clos_layer_spec_object_to_data(
    topology: BaseTopology,
    *,
    collective: str | None,
    message_size: int | None,
) -> dict[str, Any]:
    gpu_num = _positive_attr(topology, "gpu_num")
    host_local = _required_layer_spec(topology, 0)
    gpu_host = _required_layer_spec(topology, 1)
    host_leaf = _required_layer_spec(topology, 2)
    leaf_spine = _required_layer_spec(topology, 3)

    host_num = _positive_layer_value(host_local, "group_num")
    gpus_per_host = _positive_layer_value(host_local, "node_num")
    nics_per_host = _positive_layer_value(gpu_host, "node_num")
    leaf_switches = _positive_layer_value(host_leaf, "group_num")
    spine_switches = _positive_layer_value(leaf_spine, "group_num")
    if gpu_num != host_num * gpus_per_host:
        raise ValueError("gpu_num must match layer_spec_0.group_num * layer_spec_0.node_num")

    return {
        "family": "clos",
        "hosts": host_num,
        "gpus_per_host": gpus_per_host,
        "nics_per_host": nics_per_host,
        "leaf_switches": leaf_switches,
        "spine_switches": spine_switches,
        "message_size": _message_size_from_object(topology, fallback=message_size),
        "collective": _collective_from_object(topology, fallback=collective),
        "host_bw_mbpus": _bandwidth_to_mbpus(host_local.link_spec.bandwidth),
        "host_lat_us": _latency_to_us(host_local.link_spec.latency),
        "nic_bw_mbpus": _bandwidth_to_mbpus(gpu_host.link_spec.bandwidth),
        "nic_lat_us": _latency_to_us(gpu_host.link_spec.latency),
        "net_bw_mbpus": _bandwidth_to_mbpus(host_leaf.link_spec.bandwidth),
        "net_lat_us": _latency_to_us(host_leaf.link_spec.latency),
        "spine_bw_mbpus": _bandwidth_to_mbpus(leaf_spine.link_spec.bandwidth),
        "spine_lat_us": _latency_to_us(leaf_spine.link_spec.latency),
    }


def _multirail_object_to_data(
    topology: BaseTopology,
    *,
    collective: str | None,
    message_size: int | None,
) -> dict[str, Any]:
    if hasattr(topology, "layer_spec_0"):
        return _multirail_layer_spec_object_to_data(
            topology,
            collective=collective,
            message_size=message_size,
        )
    else:
        raise ValueError("MultiRailTopology must define layer_spec_0, layer_spec_1, and layer_spec_3")
    gpu_num = _positive_attr(topology, "gpu_num")
    host_num = _positive_attr(topology, "host_num")
    rails = int(getattr(topology, "rail_num", getattr(topology, "rails", 1)))
    if gpu_num % host_num != 0:
        raise ValueError("gpu_num must be divisible by host_num")
    return {
        "family": "multirail",
        "hosts": host_num,
        "gpus_per_host": gpu_num // host_num,
        "nics_per_host": int(getattr(topology, "nics_per_host", rails)),
        "rails": rails,
        "message_size": _message_size_from_object(topology, fallback=message_size),
        "collective": _collective_from_object(topology, fallback=collective),
    }


def _multirail_layer_spec_object_to_data(
    topology: BaseTopology,
    *,
    collective: str | None,
    message_size: int | None,
) -> dict[str, Any]:
    gpu_num = _positive_attr(topology, "gpu_num")
    host_local = _required_layer_spec(topology, 0)
    gpu_nic = _required_layer_spec(topology, 1)
    rail = _required_layer_spec(topology, 3)

    host_num = _positive_layer_value(host_local, "group_num")
    gpus_per_host = _positive_layer_value(host_local, "node_num")
    nics_per_host = _positive_layer_value(gpu_nic, "node_num")
    rails = _positive_layer_value(rail, "group_num")
    if gpu_num != host_num * gpus_per_host:
        raise ValueError("gpu_num must match layer_spec_0.group_num * layer_spec_0.node_num")

    return {
        "family": "multirail",
        "hosts": host_num,
        "gpus_per_host": gpus_per_host,
        "nics_per_host": nics_per_host,
        "rails": rails,
        "message_size": _message_size_from_object(topology, fallback=message_size),
        "collective": _collective_from_object(topology, fallback=collective),
        "host_bw_mbpus": _bandwidth_to_mbpus(host_local.link_spec.bandwidth),
        "host_lat_us": _latency_to_us(host_local.link_spec.latency),
        "nic_bw_mbpus": _bandwidth_to_mbpus(gpu_nic.link_spec.bandwidth),
        "nic_lat_us": _latency_to_us(gpu_nic.link_spec.latency),
        "net_bw_mbpus": _bandwidth_to_mbpus(rail.link_spec.bandwidth),
        "net_lat_us": _latency_to_us(rail.link_spec.latency),
    }


def _load_text_topology(source: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in source.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"invalid TopoDSL line: {raw_line}")
        key, value = line.split("=", 1)
        data[key.strip()] = _parse_scalar(value.strip())
    return data


def _parse_scalar(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


def _normalize_params(data: dict[str, Any]) -> TopologyParams:
    family = str(data.get("family", "")).lower()
    if family not in {"clos", "multirail"}:
        raise ValueError(f"unsupported topology family {family!r}; expected clos or multirail")

    hosts = _positive_int(data, "hosts")
    gpus_per_host = _positive_int(data, "gpus_per_host")
    nics_per_host = _positive_int(data, "nics_per_host")
    collective = str(data.get("collective", "allgather")).lower()
    message_size = int(data.get("message_size", data.get("coll_byte", 0)))
    if message_size <= 0:
        raise ValueError("message_size must be positive")
    if collective != "allgather":
        raise ValueError("this smoke workflow currently supports allgather")

    return TopologyParams(
        family=family,
        hosts=hosts,
        gpus_per_host=gpus_per_host,
        nics_per_host=nics_per_host,
        message_size=message_size,
        collective=collective,
        leaf_switches=_optional_positive_int(data, "leaf_switches"),
        spine_switches=_optional_positive_int(data, "spine_switches"),
        rails=_optional_positive_int(data, "rails"),
        host_links=str(data.get("host_links", "nvswitch")),
        host_bw_mbpus=float(data.get("host_bw_mbpus", 0.3)),
        host_lat_us=float(data.get("host_lat_us", 9.0)),
        nic_bw_mbpus=float(data.get("nic_bw_mbpus", 0.0225)),
        nic_lat_us=float(data.get("nic_lat_us", 0.0)),
        net_bw_mbpus=float(data.get("net_bw_mbpus", data.get("leaf_bw_mbpus", 0.0225))),
        net_lat_us=float(data.get("net_lat_us", data.get("leaf_lat_us", 25.0))),
        spine_bw_mbpus=float(data.get("spine_bw_mbpus", 0.36)),
        spine_lat_us=float(data.get("spine_lat_us", 25.0)),
    )


def _positive_int(data: dict[str, Any], key: str) -> int:
    value = int(data.get(key, 0))
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _optional_positive_int(data: dict[str, Any], key: str) -> int | None:
    if key not in data or data[key] is None:
        return None
    value = int(data[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _positive_attr(topology: BaseTopology, name: str) -> int:
    value = int(getattr(topology, name))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _required_layer_spec(topology: BaseTopology, layer_id: int) -> LayerSpec:
    value = getattr(topology, f"layer_spec_{layer_id}", None)
    if value is None:
        raise ValueError(f"layer_spec_{layer_id} must be provided")
    return value


def _positive_layer_value(layer_spec: LayerSpec, name: str) -> int:
    value = int(getattr(layer_spec, name))
    if value <= 0:
        raise ValueError(f"layer_spec_{layer_spec.layer_id}.{name} must be positive")
    return value


def _message_size_from_object(topology: BaseTopology, *, fallback: int | None = None) -> int:
    for name in ("message_size", "coll_bytes", "coll_byte"):
        if hasattr(topology, name):
            return _bytes_to_int(getattr(topology, name))
    if fallback is not None:
        return _bytes_to_int(fallback)
    raise ValueError("message_size must be provided by CLI or TopoDSL object")


def _collective_from_object(topology: BaseTopology, *, fallback: str | None = None) -> str:
    if hasattr(topology, "collective"):
        value = getattr(topology, "collective")
        return str(getattr(value, "value", value)).lower()
    if fallback is not None:
        return str(fallback).lower()
    return "allgather"


def _bytes_to_int(value: Any) -> int:
    if isinstance(value, (int, float)):
        size = int(value)
    else:
        text = str(value).strip().lower()
        multipliers = {
            "b": 1,
            "kb": 1024,
            "kib": 1024,
            "mb": 1024 ** 2,
            "mib": 1024 ** 2,
            "gb": 1024 ** 3,
            "gib": 1024 ** 3,
        }
        number, unit = _split_number_unit(text)
        size = int(float(number) * multipliers.get(unit or "b", 1))
    if size <= 0:
        raise ValueError("message_size must be positive")
    return size


def _bandwidth_to_mbpus(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    number, unit = _split_number_unit(text)
    amount = float(number)
    normalized = unit.replace("/s", "ps")
    if normalized in {"mbpus", "mb/us"}:
        return amount
    if normalized in {"gbps", "gb/s", "gbps"}:
        return amount / 1000.0
    if normalized in {"mbps", "mb/s"}:
        return amount / 1_000_000.0
    if normalized in {"tbps", "tb/s"}:
        return amount
    raise ValueError(f"unsupported bandwidth unit: {value!r}")


def _latency_to_us(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    number, unit = _split_number_unit(text)
    amount = float(number)
    if unit in {"us", "usec", "microsecond", "microseconds", ""}:
        return amount
    if unit in {"ns", "nsec", "nanosecond", "nanoseconds"}:
        return amount / 1000.0
    if unit in {"ms", "msec", "millisecond", "milliseconds"}:
        return amount * 1000.0
    raise ValueError(f"unsupported latency unit: {value!r}")


def _split_number_unit(text: str) -> tuple[str, str]:
    stripped = text.strip()
    index = 0
    while index < len(stripped) and (stripped[index].isdigit() or stripped[index] in ".+-"):
        index += 1
    number = stripped[:index]
    unit = stripped[index:].strip()
    if not number:
        raise ValueError(f"missing numeric value in {text!r}")
    return number, unit


def _config_id(params: TopologyParams) -> str:
    payload = json.dumps(asdict(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
