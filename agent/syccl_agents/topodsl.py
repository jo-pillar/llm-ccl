from __future__ import annotations

import ast
import hashlib
import json
import runpy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_LOADED_TOPOLOGY_INSTANCES: list["BaseTopology"] = []


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


@dataclass(frozen=True)
class LinkSpec:
    bandwidth: Any
    latency: Any

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseTopology:
    def __init__(self, params: dict[str, Any] | None = None, link_specs: dict[str, LinkSpec] | None = None) -> None:
        self.params = dict(params or {})
        self.link_specs = dict(link_specs or {})
        self.connections: list[dict[str, Any]] = []
        self.build_topology()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def wrapped_init(self, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            if not hasattr(self, "params"):
                self.params = {}
            if not hasattr(self, "link_specs"):
                self.link_specs = {}
            if not hasattr(self, "connections"):
                self.connections = []
            self.build_topology()
            _LOADED_TOPOLOGY_INSTANCES.append(self)

        cls.__init__ = wrapped_init

    def connect(self, src_node: str, dst_node: str, link_spec: LinkSpec) -> None:
        spec = link_spec.to_dict()
        self.connections.append({
            "source": src_node,
            "destination": dst_node,
            "bandwidth": spec["bandwidth"],
            "latency": spec["latency"],
        })

    def build_topology(self) -> None:
        raise NotImplementedError


def load_topodsl(
    path: str | Path,
    *,
    collective: str | None = None,
    message_size: int | None = None,
) -> TopoDSLSpec:
    topo_path = Path(path).expanduser().resolve()
    source = topo_path.read_text(encoding="utf-8")
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
        prompt_source=source,
        params=params,
        config_id=_config_id(params),
    )


def _load_python_topology(
    path: Path,
    *,
    collective: str | None,
    message_size: int | None,
) -> dict[str, Any]:
    _LOADED_TOPOLOGY_INSTANCES.clear()
    try:
        namespace = runpy.run_path(
            str(path),
            init_globals={
                "BaseTopology": BaseTopology,
                "LinkSpec": LinkSpec,
            },
        )
    finally:
        instances = list(_LOADED_TOPOLOGY_INSTANCES)
        _LOADED_TOPOLOGY_INSTANCES.clear()
    topology = namespace.get("topology")
    if not callable(topology):
        return _extract_instantiated_topology(
            namespace,
            instances,
            collective=collective,
            message_size=message_size,
        )
    data = topology()
    if not isinstance(data, dict):
        raise ValueError("topology() must return a dict")
    return data


def _extract_instantiated_topology(
    namespace: dict[str, Any],
    instances: list[BaseTopology],
    *,
    collective: str | None,
    message_size: int | None,
) -> dict[str, Any]:
    candidates = list(instances) or [
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


def _multirail_object_to_data(
    topology: BaseTopology,
    *,
    collective: str | None,
    message_size: int | None,
) -> dict[str, Any]:
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


def _message_size_from_object(topology: BaseTopology, *, fallback: int | None = None) -> int:
    for name in ("message_size", "coll_bytes", "coll_byte"):
        if hasattr(topology, name):
            return _bytes_to_int(getattr(topology, name))
    if fallback is not None:
        return _bytes_to_int(fallback)
    raise ValueError("message_size must be provided by CLI or TopoDSL object")


def _collective_from_object(topology: BaseTopology, *, fallback: str | None = None) -> str:
    if hasattr(topology, "collective"):
        return str(getattr(topology, "collective")).lower()
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
