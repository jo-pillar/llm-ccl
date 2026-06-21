from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from syccl_agents.topodsl import TopologyParams


def render_syccl_config(params: TopologyParams) -> dict[str, Any]:
    if params.family == "clos":
        return _render_clos(params)
    if params.family == "multirail":
        return _render_multirail(params)
    raise ValueError(f"unsupported topology family {params.family}")


def write_syccl_config(params: TopologyParams, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(render_syccl_config(params), indent=2), encoding="utf-8")
    return output


def topology_summary(params: TopologyParams) -> str:
    switches = (
        f"leaf_switches={params.leaf_switches or 1}, spine_switches={params.spine_switches or 1}"
        if params.family == "clos"
        else f"rails={params.rails or params.nics_per_host}"
    )
    return (
        f"family={params.family}, hosts={params.hosts}, gpus_per_host={params.gpus_per_host}, "
        f"nics_per_host={params.nics_per_host}, {switches}, total_gpus={params.hosts * params.gpus_per_host}"
    )


def layer_group_summary(params: TopologyParams) -> str:
    host_groups = [
        f"layer 1 group {host}: GPUs {host * params.gpus_per_host}-{(host + 1) * params.gpus_per_host - 1}"
        for host in range(params.hosts)
    ]
    if params.family == "clos":
        leaf_count = params.leaf_switches or 1
        hosts_per_leaf = (params.hosts + leaf_count - 1) // leaf_count
        leaf_groups = []
        for leaf in range(leaf_count):
            start_host = leaf * hosts_per_leaf
            end_host = min(params.hosts, (leaf + 1) * hosts_per_leaf)
            if start_host >= end_host:
                continue
            start_gpu = start_host * params.gpus_per_host
            end_gpu = end_host * params.gpus_per_host - 1
            leaf_groups.append(f"layer 3 group {leaf}: GPUs {start_gpu}-{end_gpu}")
        upper = [f"layer 4 group 0: all GPUs 0-{params.hosts * params.gpus_per_host - 1}"]
        return "\n".join(host_groups + leaf_groups + upper)

    rails = params.rails or params.nics_per_host
    gpu_per_nic = max(1, params.gpus_per_host // params.nics_per_host)
    rail_groups = []
    for rail in range(rails):
        members = []
        for host in range(params.hosts):
            local_begin = rail * gpu_per_nic
            local_end = min(params.gpus_per_host, local_begin + gpu_per_nic)
            members.extend(host * params.gpus_per_host + gpu for gpu in range(local_begin, local_end))
        rail_groups.append(f"layer 3 group {rail}: GPUs {members}")
    return "\n".join(host_groups + rail_groups)


def seed_sketch_hint(params: TopologyParams) -> str:
    """Return one compact valid pattern for prompt anchoring when available."""
    total_gpus = params.hosts * params.gpus_per_host
    if params.family == "clos" and total_gpus == 4 and params.hosts == 2 and params.gpus_per_host == 2:
        return (
            '[{"step": 0, "layer": 1, "group": 0, "srcs": [0], "dsts": [1]}, '
            '{"step": 1, "layer": 3, "group": 0, "srcs": [0], "dsts": [2]}, '
            '{"step": 2, "layer": 1, "group": 1, "srcs": [2], "dsts": [3]}]'
        )
    if params.family == "multirail" and total_gpus == 4 and params.hosts == 2 and params.gpus_per_host == 2:
        return (
            '[{"step": 0, "layer": 1, "group": 0, "srcs": [0], "dsts": [1]}, '
            '{"step": 1, "layer": 3, "group": 0, "srcs": [0], "dsts": [2]}, '
            '{"step": 2, "layer": 1, "group": 1, "srcs": [2], "dsts": [3]}]'
        )
    return "No seed sketch is provided for this scale; generate a compact propagation tree from the layer/group semantics."


def _base(params: TopologyParams) -> dict[str, Any]:
    return {
        "coll": {
            "name": params.collective,
            "byte": params.message_size,
            "root_sender": -1,
            "root_receiver": -1,
        },
        "hosts": {
            "host_num": params.hosts,
            "host_gpu_num": params.gpus_per_host,
            "host_nic_num": params.nics_per_host,
            "host_links": params.host_links,
        },
        "host_links": {
            "nvlink": _full_mesh(params.gpus_per_host),
        },
        "link_spec": {
            "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
            "nvlink": {"bw_mbpus": params.host_bw_mbpus, "lat_us": params.host_lat_us},
            "link_nic": {"bw_mbpus": params.nic_bw_mbpus, "lat_us": params.nic_lat_us},
            "netlink": {"bw_mbpus": params.net_bw_mbpus, "lat_us": params.net_lat_us},
            "netlink_leaf": {"bw_mbpus": params.net_bw_mbpus, "lat_us": params.net_lat_us},
            "netlink_spine": {"bw_mbpus": params.spine_bw_mbpus, "lat_us": params.spine_lat_us},
        },
        "solver": {"split_chunks": 1, "chunk_size_B": 0},
        "sketch": {
            "customize_sketch": False,
            "use_sketch_input": False,
            "save_sketch": False,
            "sketch_path": "",
        },
        "prune": {},
        "algo_solve": {},
        "topodsl_params": asdict(params),
    }


def _render_clos(params: TopologyParams) -> dict[str, Any]:
    config = _base(params)
    config["topo"] = [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {
            "layer_id": 3,
            "type": "switch",
            "switch_topo": "pod",
            "switch_num": params.leaf_switches or 1,
            "link_spec": "netlink_leaf",
        },
        {
            "layer_id": 4,
            "type": "switch",
            "switch_topo": "pod",
            "switch_num": params.spine_switches or 1,
            "link_spec": "netlink_spine",
        },
    ]
    return config


def _render_multirail(params: TopologyParams) -> dict[str, Any]:
    config = _base(params)
    config["topo"] = [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {
            "layer_id": 3,
            "type": "switch",
            "switch_topo": "multirail",
            "switch_num": params.rails or params.nics_per_host,
            "link_spec": "netlink",
        },
    ]
    return config


def _full_mesh(size: int) -> list[list[int]]:
    return [[other for other in range(size) if other != gpu] for gpu in range(size)]
