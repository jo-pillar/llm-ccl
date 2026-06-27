from __future__ import annotations

import runpy
from pathlib import Path

from syccl_agents.topodsl import load_topodsl


ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_DIR = ROOT / "agent" / "examples" / "topologies"
CLOS_TOPO = TOPOLOGY_DIR / "clos_topo.py"
MULTIRAIL_TOPO = TOPOLOGY_DIR / "multirail_topo.py"
TORUS_TOPO = TOPOLOGY_DIR / "torus_topo.py"


def test_clos_topo_example_builds_expected_connections():
    namespace = runpy.run_path(str(CLOS_TOPO))

    topology = namespace["topology"]
    connections = topology.connections

    assert len(connections) == 298
    assert _count_edges(connections, "gpu[", "gpu[") == 224
    assert _count_edges(connections, "gpu[", "host[") == 64
    assert _count_edges(connections, "host[", "leaf[") == 8
    assert _count_edges(connections, "leaf[", "spine[") == 2


def test_load_topodsl_reads_shared_base_topology_example():
    spec = load_topodsl(CLOS_TOPO)

    assert spec.params.family == "clos"
    assert spec.params.hosts == 8
    assert spec.params.gpus_per_host == 8
    assert spec.params.nics_per_host == 1
    assert spec.params.leaf_switches == 2
    assert spec.params.spine_switches == 1
    assert spec.params.message_size == 1024 * 1024
    assert spec.params.collective == "allgather"
    assert spec.params.host_bw_mbpus == 0.0325
    assert spec.params.host_lat_us == 9.0
    assert spec.params.nic_bw_mbpus == 0.0028125
    assert spec.params.net_bw_mbpus == 0.0028125
    assert spec.params.spine_bw_mbpus == 0.045
    assert spec.prompt_source.startswith("class ClosTopology(BaseTopology):")
    assert "from syccl_agents.base_topology" not in spec.prompt_source
    assert "ModuleNotFoundError" not in spec.prompt_source


def test_multirail_topo_matches_syccl_4host_8rail_config():
    namespace = runpy.run_path(str(MULTIRAIL_TOPO))

    topology = namespace["topology"]
    connections = topology.connections

    assert len(connections) == 128
    assert _count_edges(connections, "gpu[", "gpu[") == 64
    assert _count_edges(connections, "gpu[", "nic[") == 32
    assert _count_edges(connections, "nic[", "rail[") == 32

    spec = load_topodsl(MULTIRAIL_TOPO)
    assert spec.params.family == "multirail"
    assert spec.params.hosts == 4
    assert spec.params.gpus_per_host == 8
    assert spec.params.nics_per_host == 8
    assert spec.params.rails == 8
    assert spec.params.message_size == 1024 * 1024
    assert spec.params.host_bw_mbpus == 0.3
    assert spec.params.host_lat_us == 9.0
    assert spec.params.nic_bw_mbpus == 0.0225
    assert spec.params.net_bw_mbpus == 0.0225
    assert spec.params.net_lat_us == 25.0


def test_torus_topo_uses_400gbps_100ns_links():
    namespace = runpy.run_path(str(TORUS_TOPO))

    topology = namespace["topology"]
    connections = topology.connections

    assert len(connections) == 32
    assert _count_edges(connections, "gpu[", "gpu[") == 32
    assert {link.bandwidth for link in connections.values()} == {"400Gb/s"}
    assert {link.latency for link in connections.values()} == {"100ns"}


def _count_edges(connections, left_prefix: str, right_prefix: str) -> int:
    total = 0
    for src, dst in connections:
        endpoints = (src.node_id, dst.node_id)
        if endpoints[0].startswith(left_prefix) and endpoints[1].startswith(right_prefix):
            total += 1
        elif endpoints[0].startswith(right_prefix) and endpoints[1].startswith(left_prefix):
            total += 1
    return total
