from __future__ import annotations

import re
import runpy
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[3]
H800_TOPOLOGY = (
    AGENT_ROOT
    / "datasets"
    / "syccl"
    / "scheme1_direct_events"
    / "templates"
    / "H800_multirail"
    / "multirail_topo.py"
)
V100_DIR = (
    AGENT_ROOT
    / "datasets"
    / "syccl"
    / "scheme1_direct_events"
    / "templates"
    / "v100_dgx2_clos"
)
EXAMPLE_TOPOLOGY = AGENT_ROOT / "examples" / "topologies" / "clos_topo.py"


def edge_map(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    topology = runpy.run_path(path)["topology"]
    return {
        (src.node_id, dst.node_id): (spec.bandwidth, spec.latency)
        for (src, dst), spec in topology.connections.items()
    }


class TopologyTemplateTest(unittest.TestCase):
    def test_h800_uses_flat_gpu_ids_and_nvswitch_star(self):
        links = edge_map(H800_TOPOLOGY)

        self.assertIn(("gpu[0]", "nvswitch[0]"), links)
        self.assertIn(("gpu[511]", "nvswitch[63]"), links)
        self.assertFalse(any(src.startswith("gpu[") and dst.startswith("gpu[") for src, dst in links))
        self.assertFalse(self._has_nested_gpu_id(links))

    def test_v100_uses_flat_gpu_ids_and_authoritative_values(self):
        links = edge_map(V100_DIR / "clos_topo.py")

        self.assertEqual(("150GB/s", "3us"), links[("gpu[0]", "nvswitch[0]")])
        self.assertEqual(("150GB/s", "3us"), links[("gpu[63]", "nvswitch[3]")])
        self.assertEqual(("12.5GB/s", "0us"), links[("gpu[63]", "nic[3]")])
        self.assertEqual(("12.5GB/s", "3us"), links[("nic[3]", "leaf[3]")])
        self.assertEqual(("100GB/s", "0.5us"), links[("leaf[3]", "spine[0]")])
        self.assertFalse(self._has_nested_gpu_id(links))

    def test_example_clos_uses_same_flat_gpu_id_convention(self):
        links = edge_map(EXAMPLE_TOPOLOGY)

        self.assertIn(("gpu[63]", "nvswitch[7]"), links)
        self.assertIn(("gpu[63]", "host[7]"), links)
        self.assertFalse(self._has_nested_gpu_id(links))

    def test_v100_prompt_defers_link_values_to_topology(self):
        prompt = (V100_DIR / "prompt_template.txt").read_text(encoding="utf-8")

        self.assertNotIn("125 GB/s", prompt)
        self.assertNotIn("400 GB/s", prompt)
        self.assertIn("use their rendered link costs", prompt)

    @staticmethod
    def _has_nested_gpu_id(links: dict[tuple[str, str], tuple[str, str]]) -> bool:
        return any(
            re.fullmatch(r"gpu\[\d+\]\[\d+\]", endpoint)
            for edge in links
            for endpoint in edge
        )


if __name__ == "__main__":
    unittest.main()
