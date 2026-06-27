import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from syccl_agents.config_render import render_syccl_config
from syccl_agents.topodsl import load_topodsl


class SycclTopoDSLTest(unittest.TestCase):
  def test_bundled_multirail_topodsl_loads_512_gpu_topology(self):
    spec = load_topodsl(ROOT / "datasets" / "syccl" / "scheme1_direct_events" / "multirail_topo.py")

    self.assertEqual("multirail", spec.params.family)
    self.assertEqual(64, spec.params.hosts)
    self.assertEqual(8, spec.params.gpus_per_host)
    self.assertEqual(8, spec.params.nics_per_host)
    self.assertEqual(8, spec.params.rails)
    self.assertEqual(1024 * 1024, spec.params.message_size)

  def test_multirail_config_template_keeps_host_nic_and_switch_counts_consistent(self):
    spec = load_topodsl(ROOT / "datasets" / "syccl" / "scheme1_direct_events" / "multirail_topo.py")

    config = render_syccl_config(spec.params)

    self.assertEqual(64, config["hosts"]["host_num"])
    self.assertEqual(8, config["hosts"]["host_gpu_num"])
    self.assertEqual(8, config["hosts"]["host_nic_num"])
    self.assertEqual("multirail", config["topo"][-1]["switch_topo"])
    self.assertEqual(8, config["topo"][-1]["switch_num"])
    self.assertIn("mip_config", config["solver"])
    self.assertEqual(0.045, config["link_spec"]["netlink"]["bw_mbpus"])
    self.assertEqual(21.5, config["link_spec"]["netlink"]["lat_us"])


if __name__ == "__main__":
  unittest.main()
