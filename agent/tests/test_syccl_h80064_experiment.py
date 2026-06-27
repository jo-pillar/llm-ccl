import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prepare_syccl_h80064_llmelite_experiment.py"


def load_script():
  spec = importlib.util.spec_from_file_location("prepare_syccl_h80064", SCRIPT_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class SycclH80064ExperimentTest(unittest.TestCase):
  def test_case_specs_match_origin_h80064_allgather_byte_semantics(self):
    script = load_script()

    cases = script.build_case_specs()

    self.assertEqual(
        [
            65536,
            262144,
            1048576,
            4194304,
            16777216,
            67108864,
            268435456,
            1073741824,
        ],
        [case.total_message_size for case in cases],
    )
    self.assertEqual([size // 512 for size in [case.total_message_size for case in cases]], [case.config_coll_byte for case in cases])
    self.assertEqual(10.0, cases[0].net_lat_us)
    self.assertEqual(3.0, cases[0].host_lat_us)
    self.assertEqual(21.5, cases[-1].net_lat_us)
    self.assertEqual(10.5, cases[-1].host_lat_us)

  def test_prepare_experiment_writes_topodsl_configs_and_run_scripts(self):
    script = load_script()
    with tempfile.TemporaryDirectory(prefix="syccl_h80064_exp_") as tmp:
      output_root = Path(tmp) / "result" / "config"

      manifest_path = script.prepare_experiment(output_root=output_root)

      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      self.assertEqual(["linear_rank", "balance", "all"], manifest["strategies"])
      self.assertEqual(24, len(manifest["runs"]))

      first = manifest["cases"][0]
      first_config = json.loads(Path(first["config_path"]).read_text(encoding="utf-8"))
      self.assertEqual("allgather", first_config["coll"]["name"])
      self.assertEqual(128, first_config["coll"]["byte"])
      self.assertEqual({"host_num": 64, "host_gpu_num": 8, "host_nic_num": 8, "host_links": "nvswitch"}, first_config["hosts"])
      self.assertEqual(8, first_config["topo"][-1]["switch_num"])
      self.assertEqual(0.0455, first_config["link_spec"]["netlink"]["bw_mbpus"])
      self.assertEqual(10.0, first_config["link_spec"]["netlink"]["lat_us"])

      last = manifest["cases"][-1]
      last_config = json.loads(Path(last["config_path"]).read_text(encoding="utf-8"))
      self.assertEqual(2097152, last_config["coll"]["byte"])
      self.assertEqual(21.5, last_config["link_spec"]["netlink"]["lat_us"])

      first_topodsl = Path(first["topodsl_path"]).read_text(encoding="utf-8")
      self.assertIn('"message_size": 65536', first_topodsl)
      self.assertIn('"family": "multirail"', first_topodsl)

      run_one = (manifest_path.parent / "runs" / "run_one.sh").read_text(encoding="utf-8")
      self.assertIn("--selector llm_elite", run_one)
      self.assertIn("--elite-selection-strategy", run_one)
      self.assertIn("--num-chains 1", run_one)
      self.assertIn("timeout 1h", run_one)

      run_all = (manifest_path.parent / "runs" / "run_all.sh").read_text(encoding="utf-8")
      self.assertIn("xargs -P 5", run_all)


if __name__ == "__main__":
  unittest.main()
