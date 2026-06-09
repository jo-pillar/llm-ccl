import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_PATH = (
    ROOT / "agent" / "datasets" / "syccl" / "scheme1_direct_events" / "evaluator.py"
)


def load_evaluator_with_config(config_path: Path):
  spec = importlib.util.spec_from_file_location(
      f"scheme1_dynamic_{config_path.stem}",
      EVALUATOR_PATH,
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  with patch.dict("os.environ", {"SYCCL_BASE_CONFIG": str(config_path)}):
    spec.loader.exec_module(module)
  return module


def write_config(path: Path, *, coll_name: str, hosts: int, topology: str):
  if topology == "multirail":
    topo = [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {"layer_id": 3, "type": "switch", "switch_topo": "multirail", "switch_num": 8, "link_spec": "netlink"},
    ]
    link_spec = {
        "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
        "nvlink": {"bw_mbpus": 0.15, "lat_us": 3},
        "link_nic": {"bw_mbpus": 0.0455, "lat_us": 0},
        "netlink": {"bw_mbpus": 0.0455, "lat_us": 10},
    }
  else:
    topo = [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {"layer_id": 3, "type": "switch", "switch_topo": "pod", "switch_num": 2, "link_spec": "netlink_leaf"},
        {"layer_id": 4, "type": "switch", "switch_topo": "pod", "switch_num": 1, "link_spec": "netlink_spine"},
    ]
    link_spec = {
        "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
        "nvlink": {"bw_mbpus": 0.3, "lat_us": 9},
        "link_nic": {"bw_mbpus": 0.0225, "lat_us": 0},
        "netlink_leaf": {"bw_mbpus": 0.0225, "lat_us": 25},
        "netlink_spine": {"bw_mbpus": 0.36, "lat_us": 25},
    }
  payload = {
      "coll": {"name": coll_name, "byte": 4096, "root_sender": -1, "root_receiver": -1},
      "hosts": {"host_num": hosts, "host_gpu_num": 8, "host_nic_num": 8 if topology == "multirail" else 4, "host_links": "nvswitch"},
      "topo": topo,
      "host_links": {"nvlink": [[1], [0], [3], [2], [5], [4], [7], [6]]},
      "link_spec": link_spec,
      "solver": {},
      "sketch": {"customize_sketch": False, "use_sketch_input": False, "save_sketch": False, "sketch_path": ""},
      "prune": {},
      "algo_solve": {"solve_output": str(path.with_suffix(".result.json"))},
  }
  path.write_text(json.dumps(payload), encoding="utf-8")


class SycclDynamicConfigTest(unittest.TestCase):
  def test_evaluator_derives_multirail_512gpu_groups_from_config(self):
    with tempfile.TemporaryDirectory(prefix="syccl_dynamic_") as tmp:
      config_path = Path(tmp) / "multirail-512gpu-a2a-4k.json"
      write_config(config_path, coll_name="alltoall", hosts=64, topology="multirail")

      evaluator = load_evaluator_with_config(config_path)

      self.assertEqual(512, evaluator.NGPUS)
      self.assertEqual(64, evaluator.TASK_HOST_NUM)
      self.assertEqual(8, evaluator.TASK_HOST_GPU_NUM)
      self.assertEqual(set(range(8)), evaluator.LAYER_GROUPS[1][0])
      self.assertEqual(64, len(evaluator.LAYER_GROUPS[3][0]))
      self.assertTrue({0, 8, 16, 504}.issubset(evaluator.LAYER_GROUPS[3][0]))

      instruction = evaluator.render_instruction_for_config(config_path)
      self.assertIn("512 GPUs", instruction)
      self.assertIn("alltoall", instruction)
      self.assertIn("multirail", instruction)
      self.assertIn("coll.byte=4096", instruction)


  def test_evaluator_derives_clos_128gpu_groups_from_config(self):
    with tempfile.TemporaryDirectory(prefix="syccl_dynamic_") as tmp:
      config_path = Path(tmp) / "clos-128gpu-ag-4k.json"
      write_config(config_path, coll_name="allgather", hosts=16, topology="clos")

      evaluator = load_evaluator_with_config(config_path)

      self.assertEqual(128, evaluator.NGPUS)
      self.assertEqual(set(range(0, 64)), evaluator.LAYER_GROUPS[3][0])
      self.assertEqual(set(range(64, 128)), evaluator.LAYER_GROUPS[3][1])
      self.assertEqual(set(range(128)), evaluator.LAYER_GROUPS[4][0])

      instruction = evaluator.render_instruction_for_config(config_path)
      self.assertIn("128 GPUs", instruction)
      self.assertIn("allgather", instruction)
      self.assertIn("Clos", instruction)


if __name__ == "__main__":
  unittest.main()
