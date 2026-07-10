import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prepare_v100_dgx2_flow_sim_compare.py"


def load_script():
  spec = importlib.util.spec_from_file_location("prepare_v100_dgx2", SCRIPT_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def write_fake_origin_syccl(root: Path) -> None:
  scripts = root / "scripts"
  build = root / "build"
  scripts.mkdir(parents=True)
  build.mkdir(parents=True)
  (build / "synthesize").write_text("fake synthesize\n", encoding="utf-8")
  (scripts / "config_gen.py").write_text(
      '''
def allgather_coll(chunk_size_B):
  return {"name": "allgather", "byte": chunk_size_B, "root_sender": -1, "root_receiver": -1}

class ConfigGen:
  def __init__(self):
    self.SOLVE_BR = 0.2

  def set_solve_br(self, solve_br):
    self.SOLVE_BR = solve_br

  def v100conf(self, chunk_size_B, nhosts, ngpus, nnics, nleaf, nspine, prune_type, solve_output, sketch_output, a2a):
    config = {
      "coll": allgather_coll(chunk_size_B),
      "hosts": {
        "host_num": nhosts,
        "host_gpu_num": ngpus,
        "host_nic_num": nnics,
        "host_links": "nvswitch",
      },
      "topo": [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {"layer_id": 3, "type": "switch", "switch_topo": "pod", "switch_num": nleaf, "link_spec": "netlink_leaf"},
        {"layer_id": 4, "type": "switch", "switch_topo": "pod", "switch_num": nspine, "link_spec": "netlink_spine"},
      ],
      "host_links": {"nvlink": [[1, 2], [0, 2], [0, 1]]},
      "link_spec": {
        "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
        "nvlink": {"bw_mbpus": 0.128, "lat_us": 13.5},
        "link_nic": {"bw_mbpus": 0.0128, "lat_us": 0},
        "netlink_leaf": {"bw_mbpus": 0.0128, "lat_us": 25.5},
        "netlink_spine": {"bw_mbpus": 0.4096, "lat_us": 26.5},
      },
      "sketch": {"sketch_path": sketch_output},
      "algo_solve": {"solve_output": solve_output, "solve_balance_ratio": self.SOLVE_BR},
      "prune": {"must_contain_layers": [1, 3, 4], "limit_num_steps": 5},
    }
    return config
''',
      encoding="utf-8",
  )


class PrepareV100Dgx2FlowSimCompareTest(unittest.TestCase):
  def test_bw_mbpus_is_converted_from_mib_per_us_to_gb_per_s(self):
    script = load_script()

    self.assertEqual("125GB/s", script.bw_mbpus_to_gbps(0.128))
    self.assertEqual("12.5GB/s", script.bw_mbpus_to_gbps(0.0128))
    self.assertEqual("400GB/s", script.bw_mbpus_to_gbps(0.4096))

  def test_case_specs_match_dgx2_v100_allgather_sweep(self):
    script = load_script()

    cases = script.build_case_specs()

    self.assertEqual(26, len(cases))
    self.assertEqual("v100-dgx2-64gpu-allgather-1024B-prune=small", cases[0].case_id)
    self.assertEqual(64, cases[0].gpu_count)
    self.assertEqual(4, cases[0].host_count)
    self.assertEqual(16, cases[0].gpus_per_host)
    self.assertEqual(1, cases[0].nics_per_host)
    self.assertEqual(2, cases[0].leaf_count)
    self.assertEqual(1, cases[0].spine_count)
    self.assertEqual(1024 // 64, cases[0].coll_byte)

    last = cases[-1]
    self.assertEqual("v100-dgx2-128gpu-allgather-17179869184B-prune=small", last.case_id)
    self.assertEqual(128, last.gpu_count)
    self.assertEqual(8, last.host_count)
    self.assertEqual(4, last.leaf_count)
    self.assertEqual(17179869184 // 128, last.coll_byte)

  def test_prepare_writes_manifest_configs_and_task_outputs(self):
    script = load_script()
    with tempfile.TemporaryDirectory(prefix="v100_dgx2_prepare_") as tmp:
      tmp_path = Path(tmp)
      origin = tmp_path / "origin-syccl"
      write_fake_origin_syccl(origin)

      bundle = script.prepare_bundle(
          bundle_root=tmp_path / "bundles",
          launch_id="unit",
          origin_syccl=origin,
          flow_sim_bin=None,
          max_generations=7,
          k_candidates=2,
          max_parallel=3,
          simpletes_timeout="9m",
          origin_timeout="11m",
      )

      manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
      self.assertEqual("v100-dgx2-flow-sim-compare", manifest["experiment"])
      self.assertEqual(26, manifest["case_count"])
      self.assertEqual(["linear_rank", "balance", "all"], manifest["strategies"])
      self.assertEqual(str(origin), manifest["syccl"]["root"])

      config_manifest = json.loads((bundle / "configs" / "manifest.json").read_text(encoding="utf-8"))
      self.assertEqual(26, len(config_manifest["cases"]))
      first = config_manifest["cases"][0]
      first_config = json.loads(Path(first["config_path"]).read_text(encoding="utf-8"))
      self.assertEqual({"host_num": 4, "host_gpu_num": 16, "host_nic_num": 1, "host_links": "nvswitch"}, first_config["hosts"])
      self.assertEqual(16, first_config["coll"]["byte"])
      self.assertEqual([1, 3, 4], first_config["prune"]["must_contain_layers"])

      topodsl = Path(first["topodsl_path"]).read_text(encoding="utf-8")
      self.assertIn("64", topodsl)
      self.assertIn("group_num=4", topodsl)
      self.assertIn("node_num=16", topodsl)
      self.assertIn('LinkSpec("125GB/s", "13.5us")', topodsl)
      self.assertIn('LinkSpec("12.5GB/s", "0us")', topodsl)
      self.assertIn('LinkSpec("12.5GB/s", "25.5us")', topodsl)
      self.assertIn('LinkSpec("400GB/s", "26.5us")', topodsl)
      self.assertNotIn("nvlink_by_link_spec", topodsl)
      self.assertNotIn("nvlink_nv", topodsl)

      instruction = Path(first["instruction_path"]).read_text(encoding="utf-8")
      self.assertIn("Use only layers 1, 3, and 4.", instruction)

      init_program = Path(first["init_program_path"]).read_text(encoding="utf-8")
      self.assertIn("gpus_per_host=16", init_program)
      self.assertIn("hosts_per_leaf=2", init_program)

      llm_tasks = (bundle / "runs" / "llm_tasks.tsv").read_text(encoding="utf-8").strip().splitlines()
      origin_tasks = (bundle / "runs" / "origin_tasks.tsv").read_text(encoding="utf-8").strip().splitlines()
      self.assertEqual(78, len(llm_tasks))
      self.assertEqual(26, len(origin_tasks))

      run_one = (bundle / "runs" / "run_llm_one.sh").read_text(encoding="utf-8")
      self.assertIn("--selector llm_elite", run_one)
      self.assertIn("--k-candidates \"$K_CANDIDATES\"", run_one)
      self.assertIn("timeout \"$SIMPLETES_TIMEOUT\"", run_one)


if __name__ == "__main__":
  unittest.main()
