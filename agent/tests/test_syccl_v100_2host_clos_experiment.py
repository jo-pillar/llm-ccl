import importlib.util
import json
import runpy
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prepare_syccl_v100_2host_clos_experiment.py"


def load_script():
  spec = importlib.util.spec_from_file_location("prepare_syccl_v100_2host_clos", SCRIPT_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class SycclV1002HostClosExperimentTest(unittest.TestCase):
  def test_case_specs_match_requested_message_size_sweep(self):
    script = load_script()

    cases = script.build_case_specs()

    self.assertRegex(str(script.DEFAULT_OUTPUT_ROOT), r"/result/search/v100-dgx2-clos-\d{8}T\d{6}Z$")

    total_sizes = [
        65536,
        262144,
        1048576,
        4194304,
        16777216,
        67108864,
        268435456,
        1073741824,
        4294967296,
        17179869184,
        68719476736,
        274877906944,
    ]
    self.assertEqual(total_sizes, [case.total_message_size for case in cases])
    self.assertEqual([size // 32 for size in total_sizes], [case.config_coll_byte for case in cases])
    self.assertEqual("64k-total", cases[0].name)
    self.assertEqual("256g-total", cases[-1].name)

  def test_dgx2_topology_exposes_uniform_nvswitch_and_active_spine_links(self):
    namespace = runpy.run_path(
        ROOT / "datasets" / "syccl" / "scheme1_direct_events" / "templates" / "v100_dgx2_clos" / "clos_topo.py"
    )
    topology = namespace["topology"]

    self.assertEqual(276, len(topology.connections))
    links = {
        (src.node_id, dst.node_id): (spec.bandwidth, spec.latency)
        for (src, dst), spec in topology.connections.items()
    }
    self.assertEqual(("125GB/s", "3us"), links[("gpu[0]", "gpu[15]")])
    self.assertEqual(("125GB/s", "3us"), links[("gpu[16]", "gpu[31]")])
    self.assertEqual(("400GB/s", "25us"), links[("leaf[0]", "spine[0]")])
    self.assertEqual(("400GB/s", "25us"), links[("leaf[1]", "spine[0]")])

  def test_prepare_experiment_writes_topodsl_configs_and_run_scripts(self):
    script = load_script()
    with tempfile.TemporaryDirectory(prefix="syccl_v100_2host_clos_exp_") as tmp:
      output_root = Path(tmp) / "result" / "config"

      manifest_path = script.prepare_experiment(output_root=output_root)

      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      self.assertEqual("v100-dgx2-2hosts-16gpu-1nic-clos/ag", manifest["experiment"])
      self.assertEqual(2, manifest["hosts"])
      self.assertEqual(16, manifest["gpus_per_host"])
      self.assertEqual(32, manifest["total_gpus"])
      self.assertEqual(1, manifest["nics_per_host"])
      self.assertEqual(2, manifest["leaf_switches"])
      self.assertEqual(1, manifest["spine_switches"])
      self.assertEqual("allgather", manifest["collective"])
      self.assertEqual(["linear_rank", "balance", "all"], manifest["strategies"])
      self.assertEqual(2, manifest["max_parallel"])
      self.assertEqual("2h", manifest["per_run_timeout"])
      self.assertEqual(10000, manifest["max_generations"])
      self.assertEqual(4, manifest["k_candidates"])
      self.assertEqual("hosted_vllm/deepseek-ai/DeepSeek-V4-Flash", manifest["model"])
      self.assertEqual("http://127.0.0.1:8000/v1", manifest["api_base"])
      self.assertEqual(32768, manifest["max_tokens"])
      self.assertTrue(manifest["instruction_template"].endswith("/v100_dgx2_clos/prompt_template.txt"))
      self.assertTrue(manifest["resimulation_config_template"].endswith("/v100_dgx2_clos/clos_v100_config.json"))
      self.assertTrue(manifest["flow_sim_bin"].endswith("/v100-dgx2-2hosts-16gpu-1nic-clos/ag/bin/flow-sim-rs"))
      self.assertTrue(Path(manifest["flow_sim_bin"]).is_file())
      self.assertEqual("sk-nokey", manifest["openai_api_key_default"])
      self.assertEqual(12, len(manifest["cases"]))
      self.assertEqual(36, len(manifest["runs"]))

      first = manifest["cases"][0]
      first_config = json.loads(Path(first["config_path"]).read_text(encoding="utf-8"))
      self.assertEqual({"name": "allgather", "byte": 2048, "root_sender": -1, "root_receiver": -1}, first_config["coll"])
      self.assertEqual({"host_num": 2, "host_gpu_num": 16, "host_nic_num": 1, "host_links": "nvswitch"}, first_config["hosts"])
      self.assertEqual(2, first_config["topo"][-2]["switch_num"])
      self.assertEqual("pod", first_config["topo"][-1]["switch_topo"])
      self.assertEqual(1, first_config["topo"][-1]["switch_num"])
      self.assertEqual("nvswitch", first_config["topo"][1]["link_spec"])
      self.assertEqual(0.125, first_config["link_spec"]["nvswitch"]["bw_mbpus"])
      self.assertEqual(3, first_config["link_spec"]["nvswitch"]["lat_us"])
      self.assertEqual(0.0125, first_config["link_spec"]["link_nic"]["bw_mbpus"])
      self.assertEqual(0, first_config["link_spec"]["link_nic"]["lat_us"])
      self.assertEqual(0.0125, first_config["link_spec"]["netlink_leaf"]["bw_mbpus"])
      self.assertEqual(25, first_config["link_spec"]["netlink_leaf"]["lat_us"])
      self.assertEqual(0.4, first_config["link_spec"]["netlink_spine"]["bw_mbpus"])
      self.assertEqual(25, first_config["link_spec"]["netlink_spine"]["lat_us"])
      self.assertNotIn("topodsl_params", first_config)

      last = manifest["cases"][-1]
      last_config = json.loads(Path(last["config_path"]).read_text(encoding="utf-8"))
      self.assertEqual(8589934592, last_config["coll"]["byte"])

      first_topodsl = Path(first["topodsl_path"]).read_text(encoding="utf-8")
      self.assertIn("class DGX2ClosTopology", first_topodsl)
      self.assertIn("DGX-2 host-local NVSwitch fabric", first_topodsl)
      self.assertNotIn("ring_channels", first_topodsl)
      self.assertIn("topology = DGX2ClosTopology(", first_topodsl)
      self.assertIn("    32,", first_topodsl)
      self.assertIn("    65536,", first_topodsl)
      self.assertIn('LinkSpec("125GB/s", "3us")', first_topodsl)
      self.assertIn("group_num=2, node_num=16", first_topodsl)
      self.assertIn("group_num=2, node_num=1", first_topodsl)
      self.assertIn('LinkSpec("400GB/s", "25us")', first_topodsl)

      first_instruction = Path(first["instruction_path"]).read_text(encoding="utf-8")
      self.assertIn("32 GPU cluster", first_instruction)
      self.assertIn("DGX-2 host-local NVSwitch fabric", first_instruction)
      self.assertIn("topology = DGX2ClosTopology(", first_instruction)
      self.assertIn("Layer 4 crosses the Clos spine", first_instruction)
      self.assertIn("Concrete SketchDSL", first_instruction)
      self.assertIn("Use only layers 1, 3, and 4", first_instruction)
      self.assertNotIn("Use only layers 1，3", first_instruction)

      init_program = Path(first["init_program_path"]).read_text(encoding="utf-8")
      self.assertIn("gpus_per_host: int = 16", init_program)
      self.assertIn("hosts_per_leaf: int = 1", init_program)

      tasks = (manifest_path.parent / "runs" / "tasks.tsv").read_text(encoding="utf-8").splitlines()
      self.assertEqual(36, len(tasks))
      self.assertEqual(9, len(tasks[0].split("\t")))

      run_one = (manifest_path.parent / "runs" / "run_one.sh").read_text(encoding="utf-8")
      self.assertIn('SIMPLETES_TIMEOUT="${SIMPLETES_TIMEOUT:-2h}"', run_one)
      self.assertIn('K_CANDIDATES="${K_CANDIDATES:-4}"', run_one)
      self.assertIn('MAX_GENERATIONS="${MAX_GENERATIONS:-10000}"', run_one)
      self.assertIn('timeout "$SIMPLETES_TIMEOUT" uv run python main.py', run_one)
      self.assertIn("--selector llm_elite", run_one)
      self.assertIn('--elite-selection-strategy "$STRATEGY"', run_one)
      self.assertIn('--k-candidates "$K_CANDIDATES"', run_one)
      self.assertIn("--stream-k-candidates", run_one)
      self.assertIn("--save-llm-io", run_one)
      self.assertNotIn("--restart-every-n", run_one)
      self.assertIn("--model \"$MODEL_NAME\"", run_one)
      self.assertIn("--api-base \"$API_BASE\"", run_one)
      self.assertIn("--api-key \"$API_KEY\"", run_one)
      self.assertIn("--max-tokens \"$MAX_TOKENS\"", run_one)
      self.assertIn("MODEL_NAME=\"${MODEL_NAME:-hosted_vllm/deepseek-ai/DeepSeek-V4-Flash}\"", run_one)
      self.assertIn("API_BASE=\"${API_BASE:-http://127.0.0.1:8000/v1}\"", run_one)
      self.assertIn("MAX_TOKENS=\"${MAX_TOKENS:-32768}\"", run_one)
      self.assertIn("FLOW_SIM_BIN=\"${FLOW_SIM_BIN:-", run_one)
      self.assertIn("export FLOW_SIM_BIN=\"$FLOW_SIM_BIN\"", run_one)
      self.assertIn("export OPENAI_API_KEY=\"${OPENAI_API_KEY:-sk-nokey}\"", run_one)
      self.assertIn('API_KEY="${API_KEY:-${OPENAI_API_KEY:-sk-nokey}}"', run_one)
      self.assertIn("export UV_CACHE_DIR=\"${UV_CACHE_DIR:-/tmp/uv-cache}\"", run_one)
      self.assertIn("127.0.0.1 localhost", run_one)
      self.assertNotIn("sk-7452", run_one)

      run_all = (manifest_path.parent / "runs" / "run_all.sh").read_text(encoding="utf-8")
      self.assertIn('xargs -P "${LLM_MAX_PARALLEL:-2}" -n 9', run_all)

      start_commands = (output_root / "START_COMMANDS.md").read_text(encoding="utf-8")
      self.assertIn("V100 DGX-2 Clos Search Launch Commands", start_commands)
      self.assertIn(str(manifest_path.parent / "runs" / "run_all.sh"), start_commands)
      self.assertIn("API_BASE=http://127.0.0.1:8000/v1", start_commands)
      self.assertIn("36 tasks", start_commands)


if __name__ == "__main__":
  unittest.main()
