import importlib.util
import json
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prepare_syccl_v100_dgx2_clos_experiment.py"


def load_script():
  spec = importlib.util.spec_from_file_location("prepare_syccl_v100_dgx2_clos", SCRIPT_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class SycclV100ClosExperimentTest(unittest.TestCase):
  def test_case_specs_cover_four_and_eight_host_scales(self):
    script = load_script()

    four_host, eight_host = script.SCALE_SPECS
    four_host_cases = script.build_case_specs(four_host)
    eight_host_cases = script.build_case_specs(eight_host)

    self.assertEqual(
        script.REPO_ROOT / "experiments" / "v100-dgx2-clos",
        script.DEFAULT_OUTPUT_ROOT.parent,
    )
    self.assertRegex(str(script.DEFAULT_OUTPUT_ROOT), r"/experiments/v100-dgx2-clos/\d{8}-\d{6}$")

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
    self.assertEqual((4, 64, 4), (four_host.hosts, four_host.total_gpus, four_host.leaf_switches))
    self.assertEqual((8, 128, 8), (eight_host.hosts, eight_host.total_gpus, eight_host.leaf_switches))
    self.assertEqual(total_sizes, [case.total_message_size for case in four_host_cases])
    self.assertEqual(total_sizes, [case.total_message_size for case in eight_host_cases])
    self.assertEqual([size // 64 for size in total_sizes], [case.config_coll_byte for case in four_host_cases])
    self.assertEqual([size // 128 for size in total_sizes], [case.config_coll_byte for case in eight_host_cases])
    self.assertEqual("64k-total", four_host_cases[0].name)
    self.assertEqual("256g-total", eight_host_cases[-1].name)

  def test_dgx2_topology_scales_to_four_and_eight_host_clos(self):
    script = load_script()
    template_path = (
        ROOT / "datasets" / "syccl" / "scheme1_direct_events" / "templates" / "v100_dgx2_clos" / "clos_topo.py"
    )
    template_text = template_path.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="v100_dgx2_topology_") as tmp:
      for scale, expected_connections in zip(script.SCALE_SPECS, (552, 1104), strict=True):
        case = script.build_case_specs(scale)[0]
        topology_path = Path(tmp) / f"{scale.hosts}-host-topology.py"
        topology_path.write_text(script.render_topodsl(case, template_text, scale), encoding="utf-8")
        topology = runpy.run_path(topology_path)["topology"]

        self.assertEqual(expected_connections, len(topology.connections))
        links = {
            (src.node_id, dst.node_id): (spec.bandwidth, spec.latency)
            for (src, dst), spec in topology.connections.items()
        }
        last_gpu = scale.total_gpus - 1
        last_host_first_gpu = scale.total_gpus - scale.gpus_per_host
        self.assertEqual(("125GB/s", "3us"), links[(f"gpu[{last_host_first_gpu}]", f"gpu[{last_gpu}]")])
        self.assertEqual(("12.5GB/s", "0us"), links[(f"gpu[{last_host_first_gpu}]", f"nic[{scale.hosts - 1}]")])
        self.assertEqual(("12.5GB/s", "25us"), links[(f"nic[{scale.hosts - 1}]", f"leaf[{scale.leaf_switches - 1}]")])
        self.assertEqual(("400GB/s", "25us"), links[(f"leaf[{scale.leaf_switches - 1}]", "spine[0]")])

  def test_origin_source_and_generated_script_values_are_pinned(self):
    script = load_script()
    with tempfile.TemporaryDirectory(prefix="v100_origin_source_") as tmp:
      worktree = Path(tmp) / "origin-syccl"
      synthesize = worktree / "build" / "synthesize"
      synthesize.parent.mkdir(parents=True)
      synthesize.write_bytes(b"pinned-synthesize")
      subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
      subprocess.run(["git", "add", "build/synthesize"], cwd=worktree, check=True)
      subprocess.run(
          [
              "git",
              "-c",
              "user.name=V100 Test",
              "-c",
              "user.email=v100-test@example.invalid",
              "commit",
              "-q",
              "-m",
              "pin source",
          ],
          cwd=worktree,
          check=True,
      )
      head = subprocess.run(
          ["git", "rev-parse", "HEAD"],
          cwd=worktree,
          check=True,
          stdout=subprocess.PIPE,
          text=True,
      ).stdout.strip()
      sha256 = script.sha256_file(synthesize)

      self.assertEqual((head, sha256), script.verify_origin_syccl(
          worktree,
          synthesize,
          expected_commit=head,
          expected_synthesize_sha256=sha256,
      ))
      with self.assertRaisesRegex(ValueError, "HEAD mismatch"):
        script.verify_origin_syccl(
            worktree,
            synthesize,
            expected_commit="0" * 40,
            expected_synthesize_sha256=sha256,
        )
      with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
        script.verify_origin_syccl(
            worktree,
            synthesize,
            expected_commit=head,
            expected_synthesize_sha256="0" * 64,
        )
      with self.assertRaisesRegex(ValueError, "unsafe bundle path"):
        script.validate_generated_script_value(
            '/tmp/bundle"injected',
            label="bundle path",
            pattern=r"[A-Za-z0-9_./=-]+",
        )

  def test_prepare_experiment_writes_combined_four_and_eight_host_bundle(self):
    script = load_script()
    with tempfile.TemporaryDirectory(prefix="syccl_v100_dgx2_clos_exp_") as tmp:
      output_root = Path(tmp) / "result" / "search" / "v100-dgx2-clos-test"
      syccl_worktree = Path(tmp) / "origin-syccl"
      synthesize_bin = syccl_worktree / "build" / "synthesize"
      synthesize_bin.parent.mkdir(parents=True)
      synthesize_bin.write_bytes(b"test-synthesize")
      synthesize_bin.chmod(0o755)
      subprocess.run(["git", "init", "-q"], cwd=syccl_worktree, check=True)
      subprocess.run(["git", "add", "build/synthesize"], cwd=syccl_worktree, check=True)
      subprocess.run(
          [
              "git",
              "-c",
              "user.name=V100 Test",
              "-c",
              "user.email=v100-test@example.invalid",
              "commit",
              "-q",
              "-m",
              "test synthesize",
          ],
          cwd=syccl_worktree,
          check=True,
      )
      syccl_commit = subprocess.run(
          ["git", "rev-parse", "HEAD"],
          cwd=syccl_worktree,
          check=True,
          stdout=subprocess.PIPE,
          text=True,
      ).stdout.strip()
      synthesize_sha256 = script.sha256_file(synthesize_bin)

      manifest_path = script.prepare_experiment(
          output_root=output_root,
          syccl_worktree=syccl_worktree,
          syccl_commit=syccl_commit,
          synthesize_sha256=synthesize_sha256,
      )

      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      self.assertEqual(output_root / "manifest.json", manifest_path)
      self.assertEqual("v100-dgx2-clos-4host-8host/ag", manifest["experiment"])
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
      self.assertTrue(manifest["flow_sim_bin"].endswith("/v100-dgx2-clos-test/bin/flow-sim-rs"))
      self.assertTrue(Path(manifest["flow_sim_bin"]).is_file())
      self.assertEqual("sk-nokey", manifest["openai_api_key_default"])
      self.assertEqual(2, len(manifest["scales"]))
      self.assertEqual(72, len(manifest["runs"]))
      self.assertEqual(24, manifest["case_count"])
      self.assertEqual(72, manifest["llm_task_count"])
      self.assertEqual(24, manifest["origin_task_count"])
      self.assertEqual("10h", manifest["origin_timeout"])
      self.assertEqual(str(syccl_worktree), manifest["syccl"]["worktree"])
      self.assertEqual(syccl_commit, manifest["syccl"]["head"])
      bundled_synthesize = output_root / "bin" / "origin-syccl-synthesize"
      self.assertEqual(str(synthesize_bin), manifest["syccl"]["source_synthesize_path"])
      self.assertEqual(str(bundled_synthesize), manifest["syccl"]["synthesize_path"])
      self.assertEqual(synthesize_bin.read_bytes(), bundled_synthesize.read_bytes())
      self.assertTrue(manifest["syccl"]["synthesize_exists"])
      self.assertEqual(synthesize_sha256, manifest["syccl"]["synthesize_sha256"])
      self.assertTrue(manifest["syccl"]["source_verified"])

      for scale_record, expected_hosts, expected_gpus, expected_first_byte, expected_last_byte in (
          (manifest["scales"][0], 4, 64, 1024, 4294967296),
          (manifest["scales"][1], 8, 128, 512, 2147483648),
      ):
        self.assertEqual(expected_hosts, scale_record["hosts"])
        self.assertEqual(expected_gpus, scale_record["total_gpus"])
        self.assertEqual(expected_hosts, scale_record["leaf_switches"])
        self.assertEqual(12, len(scale_record["cases"]))
        self.assertTrue(Path(scale_record["manifest_path"]).is_file())

        first = scale_record["cases"][0]
        first_config = json.loads(Path(first["config_path"]).read_text(encoding="utf-8"))
        self.assertEqual(expected_first_byte, first_config["coll"]["byte"])
        self.assertEqual(
            {"host_num": expected_hosts, "host_gpu_num": 16, "host_nic_num": 1, "host_links": "nvswitch"},
            first_config["hosts"],
        )
        self.assertEqual(expected_hosts, first_config["topo"][-2]["switch_num"])
        self.assertEqual(1, first_config["topo"][-1]["switch_num"])
        self.assertEqual(0.125, first_config["link_spec"]["nvswitch"]["bw_mbpus"])
        self.assertEqual(3, first_config["link_spec"]["nvswitch"]["lat_us"])
        self.assertEqual(0.0125, first_config["link_spec"]["link_nic"]["bw_mbpus"])
        self.assertEqual(0, first_config["link_spec"]["link_nic"]["lat_us"])
        self.assertEqual(0.0125, first_config["link_spec"]["netlink_leaf"]["bw_mbpus"])
        self.assertEqual(25, first_config["link_spec"]["netlink_leaf"]["lat_us"])
        self.assertEqual(0.4, first_config["link_spec"]["netlink_spine"]["bw_mbpus"])
        self.assertEqual(25, first_config["link_spec"]["netlink_spine"]["lat_us"])
        self.assertEqual(1, first_config["solver"]["split_chunks"])
        self.assertEqual([0, 2], first_config["prune"]["ignored_layers"])
        self.assertEqual([1, 3, 4], first_config["prune"]["must_contain_layers"])
        self.assertEqual(first["origin_result_path"], first_config["algo_solve"]["solve_output"])
        self.assertEqual(first["origin_sketch_path"], first_config["sketch"]["sketch_path"])
        self.assertTrue(first_config["sketch"]["save_sketch"])
        self.assertTrue(first["config_sha256"])
        self.assertTrue(first["config_semantic_sha256"])

        last_config = json.loads(Path(scale_record["cases"][-1]["config_path"]).read_text(encoding="utf-8"))
        self.assertEqual(expected_last_byte, last_config["coll"]["byte"])

        first_topodsl = Path(first["topodsl_path"]).read_text(encoding="utf-8")
        self.assertIn(f"    {expected_gpus},", first_topodsl)
        self.assertIn(f"group_num={expected_hosts}, node_num=16", first_topodsl)
        self.assertIn('LinkSpec("125GB/s", "3us")', first_topodsl)
        self.assertEqual(2, first_topodsl.count('LinkSpec("12.5GB/s"'))
        self.assertEqual(1, first_topodsl.count('LinkSpec("400GB/s"'))

        first_instruction = Path(first["instruction_path"]).read_text(encoding="utf-8")
        self.assertIn(f"{expected_gpus} GPU cluster", first_instruction)
        self.assertIn("Layers 2 and 3 use 100 Gbps", first_instruction)
        self.assertIn("Layer 4 uses a 400 GB/s", first_instruction)
        self.assertIn("Layer 4 crosses the Clos spine", first_instruction)
        self.assertIn("Use only layers 1, 3, and 4", first_instruction)

      tasks = (manifest_path.parent / "runs" / "tasks.tsv").read_text(encoding="utf-8").splitlines()
      self.assertEqual(72, len(tasks))
      self.assertEqual(9, len(tasks[0].split("\t")))
      self.assertTrue(tasks[0].startswith("4hosts-64gpu/64k-total\t"))
      self.assertTrue(tasks[36].startswith("8hosts-128gpu/64k-total\t"))
      self.assertIn("/llm/outputs/", tasks[0])

      origin_tasks = (manifest_path.parent / "runs" / "origin_tasks.tsv").read_text(encoding="utf-8").splitlines()
      self.assertEqual(24, len(origin_tasks))
      self.assertEqual(4, len(origin_tasks[0].split("\t")))
      self.assertTrue(origin_tasks[0].startswith("4hosts-64gpu/64k-total\t"))

      config_manifest = json.loads(
          (manifest_path.parent / "configs" / "manifest.json").read_text(encoding="utf-8")
      )
      self.assertEqual(24, len(config_manifest["cases"]))

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

      run_llm_all = (manifest_path.parent / "runs" / "run_llm_all.sh").read_text(encoding="utf-8")
      self.assertIn('exec "$SCRIPT_DIR/run_all.sh"', run_llm_all)

      run_origin_one = (manifest_path.parent / "runs" / "run_origin_one.sh").read_text(encoding="utf-8")
      self.assertIn('ORIGIN_SOLVE_TIMEOUT="${ORIGIN_SOLVE_TIMEOUT:-10h}"', run_origin_one)
      self.assertIn(str(bundled_synthesize), run_origin_one)
      self.assertNotIn("${SYNTHESIZE_BIN:-", run_origin_one)

      run_origin_all = (manifest_path.parent / "runs" / "run_origin_all.sh").read_text(encoding="utf-8")
      self.assertIn("run_origin_solve_all.py", run_origin_all)

      generated_python = (
          "run_origin_solve_all.py",
          "run_origin_flow_sim_all.py",
          "summarize_llm_outputs.py",
          "build_final_report.py",
      )
      for name in generated_python:
        source = (manifest_path.parent / "runs" / name).read_text(encoding="utf-8")
        compile(source, name, "exec")
      self.assertIn(
          str(bundled_synthesize),
          (manifest_path.parent / "runs" / "run_origin_solve_all.py").read_text(encoding="utf-8"),
      )
      self.assertNotIn(
          'os.environ.get("SYNTHESIZE_BIN"',
          (manifest_path.parent / "runs" / "run_origin_solve_all.py").read_text(encoding="utf-8"),
      )

      runpy.run_path(manifest_path.parent / "runs" / "run_origin_flow_sim_all.py")
      runpy.run_path(manifest_path.parent / "runs" / "summarize_llm_outputs.py")
      runpy.run_path(manifest_path.parent / "runs" / "build_final_report.py")
      report = json.loads(
          (manifest_path.parent / "reports" / "main_case_summary.json").read_text(encoding="utf-8")
      )
      self.assertEqual(24, len(report))
      self.assertEqual("missing", report[0]["llm_status"])
      self.assertEqual("missing", report[0]["origin_status"])

      start_commands = (output_root / "START_COMMANDS.md").read_text(encoding="utf-8")
      self.assertIn("V100 DGX-2 Clos Search Launch Commands", start_commands)
      self.assertIn(str(manifest_path.parent / "runs" / "run_llm_all.sh"), start_commands)
      self.assertIn("API_BASE=http://127.0.0.1:8000/v1", start_commands)
      self.assertIn("72 llm-ccl tasks", start_commands)
      self.assertIn("24 origin-SyCCL tasks", start_commands)
      self.assertIn(str(manifest_path.parent / "runs" / "run_origin_all.sh"), start_commands)
      self.assertIn(str(manifest_path.parent / "runs" / "run_origin_flow_sim_all.py"), start_commands)
      self.assertIn(str(manifest_path.parent / "runs" / "build_final_report.py"), start_commands)
      self.assertIn("4-host and 8-host", start_commands)


if __name__ == "__main__":
  unittest.main()
