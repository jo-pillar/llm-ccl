import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "agent" / "scripts" / "run_syccl_simpletes.py"


def load_runner():
  spec = importlib.util.spec_from_file_location("run_syccl_simpletes", RUNNER_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def write_manifest(tmp_path: Path) -> Path:
  config_path = tmp_path / "clos-4host-ag-4k.json"
  config_path.write_text(
      json.dumps({
          "coll": {"name": "allgather", "byte": 4096},
          "hosts": {"host_num": 4, "host_gpu_num": 8, "host_nic_num": 4},
          "topo": [],
          "sketch": {},
          "algo_solve": {"solve_output": str(tmp_path / "result.json")},
      }),
      encoding="utf-8",
  )
  manifest_path = tmp_path / "manifest.json"
  manifest_path.write_text(
      json.dumps({
          "entries": [
              {
                  "case_id": "clos-4host",
                  "topology": "clos",
                  "collective": "allgather",
                  "coll_byte_B": 4096,
                  "config_path": str(config_path),
                  "result_path": str(tmp_path / "result.json"),
                  "log_path": str(tmp_path / "run.log"),
              }
          ]
      }),
      encoding="utf-8",
  )
  return manifest_path


class SycclSimpletesRunnerTest(unittest.TestCase):
  def test_runner_selects_manifest_entry_by_case_collective_and_coll_byte(self):
    with tempfile.TemporaryDirectory(prefix="syccl_runner_") as tmp:
      tmp_path = Path(tmp)
      runner = load_runner()
      manifest = runner.load_manifest(write_manifest(tmp_path))

      entry = runner.select_entry(
          manifest,
          case_id="clos-4host",
          collective="allgather",
          coll_byte=4096,
      )

      self.assertTrue(entry["config_path"].endswith("clos-4host-ag-4k.json"))


  def test_runner_builds_distinct_full_and_ablation_commands(self):
    with tempfile.TemporaryDirectory(prefix="syccl_runner_") as tmp:
      tmp_path = Path(tmp)
      runner = load_runner()
      manifest = runner.load_manifest(write_manifest(tmp_path))
      entry = runner.select_entry(manifest, "clos-4host", "allgather", 4096)

      full = runner.build_command(entry, condition="full", max_generations=10000)
      ablation = runner.build_command(entry, condition="ablation", max_generations=10000)

      full_cmd = full.command
      ablation_cmd = ablation.command
      self.assertIn("--init-program", full_cmd)
      self.assertIn("--evaluator", full_cmd)
      self.assertIn("--max-generations", full_cmd)
      self.assertIn("10000", full_cmd)
      self.assertIn("--include-construction", full_cmd)
      self.assertNotIn("--num-inspirations", full_cmd)

      self.assertIn("--disable-reflection", ablation_cmd)
      self.assertIn("--num-inspirations", ablation_cmd)
      self.assertIn("0", ablation_cmd)
      self.assertIn("--num-chains", ablation_cmd)
      self.assertIn("1", ablation_cmd)
      self.assertIn("--k-candidates", ablation_cmd)
      self.assertIn("1", ablation_cmd)
      self.assertNotIn("--include-construction", ablation_cmd)
      self.assertEqual(full.env["SYCCL_BASE_CONFIG"], ablation.env["SYCCL_BASE_CONFIG"])

  def test_runner_can_forward_skip_preflight(self):
    with tempfile.TemporaryDirectory(prefix="syccl_runner_") as tmp:
      tmp_path = Path(tmp)
      runner = load_runner()
      manifest = runner.load_manifest(write_manifest(tmp_path))
      entry = runner.select_entry(manifest, "clos-4host", "allgather", 4096)

      spec = runner.build_command(
          entry,
          condition="full",
          max_generations=10000,
          skip_preflight=True,
      )

      self.assertIn("--skip-preflight", spec.command)

  def test_runner_can_forward_save_llm_io(self):
    with tempfile.TemporaryDirectory(prefix="syccl_runner_") as tmp:
      tmp_path = Path(tmp)
      runner = load_runner()
      manifest = runner.load_manifest(write_manifest(tmp_path))
      entry = runner.select_entry(manifest, "clos-4host", "allgather", 4096)

      spec = runner.build_command(
          entry,
          condition="full",
          max_generations=10000,
          save_llm_io=True,
      )

      self.assertIn("--save-llm-io", spec.command)

  def test_runner_bypasses_proxy_for_private_api_base(self):
    with tempfile.TemporaryDirectory(prefix="syccl_runner_") as tmp:
      tmp_path = Path(tmp)
      env_toml = tmp_path / "env.toml"
      env_toml.write_text(
          "\n".join([
              'model = "hosted_vllm/test"',
              'api_base = "http://192.168.208.232:8000/v1"',
              'api_key = "sk-no-key-required"',
          ]),
          encoding="utf-8",
      )
      runner = load_runner()
      manifest = runner.load_manifest(write_manifest(tmp_path))
      entry = runner.select_entry(manifest, "clos-4host", "allgather", 4096)

      with patch.dict(
          "os.environ",
          {
              "HTTP_PROXY": "http://127.0.0.1:7897",
              "HTTPS_PROXY": "http://127.0.0.1:7897",
              "NO_PROXY": "127.0.0.1,localhost",
              "no_proxy": "127.0.0.1,localhost",
          },
          clear=False,
      ):
        spec = runner.build_command(
            entry,
            condition="full",
            max_generations=10000,
            env_toml=env_toml,
        )

      self.assertIn("192.168.208.232", spec.env["NO_PROXY"].split(","))
      self.assertIn("192.168.208.232", spec.env["no_proxy"].split(","))


if __name__ == "__main__":
  unittest.main()
