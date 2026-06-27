import contextlib
import io
import importlib.util
import json
import os
import subprocess
import sys
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


def write_topodsl(path: Path, *, family: str = "clos") -> None:
  if family == "clos":
    body = """
###TopoBegin
def topology():
  return {
    "family": "clos",
    "hosts": 2,
    "gpus_per_host": 4,
    "nics_per_host": 1,
    "leaf_switches": 1,
    "spine_switches": 1,
    "message_size": 4096,
    "collective": "allgather",
  }
###TopoEND
""".lstrip()
  else:
    body = """
###TopoBegin
def topology():
  return {
    "family": "multirail",
    "hosts": 2,
    "gpus_per_host": 4,
    "nics_per_host": 4,
    "rails": 4,
    "message_size": 8192,
    "collective": "allgather",
  }
###TopoEND
""".lstrip()
  path.write_text(body, encoding="utf-8")


def write_init_program(path: Path) -> None:
  path.write_text(
      """
# EVOLVE-BLOCK-START
def construct_sketches(GPU_NUM):
  return [[(0, 1, 0, 0, list(range(1, GPU_NUM)))]]
# EVOLVE-BLOCK-END
""".lstrip(),
      encoding="utf-8",
  )


class SycclSimpletesRunnerTest(unittest.TestCase):
  def test_missing_topodsl_env_is_fatal(self):
    runner = load_runner()

    with patch.dict("os.environ", {}, clear=True):
      with self.assertRaisesRegex(SystemExit, "TOPODSL"):
        runner.load_required_topodsl_from_env()

  def test_topodsl_env_must_point_to_existing_file(self):
    runner = load_runner()

    with patch.dict("os.environ", {"TOPODSL": "/tmp/does-not-exist.topo"}, clear=True):
      with self.assertRaisesRegex(SystemExit, "does not exist"):
        runner.load_required_topodsl_from_env()

  def test_build_command_generates_config_instruction_and_init_wrapper(self):
    with tempfile.TemporaryDirectory(prefix="syccl_runner_") as tmp:
      tmp_path = Path(tmp)
      topo_path = tmp_path / "topo.py"
      init_path = tmp_path / "ring_init.py"
      template_path = tmp_path / "prompt_templete.txt"
      output_root = tmp_path / "out"
      write_topodsl(topo_path)
      write_init_program(init_path)
      template_path.write_text(
          "GPU=${GPU_NUM} coll=${Collective} topo=${TOPOLOGY} bytes=${MESSAGE_SIZE}",
          encoding="utf-8",
      )
      runner = load_runner()

      with patch.dict("os.environ", {"TOPODSL": str(topo_path)}, clear=False):
        topo = runner.load_required_topodsl_from_env()
      spec = runner.build_command(
          topo,
          init_program=init_path,
          instruction_template=template_path,
          max_generations=10000,
          output_root=output_root,
          env_toml=None,
          save_llm_io=True,
      )

      env = {key: value for key, value in spec.env.items() if key.startswith("SYCCL_")}
      self.assertEqual({"SYCCL_BASE_CONFIG", "SYCCL_EVAL_ARTIFACT_DIR"}, set(env))
      self.assertIn("--save-llm-io", spec.command)
      self.assertIn("--init-program", spec.command)
      self.assertIn(str(spec.init_program_path), spec.command)
      self.assertIn("--instruction", spec.command)
      self.assertIn(str(spec.instruction_path), spec.command)

      config = json.loads(spec.config_path.read_text(encoding="utf-8"))
      self.assertEqual({"name": "allgather", "byte": 4096, "root_sender": -1, "root_receiver": -1}, config["coll"])
      self.assertEqual({"host_num": 2, "host_gpu_num": 4, "host_nic_num": 1, "host_links": "nvswitch"}, config["hosts"])
      self.assertEqual("pod", config["topo"][-1]["switch_topo"])

      instruction = spec.instruction_path.read_text(encoding="utf-8")
      self.assertIn("GPU=8", instruction)
      self.assertIn("coll=allgather", instruction)
      self.assertIn("topo=###TopoBegin", instruction)
      self.assertIn("def topology():", instruction)
      self.assertIn("###TopoEND", instruction)
      self.assertIn('"family": "clos"', instruction)
      self.assertIn("bytes=4096", instruction)
      self.assertNotIn("total_gpus=8", instruction)
      self.assertNotIn("layer 1 group 0", instruction)

      init_wrapper = spec.init_program_path.read_text(encoding="utf-8")
      self.assertIn("GPU_NUM = 8", init_wrapper)
      self.assertIn("# EVOLVE-BLOCK-START", init_wrapper)
      self.assertIn("def construct_sketches(GPU_NUM):", init_wrapper)
      self.assertIn("return construct_sketches(GPU_NUM)", init_wrapper)
      self.assertNotIn("USER_INIT_PATH", init_wrapper)
      self.assertNotIn("_INITIAL_CONSTRUCT_SKETCHES", init_wrapper)
      prefix, evolve_and_suffix = init_wrapper.split("# EVOLVE-BLOCK-START", 1)
      evolve, _suffix = evolve_and_suffix.split("# EVOLVE-BLOCK-END", 1)
      self.assertNotIn("_load_user_init", prefix)
      self.assertNotIn("_load_user_init", evolve)

  def test_multirail_topodsl_generates_multirail_config(self):
    with tempfile.TemporaryDirectory(prefix="syccl_runner_") as tmp:
      tmp_path = Path(tmp)
      topo_path = tmp_path / "topo.py"
      init_path = tmp_path / "ring_init.py"
      template_path = tmp_path / "prompt_templete.txt"
      write_topodsl(topo_path, family="multirail")
      write_init_program(init_path)
      template_path.write_text("${GPU_NUM} ${TOPOLOGY_FAMILY} ${TOPOLOGY}", encoding="utf-8")
      runner = load_runner()

      with patch.dict("os.environ", {"TOPODSL": str(topo_path)}, clear=False):
        topo = runner.load_required_topodsl_from_env()
      spec = runner.build_command(
          topo,
          init_program=init_path,
          instruction_template=template_path,
          max_generations=10,
          output_root=tmp_path / "out",
          env_toml=None,
      )

      config = json.loads(spec.config_path.read_text(encoding="utf-8"))
      self.assertEqual(8192, config["coll"]["byte"])
      self.assertEqual("multirail", config["topo"][-1]["switch_topo"])
      self.assertEqual(4, config["topo"][-1]["switch_num"])
      self.assertIn("8 multirail", spec.instruction_path.read_text(encoding="utf-8"))

  def test_rendered_init_program_accepts_build_initial_sketch_alias(self):
    with tempfile.TemporaryDirectory(prefix="syccl_runner_") as tmp:
      tmp_path = Path(tmp)
      user_init = tmp_path / "init_alias.py"
      rendered = tmp_path / "rendered_init.py"
      user_init.write_text(
          """
# EVOLVE-BLOCK-START
def construct_sketches(GPU_NUM):
  return [[(0, 1, 0, 0, GPU_NUM - 1)]]
# EVOLVE-BLOCK-END
""".lstrip(),
          encoding="utf-8",
      )
      runner = load_runner()

      runner.write_init_program(user_init, rendered, gpu_num=8)
      namespace: dict[str, object] = {}
      exec(rendered.read_text(encoding="utf-8"), namespace, namespace)

      self.assertEqual([[(0, 1, 0, 0, 7)]], namespace["run_code"]())

  def test_main_dry_run_uses_topodsl_env_and_new_cli_inputs(self):
    with tempfile.TemporaryDirectory(prefix="syccl_runner_") as tmp:
      tmp_path = Path(tmp)
      topo_path = tmp_path / "topo.py"
      init_path = tmp_path / "ring_init.py"
      template_path = tmp_path / "prompt_templete.txt"
      output_root = tmp_path / "out"
      write_topodsl(topo_path)
      write_init_program(init_path)
      template_path.write_text("GPU=${GPU_NUM}", encoding="utf-8")
      runner = load_runner()

      with patch.dict("os.environ", {"TOPODSL": str(topo_path)}, clear=False):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
          result = runner.main([
              "--init-program",
              str(init_path),
              "--instruction",
              str(template_path),
              "--max-generations",
              "1",
              "--output-root",
              str(output_root),
              "--env-toml",
              str(tmp_path / "missing-env.toml"),
              "--dry-run",
          ])

      self.assertEqual(0, result)
      self.assertIn("SimpleTES command:", stdout.getvalue())
      self.assertIn("uv run python main.py", stdout.getvalue())
      generated = list(output_root.glob("*/clos/allgather/4k/FULL/generated/flow-sim-config.json"))
      self.assertEqual(1, len(generated))

  def test_old_manifest_cli_is_rejected(self):
    runner = load_runner()

    with contextlib.redirect_stderr(io.StringIO()):
      with self.assertRaises(SystemExit):
        runner._parse_args([
            "--manifest",
            "manifest.json",
            "--case",
            "clos",
            "--coll-byte",
            "4K",
            "--condition",
            "full",
        ])

  def test_script_can_run_directly_from_scripts_path(self):
    with tempfile.TemporaryDirectory(prefix="syccl_runner_") as tmp:
      tmp_path = Path(tmp)
      topo_path = tmp_path / "topo.py"
      init_path = tmp_path / "ring_init.py"
      template_path = tmp_path / "prompt_templete.txt"
      output_root = tmp_path / "out"
      write_topodsl(topo_path)
      write_init_program(init_path)
      template_path.write_text("GPU=${GPU_NUM}", encoding="utf-8")
      env = os.environ.copy()
      env["TOPODSL"] = str(topo_path)

      proc = subprocess.run(
          [
              sys.executable,
              str(RUNNER_PATH),
              "--init-program",
              str(init_path),
              "--instruction",
              str(template_path),
              "--max-generations",
              "1",
              "--output-root",
              str(output_root),
              "--env-toml",
              str(tmp_path / "missing-env.toml"),
              "--dry-run",
          ],
          cwd=str(ROOT / "agent"),
          env=env,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
          check=False,
      )

      self.assertEqual("", proc.stderr)
      self.assertEqual(0, proc.returncode)
      self.assertIn("SYCCL_BASE_CONFIG", proc.stdout)


if __name__ == "__main__":
  unittest.main()
