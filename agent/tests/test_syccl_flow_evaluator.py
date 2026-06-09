import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


AGENT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = AGENT_ROOT / "datasets" / "syccl" / "scheme1_direct_events" / "evaluator.py"
def tiny_config():
  return {
      "coll": {"name": "allgather", "byte": 4096, "root_sender": -1, "root_receiver": -1},
      "hosts": {"host_num": 2, "host_gpu_num": 2, "host_nic_num": 2, "host_links": "nvswitch"},
      "topo": [
          {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
          {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
          {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
          {"layer_id": 3, "type": "switch", "switch_topo": "multirail", "switch_num": 2, "link_spec": "netlink"},
      ],
      "link_spec": {
          "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
          "nvlink": {"bw_mbpus": 0.15, "lat_us": 3},
          "link_nic": {"bw_mbpus": 0.0455, "lat_us": 0},
          "netlink": {"bw_mbpus": 0.0455, "lat_us": 10},
      },
  }


def load_evaluator(config_path: Path, flow_bin: Path, artifact_dir: Path):
  if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))
  spec = importlib.util.spec_from_file_location("flow_eval_evaluator", EVALUATOR_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  with patch.dict(
      "os.environ",
      {
          "SYCCL_BASE_CONFIG": str(config_path),
          "SYCCL_FLOW_SIM_BIN": str(flow_bin),
          "SYCCL_EVAL_ARTIFACT_DIR": str(artifact_dir),
      },
  ):
    spec.loader.exec_module(module)
  return module


def test_evaluator_calls_flow_sim_sketch_path_and_reads_time(tmp_path):
  config_path = tmp_path / "config.json"
  config_path.write_text(json.dumps(tiny_config()), encoding="utf-8")
  program_path = tmp_path / "program.py"
  program_path.write_text(
      "def run_code():\n"
      "  return [[(0, 1, 0, 0, 1), (0, 3, 0, 0, 2), (1, 1, 1, 2, 3)]]\n",
      encoding="utf-8",
  )
  flow_bin = tmp_path / "flow-sim-rs"
  flow_bin.write_text("#!/bin/sh\n", encoding="utf-8")
  artifact_dir = tmp_path / "artifacts"
  evaluator = load_evaluator(config_path, flow_bin, artifact_dir)

  def fake_run(command, cwd, stdout, stderr, text, timeout, check):
    assert command[0] == str(flow_bin)
    assert command[1] == "simulate-sketch"
    output_path = Path(command[command.index("--output") + 1])
    sketch_path = Path(command[command.index("--sketch") + 1])
    sketches = json.loads(sketch_path.read_text(encoding="utf-8"))
    assert sketches == [[[0, 1, 0, 0, 1], [0, 3, 0, 0, 2], [1, 1, 1, 2, 3]]]
    output_path.write_text(
        json.dumps({
            "time_us": 12.5,
            "finish_time_ns": 12500,
            "flow_count": 12,
            "channel_count": 4,
            "total_queue_wait_ns": 3,
            "critical_flow_id": 7,
            "links": [
                {"src": 0, "dst": 1, "transmissions": 2, "busy_ns": 10, "queue_wait_ns": 3, "last_finish_ns": 12}
            ],
        }),
        encoding="utf-8",
    )

    class Proc:
      returncode = 0
      stdout = "time_us=12.5"

    return Proc()

  with patch("subprocess.run", side_effect=fake_run):
    metrics = evaluator.evaluate(str(program_path))

  assert metrics["validity"] == 1.0
  assert metrics["best_time_us"] == 12.5
  assert metrics["combined_score"] == -12.5
  assert metrics["flow_count"] == 12
  assert metrics["channel_count"] == 4
  assert next(artifact_dir.rglob("candidate-sketch.json")).name == "candidate-sketch.json"
  assert not list(artifact_dir.rglob("candidate-translated.json"))
  assert not list(artifact_dir.rglob("candidate-config.json"))


def test_missing_flow_sim_binary_reports_error(tmp_path):
  config_path = tmp_path / "config.json"
  config_path.write_text(json.dumps(tiny_config()), encoding="utf-8")
  program_path = tmp_path / "program.py"
  program_path.write_text(
      "def run_code():\n"
      "  return [[(0, 1, 0, 0, 1), (0, 3, 0, 0, 2), (1, 1, 1, 2, 3)]]\n",
      encoding="utf-8",
  )
  evaluator = load_evaluator(config_path, tmp_path / "missing-flow-sim", tmp_path / "artifacts")

  metrics = evaluator.evaluate(str(program_path))

  assert metrics["validity"] == 0.0
  assert "missing flow-sim binary" in metrics["error"]
