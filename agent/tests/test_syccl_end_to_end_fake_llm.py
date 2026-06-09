import importlib.util
import json
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
EVALUATOR_PATH = ROOT / "datasets" / "syccl" / "scheme1_direct_events" / "evaluator.py"
FLOW_BIN = WORKSPACE / "Flow-Simulator" / "flow-sim-rs" / "target" / "debug" / "flow-sim-rs"


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


def tiny_a2a_config():
  config = tiny_config()
  config["coll"] = {"name": "alltoall", "byte": 4096, "root_sender": -1, "root_receiver": -1}
  return config


def load_evaluator(config_path: Path, artifact_dir: Path):
  spec = importlib.util.spec_from_file_location("e2e_flow_eval", EVALUATOR_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  with patch.dict(
      "os.environ",
      {
          "SYCCL_BASE_CONFIG": str(config_path),
          "SYCCL_FLOW_SIM_BIN": str(FLOW_BIN),
          "SYCCL_FLOW_SIM_ROOT": str(FLOW_BIN.parents[2]),
          "SYCCL_EVAL_ARTIFACT_DIR": str(artifact_dir),
      },
  ):
    spec.loader.exec_module(module)
  return module


def test_fake_llm_program_runs_through_flow_sim_evaluator(tmp_path):
  assert FLOW_BIN.exists(), f"missing debug flow-sim binary: {FLOW_BIN}"
  config_path = tmp_path / "config.json"
  config_path.write_text(json.dumps(tiny_config()), encoding="utf-8")
  program_path = tmp_path / "fake_llm_program.py"
  program_path.write_text(
      "def run_code():\n"
      "  return [[(0, 1, 0, 0, 1), (0, 3, 0, 0, 2), (1, 1, 1, 2, 3)]]\n",
      encoding="utf-8",
  )
  evaluator = load_evaluator(config_path, tmp_path / "artifacts")

  metrics = evaluator.evaluate(str(program_path))

  assert metrics["validity"] == 1.0
  assert metrics["best_time_us"] > 0
  assert metrics["combined_score"] == -metrics["best_time_us"]
  assert metrics["flow_count"] == 12
  assert metrics["channel_count"] == 4


def test_fake_llm_alltoall_program_runs_through_flow_sim_evaluator(tmp_path):
  assert FLOW_BIN.exists(), f"missing debug flow-sim binary: {FLOW_BIN}"
  config_path = tmp_path / "config.json"
  config_path.write_text(json.dumps(tiny_a2a_config()), encoding="utf-8")
  program_path = tmp_path / "fake_llm_program.py"
  program_path.write_text(
      "def run_code():\n"
      "  return [[(0, 1, 0, 0, 1), (0, 3, 0, 0, 2), (1, 1, 1, 2, 3)]]\n",
      encoding="utf-8",
  )
  evaluator = load_evaluator(config_path, tmp_path / "artifacts")

  metrics = evaluator.evaluate(str(program_path))

  assert metrics["validity"] == 1.0
  assert metrics["best_time_us"] > 0
  assert metrics["channel_count"] == 12
