import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "cross_eval_syccl_h80064.py"


def load_script():
  spec = importlib.util.spec_from_file_location("cross_eval_syccl_h80064", SCRIPT_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


class SycclCrossEvalTest(unittest.TestCase):
  def test_selects_best_llm_artifact_by_flow_sim_time(self):
    script = load_script()
    with tempfile.TemporaryDirectory(prefix="cross_eval_llm_") as tmp:
      root = Path(tmp)
      for strategy, time_us in (("linear_rank", 35.0), ("balance", 30.0), ("all", 32.0)):
        case_root = root / "outputs" / strategy / "65536B-prune=small"
        run_dir = case_root / "eval_artifacts" / "scheme1_direct_events" / f"eval_{strategy}"
        sketch_dir = run_dir / "flow-sim-inputs" / "candidate-000"
        result_dir = run_dir / "flow-sim-runs"
        sketch_dir.mkdir(parents=True)
        result_dir.mkdir(parents=True)
        (sketch_dir / "candidate-sketch.json").write_text(
            json.dumps([[0, 1, 0, [0], [1]]]),
            encoding="utf-8",
        )
        (run_dir / "candidate-config.json").write_text(json.dumps({"coll": {"byte": 128}}), encoding="utf-8")
        result_path = result_dir / "candidate-000.json"
        result_path.write_text(
            json.dumps({"time_us": time_us, "bottleneck_profile": {"critical_flow_chain": {}}}),
            encoding="utf-8",
        )
        (run_dir / "flow-sim-manifest.json").write_text(
            json.dumps({"cases": [{"rust_output": str(result_path)}]}),
            encoding="utf-8",
        )

      selected = script.select_best_llm_artifact(
          root,
          "65536B-prune=small",
          strategies=("linear_rank", "balance", "all"),
      )

      self.assertEqual("balance", selected.strategy)
      self.assertEqual(30.0, selected.flow_time_us)
      self.assertEqual(root / "outputs" / "balance" / "65536B-prune=small", selected.case_dir)

  def test_selects_best_origin_algorithm_by_alg_times(self):
    script = load_script()
    result = {
        "coll_name": "allgather",
        "ngpus": 2,
        "chunk_size_byte": 128,
        "alg_times": [5.0, 3.0, 4.0],
        "algorithms": [
            {"final_schedule": {"Time": 5.0, "Schedule": {"Events": []}}},
            {"final_schedule": {"Time": 3.0, "Schedule": {"Events": []}}},
            {"final_schedule": {"Time": 4.0, "Schedule": {"Events": []}}},
        ],
    }

    index, time_us = script.select_best_origin_algorithm(result)

    self.assertEqual(1, index)
    self.assertEqual(3.0, time_us)

  def test_wraps_single_origin_algorithm_for_flow_sim(self):
    script = load_script()
    result = {
        "coll_name": "allgather",
        "ngpus": 2,
        "chunk_size_byte": 128,
        "alg_times": [5.0, 3.0],
        "algorithms": [
            {"final_schedule": {"Time": 5.0, "Schedule": {"Events": []}}},
            {"final_schedule": {"Time": 3.0, "Schedule": {"Events": [{"src_chunk": "(0, 0)", "sends": []}]}}},
        ],
    }

    wrapped = script.origin_algorithm_as_translated(result, 1)

    self.assertEqual("allgather", wrapped["coll_name"])
    self.assertEqual(2, wrapped["ngpus"])
    self.assertEqual(128, wrapped["chunk_size_byte"])
    self.assertEqual(1, len(wrapped["algorithms"]))
    self.assertEqual(3.0, wrapped["algorithms"][0]["final_schedule"]["Time"])

  def test_compact_sketch_to_native_reconstructs_deps_and_next_edges(self):
    script = load_script()
    with tempfile.TemporaryDirectory(prefix="cross_eval_sketch_") as tmp:
      config_path = Path(tmp) / "config.json"
      config_path.write_text(
          json.dumps({"hosts": {"host_num": 1, "host_gpu_num": 4}}),
          encoding="utf-8",
      )

      native = script.compact_sketch_to_native(
          [
              [0, 1, 0, 0, [1, 2]],
              [1, 1, 0, 1, [3]],
          ],
          config_path,
      )

      nodes = native[0]["nodes"]
      self.assertEqual([], nodes[0]["deps"])
      self.assertEqual([1], nodes[0]["next"])
      self.assertEqual([0], nodes[1]["deps"])
      self.assertEqual([], nodes[1]["next"])

  def test_read_flow_time_accepts_simulate_solutions_output(self):
    script = load_script()
    with tempfile.TemporaryDirectory(prefix="cross_eval_flow_") as tmp:
      output_path = Path(tmp) / "origin-flow.json"
      output_path.write_text(
          json.dumps(
              {
                  "solutions": [
                      {
                          "solution_index": 0,
                          "syccl_time_us": 26.0,
                          "rust_time_us": 219.0,
                          "result": {"time_us": 219.0},
                      }
                  ]
              }
          ),
          encoding="utf-8",
      )

      self.assertEqual(219.0, script.read_flow_time(output_path))

  def test_read_resim_time_scans_top_level_time_without_full_json_load(self):
    script = load_script()
    with tempfile.TemporaryDirectory(prefix="cross_eval_resim_") as tmp:
      output_path = Path(tmp) / "resim.json"
      output_path.write_text(
          '{\n  "LinkTrace": [\n    {"arrival_time_ns": 1}\n  ],\n  "Time": 26.246\n}\n',
          encoding="utf-8",
      )

      self.assertEqual(26.246, script.read_resim_time(output_path))


if __name__ == "__main__":
  unittest.main()
