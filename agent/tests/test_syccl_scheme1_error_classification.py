import importlib.util
import unittest
from pathlib import Path


EVALUATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "syccl"
    / "scheme1_direct_events"
    / "evaluator.py"
)


def load_evaluator():
  spec = importlib.util.spec_from_file_location("scheme1_error_classification", EVALUATOR_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class SycclScheme1ErrorClassificationTest(unittest.TestCase):
  def test_flow_sim_sketch_layer_error_is_not_fatal(self):
    evaluator = load_evaluator()

    metrics = evaluator._error_result(
        "flow-sim-rs batch-sketch exited with code 1: "
        "Error: unsupported layer 4; allowed layers are [1, 3]"
    )

    self.assertEqual("flow_sim_sketch_error", metrics["failure_category"])
    self.assertNotIn("simpletes_fatal", metrics)
    self.assertIn("unsupported layer 4", metrics["failure_feedback"])
    self.assertEqual(0.0, metrics["validity"])


if __name__ == "__main__":
  unittest.main()
