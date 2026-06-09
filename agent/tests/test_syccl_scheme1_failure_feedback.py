import importlib.util
import math
from pathlib import Path


EVALUATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "syccl"
    / "scheme1_direct_events"
    / "evaluator.py"
)
SPEC = importlib.util.spec_from_file_location("scheme1_direct_events_evaluator", EVALUATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def test_error_result_returns_finite_penalty_and_actionable_feedback():
    metrics = EVALUATOR._error_result(
        "invalid sketch: source GPU 16 has not been reached before step 1"
    )

    assert math.isfinite(metrics["combined_score"])
    assert math.isfinite(metrics["best_time_us"])
    assert metrics["validity"] == 0.0
    assert metrics["failure_category"] == "dependency_error"
    assert "GPU 16" in metrics["failure_feedback"]
    assert "step 1" in metrics["failure_feedback"]


def test_dependency_error_mentions_same_step_delivery():
    sketch = [
        (0, 1, 0, 0, [1, 2, 3, 4, 5, 6, 7]),
        (0, 4, 0, 0, 8),
        (1, 4, 0, 0, 16),
        (1, 1, 2, 16, [17, 18, 19]),
    ]

    try:
        EVALUATOR._normalize_sketches([sketch])
    except EVALUATOR.SketchValidationError as exc:
        msg = str(exc)
    else:
        raise AssertionError("expected SketchValidationError")

    assert "GPU 16" in msg
    assert "first reached at step 1" in msg
    assert "strictly later step" in msg


def test_nested_dst_list_error_explains_flat_list_requirement():
    sketch = [
        (0, 1, 0, 0, [1, 2, 3]),
        (1, 1, 0, [0, 1, 2, 3], [[4, 5], [6, 7]]),
    ]

    try:
        EVALUATOR._normalize_sketches([sketch])
    except EVALUATOR.SketchValidationError as exc:
        msg = str(exc)
    else:
        raise AssertionError("expected SketchValidationError")

    assert "flat list" in msg
    assert "nested" in msg


def test_syccl_expand_size_mismatch_gets_specific_feedback():
    metrics = EVALUATOR._error_result(
        "SyCCL direct resim exited with code -11: "
        "Failed to map node 0 to 1, size mismatch"
    )

    assert metrics["failure_category"] == "syccl_expand_incompatible"
    assert "symmetry expansion" in metrics["failure_feedback"]
    assert "split" in metrics["failure_feedback"]


def test_syccl_expand_over_broad_layer_gets_specific_feedback():
    metrics = EVALUATOR._error_result(
        "SyCCL direct resim exited with code -11: "
        "Failed to expand sketch node #0 (step=0, layer=4, group=0) "
        "from root GPU 0 to target GPU 1: mapped src/dst set sizes changed "
        "from 1/31 to 1/1. The sketch likely assigned traffic to an "
        "over-broad complete/top layer instead of the most specific valid "
        "layer/group; for example, same-host sends such as 0->1 should use "
        "the host layer, not the complete layer."
    )

    assert metrics["failure_category"] == "syccl_expand_incompatible"
    assert "proper layer" in metrics["failure_feedback"]
    assert "host layer" in metrics["failure_feedback"]
