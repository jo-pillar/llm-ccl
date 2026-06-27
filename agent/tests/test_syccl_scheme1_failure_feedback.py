import importlib.util
import json
import math
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch


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
    assert metrics["validity"] == 0.0
    assert metrics["bottleneck_profile"] is None
    assert metrics["failure_category"] == "dependency_error"
    assert "GPU 16" in metrics["failure_feedback"]
    assert "step 1" in metrics["failure_feedback"]
    assert "validity_status" not in metrics
    assert "best_time_us" not in metrics
    assert "completion_time" not in metrics
    assert "expanded_events" not in metrics


def test_write_flow_sim_inputs_writes_run_code_output_without_python_sketch_validation():
    raw = [
        [(0, 999, 123, 0, [1, 2, 3])],
        [(0, 1, 0, 0, [[4, 5], [6, 7]])],
    ]

    with tempfile.TemporaryDirectory(prefix="syccl_flow_inputs_") as tmp:
        config_path, input_dir, output_dir, manifest_path, summary_path = (
            EVALUATOR._write_flow_sim_inputs(raw, Path(tmp))
        )

        assert config_path.name == "candidate-config.json"
        assert input_dir.name == "flow-sim-inputs"
        assert output_dir.name == "flow-sim-runs"
        assert manifest_path.name == "flow-sim-manifest.json"
        assert summary_path.name == "flow-sim-summary.csv"
        assert json.loads((input_dir / "candidate-000" / "candidate-sketch.json").read_text()) == raw
        assert not (input_dir / "candidate-001").exists()


def test_read_best_flow_sim_case_selects_lowest_time_and_requires_profile():
    with tempfile.TemporaryDirectory(prefix="syccl_flow_outputs_") as tmp:
        root = Path(tmp)
        fast = root / "fast.json"
        slow = root / "slow.json"
        fast.write_text(json.dumps({
            "name": "candidate-001",
            "time_us": 7.5,
            "bottleneck_profile": {"critical_flow_chain": {"latest_flow_id": 1}},
        }), encoding="utf-8")
        slow.write_text(json.dumps({
            "name": "candidate-000",
            "time_us": 12.0,
            "bottleneck_profile": {"critical_flow_chain": {"latest_flow_id": 2}},
        }), encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "cases": [
                {"name": "candidate-000", "sketch_index": 0, "rust_output": str(slow)},
                {"name": "candidate-001", "sketch_index": 0, "rust_output": str(fast)},
            ]
        }), encoding="utf-8")

        selected = EVALUATOR._read_best_flow_sim_case(manifest)

        assert selected.output["time_us"] == 7.5
        assert selected.case["name"] == "candidate-001"
        assert selected.case["sketch_index"] == 0


def test_read_best_flow_sim_case_rejects_missing_bottleneck_profile():
    with tempfile.TemporaryDirectory(prefix="syccl_flow_bad_output_") as tmp:
        root = Path(tmp)
        output = root / "bad.json"
        output.write_text(json.dumps({"time_us": 1.0}), encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "cases": [{"name": "candidate-000", "rust_output": str(output)}]
        }), encoding="utf-8")

        try:
            EVALUATOR._read_best_flow_sim_case(manifest)
        except EVALUATOR.FlowSimOutputError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected FlowSimOutputError")

        assert "bottleneck_profile" in message


def test_success_metrics_scores_bytes_per_simulated_microsecond_and_prunes_fields():
    output = {
        "time_us": 7.5,
        "bottleneck_profile": {"critical_flow_chain": {"latest_flow_id": 3}},
    }
    best = EVALUATOR.FlowSimCaseResult(
        case={"name": "candidate-001", "sketch_index": 1},
        output=output,
    )

    metrics = EVALUATOR._success_metrics(best=best)

    assert metrics == {
        "combined_score": EVALUATOR.COLL_BYTE / 7.5,
        "validity": 1.0,
        "bottleneck_profile": output["bottleneck_profile"],
    }
    assert "best_time_us" not in metrics
    assert "flow_count" not in metrics
    assert "finish_time_ns" not in metrics


def test_flow_sim_sketch_errors_are_nonfatal():
    metrics = EVALUATOR._error_result(
        "flow-sim-rs batch-sketch exited with code 1: Error: unsupported layer 4; allowed layers are [1, 3]"
    )

    assert metrics["failure_category"] == "flow_sim_sketch_error"
    assert "unsupported layer 4" in metrics["failure_feedback"]
    assert "simpletes_fatal" not in metrics


def test_flow_sim_config_errors_are_fatal():
    metrics = EVALUATOR._error_result("missing flow-sim-rs binary: /opt/flow-sim-rs")

    assert metrics["failure_category"] == "flow_sim_config_error"
    assert metrics["simpletes_fatal"] is True


def test_flow_sim_output_errors_are_fatal():
    metrics = EVALUATOR._error_result(
        "flow-sim output /tmp/case.json missing required bottleneck_profile object"
    )

    assert metrics["failure_category"] == "flow_sim_output_error"
    assert metrics["simpletes_fatal"] is True


def test_run_flow_sim_batch_sketch_builds_batch_command():
    with tempfile.TemporaryDirectory(prefix="syccl_flow_cmd_") as tmp:
        root = Path(tmp)
        config = root / "candidate-config.json"
        input_dir = root / "flow-sim-inputs"
        output_dir = root / "flow-sim-runs"
        manifest = root / "flow-sim-manifest.json"
        summary = root / "flow-sim-summary.csv"
        input_dir.mkdir()
        output_dir.mkdir()

        with patch.object(EVALUATOR, "FLOW_SIM_BIN", Path("/opt/flow-sim-rs")):
            with patch.object(EVALUATOR.subprocess, "run") as run:
                run.return_value = subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="simulated 2 sketch cases",
                )

                returncode, _wall, log = EVALUATOR._run_flow_sim_batch_sketch(
                    config,
                    input_dir,
                    output_dir,
                    manifest,
                    summary,
                )

        command = run.call_args.args[0]
        assert returncode == 0
        assert log == "simulated 2 sketch cases"
        assert command == [
            "/opt/flow-sim-rs",
            "batch-sketch",
            "--config",
            str(config),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest),
            "--summary",
            str(summary),
        ]
