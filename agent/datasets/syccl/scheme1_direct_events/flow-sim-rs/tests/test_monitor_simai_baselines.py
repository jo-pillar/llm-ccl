import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "monitor_simai_baselines.py"
SPEC = importlib.util.spec_from_file_location("monitor_simai_baselines", SCRIPT)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_count_running_simai_processes_uses_root_boundary(monkeypatch):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(
            stdout="\n".join(
                [
                    "AstraSimNetwork-debug -w /tmp/16m/simai-cache/case/workload.txt",
                    "AstraSimNetwork-debug -w /tmp/16m-tail/simai-cache/case/workload.txt",
                    "AstraSimNetwork-debug -w /tmp/16m-debug-t8/simai-cache/case/workload.txt",
                    "python3 scripts/run_simai_baselines.py --root /tmp/16m",
                ]
            )
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.count_running_simai_processes(Path("/tmp/16m")) == 1


def test_collect_root_status_reports_fct_progress_without_valid_baseline(tmp_path, monkeypatch):
    root = tmp_path / "run"
    case_dir = root / "simai-cache" / "case_a"
    case_dir.mkdir(parents=True)
    (case_dir / "fct.txt").write_text("a\nb\nc\n", encoding="utf-8")
    (case_dir / "emptyEndToEnd.csv").write_text("", encoding="utf-8")
    (case_dir / "simai.lock").write_text("999999\n", encoding="utf-8")
    (root / "simai-cache" / "baseline-manifest.json").write_text(
        module.json.dumps({"cases": []}), encoding="utf-8"
    )
    monkeypatch.setattr(module, "count_running_simai_processes", lambda root: 2)

    status = module.collect_root_status(root, expected_fct_lines=6)

    assert status["root"] == str(root)
    assert status["manifest_exists"] is True
    assert status["manifest_cases"] == 0
    assert status["valid_cases"] == 0
    assert status["nonempty_end_to_end_files"] == 0
    assert status["active_locks"] == 1
    assert status["running_simai_processes"] == 2
    assert status["fct_files"] == 1
    assert status["max_fct_lines"] == 3
    assert status["fct_line_total"] == 3
    assert status["fct_line_target"] == 6
    assert status["fct_progress_rate"] == 0.5


def test_monitor_writes_merged_manifest_and_runs_compare_for_valid_cases(tmp_path, monkeypatch):
    main = tmp_path / "main"
    tail = tmp_path / "tail"
    (main / "simai-cache").mkdir(parents=True)
    (tail / "simai-cache").mkdir(parents=True)
    rust_output = tmp_path / "rust.json"
    translated = tmp_path / "candidate-translated.json"
    config = tmp_path / "config.json"
    for path in (rust_output, translated, config):
        path.write_text("{}", encoding="utf-8")
    (main / "simai-cache" / "baseline-manifest.json").write_text(
        module.json.dumps(
            {
                "cases": [
                    {
                        "name": "case_a",
                        "config": str(config),
                        "translated": str(translated),
                        "rust_output": str(rust_output),
                        "simai_time_us": None,
                        "simai_end_to_end_csv": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tail / "simai-cache" / "baseline-manifest.json").write_text(
        module.json.dumps(
            {
                "cases": [
                    {
                        "name": "case_a",
                        "config": str(config),
                        "translated": str(translated),
                        "rust_output": str(rust_output),
                        "simai_time_us": 10.0,
                        "simai_end_to_end_csv": str(tail / "simai-cache" / "case_a" / "EndToEnd.csv"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run(cmd, check):
        calls.append((cmd, check))

    monkeypatch.setattr(module, "count_running_simai_processes", lambda root: 0)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output_manifest = tmp_path / "merged.json"
    compare_output = tmp_path / "compare.json"

    result = module.monitor_once(
        [main, tail],
        output_manifest,
        compare_output,
        Path("/bin/flow-sim-rs"),
    )

    assert result["merged_valid_cases"] == 1
    assert module.json.loads(output_manifest.read_text(encoding="utf-8"))["cases"][0][
        "simai_time_us"
    ] == 10.0
    assert calls == [
        (
            [
                "/bin/flow-sim-rs",
                "compare",
                "--manifest",
                str(output_manifest),
                "--output",
                str(compare_output),
            ],
            True,
        )
    ]
    assert result["compare_status"] == "run"
