import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_simai_baselines.py"
SPEC = importlib.util.spec_from_file_location("run_simai_baselines", SCRIPT)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_select_cases_all_applies_limit():
    rows = [{"name": f"case_{idx}"} for idx in range(4)]

    selected = module.select_cases(rows, "all", 2)

    assert [row["name"] for row in selected] == ["case_0", "case_1"]


def test_update_manifest_preserves_completed_case_metadata(tmp_path):
    rust_output = tmp_path / "rust-runs" / "case_a.json"
    rust_output.parent.mkdir()
    translated = tmp_path / "input" / "case_a" / "candidate-translated.json"
    translated.parent.mkdir(parents=True)
    translated.write_text("{}", encoding="utf-8")
    benchmarks = {
        "cases": [
            {
                "name": "case_a",
                "config": str(tmp_path / "config.json"),
                "translated": str(translated),
                "rust_output": str(rust_output),
                "simai_time_us": None,
                "simai_end_to_end_csv": None,
            }
        ]
    }
    (tmp_path / "benchmarks.json").write_text(
        module.json.dumps(benchmarks), encoding="utf-8"
    )

    manifest_path = module.update_manifest(
        tmp_path,
        [{"name": "case_a", "status": "run", "simai_time_us": 123.0}],
    )

    data = module.json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["cases"][0]["simai_time_us"] == 123.0
    assert data["cases"][0]["simai_end_to_end_csv"].endswith(
        "simai-cache/case_a/EndToEnd.csv"
    )


def test_has_failed_results_detects_error_status():
    assert module.has_failed_results(
        [
            {"name": "case_a", "status": "cached", "simai_time_us": 1.0},
            {"name": "case_b", "status": "error", "error": "failed"},
        ]
    )


def test_has_failed_results_rejects_missing_simai_time_for_real_run():
    assert module.has_failed_results([{"name": "case_a", "status": "run"}])


def test_has_failed_results_accepts_locked_cases():
    assert not module.has_failed_results([{"name": "case_a", "status": "locked"}])


def test_prepare_case_uses_configured_collective_bytes(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(
        module.json.dumps(
            {
                "coll": {"name": "allgather", "byte": 16777216},
                "hosts": {"host_num": 64, "host_gpu_num": 8, "host_nic_num": 8},
            }
        ),
        encoding="utf-8",
    )
    translated = tmp_path / "input" / "case_16m" / "candidate-translated.json"
    translated.parent.mkdir(parents=True)
    translated.write_text("{}", encoding="utf-8")
    template = tmp_path / "simai-template.conf"
    template.write_text("FCT_OUTPUT_FILE unused\n", encoding="utf-8")

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "SIMAI_CONF_TEMPLATE", template)
    args = module.argparse.Namespace(
        root=tmp_path,
        simai_root=tmp_path / "SimAI",
        config=config,
    )

    case_dir = module.prepare_case(
        {"name": "case_16m", "translated": str(translated)},
        args,
    )

    assert "syccl-flow-size: 16777216" in (case_dir / "syccl-sys.txt").read_text(
        encoding="utf-8"
    )
    assert "ALLGATHER 16777216" in (
        case_dir / "workload-allgather.txt"
    ).read_text(encoding="utf-8")


def test_prepare_case_prefers_translated_chunk_size(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(
        module.json.dumps({"coll": {"name": "allgather", "byte": 4096}}),
        encoding="utf-8",
    )
    translated = tmp_path / "input" / "case_chunk" / "candidate-translated.json"
    translated.parent.mkdir(parents=True)
    translated.write_text(
        module.json.dumps({"chunk_size_byte": 1048576}),
        encoding="utf-8",
    )
    template = tmp_path / "simai-template.conf"
    template.write_text("FCT_OUTPUT_FILE unused\n", encoding="utf-8")

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "SIMAI_CONF_TEMPLATE", template)
    args = module.argparse.Namespace(
        root=tmp_path,
        simai_root=tmp_path / "SimAI",
        config=config,
    )

    case_dir = module.prepare_case(
        {"name": "case_chunk", "translated": str(translated)},
        args,
    )

    assert "syccl-flow-size: 1048576" in (case_dir / "syccl-sys.txt").read_text(
        encoding="utf-8"
    )


def test_run_case_writes_simai_stdout_to_case_log(tmp_path, monkeypatch):
    case_dir = tmp_path / "simai-cache" / "case_a"
    case_dir.mkdir(parents=True)
    (case_dir / "newEndToEnd.csv").write_text(
        "layer_name,workload finished at\ncase_a,12.5\n",
        encoding="utf-8",
    )
    calls = {}

    monkeypatch.setattr(module, "prepare_case", lambda row, args: case_dir)

    def fake_run_simai_subprocess(cmd, **kwargs):
        kwargs["log"].write("simai stdout\n")
        kwargs["log"].write("simai stderr\n")
        calls["cmd"] = cmd
        return None

    monkeypatch.setattr(module, "run_simai_subprocess", fake_run_simai_subprocess)
    args = module.argparse.Namespace(
        root=tmp_path,
        simai_root=tmp_path / "SimAI",
        simai_binary=tmp_path / "AstraSimNetwork-debug",
        mocknccl_log_dir=tmp_path / "mocknccl-logs",
        force=True,
        dry_run=False,
    )

    result = module.run_case({"name": "case_a"}, args)

    assert result["simai_time_us"] == 12.5
    assert "simai stdout" in (case_dir / "simai-run.log").read_text(encoding="utf-8")
    assert "simai stderr" in (case_dir / "simai-run.log").read_text(encoding="utf-8")
    assert calls["cmd"][0] == str(tmp_path / "AstraSimNetwork-debug")


def test_run_case_ignores_empty_end_to_end_files(tmp_path, monkeypatch):
    case_dir = tmp_path / "simai-cache" / "case_a"
    case_dir.mkdir(parents=True)
    (case_dir / "oldEndToEnd.csv").write_text(
        "layer_name,workload finished at\ncase_a,10.0\n",
        encoding="utf-8",
    )
    empty = case_dir / "newerEmptyEndToEnd.csv"
    empty.write_text("", encoding="utf-8")

    monkeypatch.setattr(module, "prepare_case", lambda row, args: case_dir)
    monkeypatch.setattr(module, "run_simai_subprocess", lambda *args, **kwargs: None)
    args = module.argparse.Namespace(
        root=tmp_path,
        simai_root=tmp_path / "SimAI",
        simai_binary=tmp_path / "AstraSimNetwork-debug",
        mocknccl_log_dir=tmp_path / "mocknccl-logs",
        force=True,
        dry_run=False,
    )

    result = module.run_case({"name": "case_a"}, args)

    assert result["simai_time_us"] == 10.0
    assert (case_dir / "EndToEnd.csv").read_text(encoding="utf-8") == (
        "layer_name,workload finished at\ncase_a,10.0\n"
    )


def test_run_case_skips_case_with_active_lock(tmp_path, monkeypatch):
    case_dir = tmp_path / "simai-cache" / "case_a"
    case_dir.mkdir(parents=True)
    (case_dir / "simai.lock").write_text("999999\n", encoding="utf-8")
    calls = {"run": 0}

    monkeypatch.setattr(module, "prepare_case", lambda row, args: case_dir)

    def fake_process_is_running(pid):
        return pid == 999999

    def fake_run(*args, **kwargs):
        calls["run"] += 1

    monkeypatch.setattr(module, "process_is_running", fake_process_is_running)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = module.argparse.Namespace(
        root=tmp_path,
        simai_root=tmp_path / "SimAI",
        simai_binary=tmp_path / "AstraSimNetwork-debug",
        mocknccl_log_dir=tmp_path / "mocknccl-logs",
        force=False,
        dry_run=False,
    )

    result = module.run_case({"name": "case_a"}, args)

    assert result == {"name": "case_a", "status": "locked", "pid": 999999}
    assert calls["run"] == 0


def test_active_lock_pid_accepts_json_lock_with_child_pid(tmp_path, monkeypatch):
    lock_path = tmp_path / "simai.lock"
    lock_path.write_text(
        module.json.dumps({"pid": 4242, "owner_pid": 3131, "case": "case_a"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "process_is_running", lambda pid: pid == 4242)

    assert module.active_lock_pid(lock_path) == 4242


def test_run_simai_subprocess_writes_child_pid_lock(tmp_path, monkeypatch):
    lock_path = tmp_path / "simai.lock"
    log_path = tmp_path / "simai-run.log"
    calls = {}

    class FakePopen:
        pid = 4242

        def __init__(self, cmd, **kwargs):
            calls["cmd"] = cmd
            calls["kwargs"] = kwargs
            kwargs["stdout"].write("child output\n")

        def wait(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", FakePopen)

    with log_path.open("w", encoding="utf-8") as log:
        module.run_simai_subprocess(
            ["AstraSimNetwork-debug", "-t", "1"],
            cwd=tmp_path,
            env={"AS_LOG_LEVEL": "3"},
            log=log,
            lock_path=lock_path,
            case="case_a",
        )

    lock = module.json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["pid"] == 4242
    assert lock["owner_pid"] == module.os.getpid()
    assert lock["case"] == "case_a"
    assert lock["cmd"] == ["AstraSimNetwork-debug", "-t", "1"]
    assert calls["kwargs"]["cwd"] == tmp_path
    assert "child output" in log_path.read_text(encoding="utf-8")


def test_run_case_removes_stale_lock_before_running(tmp_path, monkeypatch):
    case_dir = tmp_path / "simai-cache" / "case_a"
    case_dir.mkdir(parents=True)
    (case_dir / "simai.lock").write_text("123\n", encoding="utf-8")
    (case_dir / "newEndToEnd.csv").write_text(
        "layer_name,workload finished at\ncase_a,12.5\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "prepare_case", lambda row, args: case_dir)
    monkeypatch.setattr(module, "process_is_running", lambda pid: False)
    monkeypatch.setattr(module, "run_simai_subprocess", lambda *args, **kwargs: None)
    args = module.argparse.Namespace(
        root=tmp_path,
        simai_root=tmp_path / "SimAI",
        simai_binary=tmp_path / "AstraSimNetwork-debug",
        mocknccl_log_dir=tmp_path / "mocknccl-logs",
        force=True,
        dry_run=False,
    )

    result = module.run_case({"name": "case_a"}, args)

    assert result["simai_time_us"] == 12.5
    assert not (case_dir / "simai.lock").exists()


def test_run_case_reports_subprocess_signal_as_structured_error(tmp_path, monkeypatch):
    case_dir = tmp_path / "simai-cache" / "case_a"
    case_dir.mkdir(parents=True)

    monkeypatch.setattr(module, "prepare_case", lambda row, args: case_dir)

    def fake_run_simai_subprocess(cmd, **kwargs):
        raise module.subprocess.CalledProcessError(-11, cmd)

    monkeypatch.setattr(module, "run_simai_subprocess", fake_run_simai_subprocess)
    args = module.argparse.Namespace(
        root=tmp_path,
        simai_root=tmp_path / "SimAI",
        simai_binary=tmp_path / "AstraSimNetwork-optimized",
        mocknccl_log_dir=tmp_path / "mocknccl-logs",
        force=True,
        dry_run=False,
    )

    result = module.run_case({"name": "case_a"}, args)

    assert result["status"] == "error"
    assert result["returncode"] == -11
    assert result["signal"] == 11
    assert "SIGSEGV" in result["signal_name"]
    assert not (case_dir / "simai.lock").exists()


def test_run_case_uses_configured_simai_threads(tmp_path, monkeypatch):
    case_dir = tmp_path / "simai-cache" / "case_a"
    case_dir.mkdir(parents=True)
    (case_dir / "newEndToEnd.csv").write_text(
        "layer_name,workload finished at\ncase_a,12.5\n",
        encoding="utf-8",
    )
    calls = {}

    monkeypatch.setattr(module, "prepare_case", lambda row, args: case_dir)

    def fake_run_simai_subprocess(cmd, **kwargs):
        calls["cmd"] = cmd

    monkeypatch.setattr(module, "run_simai_subprocess", fake_run_simai_subprocess)
    args = module.argparse.Namespace(
        root=tmp_path,
        simai_root=tmp_path / "SimAI",
        simai_binary=tmp_path / "AstraSimNetwork-debug",
        mocknccl_log_dir=tmp_path / "mocknccl-logs",
        simai_threads=8,
        force=True,
        dry_run=False,
    )

    result = module.run_case({"name": "case_a"}, args)

    assert result["simai_time_us"] == 12.5
    assert calls["cmd"][calls["cmd"].index("-t") + 1] == "8"


def test_resolve_simai_binary_prefers_stable_debug_build(tmp_path):
    sim_dir = (
        tmp_path
        / "SimAI"
        / "astra-sim-alibabacloud"
        / "extern"
        / "network_backend"
        / "ns3-interface"
        / "simulation"
        / "build"
        / "scratch"
    )
    sim_dir.mkdir(parents=True)
    debug = sim_dir / "ns3.36.1-AstraSimNetwork-debug"
    optimized = sim_dir / "ns3.36.1-AstraSimNetwork-optimized"
    debug.write_text("debug", encoding="utf-8")
    optimized.write_text("optimized", encoding="utf-8")

    resolved = module.resolve_simai_binary(tmp_path / "SimAI", None)

    assert resolved == debug


def test_build_simai_env_sets_mocknccl_log_dir(tmp_path):
    args = module.argparse.Namespace(
        mocknccl_log_dir=tmp_path / "mocknccl-logs",
        simai_log_level="3",
    )

    env = module.build_simai_env(args)

    assert env["AS_SEND_LAT"] == "3"
    assert env["AS_NVLS_ENABLE"] == "1"
    assert env["AS_LOG_LEVEL"] == "3"
    assert env["AS_LOG_DIR"] == str(tmp_path / "mocknccl-logs")
    assert env["SIMAI_LOG_PATH"] == str(tmp_path / "mocknccl-logs")
    assert (tmp_path / "mocknccl-logs").is_dir()


def test_build_simai_env_allows_log_level_override(tmp_path):
    args = module.argparse.Namespace(
        mocknccl_log_dir=tmp_path / "mocknccl-logs",
        simai_log_level="1",
    )

    env = module.build_simai_env(args)

    assert env["AS_LOG_LEVEL"] == "1"


def test_truncate_mocknccl_logs_only_matches_known_log_patterns(tmp_path):
    keep = tmp_path / "keep.txt"
    keep.write_text("keep", encoding="utf-8")
    matched = [
        tmp_path / "SIMAI_pid1_20260518.log",
        tmp_path / "SimAI.log.full",
        tmp_path / "SimAI_123_456.log",
    ]
    for path in matched:
        path.write_text("x" * 10, encoding="utf-8")

    truncated = module.truncate_mocknccl_logs(tmp_path)

    assert truncated == len(matched)
    assert keep.read_text(encoding="utf-8") == "keep"
    assert all(path.stat().st_size == 0 for path in matched)
