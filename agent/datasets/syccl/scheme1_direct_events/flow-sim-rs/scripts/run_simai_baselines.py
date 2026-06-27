#!/usr/bin/env python3
"""Run cached SimAI baselines for Rust flow-simulator cases.

The script is intentionally resumable: completed cases with an EndToEnd.csv are
skipped unless --force is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_ROOT = Path("/home/antl/mntdisk/new-simulator")
DEFAULT_SIMAI_ROOT = Path("/home/antl/wzd/SimAI")
DEFAULT_CONFIG = Path(
    "/home/antl/mntdisk/syccl-llm-scheme1-direct-events/"
    "configs/multirail-512gpu/allgather/multirail-512gpu-ag-4k.json"
)
SIMAI_CONF_TEMPLATE = Path("/tmp/simai_syccl_512gpu_result/simai.conf")
DEFAULT_MOCKNCCL_LOG_DIR = Path("/home/antl/mntdisk/new-simulator/mocknccl-logs")
LEGACY_MOCKNCCL_LOG_DIR = Path("/etc/astra-sim")
DEFAULT_SIMAI_LOG_LEVEL = "3"


KMAX_MAP = (
    "KMAX_MAP 7 100000000000 400 364000000000 3200 "
    "381681664000 3200 400000000000 3200 1200000000000 4800 "
    "1258291200000 4800 2400000000000 4800"
)
KMIN_MAP = (
    "KMIN_MAP 7 100000000000 100 364000000000 800 "
    "381681664000 800 400000000000 800 1200000000000 1200 "
    "1258291200000 1200 2400000000000 1200"
)
PMAX_MAP = (
    "PMAX_MAP 7 100000000000 0.2 364000000000 0.2 "
    "381681664000 0.2 400000000000 0.2 1200000000000 0.2 "
    "1258291200000 0.2 2400000000000 0.2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--simai-root", type=Path, default=DEFAULT_SIMAI_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected cases")
    parser.add_argument("--jobs", type=int, default=1, help="number of SimAI cases to run concurrently")
    parser.add_argument(
        "--selection",
        choices=("representative", "all", "head", "tail"),
        default="representative",
    )
    parser.add_argument(
        "--simai-binary",
        type=Path,
        default=None,
        help="override AstraSimNetwork binary; defaults to the stable debug build",
    )
    parser.add_argument(
        "--mocknccl-log-dir",
        type=Path,
        default=DEFAULT_MOCKNCCL_LOG_DIR,
        help="directory for SimAI MockNcclLog files when the SimAI build supports AS_LOG_DIR",
    )
    parser.add_argument(
        "--simai-log-level",
        default=DEFAULT_SIMAI_LOG_LEVEL,
        help="AS_LOG_LEVEL for SimAI MockNcclLog; 3 keeps ERROR logs only",
    )
    parser.add_argument(
        "--simai-threads",
        type=int,
        default=1,
        help="SimAI -t thread count; default preserves the sequential baseline runner behavior",
    )
    parser.add_argument(
        "--truncate-legacy-mocknccl-logs",
        action="store_true",
        help="truncate old /etc/astra-sim MockNcclLog files before launching cases",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_summary(root: Path) -> list[dict[str, str]]:
    path = root / "reports" / "rust-summary.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_cases(rows: list[dict[str, str]], selection: str, limit: int) -> list[dict[str, str]]:
    if selection == "all":
        chosen = rows
    elif selection == "head":
        chosen = rows[: max(limit, 1)]
    elif selection == "tail":
        chosen = rows[-max(limit, 1) :]
    else:
        indexes = {
            0,
            1,
            2,
            len(rows) // 4,
            len(rows) // 2,
            (len(rows) * 3) // 4,
            len(rows) - 3,
            len(rows) - 2,
            len(rows) - 1,
        }
        chosen = [rows[i] for i in sorted(indexes) if 0 <= i < len(rows)]
        if limit:
            chosen = chosen[:limit]
    if selection != "representative" and limit:
        chosen = chosen[:limit]
    return chosen


def write_simai_conf(template: Path, output: Path, case_dir: Path) -> None:
    replacements = {
        "FLOW_FILE": f"FLOW_FILE {case_dir / 'flow.txt'}",
        "TRACE_FILE": f"TRACE_FILE {case_dir / 'trace.txt'}",
        "TRACE_OUTPUT_FILE": f"TRACE_OUTPUT_FILE {case_dir / 'trace.tr'}",
        "FCT_OUTPUT_FILE": f"FCT_OUTPUT_FILE {case_dir / 'fct.txt'}",
        "PFC_OUTPUT_FILE": f"PFC_OUTPUT_FILE {case_dir / 'pfc.txt'}",
        "QLEN_MON_FILE": f"QLEN_MON_FILE {case_dir / 'qlen.txt'}",
        "BW_MON_FILE": f"BW_MON_FILE {case_dir / 'bw.txt'}",
        "RATE_MON_FILE": f"RATE_MON_FILE {case_dir / 'rate.txt'}",
        "CNP_MON_FILE": f"CNP_MON_FILE {case_dir / 'cnp.txt'}",
        "KMAX_MAP": KMAX_MAP,
        "KMIN_MAP": KMIN_MAP,
        "PMAX_MAP": PMAX_MAP,
    }
    lines = []
    for line in template.read_text(encoding="utf-8").splitlines():
        key = line.split(maxsplit=1)[0] if line.strip() else ""
        lines.append(replacements.get(key, line))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_collective_bytes(config: Path) -> int:
    data = json.loads(config.read_text(encoding="utf-8"))
    return int(data["coll"]["byte"])


def read_flow_size(row: dict[str, str], config: Path) -> int:
    translated = row.get("translated")
    if translated:
        data = json.loads(Path(translated).read_text(encoding="utf-8"))
        if data.get("chunk_size_byte"):
            return int(data["chunk_size_byte"])
    return read_collective_bytes(config)


def prepare_case(row: dict[str, str], args: argparse.Namespace) -> Path:
    case = row["name"]
    case_dir = args.root / "simai-cache" / case
    case_dir.mkdir(parents=True, exist_ok=True)
    flow_size = read_flow_size(row, args.config)

    topo_script = args.simai_root / "astra-sim-alibabacloud/inputs/topo/gen_Syccl_Multirail_Topo.py"
    subprocess.run(
        [
            "python3",
            str(topo_script),
            "-c",
            str(args.config),
            "-o",
            str(case_dir / "topofile"),
            "-gt",
            "A100",
        ],
        check=True,
    )
    (case_dir / "syccl-sys.txt").write_text(
        "\n".join(
            [
                "all-gather-implementation: sycclFlowModel",
                f"syccl-translated-path: {row['translated']}",
                f"syccl-flow-size: {flow_size}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (case_dir / "workload-allgather.txt").write_text(
        "\n".join(
            [
                "HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 512 ep: 1 pp: 1 vpp: 1 ga: 1 all_gpus: 512 checkpoints: 0 checkpoint_initiates: 0",
                "1",
                f"{case} -1 1 ALLGATHER {flow_size} 1 NONE 0 1 NONE 0 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_simai_conf(SIMAI_CONF_TEMPLATE, case_dir / "simai.conf", case_dir)
    return case_dir


def simulation_dir(simai_root: Path) -> Path:
    return (
        simai_root
        / "astra-sim-alibabacloud"
        / "extern"
        / "network_backend"
        / "ns3-interface"
        / "simulation"
    )


def resolve_simai_binary(simai_root: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    scratch = simulation_dir(simai_root) / "build" / "scratch"
    candidates = [
        scratch / "ns3.36.1-AstraSimNetwork-debug",
        scratch / "ns3.36.1-AstraSimNetwork-default",
        scratch / "ns3.36.1-AstraSimNetwork-release",
        scratch / "ns3.36.1-AstraSimNetwork-optimized",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find AstraSimNetwork binary under {scratch}")


def build_simai_env(args: argparse.Namespace) -> dict[str, str]:
    env = {**os.environ, "AS_SEND_LAT": "3", "AS_NVLS_ENABLE": "1"}
    simai_log_level = getattr(args, "simai_log_level", DEFAULT_SIMAI_LOG_LEVEL)
    if simai_log_level:
        env["AS_LOG_LEVEL"] = str(simai_log_level)
    mocknccl_log_dir = getattr(args, "mocknccl_log_dir", DEFAULT_MOCKNCCL_LOG_DIR)
    if mocknccl_log_dir:
        mocknccl_log_dir.mkdir(parents=True, exist_ok=True)
        env["AS_LOG_DIR"] = str(mocknccl_log_dir)
        env["SIMAI_LOG_PATH"] = str(mocknccl_log_dir)
    return env


def truncate_mocknccl_logs(log_dir: Path) -> int:
    if not log_dir.exists():
        return 0
    patterns = ("SIMAI_pid*.log", "SimAI.log*", "SimAI_*.log", "SimAi_*.log")
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in log_dir.glob(pattern) if path.is_file())
    for path in paths:
        path.write_text("", encoding="utf-8")
    return len(paths)


def read_simai_time(end_to_end: Path) -> float:
    lines = end_to_end.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        parts = [part.strip() for part in line.split(",")]
        if "workload finished at" in parts:
            value_idx = parts.index("workload finished at")
            if idx + 1 < len(lines):
                values = [part.strip() for part in lines[idx + 1].split(",")]
                if value_idx < len(values):
                    return float(values[value_idx])
    for line in lines:
        if "workload finished at" in line and not line.startswith("layer_name"):
            parts = [part.strip() for part in line.split(",")]
            return float(parts[-1])
    raise RuntimeError(f"could not parse workload finished at from {end_to_end}")


def latest_valid_end_to_end(paths: list[Path]) -> tuple[Path, float] | None:
    for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            return path, read_simai_time(path)
        except (OSError, RuntimeError, ValueError):
            continue
    return None


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def active_lock_pid(lock_path: Path) -> int | None:
    if not lock_path.exists():
        return None
    text = lock_path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
        pid = int(data["pid"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        try:
            pid = int(text)
        except ValueError:
            lock_path.unlink(missing_ok=True)
            return None
    if process_is_running(pid):
        return pid
    lock_path.unlink(missing_ok=True)
    return None


def write_lock(lock_path: Path, *, case: str, pid: int | None, cmd: list[str]) -> None:
    if pid is None:
        lock_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        return
    lock_path.write_text(
        json.dumps(
            {
                "pid": pid,
                "owner_pid": os.getpid(),
                "case": case,
                "cmd": cmd,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_simai_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log,
    lock_path: Path,
    case: str,
) -> None:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=log,
        text=True,
    )
    write_lock(lock_path, case=case, pid=process.pid, cmd=cmd)
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, cmd)


def run_case(row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    case = row["name"]
    case_dir = prepare_case(row, args)
    lock_path = case_dir / "simai.lock"
    locked_pid = active_lock_pid(lock_path)
    if locked_pid is not None:
        return {"name": case, "status": "locked", "pid": locked_pid}
    cached_end = case_dir / "EndToEnd.csv"
    if cached_end.exists() and not args.force:
        cached_valid = latest_valid_end_to_end([cached_end])
        if cached_valid is not None:
            _, simai_time = cached_valid
            return {"name": case, "status": "cached", "simai_time_us": simai_time}

    simai_binary = resolve_simai_binary(args.simai_root, getattr(args, "simai_binary", None))
    before = set(case_dir.glob("*EndToEnd.csv"))
    cmd = [
        str(simai_binary),
        "-t",
        str(getattr(args, "simai_threads", 1)),
        "-w",
        str(case_dir / "workload-allgather.txt"),
        "-s",
        str(case_dir / "syccl-sys.txt"),
        "-n",
        str(case_dir / "topofile"),
        "-c",
        str(case_dir / "simai.conf"),
    ]
    if args.dry_run:
        return {"name": case, "status": "dry-run", "cmd": cmd}

    write_lock(lock_path, case=case, pid=None, cmd=cmd)
    try:
        with (case_dir / "simai-run.log").open("a", encoding="utf-8") as log:
            log.write(f"$ {' '.join(cmd)}\n")
            log.flush()
            try:
                run_simai_subprocess(
                    cmd,
                    cwd=case_dir,
                    env=build_simai_env(args),
                    log=log,
                    lock_path=lock_path,
                    case=case,
                )
            except subprocess.CalledProcessError as err:
                return subprocess_error_result(case, err)
        after = set(case_dir.glob("*EndToEnd.csv"))
        valid = latest_valid_end_to_end(list(after - before))
        if valid is None:
            valid = latest_valid_end_to_end(list(after))
        if valid is None:
            raise RuntimeError(f"could not find a valid EndToEnd.csv under {case_dir}")
        valid_end, simai_time = valid
        shutil.copy2(valid_end, cached_end)
        return {"name": case, "status": "run", "simai_time_us": simai_time}
    finally:
        lock_path.unlink(missing_ok=True)


def subprocess_error_result(case: str, err: subprocess.CalledProcessError) -> dict[str, object]:
    result: dict[str, object] = {
        "name": case,
        "status": "error",
        "returncode": err.returncode,
        "error": str(err),
    }
    if err.returncode < 0:
        signum = -err.returncode
        result["signal"] = signum
        try:
            result["signal_name"] = signal.Signals(signum).name
        except ValueError:
            result["signal_name"] = f"SIG{signum}"
    return result


def update_manifest(root: Path, results: list[dict[str, object]]) -> Path:
    original = json.loads((root / "benchmarks.json").read_text(encoding="utf-8"))
    out = root / "simai-cache" / "baseline-manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    by_name = {}
    if out.exists():
        previous = json.loads(out.read_text(encoding="utf-8"))
        for case in previous.get("cases", []):
            if "simai_time_us" in case:
                by_name[case["name"]] = case
    by_name.update({r["name"]: r for r in results if "simai_time_us" in r})

    cases = []
    for case in original["cases"]:
        result = by_name.get(case["name"])
        if result:
            case["simai_time_us"] = result["simai_time_us"]
            case["simai_end_to_end_csv"] = str(root / "simai-cache" / case["name"] / "EndToEnd.csv")
            cases.append(case)
    out.write_text(json.dumps({"cases": cases}, indent=2), encoding="utf-8")
    return out


def has_failed_results(results: list[dict[str, object]]) -> bool:
    for result in results:
        if result.get("status") == "error":
            return True
        if result.get("status") not in {"dry-run", "locked"} and "simai_time_us" not in result:
            return True
    return False


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.truncate_legacy_mocknccl_logs:
        truncated = truncate_mocknccl_logs(LEGACY_MOCKNCCL_LOG_DIR)
        print(
            f"truncated {truncated} legacy MockNcclLog files under {LEGACY_MOCKNCCL_LOG_DIR}",
            file=sys.stderr,
            flush=True,
        )
    rows = load_summary(args.root)
    selected = select_cases(rows, args.selection, args.limit)
    results = []
    if args.jobs == 1:
        for idx, row in enumerate(selected, 1):
            print(f"[{idx}/{len(selected)}] {row['name']} rust_time_us={row['time_us']}", flush=True)
            result = run_case(row, args)
            print(json.dumps(result), flush=True)
            results.append(result)
            update_manifest(args.root, results)
        if has_failed_results(results):
            raise SystemExit(1)
        return

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_case, row, args): (idx, row)
            for idx, row in enumerate(selected, 1)
        }
        for future in as_completed(futures):
            idx, row = futures[future]
            try:
                result = future.result()
            except Exception as err:
                result = {"name": row["name"], "status": "error", "error": str(err)}
            print(f"[{idx}/{len(selected)}] {row['name']} rust_time_us={row['time_us']}", flush=True)
            print(json.dumps(result), flush=True)
            results.append(result)
            update_manifest(args.root, results)
    if has_failed_results(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
