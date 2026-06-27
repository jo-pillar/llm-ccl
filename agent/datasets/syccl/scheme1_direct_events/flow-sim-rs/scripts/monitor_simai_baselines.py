#!/usr/bin/env python3
"""Monitor SimAI baseline runs and compare when valid baselines exist."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_baseline_manifests import merge_manifest_data


DEFAULT_ROOTS = [
    Path("/home/antl/mntdisk/new-simulator/16m"),
    Path("/home/antl/mntdisk/new-simulator/16m-tail"),
]
DEFAULT_OUTPUT_MANIFEST = Path(
    "/home/antl/mntdisk/new-simulator/16m/reports/merged-baseline-manifest.json"
)
DEFAULT_COMPARE_OUTPUT = Path(
    "/home/antl/mntdisk/new-simulator/16m/reports/merged-baseline-compare.json"
)
DEFAULT_FLOW_SIM = Path("/home/antl/mntdisk/new-simulator/cargo-target/release/flow-sim-rs")
DEFAULT_EXPECTED_FCT_LINES = 261632


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", default=None)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--compare-output", type=Path, default=DEFAULT_COMPARE_OUTPUT)
    parser.add_argument("--flow-sim", type=Path, default=DEFAULT_FLOW_SIM)
    parser.add_argument(
        "--expected-fct-lines",
        type=int,
        default=DEFAULT_EXPECTED_FCT_LINES,
        help="expected complete fct.txt line count for one SimAI case; set 0 to disable progress rate",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="only write monitor JSON and merged manifest; do not run flow-sim-rs compare",
    )
    return parser.parse_args()


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def count_running_simai_processes(root: Path) -> int:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "cmd"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return 0
    cache_text = str(root / "simai-cache") + "/"
    return sum(
        1
        for line in completed.stdout.splitlines()
        if "AstraSimNetwork" in line and cache_text in line
    )


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"cases": []}
    return json.loads(path.read_text(encoding="utf-8"))


def valid_case_count(manifest: dict[str, Any]) -> int:
    return sum(1 for case in manifest.get("cases", []) if case.get("simai_time_us") is not None)


def collect_root_status(root: Path, expected_fct_lines: int = DEFAULT_EXPECTED_FCT_LINES) -> dict[str, Any]:
    simai_cache = root / "simai-cache"
    manifest_path = simai_cache / "baseline-manifest.json"
    manifest = read_manifest(manifest_path)
    fct_files = list(simai_cache.glob("*/fct.txt")) if simai_cache.exists() else []
    fct_line_counts = [count_lines(path) for path in fct_files]
    fct_line_total = sum(fct_line_counts)
    fct_line_target = expected_fct_lines * len(fct_files) if expected_fct_lines > 0 else 0
    fct_progress_rate = (fct_line_total / fct_line_target) if fct_line_target else None
    locks = list(simai_cache.glob("*/simai.lock")) if simai_cache.exists() else []
    nonempty_end_to_end = (
        [path for path in simai_cache.glob("*/*EndToEnd.csv") if path.stat().st_size > 0]
        if simai_cache.exists()
        else []
    )
    return {
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "manifest_cases": len(manifest.get("cases", [])),
        "valid_cases": valid_case_count(manifest),
        "nonempty_end_to_end_files": len(nonempty_end_to_end),
        "active_locks": len(locks),
        "running_simai_processes": count_running_simai_processes(root),
        "fct_files": len(fct_files),
        "max_fct_lines": max(fct_line_counts, default=0),
        "min_fct_lines": min(fct_line_counts, default=0),
        "fct_line_total": fct_line_total,
        "fct_line_target": fct_line_target,
        "fct_progress_rate": fct_progress_rate,
    }


def load_existing_manifests(roots: list[Path]) -> list[dict[str, Any]]:
    manifests = []
    for root in roots:
        path = root / "simai-cache" / "baseline-manifest.json"
        if path.exists():
            manifests.append(read_manifest(path))
    return manifests


def monitor_once(
    roots: list[Path],
    output_manifest: Path,
    compare_output: Path,
    flow_sim: Path,
    *,
    run_compare: bool = True,
    expected_fct_lines: int = DEFAULT_EXPECTED_FCT_LINES,
) -> dict[str, Any]:
    statuses = [collect_root_status(root, expected_fct_lines) for root in roots]
    merged = merge_manifest_data(load_existing_manifests(roots))
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    merged_valid_cases = valid_case_count(merged)
    result: dict[str, Any] = {
        "roots": statuses,
        "merged_manifest": str(output_manifest),
        "merged_cases": len(merged.get("cases", [])),
        "merged_valid_cases": merged_valid_cases,
        "compare_output": str(compare_output),
        "compare_status": "skipped",
    }
    if run_compare and merged_valid_cases > 0:
        compare_output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(flow_sim),
                "compare",
                "--manifest",
                str(output_manifest),
                "--output",
                str(compare_output),
            ],
            check=True,
        )
        result["compare_status"] = "run"
    return result


def main() -> None:
    args = parse_args()
    roots = args.root if args.root else DEFAULT_ROOTS
    result = monitor_once(
        roots,
        args.output_manifest,
        args.compare_output,
        args.flow_sim,
        run_compare=not args.no_compare,
        expected_fct_lines=args.expected_fct_lines,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
