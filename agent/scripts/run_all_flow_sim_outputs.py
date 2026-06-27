#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


AGENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = (
    AGENT_ROOT / "result" / "config" / "h800-64hosts-8gpu-8nic-rail" / "ag"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_EXPERIMENT_ROOT / "all_flow_sim_after_fix"
DEFAULT_FLOW_SIM_BIN = Path(
    "/home/antl/wzd/llm-ccl/Flow-Simulator/flow-sim-rs/target/release/flow-sim-rs"
)


@dataclass(frozen=True)
class FlowCase:
    strategy: str
    case_name: str
    eval_id: str
    candidate_name: str
    eval_dir: Path
    config_path: Path
    sketch_path: Path
    output_path: Path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_time_us(path: Path) -> float | None:
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        value = data.get("time_us")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def extract_chain_len(path: Path) -> int | None:
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    chain = (
        data.get("bottleneck_profile", {})
        .get("critical_flow_chain", {})
        .get("chain")
    )
    if isinstance(chain, list):
        return len(chain)
    return None


def iter_cases(experiment_root: Path, output_root: Path) -> list[FlowCase]:
    outputs_root = experiment_root / "outputs"
    cases: list[FlowCase] = []
    for config_path in sorted(outputs_root.rglob("candidate-config.json")):
        eval_dir = config_path.parent
        try:
            rel = eval_dir.relative_to(outputs_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 5:
            continue
        strategy = parts[0]
        case_name = parts[1]
        eval_id = parts[-1]
        input_root = eval_dir / "flow-sim-inputs"
        for sketch_path in sorted(input_root.glob("candidate-*/candidate-sketch.json")):
            candidate_name = sketch_path.parent.name
            output_path = (
                output_root
                / "runs"
                / strategy
                / case_name
                / eval_id
                / f"{candidate_name}.json"
            )
            cases.append(
                FlowCase(
                    strategy=strategy,
                    case_name=case_name,
                    eval_id=eval_id,
                    candidate_name=candidate_name,
                    eval_dir=eval_dir,
                    config_path=config_path,
                    sketch_path=sketch_path,
                    output_path=output_path,
                )
            )
    return cases


def run_one(flow_sim_bin: Path, case: FlowCase, timeout_s: int) -> dict[str, Any]:
    case.output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cmd = [
        str(flow_sim_bin),
        "simulate-sketch",
        "--config",
        str(case.config_path),
        "--sketch",
        str(case.sketch_path),
        "--output",
        str(case.output_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(AGENT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        elapsed_s = time.time() - started
        ok = proc.returncode == 0
        output_text = proc.stdout[-1000:] if proc.stdout else ""
        error = "" if ok else output_text
    except subprocess.TimeoutExpired as exc:
        elapsed_s = time.time() - started
        ok = False
        output_text = (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else ""
        error = f"timeout after {timeout_s}s {output_text}".strip()

    time_us = extract_time_us(case.output_path) if ok else None
    chain_len = extract_chain_len(case.output_path) if ok else None
    return {
        "strategy": case.strategy,
        "case_name": case.case_name,
        "eval_id": case.eval_id,
        "candidate_name": case.candidate_name,
        "status": "ok" if ok and time_us is not None else "failed",
        "time_us": time_us,
        "critical_chain_len": chain_len,
        "elapsed_s": elapsed_s,
        "config": str(case.config_path),
        "sketch": str(case.sketch_path),
        "output": str(case.output_path),
        "error": error,
    }


def write_rows(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows_json = output_root / "all_results.json"
    rows_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = [
        "strategy",
        "case_name",
        "eval_id",
        "candidate_name",
        "status",
        "time_us",
        "critical_chain_len",
        "elapsed_s",
        "config",
        "sketch",
        "output",
        "error",
    ]
    with (output_root / "all_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in ok_rows:
        groups.setdefault((row["strategy"], row["case_name"]), []).append(row)
    for (strategy, case_name), group in sorted(groups.items()):
        times = sorted(float(row["time_us"]) for row in group)
        best = min(group, key=lambda row: float(row["time_us"]))
        worst = max(group, key=lambda row: float(row["time_us"]))
        summary_rows.append(
            {
                "strategy": strategy,
                "case_name": case_name,
                "count": len(group),
                "best_time_us": times[0],
                "median_time_us": times[len(times) // 2],
                "worst_time_us": times[-1],
                "best_eval_id": best["eval_id"],
                "best_candidate": best["candidate_name"],
                "best_chain_len": best.get("critical_chain_len"),
                "worst_eval_id": worst["eval_id"],
                "worst_candidate": worst["candidate_name"],
            }
        )

    fieldnames = [
        "strategy",
        "case_name",
        "count",
        "best_time_us",
        "median_time_us",
        "worst_time_us",
        "best_eval_id",
        "best_candidate",
        "best_chain_len",
        "worst_eval_id",
        "worst_candidate",
    ]
    with (output_root / "summary_by_strategy_case.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    (output_root / "summary_by_strategy_case.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run flow-sim-rs simulate-sketch for every generated output artifact."
    )
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--flow-sim-bin", type=Path, default=DEFAULT_FLOW_SIM_BIN)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = iter_cases(args.experiment_root.resolve(), args.output_root.resolve())
    if args.limit > 0:
        cases = cases[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = [
        {
            "strategy": case.strategy,
            "case_name": case.case_name,
            "eval_id": case.eval_id,
            "candidate_name": case.candidate_name,
            "config": str(case.config_path),
            "sketch": str(case.sketch_path),
            "output": str(case.output_path),
        }
        for case in cases
    ]
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [
            pool.submit(run_one, args.flow_sim_bin.resolve(), case, args.timeout_s)
            for case in cases
        ]
        total = len(futures)
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            if index % 25 == 0 or index == total:
                ok = sum(1 for item in rows if item.get("status") == "ok")
                print(f"completed {index}/{total} ok={ok}", flush=True)

    rows.sort(key=lambda row: (row["strategy"], row["case_name"], row["eval_id"], row["candidate_name"]))
    write_rows(args.output_root, rows)
    write_summary(args.output_root, rows)
    failed = [row for row in rows if row.get("status") != "ok"]
    print(
        f"wrote {args.output_root / 'all_results.csv'} "
        f"ok={len(rows) - len(failed)} failed={len(failed)}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
