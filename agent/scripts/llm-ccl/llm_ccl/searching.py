from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .manifest import ManifestStore


AGENT_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = AGENT_ROOT / "datasets" / "syccl" / "scheme1_direct_events" / "evaluator.py"
DEFAULT_FLOW_SIM_BIN = (
    AGENT_ROOT
    / "datasets"
    / "syccl"
    / "scheme1_direct_events"
    / "flow-sim-rs"
    / "target"
    / "release"
    / "flow-sim-rs"
)

Runner = Callable[..., int]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_search_command(
    case_dir: Path,
    attempt_dir: Path,
    *,
    model: str,
    max_generations: int,
    k_candidates: int,
    eval_concurrency: int,
    gen_concurrency: int,
    llm_policy_pool_size: int,
    extra_args: Sequence[str] = (),
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "main.py",
        "--init-program",
        str(case_dir / "init_program.py"),
        "--evaluator",
        str(EVALUATOR),
        "--instruction",
        str(case_dir / "instruction.txt"),
        "--selector",
        "llm_elite",
        "--elite-selection-strategy",
        "all",
        "--num-chains",
        "1",
        "--k-candidates",
        str(k_candidates),
        "--stream-k-candidates",
        "--max-generations",
        str(max_generations),
        "--eval-concurrency",
        str(eval_concurrency),
        "--gen-concurrency",
        str(gen_concurrency),
        "--init-eval-repeats",
        "1",
        "--llm-policy-pool-size",
        str(llm_policy_pool_size),
        "--output-path",
        str(attempt_dir / "checkpoints"),
        "--model",
        model,
        "--save-llm-io",
        "--disable-reflection",
        "--skip-preflight",
        *extra_args,
    ]


def _subprocess_runner(command, *, env, cwd, log_path, timeout) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    return completed.returncode


def _run_one(runner: Runner, command: list[str], env: dict[str, str], log_path: Path, timeout: float | None) -> int:
    try:
        return runner(command, env=env, cwd=AGENT_ROOT, log_path=log_path, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(f"search timed out: {exc}\n", encoding="utf-8")
        return 124
    except Exception as exc:
        log_path.write_text(f"search failed before completion: {exc}\n", encoding="utf-8")
        return 1


def search_all(
    bundle: Path,
    *,
    model: str,
    jobs: int = 1,
    max_generations: int = 10000,
    k_candidates: int = 4,
    eval_concurrency: int = 1,
    gen_concurrency: int = 1,
    llm_policy_pool_size: int = 100,
    flow_sim_bin: Path = DEFAULT_FLOW_SIM_BIN,
    timeout: float | None = None,
    extra_args: Sequence[str] = (),
    runner: Runner = _subprocess_runner,
) -> dict:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    bundle = bundle.resolve()
    store = ManifestStore(bundle / "manifest.json")
    manifest = store.load()
    scheduled: list[tuple[str, str, list[str], dict[str, str], Path]] = []

    for case_id, case in manifest["cases"].items():
        search = case["search"]
        if search["status"] == "succeeded":
            continue
        for old_attempt in search["attempts"]:
            if old_attempt["status"] == "running":
                old_attempt["status"] = "failed"
                old_attempt["finished_at"] = _now()
                old_attempt["error"] = "interrupted before completion"

        attempt_id = f"attempt-{len(search['attempts']) + 1:04d}"
        case_dir = bundle / "cases" / case_id
        attempt_dir = case_dir / "search" / attempt_id
        (attempt_dir / "checkpoints").mkdir(parents=True)
        (attempt_dir / "eval_artifacts").mkdir()
        attempt = {
            "id": attempt_id,
            "status": "running",
            "path": str(attempt_dir.relative_to(bundle)),
            "started_at": _now(),
        }
        search["attempts"].append(attempt)
        search["status"] = "running"

        env = os.environ.copy()
        env["SYCCL_BASE_CONFIG"] = str(case_dir / "config.json")
        env["SYCCL_EVAL_ARTIFACT_DIR"] = str(attempt_dir / "eval_artifacts")
        env["FLOW_SIM_BIN"] = str(flow_sim_bin)
        command = build_search_command(
            case_dir,
            attempt_dir,
            model=model,
            max_generations=max_generations,
            k_candidates=k_candidates,
            eval_concurrency=eval_concurrency,
            gen_concurrency=gen_concurrency,
            llm_policy_pool_size=llm_policy_pool_size,
            extra_args=extra_args,
        )
        scheduled.append((case_id, attempt_id, command, env, attempt_dir / "search.log"))

    if not scheduled:
        return manifest
    store.save(manifest)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_run_one, runner, command, env, log_path, timeout): (case_id, attempt_id)
            for case_id, attempt_id, command, env, log_path in scheduled
        }
        for future in as_completed(futures):
            case_id, attempt_id = futures[future]
            exit_code = future.result()

            def finish(payload: dict) -> None:
                search = payload["cases"][case_id]["search"]
                attempt = next(item for item in search["attempts"] if item["id"] == attempt_id)
                attempt["status"] = "succeeded" if exit_code == 0 else "failed"
                attempt["exit_code"] = exit_code
                attempt["finished_at"] = _now()
                search["status"] = attempt["status"]
                if exit_code == 0:
                    search["latest_successful_attempt"] = attempt_id

            manifest = store.update(finish)

    return manifest
