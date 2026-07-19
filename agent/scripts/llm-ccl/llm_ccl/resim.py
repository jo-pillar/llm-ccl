from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from .manifest import ManifestStore
from .searching import AGENT_ROOT, DEFAULT_FLOW_SIM_BIN


Runner = Callable[..., int]


def _subprocess_runner(command, *, cwd, log_path, timeout) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    return completed.returncode


def _run(runner: Runner, command: list[str], *, cwd: Path, log_path: Path, timeout: float | None) -> int:
    try:
        return runner(command, cwd=cwd, log_path=log_path, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(f"command timed out: {exc}\n", encoding="utf-8")
        return 124
    except Exception as exc:
        log_path.write_text(f"command failed before completion: {exc}\n", encoding="utf-8")
        return 1


def resim_selected(
    bundle: Path,
    *,
    synthesize_bin: Path,
    flow_sim_bin: Path = DEFAULT_FLOW_SIM_BIN,
    timeout: float | None = None,
    runner: Runner = _subprocess_runner,
) -> dict:
    bundle = bundle.resolve()
    store = ManifestStore(bundle / "manifest.json")
    manifest = store.load()

    unfinished_searches = [
        case_id
        for case_id, case in manifest["cases"].items()
        if case["search"]["status"] not in {"succeeded", "failed"}
    ]
    if unfinished_searches:
        raise RuntimeError(f"searches are not finished: {unfinished_searches}")

    unfinished_selections = [
        case_id
        for case_id, case in manifest["cases"].items()
        if case["search"]["status"] == "succeeded"
        and case["selection"]["status"] not in {"succeeded", "failed"}
    ]
    if unfinished_selections:
        raise RuntimeError(f"selections are not finished: {unfinished_selections}")

    for case_id, case in manifest["cases"].items():
        if case["search"]["status"] == "failed":
            case["resim"] = {"status": "skipped", "reason": "search failed"}
        elif case["selection"]["status"] != "succeeded":
            case["resim"] = {"status": "skipped", "reason": "selection failed"}
    store.save(manifest)

    for case_id, case in manifest["cases"].items():
        if case["selection"]["status"] != "succeeded":
            continue
        existing_output = case["resim"].get("output")
        if (
            case["resim"]["status"] == "succeeded"
            and isinstance(existing_output, str)
            and (bundle / existing_output).is_file()
        ):
            continue
        case_dir = bundle / "cases" / case_id
        best_dir = case_dir / "best"
        config = best_dir / "candidate-config.json"
        sketch = best_dir / "candidate-sketch.json"
        if not config.is_file() or not sketch.is_file():
            case["resim"] = {"status": "failed", "reason": "best candidate files are missing"}
            store.save(manifest)
            continue

        resim_dir = case_dir / "resim"
        resim_dir.mkdir(exist_ok=True)
        flow_output = resim_dir / "flow-sim.json"
        translated = resim_dir / "translated.json"
        syccl_output = resim_dir / "resim.json"
        case["resim"] = {"status": "running"}
        store.save(manifest)

        flow_command = [
            str(flow_sim_bin),
            "simulate-sketch",
            "--config",
            str(config),
            "--sketch",
            str(sketch),
            "--output",
            str(flow_output),
            "--dump-translated",
            str(translated),
        ]
        flow_status = _run(
            runner,
            flow_command,
            cwd=AGENT_ROOT,
            log_path=resim_dir / "flow-sim.log",
            timeout=timeout,
        )
        if flow_status != 0 or not translated.is_file():
            case["resim"] = {
                "status": "failed",
                "stage": "flow-sim",
                "exit_code": flow_status,
            }
            store.save(manifest)
            continue

        syccl_command = [
            str(synthesize_bin),
            "-f",
            str(config),
            "resim",
            "-i",
            str(translated),
            "-o",
            str(syccl_output),
        ]
        syccl_status = _run(
            runner,
            syccl_command,
            cwd=synthesize_bin.parent.parent,
            log_path=resim_dir / "syccl.log",
            timeout=timeout,
        )
        case["resim"] = (
            {
                "status": "succeeded",
                "flow_sim_output": str(flow_output.relative_to(bundle)),
                "translated": str(translated.relative_to(bundle)),
                "output": str(syccl_output.relative_to(bundle)),
            }
            if syccl_status == 0 and syccl_output.is_file()
            else {"status": "failed", "stage": "syccl", "exit_code": syccl_status}
        )
        store.save(manifest)

    return manifest
