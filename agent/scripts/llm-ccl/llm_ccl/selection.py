from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from .manifest import ManifestStore


def _read_time(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("time_us") if isinstance(payload, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value > 0 and math.isfinite(value) else None


def _resolve_output(eval_dir: Path, raw: object) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else eval_dir / path


def _resolve_sketch(eval_dir: Path, item: dict) -> tuple[str, Path] | None:
    name = item.get("name")
    if isinstance(name, str) and name:
        sketch = eval_dir / "flow-sim-inputs" / name / "candidate-sketch.json"
        return (name, sketch) if sketch.is_file() else None
    inputs = eval_dir / "flow-sim-inputs"
    sketches = sorted(inputs.glob("*/candidate-sketch.json")) if inputs.is_dir() else []
    if len(sketches) != 1:
        return None
    return sketches[0].parent.name, sketches[0]


def _best_candidate(attempt_dir: Path) -> dict | None:
    root = attempt_dir / "eval_artifacts" / "scheme1_direct_events"
    best: dict | None = None
    for manifest_path in sorted(root.glob("*/flow-sim-manifest.json")):
        eval_dir = manifest_path.parent
        config = eval_dir / "candidate-config.json"
        if not config.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cases = manifest.get("cases") if isinstance(manifest, dict) else None
        if not isinstance(cases, list):
            continue
        for item in cases:
            if not isinstance(item, dict):
                continue
            output = _resolve_output(eval_dir, item.get("rust_output"))
            sketch_info = _resolve_sketch(eval_dir, item)
            if output is None or sketch_info is None or not output.is_file():
                continue
            time_us = _read_time(output)
            if time_us is None:
                continue
            name, sketch = sketch_info
            candidate = {
                "candidate": name,
                "time_us": time_us,
                "config": config,
                "sketch": sketch,
                "flow_sim": output,
                "eval_dir": eval_dir,
            }
            if best is None or time_us < best["time_us"]:
                best = candidate
    return best


def select_all(bundle: Path) -> dict:
    bundle = bundle.resolve()
    store = ManifestStore(bundle / "manifest.json")
    manifest = store.load()

    for case_id, case in manifest["cases"].items():
        search = case["search"]
        selection = case["selection"]
        if search["status"] != "succeeded":
            if search["status"] == "failed":
                selection.clear()
                selection.update({"status": "skipped", "reason": "search failed"})
            continue
        attempt_id = search.get("latest_successful_attempt")
        if not isinstance(attempt_id, str):
            selection.clear()
            selection.update({"status": "failed", "reason": "successful search has no attempt"})
            continue

        case_dir = bundle / "cases" / case_id
        best = _best_candidate(case_dir / "search" / attempt_id)
        if best is None:
            selection.clear()
            selection.update({"status": "failed", "attempt_id": attempt_id, "reason": "no valid candidate"})
            continue

        best_dir = case_dir / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        best_dir.mkdir()
        shutil.copy2(best["config"], best_dir / "candidate-config.json")
        shutil.copy2(best["sketch"], best_dir / "candidate-sketch.json")
        shutil.copy2(best["flow_sim"], best_dir / "flow-sim.json")
        selection.clear()
        selection.update(
            {
                "status": "succeeded",
                "attempt_id": attempt_id,
                "candidate": best["candidate"],
                "time_us": best["time_us"],
                "source_eval_dir": str(best["eval_dir"].relative_to(bundle)),
            }
        )

    store.save(manifest)
    return manifest
