from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .manifest import ManifestStore


def write_report(bundle: Path) -> tuple[Path, Path]:
    bundle = bundle.resolve()
    manifest = ManifestStore(bundle / "manifest.json").load()
    rows: list[dict] = []
    for case_id, case in manifest["cases"].items():
        selection = case["selection"]
        resim = case["resim"]
        rows.append(
            {
                "case_id": case_id,
                "scale": case["scale"],
                "gpu_count": case["gpu_count"],
                "collective": case["collective"],
                "total_message_size": case["total_message_size"],
                "coll_byte": case["coll_byte"],
                "search_status": case["search"]["status"],
                "selection_status": selection["status"],
                "candidate": selection.get("candidate", ""),
                "flow_sim_time_us": selection.get("time_us", ""),
                "resim_status": resim["status"],
                "resim_output": resim.get("output", ""),
                "reason": resim.get("reason", selection.get("reason", "")),
            }
        )

    search_counts = Counter(row["search_status"] for row in rows)
    selection_counts = Counter(row["selection_status"] for row in rows)
    resim_counts = Counter(row["resim_status"] for row in rows)
    summary = {
        "total_cases": len(rows),
        "search_succeeded": search_counts["succeeded"],
        "search_failed": search_counts["failed"],
        "selection_succeeded": selection_counts["succeeded"],
        "selection_failed": selection_counts["failed"],
        "resim_succeeded": resim_counts["succeeded"],
        "resim_failed": resim_counts["failed"],
        "resim_skipped": resim_counts["skipped"],
    }

    report_dir = bundle / "reports"
    report_dir.mkdir(exist_ok=True)
    json_path = report_dir / "report.json"
    csv_path = report_dir / "report.csv"
    json_path.write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path
