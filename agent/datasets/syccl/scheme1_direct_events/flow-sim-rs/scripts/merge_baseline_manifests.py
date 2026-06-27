#!/usr/bin/env python3
"""Merge baseline manifests produced by independent SimAI runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("manifests", type=Path, nargs="+")
    return parser.parse_args()


def has_simai_time(case: dict[str, Any]) -> bool:
    return case.get("simai_time_us") is not None


def merge_manifest_data(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for manifest in manifests:
        for case in manifest.get("cases", []):
            name = case["name"]
            if name not in by_name:
                by_name[name] = dict(case)
                order.append(name)
                continue
            if has_simai_time(case) and not has_simai_time(by_name[name]):
                by_name[name] = dict(case)
    return {"cases": [by_name[name] for name in order]}


def main() -> None:
    args = parse_args()
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.manifests
        if path.exists()
    ]
    merged = merge_manifest_data(manifests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"wrote {len(merged['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
