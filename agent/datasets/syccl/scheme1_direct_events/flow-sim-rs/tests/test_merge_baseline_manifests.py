import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_baseline_manifests.py"
SPEC = importlib.util.spec_from_file_location("merge_baseline_manifests", SCRIPT)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_merge_manifests_prefers_cases_with_simai_time(tmp_path):
    base = {
        "cases": [
            {
                "name": "case_a",
                "rust_output": "/runs/case_a.json",
                "translated": "/inputs/case_a/candidate-translated.json",
                "simai_time_us": None,
                "simai_end_to_end_csv": None,
            },
            {
                "name": "case_b",
                "rust_output": "/runs/case_b.json",
                "translated": "/inputs/case_b/candidate-translated.json",
                "simai_time_us": None,
                "simai_end_to_end_csv": None,
            },
        ]
    }
    tail = {
        "cases": [
            {
                "name": "case_b",
                "rust_output": "/runs/case_b.json",
                "translated": "/inputs/case_b/candidate-translated.json",
                "simai_time_us": 22.0,
                "simai_end_to_end_csv": "/tail/case_b/EndToEnd.csv",
            }
        ]
    }

    merged = module.merge_manifest_data([base, tail])

    assert [case["name"] for case in merged["cases"]] == ["case_a", "case_b"]
    assert merged["cases"][0]["simai_time_us"] is None
    assert merged["cases"][1]["simai_time_us"] == 22.0
    assert merged["cases"][1]["simai_end_to_end_csv"] == "/tail/case_b/EndToEnd.csv"


def test_merge_manifests_keeps_existing_simai_when_later_case_is_empty():
    first = {
        "cases": [
            {
                "name": "case_a",
                "rust_output": "/runs/case_a.json",
                "translated": "/inputs/case_a/candidate-translated.json",
                "simai_time_us": 10.0,
                "simai_end_to_end_csv": "/main/case_a/EndToEnd.csv",
            }
        ]
    }
    later_empty = {
        "cases": [
            {
                "name": "case_a",
                "rust_output": "/runs/case_a.json",
                "translated": "/inputs/case_a/candidate-translated.json",
                "simai_time_us": None,
                "simai_end_to_end_csv": None,
            }
        ]
    }

    merged = module.merge_manifest_data([first, later_empty])

    assert merged["cases"][0]["simai_time_us"] == 10.0
    assert merged["cases"][0]["simai_end_to_end_csv"] == "/main/case_a/EndToEnd.csv"
