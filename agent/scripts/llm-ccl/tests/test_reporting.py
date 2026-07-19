from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from llm_ccl.manifest import ManifestStore
from llm_ccl.preparation import prepare_bundle
from llm_ccl.project_loader import discover_projects
from llm_ccl.reporting import write_report


class ReportingTest(unittest.TestCase):
    def test_report_contains_selected_and_failed_search_cases(self):
        project = discover_projects()["v100_dgx2_clos"]
        selected_id = project.cases[0].case_id
        failed_id = project.cases[1].case_id
        with tempfile.TemporaryDirectory() as tmp:
            bundle = prepare_bundle(project, Path(tmp), "unit", {selected_id, failed_id})

            def finish(payload):
                selected = payload["cases"][selected_id]
                selected["search"]["status"] = "succeeded"
                selected["selection"] = {"status": "succeeded", "candidate": "candidate-003", "time_us": 7.5}
                selected["resim"] = {"status": "succeeded", "output": "cases/a/resim/resim.json"}
                failed = payload["cases"][failed_id]
                failed["search"]["status"] = "failed"
                failed["selection"] = {"status": "skipped", "reason": "search failed"}
                failed["resim"] = {"status": "skipped", "reason": "search failed"}

            ManifestStore(bundle / "manifest.json").update(finish)

            json_path, csv_path = write_report(bundle)

            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(2, report["summary"]["total_cases"])
            self.assertEqual(1, report["summary"]["search_failed"])
            self.assertEqual(1, report["summary"]["resim_succeeded"])
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({selected_id, failed_id}, {row["case_id"] for row in rows})


if __name__ == "__main__":
    unittest.main()
