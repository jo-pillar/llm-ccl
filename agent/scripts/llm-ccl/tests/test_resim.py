from __future__ import annotations

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
from llm_ccl.resim import resim_selected


class ResimTest(unittest.TestCase):
    def test_resim_waits_until_every_case_has_finished_searching(self):
        project = discover_projects()["v100_dgx2_clos"]
        case_ids = {project.cases[0].case_id, project.cases[1].case_id}
        with tempfile.TemporaryDirectory() as tmp:
            bundle = prepare_bundle(project, Path(tmp), "unit", case_ids)
            calls: list[list[str]] = []

            def runner(command, *, cwd, log_path, timeout):
                calls.append(command)
                return 0

            with self.assertRaisesRegex(RuntimeError, "searches are not finished"):
                resim_selected(bundle, synthesize_bin=Path("/tmp/synthesize"), runner=runner)

            self.assertEqual([], calls)

    def test_resims_only_the_selected_best_candidate_after_searches_finish(self):
        project = discover_projects()["v100_dgx2_clos"]
        selected_id = project.cases[0].case_id
        failed_id = project.cases[1].case_id
        with tempfile.TemporaryDirectory() as tmp:
            bundle = prepare_bundle(project, Path(tmp), "unit", {selected_id, failed_id})
            best = bundle / "cases" / selected_id / "best"
            best.mkdir()
            (best / "candidate-config.json").write_text("{}\n", encoding="utf-8")
            (best / "candidate-sketch.json").write_text("[]\n", encoding="utf-8")

            def finish_manifest(payload):
                selected = payload["cases"][selected_id]
                selected["search"].update(
                    {"status": "succeeded", "latest_successful_attempt": "attempt-0001"}
                )
                selected["selection"].update({"status": "succeeded", "candidate": "candidate-007"})
                failed = payload["cases"][failed_id]
                failed["search"]["status"] = "failed"
                failed["selection"] = {"status": "skipped", "reason": "search failed"}

            ManifestStore(bundle / "manifest.json").update(finish_manifest)
            calls: list[list[str]] = []

            def runner(command, *, cwd, log_path, timeout):
                calls.append(command)
                if "simulate-sketch" in command:
                    output = Path(command[command.index("--output") + 1])
                    translated = Path(command[command.index("--dump-translated") + 1])
                    output.write_text('{"time_us": 9.5}\n', encoding="utf-8")
                    translated.write_text("{}\n", encoding="utf-8")
                else:
                    output = Path(command[command.index("-o") + 1])
                    output.write_text('{"Time": 8.0}\n', encoding="utf-8")
                return 0

            resim_selected(
                bundle,
                flow_sim_bin=Path("/tmp/flow-sim-rs"),
                synthesize_bin=Path("/tmp/synthesize"),
                runner=runner,
            )
            resim_selected(
                bundle,
                flow_sim_bin=Path("/tmp/flow-sim-rs"),
                synthesize_bin=Path("/tmp/synthesize"),
                runner=runner,
            )

            self.assertEqual(2, len(calls))
            self.assertEqual("simulate-sketch", calls[0][1])
            self.assertIn("--dump-translated", calls[0])
            self.assertEqual(["/tmp/synthesize", "-f"], calls[1][:2])
            self.assertIn("resim", calls[1])
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("succeeded", manifest["cases"][selected_id]["resim"]["status"])
            self.assertEqual("skipped", manifest["cases"][failed_id]["resim"]["status"])
            self.assertTrue((bundle / "cases" / selected_id / "resim" / "resim.json").is_file())
            self.assertFalse((bundle / "cases" / failed_id / "resim").exists())


if __name__ == "__main__":
    unittest.main()
