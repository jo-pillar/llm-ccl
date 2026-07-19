from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from llm_ccl.preparation import prepare_bundle
from llm_ccl.project_loader import discover_projects
from llm_ccl.searching import search_all
from llm_ccl.selection import select_all


class SelectionTest(unittest.TestCase):
    def _write_candidate(self, eval_dir: Path, name: str, time_us: float, config_source: Path) -> None:
        eval_dir.mkdir(parents=True)
        shutil.copy2(config_source, eval_dir / "candidate-config.json")
        sketch_dir = eval_dir / "flow-sim-inputs" / name
        sketch_dir.mkdir(parents=True)
        (sketch_dir / "candidate-sketch.json").write_text("[[0, 0, 0, 0, 1]]\n", encoding="utf-8")
        output = eval_dir / f"{name}-flow-sim.json"
        output.write_text(json.dumps({"time_us": time_us}) + "\n", encoding="utf-8")
        (eval_dir / "flow-sim-manifest.json").write_text(
            json.dumps({"cases": [{"name": name, "rust_output": str(output)}]}) + "\n",
            encoding="utf-8",
        )

    def test_selects_fastest_finite_candidate_from_latest_successful_attempt(self):
        project = discover_projects()["v100_dgx2_clos"]
        case_id = project.cases[0].case_id
        with tempfile.TemporaryDirectory() as tmp:
            bundle = prepare_bundle(project, Path(tmp), "unit", {case_id})

            def create_search_outputs(command, *, env, cwd, log_path, timeout):
                attempt = Path(env["SYCCL_EVAL_ARTIFACT_DIR"]).parent
                root = attempt / "eval_artifacts" / "scheme1_direct_events"
                config = bundle / "cases" / case_id / "config.json"
                self._write_candidate(root / "eval_slow", "candidate-000", 30.0, config)
                self._write_candidate(root / "eval_fast", "candidate-001", 10.0, config)
                self._write_candidate(root / "eval_invalid", "candidate-002", float("inf"), config)
                return 0

            search_all(bundle, model="test-model", runner=create_search_outputs)

            # A faster file outside the manifest-selected attempt must be ignored.
            old_root = (
                bundle
                / "cases"
                / case_id
                / "search"
                / "attempt-0000"
                / "eval_artifacts"
                / "scheme1_direct_events"
            )
            self._write_candidate(
                old_root / "eval_old",
                "candidate-000",
                1.0,
                bundle / "cases" / case_id / "config.json",
            )

            select_all(bundle)

            best = bundle / "cases" / case_id / "best"
            self.assertTrue((best / "candidate-config.json").is_file())
            self.assertTrue((best / "candidate-sketch.json").is_file())
            self.assertEqual(10.0, json.loads((best / "flow-sim.json").read_text())["time_us"])
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            selection = manifest["cases"][case_id]["selection"]
            self.assertEqual("succeeded", selection["status"])
            self.assertEqual("attempt-0001", selection["attempt_id"])
            self.assertEqual("candidate-001", selection["candidate"])
            self.assertEqual(10.0, selection["time_us"])


if __name__ == "__main__":
    unittest.main()
