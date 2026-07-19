from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from llm_ccl.preparation import prepare_bundle
from llm_ccl.project_loader import discover_projects
from llm_ccl.searching import search_all


class SearchingTest(unittest.TestCase):
    def _bundle(self, root: Path) -> tuple[Path, str]:
        project = discover_projects()["v100_dgx2_clos"]
        case_id = project.cases[0].case_id
        return prepare_bundle(project, root, "unit", {case_id}), case_id

    def test_successful_case_uses_only_llm_elite_all_and_is_not_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, case_id = self._bundle(Path(tmp))
            commands: list[list[str]] = []

            def succeed(command, *, env, cwd, log_path, timeout):
                commands.append(command)
                self.assertEqual(str(bundle / "cases" / case_id / "config.json"), env["SYCCL_BASE_CONFIG"])
                self.assertEqual(
                    str(bundle / "cases" / case_id / "search" / "attempt-0001" / "eval_artifacts"),
                    env["SYCCL_EVAL_ARTIFACT_DIR"],
                )
                return 0

            search_all(bundle, model="test-model", runner=succeed)
            search_all(bundle, model="test-model", runner=succeed)

            self.assertEqual(1, len(commands))
            command = commands[0]
            self.assertEqual("llm_elite", command[command.index("--selector") + 1])
            self.assertEqual("all", command[command.index("--elite-selection-strategy") + 1])
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            search = manifest["cases"][case_id]["search"]
            self.assertEqual("succeeded", search["status"])
            self.assertEqual("attempt-0001", search["latest_successful_attempt"])
            self.assertEqual(1, len(search["attempts"]))

    def test_failed_case_gets_a_new_attempt_on_the_next_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, case_id = self._bundle(Path(tmp))
            exit_codes = iter((17, 0))

            def run(command, *, env, cwd, log_path, timeout):
                return next(exit_codes)

            search_all(bundle, model="test-model", runner=run)
            search_all(bundle, model="test-model", runner=run)

            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            search = manifest["cases"][case_id]["search"]
            self.assertEqual(["failed", "succeeded"], [item["status"] for item in search["attempts"]])
            self.assertEqual(["attempt-0001", "attempt-0002"], [item["id"] for item in search["attempts"]])
            self.assertEqual("attempt-0002", search["latest_successful_attempt"])


if __name__ == "__main__":
    unittest.main()
