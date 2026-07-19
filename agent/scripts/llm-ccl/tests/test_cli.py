from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from llm_ccl.project_loader import discover_projects


class CliTest(unittest.TestCase):
    def test_prepare_command_creates_a_bundle(self):
        case_id = discover_projects()["v100_dgx2_clos"].cases[0].case_id
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_ROOT / "run.py"),
                    "prepare",
                    "v100_dgx2_clos",
                    "--bundle-root",
                    tmp,
                    "--launch-id",
                    "unit",
                    "--case",
                    case_id,
                ],
                cwd=SCRIPT_ROOT.parents[1],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertTrue((Path(tmp) / "v100_dgx2_clos" / "unit" / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
