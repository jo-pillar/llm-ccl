from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from llm_ccl.models import CaseSpec, LayerShape, ScaleSpec, case_matrix
from llm_ccl.preparation import prepare_bundle
from llm_ccl.project_loader import discover_projects


class ModelAndProjectTest(unittest.TestCase):
    def test_case_matrix_converts_total_bytes_once(self):
        scale = ScaleSpec("4hosts-64gpu", 64, (LayerShape(1, 4, 16),))

        cases = case_matrix((scale,), ("allgather",), (65536,))

        self.assertEqual("4hosts-64gpu-allgather-65536B", cases[0].case_id)
        self.assertEqual(1024, cases[0].coll_byte)

    def test_case_rejects_non_divisible_total_size(self):
        scale = ScaleSpec("8gpu", 8, (LayerShape(0, 1, 8),))

        with self.assertRaisesRegex(ValueError, "divisible"):
            CaseSpec("bad", scale, "allgather", 10)

    def test_initial_projects_are_discovered_without_registry(self):
        projects = discover_projects()

        self.assertEqual({"h800_multirail", "v100_dgx2_clos"}, set(projects))
        self.assertEqual(35, len(projects["h800_multirail"].cases))
        self.assertEqual(24, len(projects["v100_dgx2_clos"].cases))


class PreparationTest(unittest.TestCase):
    def test_prepares_one_v100_case_from_topology_values(self):
        project = discover_projects()["v100_dgx2_clos"]
        case_id = "4hosts-64gpu-allgather-65536B"

        with tempfile.TemporaryDirectory() as tmp:
            bundle = prepare_bundle(
                project,
                bundle_root=Path(tmp),
                launch_id="unit",
                case_ids={case_id},
            )
            case_dir = bundle / "cases" / case_id
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            config = json.loads((case_dir / "config.json").read_text(encoding="utf-8"))

            self.assertEqual(1, manifest["schema_version"])
            self.assertEqual([case_id], list(manifest["cases"]))
            self.assertEqual(65536, manifest["cases"][case_id]["total_message_size"])
            self.assertEqual(1024, manifest["cases"][case_id]["coll_byte"])
            self.assertEqual("pending", manifest["cases"][case_id]["search"]["status"])
            self.assertEqual(1024, config["coll"]["byte"])
            self.assertEqual({"bw_mbpus": 0.15, "lat_us": 3.0}, config["link_spec"]["nvlink"])
            self.assertEqual({"bw_mbpus": 0.0125, "lat_us": 3.0}, config["link_spec"]["netlink_leaf"])
            self.assertEqual({"bw_mbpus": 0.1, "lat_us": 0.5}, config["link_spec"]["netlink_spine"])
            self.assertIn("group_num=4, node_num=16", (case_dir / "topodsl.py").read_text(encoding="utf-8"))
            self.assertIn("GPU_NUM = 64", (case_dir / "init_program.py").read_text(encoding="utf-8"))

    def test_existing_launch_is_never_overwritten(self):
        project = discover_projects()["v100_dgx2_clos"]

        with tempfile.TemporaryDirectory() as tmp:
            prepare_bundle(project, Path(tmp), "same", {project.cases[0].case_id})
            with self.assertRaises(FileExistsError):
                prepare_bundle(project, Path(tmp), "same", {project.cases[0].case_id})


if __name__ == "__main__":
    unittest.main()
