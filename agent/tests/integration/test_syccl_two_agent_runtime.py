from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

from tests.test_syccl_two_agent import (
    _build_engine,
    _fake_bottleneck_profile,
    _fake_sketch_evolve_block,
    _write_python_topodsl,
)


@unittest.skipUnless(
    os.environ.get("RUN_SYCCL_INTEGRATION") == "1",
    "set RUN_SYCCL_INTEGRATION=1 to run SyCCL integration tests",
)
class SycclTwoAgentRuntimeIntegrationTest(unittest.TestCase):
    def test_experiment_records_have_required_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="syccl_two_agent_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=1,
                completions=[_fake_sketch_evolve_block()],
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )

            asyncio.run(engine.run())

            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            round_record = json.loads((artifacts_dir / "rounds.jsonl").read_text().splitlines()[0])
            llm_records = [json.loads(line) for line in (artifacts_dir / "llm_calls.jsonl").read_text().splitlines()]
            proposal_record = next(record for record in llm_records if record["agent_type"] == "proposal")
            record_agent_record = next(record for record in llm_records if record["agent_type"] == "record")

            required_round_fields = {
                "topodsl_config_id",
                "collective",
                "message_size",
                "round_id",
                "prompt_hash",
                "input_tokens",
                "output_tokens",
                "combined_score",
                "candidate_sketch",
                "validity_status",
                "expanded_events",
                "completion_time",
                "algorithm_bandwidth",
                "bottleneck_profile",
                "best_so_far",
            }
            required_llm_fields = {
                "run_id",
                "agent_type",
                "round_id",
                "model_name",
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "reasoning_tokens",
                "wall_clock_time_ms",
                "api_cost_usd",
                "prompt_hash",
                "completion_hash",
            }
            self.assertLessEqual(required_round_fields, set(round_record.keys()))
            self.assertLessEqual(required_llm_fields, set(proposal_record.keys()))
            self.assertLessEqual(required_llm_fields, set(record_agent_record.keys()))
            self.assertEqual(round_record["validity_status"], "ok")
            self.assertEqual(round_record["combined_score"], -10.0)
            self.assertEqual(round_record["expanded_events"], 12)
            self.assertEqual(round_record["bottleneck_profile"]["status"], "ok")
            self.assertEqual(
                round_record["bottleneck_profile"]["top_transmission_bottlenecks"][0]["step"],
                1,
            )
            self.assertNotIn("critical_path", round_record)
            self.assertNotIn("bottleneck_links", round_record)
            self.assertEqual(proposal_record["agent_type"], "proposal")
            self.assertEqual(record_agent_record["agent_type"], "record")
