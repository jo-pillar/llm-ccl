from __future__ import annotations

import asyncio
import os
import shutil
import json
import importlib.util
import math
import runpy
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from simpletes.config import EngineConfig
from simpletes.engine import SimpleTESEngine
from simpletes.engine.syccl_two_agent import SycclTwoAgentRuntime
from syccl_agents.flow_sim import FlowSimRunner
from syccl_agents.llm import FakeLLMBackend
from syccl_agents.record_agent import RecordAgent
from syccl_agents.sketch_dsl import parse_sketch_dsl
from syccl_agents.topodsl import load_topodsl


ROOT = Path(__file__).resolve().parents[2]
RUNNER_SCRIPT = ROOT / "agent" / "scripts" / "run_syccl_two_agent.py"


def _load_runner_script():
    spec = importlib.util.spec_from_file_location("run_syccl_two_agent", RUNNER_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_python_topodsl(path: Path) -> None:
    path.write_text(
        """
def topology():
    return {
        "family": "clos",
        "hosts": 2,
        "gpus_per_host": 2,
        "nics_per_host": 1,
        "leaf_switches": 1,
        "spine_switches": 1,
        "message_size": 4096,
        "collective": "allgather",
    }
""".lstrip(),
        encoding="utf-8",
    )


def _write_derived_class_topodsl(path: Path) -> None:
    path.write_text(
        '''
class ClosTopology(BaseTopology):
    def __init__(self, gpu_num, host_num, leaf_num, spine_num,
                 gpu_host_bw, gpu_host_lat,
                 host_leaf_bw, host_leaf_lat,
                 leaf_spine_bw, leaf_spine_lat,
                 coll_bytes, collective):
        self.gpu_num = gpu_num
        self.host_num = host_num
        self.leaf_num = leaf_num
        self.spine_num = spine_num
        self.coll_bytes = coll_bytes
        self.collective = collective
        self.link_spec_0 = LinkSpec(bandwidth=gpu_host_bw, latency=gpu_host_lat)
        self.link_spec_1 = LinkSpec(bandwidth=host_leaf_bw, latency=host_leaf_lat)
        self.link_spec_2 = LinkSpec(bandwidth=leaf_spine_bw, latency=leaf_spine_lat)

    def build_topology(self):
        self.f_gpu_host()
        self.f_host_leaf()
        self.f_leaf_spine()

    def f_gpu_host(self):
        gpu_per_host = self.gpu_num // self.host_num
        for host_id in range(self.host_num):
            for gpu_i in range(gpu_per_host):
                for gpu_j in range(gpu_i + 1, gpu_per_host):
                    self.connect(f"gpu[{host_id}][{gpu_i}]", f"gpu[{host_id}][{gpu_j}]", self.link_spec_0)

    def f_host_leaf(self):
        host_per_leaf = self.host_num // self.leaf_num
        for leaf_id in range(self.leaf_num):
            for host_id in range(leaf_id * host_per_leaf, (leaf_id + 1) * host_per_leaf):
                self.connect(f"host[{host_id}]", f"leaf[{leaf_id}]", self.link_spec_1)

    def f_leaf_spine(self):
        for leaf_id in range(self.leaf_num):
            for spine_id in range(self.spine_num):
                self.connect(f"leaf[{leaf_id}]", f"spine[{spine_id}]", self.link_spec_2)

target_topo = ClosTopology(32, 4, 2, 1, "32.5GBps", "9us", "2.8125GBps", "25us", "45GBps", "25us", "1MB", "allgather")
'''.lstrip(),
        encoding="utf-8",
    )


def _fake_sketch() -> list[dict[str, object]]:
    return [
        {"step": 0, "layer": 1, "group": 0, "srcs": [0], "dsts": [1]},
        {"step": 1, "layer": 3, "group": 0, "srcs": [0], "dsts": [2]},
        {"step": 2, "layer": 1, "group": 1, "srcs": [2], "dsts": [3]},
    ]


def _fake_bottleneck_profile() -> dict[str, object]:
    return {
        "status": "ok",
        "diagnosis": "transmission 1 step 1 layer 3 group 0 has the largest aggregate delay",
        "critical_transmission": {
            "transmission_index": 1,
            "step": 1,
            "layer": 3,
            "group": 0,
            "srcs": [0],
            "dsts": [2],
            "delay_ns": 120,
            "reason": "transmission 1 is on the critical receive chain",
        },
        "critical_chain": [
            {
                "transmission_index": 0,
                "step": 0,
                "layer": 1,
                "group": 0,
                "srcs": [0],
                "dsts": [1],
                "dependency": "recv",
                "finish_ns": 40,
            },
            {
                "transmission_index": 1,
                "step": 1,
                "layer": 3,
                "group": 0,
                "srcs": [0],
                "dsts": [2],
                "dependency": "critical",
                "finish_ns": 120,
            },
        ],
        "top_transmission_bottlenecks": [
            {
                "transmission_index": 1,
                "step": 1,
                "layer": 3,
                "group": 0,
                "srcs": [0],
                "dsts": [2],
                "reason": "transmission 1 has the largest aggregate delay",
                "delay_ns": 120,
                "affected_rotated_flows": 4,
            }
        ],
        "stage_pressure": [
            {
                "step": 1,
                "layer": 3,
                "group": 0,
                "delay_ns": 120,
                "affected_rotated_flows": 4,
            }
        ],
    }


def _fake_sketch_evolve_block() -> str:
    return """
```python
# EVOLVE-BLOCK-START
def tx(step, layer, group, srcs, dsts):
    return (step, layer, group, srcs, dsts)


def construct_sketches():
    return [
        tx(0, 1, 0, 0, [1]),
        tx(1, 3, 0, 0, [2]),
        tx(2, 1, 1, 2, [3]),
    ]
# EVOLVE-BLOCK-END
```
""".strip()


def _record_completion(round_id: int) -> str:
    summary = "\n".join(
        [
            f"Trail id: {round_id}",
            f"- transmissions strategy: async record for round {round_id}",
            "- compact dsl: [(0, 1, 0, 0, [1])]",
            f"- profile data: time_us={10.0 + round_id}",
        ]
    )
    return json.dumps({"summary": summary, "direction_hint": f"avoid bottleneck from round {round_id}"})


class DelayedRecordLLMBackend:
    def __init__(self, *, record_delay_s: float | dict[int, float] = 0.0, proposal_delay_s: float = 0.0) -> None:
        self.model_name = "delayed-record-llm"
        self.prompts: list[str] = []
        self.calls: list[tuple[str, int, str]] = []
        self.record_delay_s = record_delay_s
        self.proposal_delay_s = proposal_delay_s
        self.record_started_at: dict[int, float] = {}
        self.record_finished_at: dict[int, float] = {}
        self.max_concurrent_records = 0
        self._active_records = 0
        self._lock = threading.Lock()

    def generate(self, prompt: str, *, agent_type: str, round_id: int):
        with self._lock:
            self.prompts.append(prompt)
            self.calls.append((agent_type, round_id, prompt))
        if agent_type == "proposal":
            time.sleep(self.proposal_delay_s)
            return _fake_llm_response(_fake_sketch_evolve_block(), self.model_name, prompt)
        if agent_type == "record":
            delay = (
                self.record_delay_s.get(round_id, 0.0)
                if isinstance(self.record_delay_s, dict)
                else self.record_delay_s
            )
            with self._lock:
                self._active_records += 1
                self.max_concurrent_records = max(self.max_concurrent_records, self._active_records)
                self.record_started_at[round_id] = time.perf_counter()
            time.sleep(delay)
            with self._lock:
                self.record_finished_at[round_id] = time.perf_counter()
                self._active_records -= 1
            return _fake_llm_response(_record_completion(round_id), self.model_name, prompt)
        raise AssertionError(f"unexpected agent_type={agent_type!r}")


class AsyncDelayedRecordLLMBackend:
    def __init__(self, *, record_delay_s: float | dict[int, float] = 0.0, proposal_delay_s: float = 0.0) -> None:
        self.model_name = "async-delayed-record-llm"
        self.prompts: list[str] = []
        self.calls: list[tuple[str, int, str]] = []
        self.record_delay_s = record_delay_s
        self.proposal_delay_s = proposal_delay_s
        self.record_started_at: dict[int, float] = {}
        self.record_finished_at: dict[int, float] = {}
        self.max_concurrent_records = 0
        self._active_records = 0

    async def generate(self, prompt: str, *, agent_type: str, round_id: int):
        self.prompts.append(prompt)
        self.calls.append((agent_type, round_id, prompt))
        if agent_type == "proposal":
            await asyncio.sleep(self.proposal_delay_s)
            return _fake_llm_response(_fake_sketch_evolve_block(), self.model_name, prompt)
        if agent_type == "record":
            delay = (
                self.record_delay_s.get(round_id, 0.0)
                if isinstance(self.record_delay_s, dict)
                else self.record_delay_s
            )
            self._active_records += 1
            self.max_concurrent_records = max(self.max_concurrent_records, self._active_records)
            self.record_started_at[round_id] = time.perf_counter()
            await asyncio.sleep(delay)
            self.record_finished_at[round_id] = time.perf_counter()
            self._active_records -= 1
            return _fake_llm_response(_record_completion(round_id), self.model_name, prompt)
        raise AssertionError(f"unexpected agent_type={agent_type!r}")


class AsyncFailingRecordLLMBackend(AsyncDelayedRecordLLMBackend):
    def __init__(self, *, failing_rounds: set[int]) -> None:
        super().__init__(record_delay_s=0.0, proposal_delay_s=0.01)
        self.failing_rounds = set(failing_rounds)

    async def generate(self, prompt: str, *, agent_type: str, round_id: int):
        if agent_type == "record" and round_id in self.failing_rounds:
            self.prompts.append(prompt)
            self.calls.append((agent_type, round_id, prompt))
            self.record_started_at[round_id] = time.perf_counter()
            self.record_finished_at[round_id] = time.perf_counter()
            raise RuntimeError(f"record round {round_id} failed")
        return await super().generate(prompt, agent_type=agent_type, round_id=round_id)


def _fake_llm_response(text: str, model_name: str, prompt: str):
    from syccl_agents.llm import LLMResponse

    return LLMResponse(
        text=text,
        model_name=model_name,
        input_tokens=len(prompt.split()),
        output_tokens=len(text.split()),
        cached_input_tokens=None,
        reasoning_tokens=None,
        wall_clock_time_ms=0,
        api_cost_usd=None,
        raw_output=text,
    )


def _write_minimal_simpletes_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    init_program = tmp_path / "init_program.py"
    init_program.write_text(
        """
# EVOLVE-BLOCK-START
def construct_sketches():
    return []
# EVOLVE-BLOCK-END


def run_code():
    return construct_sketches()
""".lstrip(),
        encoding="utf-8",
    )
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        """
def evaluate(program_path):
    return {"combined_score": -1000000000000.0}
""".lstrip(),
        encoding="utf-8",
    )
    instruction = tmp_path / "instruction.txt"
    instruction.write_text("SyCCL two-agent smoke instruction", encoding="utf-8")
    return init_program, evaluator, instruction


def _read_latest_checkpoint_nodes(engine: SimpleTESEngine) -> list[dict[str, object]]:
    checkpoint_dirs = sorted(Path(engine.checkpoint_dir).glob("db_state_*"))
    if not checkpoint_dirs:
        raise AssertionError("expected at least one SimpleTES checkpoint")
    return json.loads((checkpoint_dirs[-1] / "nodes.json").read_text())


def _build_engine(
    tmp_path: Path,
    *,
    topo_path: Path,
    rounds: int,
    completions: list[str],
    flow_sim_fn,
    save_llm_io: bool = False,
    collective: str | None = None,
    message_size: int | None = None,
) -> tuple[SimpleTESEngine, FakeLLMBackend]:
    init_program, evaluator, instruction = _write_minimal_simpletes_files(tmp_path)
    config = EngineConfig(
        init_program=str(init_program),
        evaluator_path=str(evaluator),
        instruction_path=str(instruction),
        max_generations=rounds,
        init_eval_repeats=1,
        output_path=str(tmp_path / "checkpoints"),
        save_llm_io=save_llm_io,
        log_interval=1,
        db_show_interval=0,
        num_inspirations=0,
        num_chains=1,
        k_candidates=1,
        gen_concurrency=1,
        eval_concurrency=1,
    )
    llm_completions = list(completions)
    if len(llm_completions) == rounds:
        expanded: list[str] = []
        for idx, completion in enumerate(llm_completions):
            expanded.append(completion)
            expanded.append(_record_completion(idx))
        llm_completions = expanded
    llm = FakeLLMBackend(llm_completions)
    runtime = SycclTwoAgentRuntime(
        topo_path=topo_path,
        collective=collective,
        message_size=message_size,
        flow_sim_bin="flow-sim-rs",
        llm=llm,
        flow_sim_fn=flow_sim_fn,
    )
    return SimpleTESEngine(config, runtime=runtime), llm


def _run_engine_without_slow_finalization(engine: SimpleTESEngine) -> None:
    write_checkpoint = AsyncMock()
    finalize_run = AsyncMock()
    with (
        patch.dict(os.environ, {"MPLCONFIGDIR": str(Path(engine.checkpoint_dir) / "matplotlib")}),
        patch.object(SimpleTESEngine, "_write_checkpoint", new=write_checkpoint),
        patch.object(SimpleTESEngine, "_finalize_run", new=finalize_run),
    ):
        asyncio.run(engine.run())


def _build_engine_with_log_interval(
    tmp_path: Path,
    *,
    topo_path: Path,
    rounds: int,
    completions: list[str],
    flow_sim_fn,
    log_interval: int,
) -> tuple[SimpleTESEngine, FakeLLMBackend]:
    init_program, evaluator, instruction = _write_minimal_simpletes_files(tmp_path)
    config = EngineConfig(
        init_program=str(init_program),
        evaluator_path=str(evaluator),
        instruction_path=str(instruction),
        max_generations=rounds,
        init_eval_repeats=1,
        output_path=str(tmp_path / "checkpoints"),
        log_interval=log_interval,
        db_show_interval=0,
        num_inspirations=0,
        num_chains=1,
        k_candidates=1,
        gen_concurrency=1,
        eval_concurrency=1,
    )
    llm = FakeLLMBackend(completions)
    runtime = SycclTwoAgentRuntime(
        topo_path=topo_path,
        flow_sim_bin="flow-sim-rs",
        llm=llm,
        flow_sim_fn=flow_sim_fn,
    )
    return SimpleTESEngine(config, runtime=runtime), llm


class SycclTwoAgentTest(unittest.TestCase):
    def test_syccl_two_agent_prompts_are_loaded_from_agent_prompt_dir(self) -> None:
        from syccl_agents import prompts as prompt_assets

        prompt_root = ROOT / "agent" / "prompt" / "syccl_two_agent"
        self.assertEqual(prompt_assets.PROMPT_ROOT, prompt_root)
        self.assertIn("$Collective", prompt_assets.load_prompt("proposal_base.txt"))
        self.assertIn("{message_size}", prompt_assets.load_prompt("proposal_dynamic_context.txt"))
        self.assertIn("{payload_json}", prompt_assets.load_prompt("record_agent.txt"))
        self.assertIn("def run_code()", prompt_assets.load_init_program())

    def test_init_program_seed_is_double_ring_broadcast(self) -> None:
        from syccl_agents.prompts import load_init_program
        from syccl_agents.sketch_dsl import parse_sketch_dsl

        prompt_root = ROOT / "agent" / "prompt" / "syccl_two_agent"
        namespace = runpy.run_path(str(prompt_root / "init_program.py"))

        sketch = namespace["construct_sketches"](ngpus=32, root_gpu=0, gpus_per_host=8, hosts_per_leaf=2)

        self.assertEqual(len(sketch), 31)
        self.assertEqual(sketch[0], (0, 1, 0, 0, 1))
        self.assertEqual(sketch[1], (0, 4, 0, 0, 31))
        self.assertEqual(sketch[2], (1, 1, 0, 1, 2))
        self.assertEqual(sketch[3], (1, 1, 3, 31, 30))
        self.assertEqual(sketch[14], (7, 3, 0, 7, 8))
        self.assertEqual(sketch[15], (7, 1, 3, 25, 24))
        self.assertEqual(sketch[-1], (15, 4, 0, 15, 16))
        received = [dst for _, _, _, _, dsts in sketch for dst in (dsts if isinstance(dsts, list) else [dsts])]
        self.assertEqual(sorted(received), list(range(1, 32)))
        self.assertEqual(len(received), len(set(received)))

        parsed = parse_sketch_dsl(repr(sketch))
        self.assertEqual(len(parsed), len(sketch))
        self.assertEqual(parsed[0], {"step": 0, "layer": 1, "group": 0, "srcs": [0], "dsts": [1]})
        self.assertEqual(parsed[1], {"step": 0, "layer": 4, "group": 0, "srcs": [0], "dsts": [31]})
        self.assertEqual(parsed[-1], {"step": 15, "layer": 4, "group": 0, "srcs": [15], "dsts": [16]})

        default_parsed = parse_sketch_dsl(load_init_program())
        default_ngpus = max(gpu for tx in default_parsed for gpu in tx["dsts"]) + 1
        self.assertEqual(default_parsed[0], {"step": 0, "layer": 1, "group": 0, "srcs": [0], "dsts": [1]})
        self.assertEqual(default_parsed[1], {"step": 0, "layer": 4, "group": 0, "srcs": [0], "dsts": [default_ngpus - 1]})

    def test_init_program_double_ring_runs_real_flow_sim_when_available(self) -> None:
        from syccl_agents.config_render import write_syccl_config
        from syccl_agents.prompts import load_init_program
        from syccl_agents.sketch_dsl import parse_sketch_dsl, write_compact_sketch
        from syccl_agents.topodsl import TopologyParams

        configured_binary = os.environ.get("SYCCL_FLOW_SIM_BIN")
        flow_sim_bin = configured_binary or shutil.which("flow-sim-rs")
        if flow_sim_bin is None:
            self.skipTest("flow-sim-rs is not available; set SYCCL_FLOW_SIM_BIN to run this integration test")
        if configured_binary is not None and not Path(configured_binary).exists():
            self.fail(f"SYCCL_FLOW_SIM_BIN does not exist: {configured_binary}")
        ngpus=2048
        with self._tmpdir() as tmp_path:
            params = TopologyParams(
                family="clos",
                hosts=ngpus//8,
                gpus_per_host=8,
                nics_per_host=1,
                leaf_switches=2,
                spine_switches=1,
                message_size=1024*1024* 1024,
                collective="allgather",
                host_bw_mbpus=32.5 / 1000.0,
                host_lat_us=9.0,
                net_bw_mbpus=2.8125 / 1000.0,
                net_lat_us=25.0,
                spine_bw_mbpus=45.0 / 1000.0,
                spine_lat_us=25.0,
            )
            prompt_root = ROOT / "agent" / "prompt" / "syccl_two_agent"
            namespace = runpy.run_path(str(prompt_root / "init_program.py"))
            raw_sketch = namespace["construct_sketches"](ngpus=ngpus, root_gpu=0, gpus_per_host=8, hosts_per_leaf= ngpus//(8*2))
            sketch = parse_sketch_dsl(repr(raw_sketch))
            config_path = write_syccl_config(params, tmp_path / "flow-sim-config.json")
            sketch_path = write_compact_sketch(sketch, tmp_path / "double-ring-sketch.json")

            result = FlowSimRunner(flow_sim_bin, timeout_s=600).simulate_sketch(
                config_path=config_path,
                sketch_path=sketch_path,
                output_path=tmp_path / "double-ring-flow-sim.json",
            )
        # output_file = Path(os.curdir) / "test-double-ring-flow-sim.json"
        # with output_file.open("w", encoding="utf-8") as fp:
        #     json.dump(result, fp, indent=2)
        self.assertGreater(float(result["time_us"]), 0.0)
        self.assertGreater(int(result.get("flow_count", 0)), 0)
        self.assertTrue(math.isfinite(float(result["time_us"])))
        
    def test_one_round_full_runtime_runs_real_flow_sim_when_available(self) -> None:
        configured_binary = os.environ.get("SYCCL_FLOW_SIM_BIN")
        flow_sim_bin = configured_binary or shutil.which("flow-sim-rs")
        if flow_sim_bin is None:
            self.skipTest("flow-sim-rs is not available; set SYCCL_FLOW_SIM_BIN to run this integration test")
        if configured_binary is not None and not Path(configured_binary).exists():
            self.fail(f"SYCCL_FLOW_SIM_BIN does not exist: {configured_binary}")

        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_32gpu.py"
            _write_derived_class_topodsl(topo_path)
            runner_script = _load_runner_script()
            init_program, evaluator, instruction = runner_script._write_task_files(tmp_path / "simpletes_task")
            candidate = """
```python
# EVOLVE-BLOCK-START
def tx(step, layer, group, srcs, dsts):
    return (step, layer, group, srcs, dsts)


def construct_sketches():
    sketch = []
    ngpus = 32
    root_gpu = 0
    gpus_per_host = 8
    hosts_per_leaf = 2

    def ring_gpu(offset):
        return (root_gpu + offset) % ngpus

    def host_id(gpu):
        return gpu // gpus_per_host

    def leaf_id(gpu):
        return host_id(gpu) // hosts_per_leaf

    def nearest_layer_group(src, dst):
        if host_id(src) == host_id(dst):
            return 1, host_id(src)
        if leaf_id(src) == leaf_id(dst):
            return 3, leaf_id(src)
        return 4, 0

    left_front = root_gpu
    right_front = root_gpu
    remaining = ngpus - 1
    step = 0
    while remaining > 0:
        if remaining >= 1:
            left_dst = ring_gpu(step + 1)
            layer, group = nearest_layer_group(left_front, left_dst)
            sketch.append(tx(step, layer, group, left_front, left_dst))
            left_front = left_dst
            remaining -= 1
        if remaining >= 1:
            right_dst = ring_gpu(-(step + 1))
            layer, group = nearest_layer_group(right_front, right_dst)
            sketch.append(tx(step, layer, group, right_front, right_dst))
            right_front = right_dst
            remaining -= 1
        step += 1
    return sketch
# EVOLVE-BLOCK-END
```
""".strip()
            record_completion = "\n".join(
                [
                    "Trail id: 0",
                    "- transmissions strategy: double ring with nearest legal layer per edge",
                    "- compact dsl: double-ring-32gpu",
                    "- profile data: real flow-sim integration run",
                ]
            )
            llm = FakeLLMBackend([candidate, json.dumps({"summary": record_completion, "direction_hint": ""})])
            config = EngineConfig(
                init_program=str(init_program),
                evaluator_path=str(evaluator),
                instruction_path=str(instruction),
                max_generations=1,
                init_eval_repeats=1,
                output_path=str(tmp_path / "checkpoints"),
                save_llm_io=True,
                log_interval=0,
                db_show_interval=0,
                num_inspirations=0,
                num_chains=1,
                k_candidates=1,
                gen_concurrency=1,
                eval_concurrency=1,
                eval_timeout=120,
            )
            runtime = SycclTwoAgentRuntime(
                topo_path=topo_path,
                flow_sim_bin=flow_sim_bin,
                llm=llm,
            )
            engine = SimpleTESEngine(config, runtime=runtime)

            import asyncio

            write_checkpoint = AsyncMock()
            finalize_run = AsyncMock()
            with (
                patch.dict(os.environ, {"MPLCONFIGDIR": str(tmp_path / "matplotlib")}),
                patch.object(SimpleTESEngine, "_write_checkpoint", new=write_checkpoint),
                patch.object(SimpleTESEngine, "_finalize_run", new=finalize_run),
            ):
                asyncio.run(engine.run())
            write_checkpoint.assert_not_awaited()
            finalize_run.assert_awaited_once()

            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            summary = json.loads((artifacts_dir / "summary.json").read_text())
            round_record = json.loads((artifacts_dir / "rounds.jsonl").read_text().splitlines()[0])
            llm_records = [json.loads(line) for line in (artifacts_dir / "llm_calls.jsonl").read_text().splitlines()]

            self.assertEqual(summary["rounds_completed"], 1)
            self.assertEqual(round_record["validity_status"], "ok")
            self.assertGreater(float(round_record["completion_time"]), 0.0)
            self.assertGreater(int(round_record["expanded_events"]), 0)
            self.assertEqual(round_record["candidate_code_path"], "rounds/round-0001-candidate.py")
            self.assertTrue((artifacts_dir / "rounds" / "round-0001-candidate.py").exists())
            self.assertTrue((artifacts_dir / "rounds" / "round-0001-sketch.json").exists())
            self.assertTrue((artifacts_dir / "rounds" / "round-0001-flow-sim.json").exists())
            self.assertEqual([record["agent_type"] for record in llm_records], ["proposal", "record"])
            self.assertIn(record_completion, engine.runtime.record_agent.summary)

    def test_central_prompt_renderer_builds_proposal_and_record_prompts(self) -> None:
        from syccl_agents.prompts import render_proposal_prompt, render_record_prompt

        proposal = render_proposal_prompt(
            topo_source="def topology():\n    return {}",
            collective="allgather",
            gpu_num=4,
            message_size=4096,
            topo_summary="family=clos, total_gpus=4",
            layer_summary="layer 1 group 0: GPUs 0-1",
            seed_hint="[(0, 1, 0, 0, [1])]",
            enumeration_note="Sketch enumeration: skipped",
            record_summary="round 1: best",
            direction_hint="try a different relay",
            init_program_reference=(
                "--- Inspiration 1 ---\n"
                "Score: -1.000000\n"
                "Metrics:\n"
                "  combined_score: -1.000000\n"
                "Code:\n"
                "```python\n"
                "# EVOLVE-BLOCK-START\n"
                "def construct_sketches():\n"
                "    return []\n"
                "# EVOLVE-BLOCK-END\n"
                "```"
            ),
        )
        self.assertIn("The target collective is allgather", proposal)
        self.assertIn("rotated across all 4 roots", proposal)
        self.assertIn("def topology():", proposal)
        self.assertIn("--- Inspiration 1 ---", proposal)
        self.assertIn("def construct_sketches():", proposal)
        self.assertIn("- message_size_bytes: 4096", proposal)
        self.assertIn("round 1: best", proposal)
        self.assertNotIn("$Collective", proposal)
        self.assertNotIn("$GPU_NUM", proposal)

        record = render_record_prompt(
            round_id=3,
            candidate=[{"step": 0, "layer": 1, "group": 0, "srcs": [0], "dsts": [1]}],
            metrics={
                "validity_status": "ok",
                "combined_score": -10.0,
                "completion_time": 10.0,
                "expanded_events": 12,
                "bottleneck_profile": _fake_bottleneck_profile(),
                "best_so_far": True,
                "proposal_output": "raw proposal",
                "code_extract_reason": "evolve_block_merged",
                "candidate_code_path": "rounds/round-0003-candidate.py",
            },
            current_summary="round 2: bottleneck=0->1",
            direction_hint="avoid 0->1",
        )
        self.assertIn("Return exactly one record for this attempt.", record)
        self.assertIn("Trail id: 2", record)
        self.assertIn("Proposal LLM output:\nraw proposal", record)
        self.assertIn("bottleneck_profile", record)
        self.assertIn("top_transmission_bottlenecks", record)
        self.assertNotIn("critical path", record.lower())
        self.assertIn('"current_direction_hint": "avoid 0->1"', record)
        print(f"Proposal: {proposal}")
        print(f"Record: {record}")

    def test_record_prompt_rejects_missing_required_metrics(self) -> None:
        from syccl_agents.prompts import render_record_prompt

        with self.assertRaisesRegex(ValueError, "bottleneck_profile"):
            render_record_prompt(
                round_id=1,
                candidate=_fake_sketch(),
                metrics={
                    "validity_status": "ok",
                    "combined_score": -10.0,
                    "completion_time": 10.0,
                    "expanded_events": 12,
                    "best_so_far": True,
                },
                current_summary="",
                direction_hint="",
            )

    def test_round_record_validation_rejects_null_required_fields(self) -> None:
        from syccl_agents.records import validate_round_record

        record = {
            "topodsl_config_id": "topo",
            "collective": "allgather",
            "message_size": 4096,
            "round_id": 1,
            "prompt_hash": "a" * 64,
            "input_tokens": 1,
            "output_tokens": 1,
            "candidate_sketch": _fake_sketch(),
            "validity_status": "ok",
            "expanded_events": 12,
            "combined_score": -10.0,
            "completion_time": 10.0,
            "algorithm_bandwidth": 409.6,
            "bottleneck_profile": None,
            "best_so_far": True,
        }

        with self.assertRaisesRegex(ValueError, "bottleneck_profile"):
            validate_round_record(record)

    def test_cli_task_files_are_loaded_from_syccl_prompt_assets(self) -> None:
        with self._tmpdir() as tmp_path:
            runner_script = _load_runner_script()
            init_program, evaluator, instruction = runner_script._write_task_files(tmp_path / "task")

            prompt_dir = ROOT / "agent" / "prompt" / "syccl_two_agent"
            self.assertEqual(
                init_program.read_text(encoding="utf-8"),
                (prompt_dir / "init_program.py").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                instruction.read_text(encoding="utf-8"),
                (prompt_dir / "proposal_base.txt").read_text(encoding="utf-8"),
            )
            self.assertIn("combined_score", evaluator.read_text(encoding="utf-8"))

    def test_topodsl_file_entry_preserves_python_prompt_source_and_params(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)

            spec = load_topodsl(topo_path)

            self.assertTrue(spec.prompt_source.startswith("def topology():"))
            self.assertFalse(spec.prompt_source.lstrip().startswith("{"))
            self.assertEqual(spec.params.family, "clos")
            self.assertEqual(spec.params.hosts, 2)
            self.assertEqual(spec.params.gpus_per_host, 2)
            self.assertEqual(spec.params.nics_per_host, 1)
            self.assertEqual(spec.params.message_size, 4096)
            self.assertEqual(len(spec.config_id), 16)

    def test_topodsl_loader_injects_framework_for_user_derived_class(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_derived.py"
            _write_derived_class_topodsl(topo_path)

            spec = load_topodsl(topo_path)

            self.assertTrue(spec.prompt_source.startswith("class ClosTopology(BaseTopology):"))
            self.assertNotIn("class BaseTopology", spec.prompt_source)
            self.assertEqual(spec.params.family, "clos")
            self.assertEqual(spec.params.hosts, 4)
            self.assertEqual(spec.params.gpus_per_host, 8)
            self.assertEqual(spec.params.nics_per_host, 1)
            self.assertEqual(spec.params.leaf_switches, 2)
            self.assertEqual(spec.params.spine_switches, 1)
            self.assertEqual(spec.params.collective, "allgather")
            self.assertEqual(spec.params.message_size, 1024 * 1024)
            self.assertAlmostEqual(spec.params.host_bw_mbpus, 32.5 / 1000.0)
            self.assertAlmostEqual(spec.params.net_bw_mbpus, 2.8125 / 1000.0)
            self.assertAlmostEqual(spec.params.spine_bw_mbpus, 45 / 1000.0)

    def test_record_agent_summary_update_and_new_direction_hint(self) -> None:
        agent = RecordAgent(stagnation_rounds=2)

        first = agent.update(
            round_id=1,
            validity_status="ok",
            completion_time=20.0,
            bottleneck_profile=_fake_bottleneck_profile(),
            candidate_sketch=_fake_sketch(),
            best_so_far=True,
        )
        second = agent.update(
            round_id=2,
            validity_status="ok",
            completion_time=22.0,
            bottleneck_profile=_fake_bottleneck_profile(),
            candidate_sketch=_fake_sketch(),
            best_so_far=False,
        )
        third = agent.update(
            round_id=3,
            validity_status="ok",
            completion_time=21.0,
            bottleneck_profile=_fake_bottleneck_profile(),
            candidate_sketch=_fake_sketch(),
            best_so_far=False,
        )

        self.assertIn("round 1", first.summary.lower())
        self.assertIn("best", first.summary.lower())
        self.assertEqual(second.direction_hint, first.direction_hint)
        self.assertNotEqual(third.direction_hint, first.direction_hint)
        self.assertIn("stagnated", third.direction_hint.lower())
        self.assertIn("tx1 step 1 layer 3 group 0", third.summary)

    def test_simpletes_runtime_fails_loud_when_simulator_omits_time_us(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=1,
                completions=[_fake_sketch_evolve_block()],
                flow_sim_fn=lambda **_: {
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )

            with self.assertRaisesRegex(RuntimeError, "time_us"):
                _run_engine_without_slow_finalization(engine)

    def test_simpletes_runtime_fails_loud_when_simulator_omits_bottleneck_profile(self) -> None:
        with self._tmpdir() as tmp_path:
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
                    "links": [{"src": 0, "dst": 4, "queue_wait_ns": 120}],
                },
            )

            with self.assertRaisesRegex(RuntimeError, "bottleneck_profile.*legacy links"):
                _run_engine_without_slow_finalization(engine)

    def test_proposal_output_uses_simpletes_evolve_block_extraction(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            completion = _fake_sketch_evolve_block()
            seen: dict[str, object] = {}

            def flow_sim_fn(*, sketch_path: Path, **_: object) -> dict[str, object]:
                seen["sketch"] = json.loads(sketch_path.read_text())
                return {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                }

            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=1,
                completions=[completion],
                flow_sim_fn=flow_sim_fn,
            )

            _run_engine_without_slow_finalization(engine)

            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            round_record = json.loads((artifacts_dir / "rounds.jsonl").read_text().splitlines()[0])
            candidate_path = artifacts_dir / "rounds" / "round-0001-candidate.py"
            self.assertTrue(candidate_path.exists())
            candidate_code = candidate_path.read_text()
            self.assertIn("# EVOLVE-BLOCK-START", candidate_code)
            self.assertIn("# EVOLVE-BLOCK-END", candidate_code)
            self.assertIn("def run_code():", candidate_code)
            self.assertEqual(round_record["candidate_code_path"], "rounds/round-0001-candidate.py")
            self.assertEqual(round_record["code_extract_reason"], "evolve_block_merged")
            self.assertEqual(round_record["validity_status"], "ok")
            self.assertEqual(seen["sketch"], _fake_sketch())

    def test_invalid_proposal_record_includes_raw_llm_output(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            bad_completion = "I will describe the idea but forgot the evolve block."
            engine, llm = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=1,
                completions=[bad_completion],
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )

            _run_engine_without_slow_finalization(engine)

            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            round_record = json.loads((artifacts_dir / "rounds.jsonl").read_text().splitlines()[0])
            self.assertIn("missing_evolve_block_markers", round_record["validity_status"])
            self.assertEqual(round_record["proposal_output"], bad_completion)
            self.assertEqual(round_record["code_extract_reason"], "missing_evolve_block_markers")
            self.assertTrue(math.isinf(round_record["completion_time"]))
            self.assertLess(round_record["combined_score"], -1_000_000_000.0)
            self.assertIn("Proposal LLM output:", llm.prompts[1])
            self.assertIn(bad_completion, llm.prompts[1])

    def test_flow_sim_rejected_proposal_is_recorded_as_invalid_feedback(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            flow_sim_error = (
                "flow-sim-rs simulate-sketch failed with 1: "
                "Error: transmission 64.step must be in [0, 31]"
            )
            calls = 0

            def flow_sim_fn(**_: object) -> dict[str, object]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {
                        "time_us": 10.0,
                        "flow_count": 12,
                        "bottleneck_profile": _fake_bottleneck_profile(),
                    }
                raise RuntimeError(flow_sim_error)

            engine, llm = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=1,
                completions=[_fake_sketch_evolve_block(), _record_completion(0)],
                flow_sim_fn=flow_sim_fn,
            )

            _run_engine_without_slow_finalization(engine)

            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            round_record = json.loads((artifacts_dir / "rounds.jsonl").read_text().splitlines()[0])
            self.assertIn("flow-sim-rs simulate-sketch failed", round_record["validity_status"])
            self.assertEqual(round_record["bottleneck_profile"]["status"], "invalid")
            self.assertIn("transmission 64.step", round_record["bottleneck_profile"]["diagnosis"])
            self.assertTrue(math.isinf(round_record["completion_time"]))
            self.assertLess(round_record["combined_score"], -1_000_000_000.0)
            record_prompt = llm.prompts[1]
            self.assertIn("flow-sim-rs simulate-sketch failed", record_prompt)
            self.assertIn("transmission 64.step must be in [0, 31]", record_prompt)
            self.assertIn('"validity_status": "invalid:', record_prompt)

    def test_record_agent_prompt_requests_json_record_and_applies_repaired_output(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            completion = _fake_sketch_evolve_block()
            record_summary = "\n".join(
                [
                    "Trail id: 0",
                    "- transmissions strategy: local fanout, cross layer, remote fanout",
                    "- compact dsl: [(0, 1, 0, 0, [1])]",
                    "- profile data: time_us=10.0, bottleneck=0->4:120ns",
                ]
            )
            record_completion = (
                "{summary: "
                + json.dumps(record_summary)
                + ", direction_hint: \"avoid repaired bottleneck\"}"
            )
            engine, llm = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=1,
                completions=[completion, record_completion],
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )

            _run_engine_without_slow_finalization(engine)

            self.assertEqual(len(llm.prompts), 2)
            record_prompt = llm.prompts[1]
            self.assertIn("Return exactly one record for this attempt.", record_prompt)
            self.assertIn('"summary"', record_prompt)
            self.assertIn('"direction_hint"', record_prompt)
            self.assertIn("top_transmission_bottlenecks", record_prompt)
            self.assertIn("step 1 layer 3 group 0", record_prompt)
            self.assertNotIn("0->4:120ns", record_prompt)
            self.assertIn("Output only one JSON object", record_prompt)
            self.assertIn(record_summary, engine.runtime.record_agent.summary)
            self.assertEqual(engine.runtime.record_agent.direction_hint, "avoid repaired bottleneck")

    def test_record_agent_retries_unrepairable_json_and_logs_failure(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            bad_record = "[[["
            good_summary = "Trail id: 0\n- transmissions strategy: retry recovered\n- compact dsl: []\n- profile data: ok"
            good_record = json.dumps({"summary": good_summary, "direction_hint": "retry recovered"})
            engine, llm = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=1,
                completions=[_fake_sketch_evolve_block(), bad_record, good_record],
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )

            _run_engine_without_slow_finalization(engine)

            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            summary = json.loads((artifacts_dir / "summary.json").read_text())
            errors = [json.loads(line) for line in (artifacts_dir / "record_errors.jsonl").read_text().splitlines()]
            self.assertEqual(summary["record_tasks_failed"], 0)
            self.assertEqual(summary["record_tasks_retried"], 1)
            self.assertEqual(errors[0]["round_id"], 1)
            self.assertEqual(errors[0]["attempt"], 1)
            self.assertIn("RecordAgentOutputError", errors[0]["error_type"])
            self.assertIn("retry recovered", engine.runtime.record_agent.summary)
            llm_records = [json.loads(line) for line in (artifacts_dir / "llm_calls.jsonl").read_text().splitlines()]
            self.assertEqual(len(llm.prompts), 3)
            self.assertEqual([record["agent_type"] for record in llm_records], ["proposal", "record", "record"])
            self.assertIn("Record agent output failed", (Path(engine.checkpoint_dir) / "run.log").read_text())

    def test_record_agent_preserves_failed_retry_attempt_artifacts(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=1,
                completions=[_fake_sketch_evolve_block(), "not json", "{summary:"],
                save_llm_io=True,
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )

            _run_engine_without_slow_finalization(engine)

            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            summary = json.loads((artifacts_dir / "summary.json").read_text())
            errors = [json.loads(line) for line in (artifacts_dir / "record_errors.jsonl").read_text().splitlines()]
            llm_records = [json.loads(line) for line in (artifacts_dir / "llm_calls.jsonl").read_text().splitlines()]
            record_llm_records = [record for record in llm_records if record["agent_type"] == "record"]

            self.assertEqual(summary["record_tasks_failed"], 1)
            self.assertEqual(summary["record_tasks_retried"], 1)
            self.assertEqual([record["will_retry"] for record in errors], [True, False])
            self.assertEqual([record["attempt"] for record in errors], [1, 2])
            self.assertEqual([record["attempt"] for record in record_llm_records], [1, 2])
            self.assertEqual(len(llm_records), 3)
            self.assertNotEqual(record_llm_records[0]["completion_path"], record_llm_records[1]["completion_path"])
            self.assertEqual((artifacts_dir / record_llm_records[0]["completion_path"]).read_text(), "not json")
            self.assertEqual((artifacts_dir / record_llm_records[1]["completion_path"]).read_text(), "{summary:")
            run_log = (Path(engine.checkpoint_dir) / "run.log").read_text()
            self.assertIn("Record agent output failed for round 1 attempt 1; retrying", run_log)
            self.assertIn("Record agent output failed for round 1 attempt 2; giving up", run_log)

    def test_proposal_prompt_includes_initial_program_reference(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            engine, llm = _build_engine(
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

            _run_engine_without_slow_finalization(engine)

            proposal_prompt = llm.prompts[0]
            self.assertIn("--- Inspiration 1 ---", proposal_prompt)
            self.assertIn("Initial program reference", proposal_prompt)
            self.assertIn("def run_code():", proposal_prompt)
            self.assertIn("def construct_sketches():", proposal_prompt)
            self.assertIn("# EVOLVE-BLOCK-START", proposal_prompt)
            self.assertIn("# EVOLVE-BLOCK-END", proposal_prompt)

    def test_initial_program_is_evaluated_with_flow_sim_baseline(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            calls: list[Path] = []

            def flow_sim_fn(*, sketch_path: Path, **_: object) -> dict[str, object]:
                calls.append(sketch_path)
                return {
                    "time_us": 7.0 if len(calls) == 1 else 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                }

            engine, llm = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=1,
                completions=[_fake_sketch_evolve_block()],
                flow_sim_fn=flow_sim_fn,
            )

            _run_engine_without_slow_finalization(engine)

            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            summary = json.loads((artifacts_dir / "summary.json").read_text())
            root = next(node for node in engine.db.nodes.values() if not node.parent_ids)

            self.assertEqual(len(calls), 2)
            self.assertEqual(root.metrics["validity_status"], "ok")
            self.assertEqual(root.metrics["completion_time"], 7.0)
            self.assertEqual(root.score, -7.0)
            self.assertEqual(engine.best_score, -7.0)
            self.assertEqual(summary["initial_time_us"], 7.0)
            self.assertEqual(summary["best_time_us"], 7.0)
            self.assertEqual(summary["best_source"], "initial")
            self.assertEqual(summary["best_source_round"], 0)
            self.assertIn("Score: -7.0", llm.prompts[0])
            self.assertNotIn("-1000000000000", llm.prompts[0])

    def test_runtime_initial_program_uses_topodsl_gpu_count(self) -> None:
        from syccl_agents.prompts import load_init_program

        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_64gpu.py"
            topo_path.write_text(
                """
def topology():
    return {
        "family": "clos",
        "hosts": 8,
        "gpus_per_host": 8,
        "nics_per_host": 1,
        "leaf_switches": 2,
        "spine_switches": 1,
        "message_size": 1048576,
        "collective": "allgather",
    }
""".lstrip(),
                encoding="utf-8",
            )
            init_program, evaluator, instruction = _write_minimal_simpletes_files(tmp_path)
            init_program.write_text(load_init_program(), encoding="utf-8")
            config = EngineConfig(
                init_program=str(init_program),
                evaluator_path=str(evaluator),
                instruction_path=str(instruction),
                max_generations=1,
                init_eval_repeats=1,
                output_path=str(tmp_path / "checkpoints"),
                log_interval=1,
                db_show_interval=0,
                num_inspirations=0,
                num_chains=1,
                k_candidates=1,
                gen_concurrency=1,
                eval_concurrency=1,
            )

            def flow_sim_fn(*, sketch_path: Path, **_: object) -> dict[str, object]:
                sketch = json.loads(sketch_path.read_text())
                seen_gpus = [gpu for tx in sketch for gpu in tx["srcs"] + tx["dsts"]]
                if max(seen_gpus, default=0) >= 64:
                    raise AssertionError("initial sketch used a GPU id outside the 64-GPU topology")
                return {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                }

            llm = FakeLLMBackend([_fake_sketch_evolve_block(), _record_completion(0)])
            runtime = SycclTwoAgentRuntime(
                topo_path=topo_path,
                flow_sim_bin="flow-sim-rs",
                llm=llm,
                flow_sim_fn=flow_sim_fn,
            )
            engine = SimpleTESEngine(config, runtime=runtime)

            _run_engine_without_slow_finalization(engine)

            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            initial_sketch = json.loads((artifacts_dir / "initial-sketch.json").read_text())
            self.assertEqual(len(initial_sketch), 63)
            self.assertLess(max(gpu for tx in initial_sketch for gpu in tx["srcs"] + tx["dsts"]), 64)

    def test_runtime_leaves_final_checkpoint_to_finalize_run(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            engine, _ = _build_engine_with_log_interval(
                tmp_path,
                topo_path=topo_path,
                rounds=1,
                completions=[_fake_sketch_evolve_block(), _record_completion(0)],
                log_interval=0,
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )
            write_checkpoint = AsyncMock()
            finalize_run = AsyncMock()

            with (
                patch.object(SimpleTESEngine, "_write_checkpoint", new=write_checkpoint),
                patch.object(SimpleTESEngine, "_finalize_run", new=finalize_run),
            ):
                asyncio.run(engine.run())

            write_checkpoint.assert_not_awaited()
            finalize_run.assert_awaited_once()

    def test_generated_nodes_store_full_python_and_root_parent(self) -> None:
        with self._tmpdir() as tmp_path:
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

            _run_engine_without_slow_finalization(engine)

            nodes = [node.to_dict(include_llm_io=False) for node in engine.db.nodes.values()]
            root = next(node for node in nodes if not node.get("parent_ids"))
            generated = [node for node in nodes if node.get("parent_ids")]
            self.assertEqual(len(generated), 1)
            self.assertEqual(generated[0]["parent_ids"], [root["id"]])
            self.assertIn("def run_code():", generated[0]["code"])
            self.assertIn("# EVOLVE-BLOCK-START", generated[0]["code"])
            self.assertNotEqual(generated[0]["code"].lstrip()[0], "[")

    def test_all_generated_nodes_use_root_parent(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=2,
                completions=[_fake_sketch_evolve_block(), _fake_sketch_evolve_block()],
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )

            _run_engine_without_slow_finalization(engine)

            nodes = [node.to_dict(include_llm_io=False) for node in engine.db.nodes.values()]
            root = next(node for node in nodes if not node.get("parent_ids"))
            generated = [node for node in nodes if node.get("parent_ids")]
            self.assertEqual(len(generated), 2)
            self.assertEqual([node["parent_ids"] for node in generated], [[root["id"]], [root["id"]]])

    def test_proposal_context_uses_only_applied_llm_record_summary(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            llm = AsyncDelayedRecordLLMBackend(record_delay_s={1: 0.05, 2: 0.0})
            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=2,
                completions=[],
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )
            engine.runtime.llm = llm

            _run_engine_without_slow_finalization(engine)

            proposal2_prompt = next(prompt for agent_type, round_id, prompt in llm.calls if (agent_type, round_id) == ("proposal", 2))
            self.assertNotIn("round 1: best", proposal2_prompt)
            self.assertNotIn("sketch_bottlenecks=", proposal2_prompt)
            self.assertNotIn("Trail id: 1", proposal2_prompt)

    def test_async_record_does_not_block_next_proposal(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            llm = AsyncDelayedRecordLLMBackend(record_delay_s={1: 0.05, 2: 0.0})
            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=2,
                completions=[],
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )
            engine.runtime.llm = llm

            _run_engine_without_slow_finalization(engine)

            proposal2 = next(index for index, call in enumerate(llm.calls) if call[:2] == ("proposal", 2))
            record1 = next(index for index, call in enumerate(llm.calls) if call[:2] == ("record", 1))
            self.assertLess(record1, proposal2)
            self.assertLess(llm.record_started_at[1], llm.record_started_at[2])
            self.assertNotIn(1, llm.record_finished_at)

    def test_async_record_backlog_waits_when_five_tasks_are_pending(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            llm = AsyncDelayedRecordLLMBackend(record_delay_s={idx: 0.05 for idx in range(1, 7)})
            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=6,
                completions=[],
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )
            engine.runtime.llm = llm

            _run_engine_without_slow_finalization(engine)

            self.assertLessEqual(llm.max_concurrent_records, 5)
            self.assertGreaterEqual(llm.record_started_at[6], min(llm.record_finished_at.values()))

    def test_async_record_backlog_wait_is_reported_in_summary(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            llm = AsyncDelayedRecordLLMBackend(record_delay_s={idx: 0.05 for idx in range(1, 7)})
            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=6,
                completions=[],
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )
            engine.runtime.llm = llm

            _run_engine_without_slow_finalization(engine)

            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            summary = json.loads((artifacts_dir / "summary.json").read_text())
            self.assertGreater(summary["proposal_wait_for_record_backlog_events"], 0)
            self.assertGreater(summary["proposal_wait_for_record_backlog_ms_total"], 0)
            self.assertGreater(summary["proposal_wait_for_record_backlog_ms_max"], 0)

    def test_async_record_backlog_counts_out_of_order_completed_records(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            llm = AsyncDelayedRecordLLMBackend(
                record_delay_s={1: 0.08, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0},
                proposal_delay_s=0.005,
            )
            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=6,
                completions=[],
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )
            engine.runtime.llm = llm

            _run_engine_without_slow_finalization(engine)

            self.assertGreater(llm.record_started_at[6], llm.record_finished_at[1])

    def test_async_record_applies_completed_results_in_strict_round_order(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            llm = AsyncDelayedRecordLLMBackend(
                record_delay_s={1: 0.035, 2: 0.0, 3: 0.0},
                proposal_delay_s=0.02,
            )
            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=4,
                completions=[],
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )
            engine.runtime.llm = llm

            _run_engine_without_slow_finalization(engine)

            proposal3_prompt = next(prompt for agent_type, round_id, prompt in llm.calls if (agent_type, round_id) == ("proposal", 3))
            proposal4_prompt = next(prompt for agent_type, round_id, prompt in llm.calls if (agent_type, round_id) == ("proposal", 4))
            self.assertNotIn("Trail id: 2", proposal3_prompt)
            self.assertIn("Trail id: 1", proposal4_prompt)
            self.assertIn("Trail id: 2", proposal4_prompt)

    def test_async_record_failure_does_not_block_later_records(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            llm = AsyncFailingRecordLLMBackend(failing_rounds={1})
            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=3,
                completions=[],
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )
            engine.runtime.llm = llm

            _run_engine_without_slow_finalization(engine)

            proposal3_prompt = next(prompt for agent_type, round_id, prompt in llm.calls if (agent_type, round_id) == ("proposal", 3))
            self.assertNotIn("Trail id: 1", proposal3_prompt)
            self.assertIn("Trail id: 2", proposal3_prompt)
            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            summary = json.loads((artifacts_dir / "summary.json").read_text())
            self.assertEqual(summary["record_tasks_failed"], 1)
            errors = [json.loads(line) for line in (artifacts_dir / "record_errors.jsonl").read_text().splitlines()]
            self.assertEqual(errors[0]["round_id"], 1)
            self.assertEqual(errors[0]["error_type"], "RuntimeError")
            self.assertIn("record round 1 failed", errors[0]["message"])

    def test_llm_prompt_and_completion_are_saved_for_inspection(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            completion = _fake_sketch_evolve_block()
            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=1,
                completions=[completion],
                save_llm_io=True,
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )

            _run_engine_without_slow_finalization(engine)

            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            llm_record = json.loads((artifacts_dir / "llm_calls.jsonl").read_text().splitlines()[0])
            prompt_path = artifacts_dir / llm_record["prompt_path"]
            completion_path = artifacts_dir / llm_record["completion_path"]
            self.assertTrue(prompt_path.exists())
            self.assertTrue(completion_path.exists())
            prompt = prompt_path.read_text()
            self.assertIn("topo_dsl:\n```python", prompt)
            self.assertIn("Implement `construct_sketches()`", prompt)
            self.assertIn("# EVOLVE-BLOCK-START", prompt)
            self.assertIn("# EVOLVE-BLOCK-END", prompt)
            self.assertIn("def topology():", prompt)
            self.assertEqual(completion_path.read_text(), completion)

    def test_cli_exposes_save_llm_io_flag(self) -> None:
        runner_script = _load_runner_script()

        args = runner_script.parse_args([
            "--topo",
            "examples/topologies/clos_2host.py",
            "--output-dir",
            "/tmp/out",
            "--save-llm-io",
        ])

        self.assertTrue(args.save_llm_io)
        self.assertIsNone(args.message_size)
        self.assertIsNone(args.collective)

    def test_cli_env_toml_missing_path_fails_loud(self) -> None:
        runner_script = _load_runner_script()

        with self.assertRaisesRegex(FileNotFoundError, "env.toml"):
            runner_script.load_env_toml("/tmp/definitely-missing-syccl-env.toml")

    def test_cli_prints_syccl_summary_fields_when_summary_exists(self) -> None:
        runner_script = _load_runner_script()

        with self._tmpdir() as tmp_path:
            checkpoint_dir = tmp_path / "checkpoints" / "2026-06-22" / "instance-test"
            summary_dir = checkpoint_dir / "syccl_two_agent"
            summary_dir.mkdir(parents=True)
            (summary_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "best_time_us": 12.5,
                        "best_source": "initial",
                        "best_source_round": 0,
                    }
                ),
                encoding="utf-8",
            )
            engine = type(
                "Engine",
                (),
                {
                    "instance_id": "test",
                    "checkpoint_dir": str(checkpoint_dir),
                    "best_score": -12.5,
                },
            )()

            lines = runner_script._format_final_output(engine)

            self.assertIn("instance_id=test", lines)
            self.assertIn(f"checkpoint_dir={checkpoint_dir}", lines)
            self.assertIn("best_score=-12.5", lines)
            self.assertIn("syccl_best_time_us=12.5", lines)
            self.assertIn("syccl_best_source=initial", lines)
            self.assertIn("syccl_best_source_round=0", lines)
            self.assertIn(f"syccl_summary_json={summary_dir / 'summary.json'}", lines)

    def test_runtime_uses_collective_and_message_size_from_topodsl_by_default(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_derived.py"
            _write_derived_class_topodsl(topo_path)
            engine, llm = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=1,
                completions=[_fake_sketch_evolve_block()],
                collective=None,
                message_size=None,
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )

            _run_engine_without_slow_finalization(engine)

            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            round_record = json.loads((artifacts_dir / "rounds.jsonl").read_text().splitlines()[0])
            self.assertEqual(round_record["collective"], "allgather")
            self.assertEqual(round_record["message_size"], 1024 * 1024)
            self.assertIn("- message_size_bytes: 1048576", llm.prompts[0])
            self.assertIn("The target collective is allgather", llm.prompts[0])

    def test_simpletes_runtime_preserves_checkpoint_llm_io(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            completion = _fake_sketch_evolve_block()
            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=1,
                completions=[completion],
                save_llm_io=True,
                flow_sim_fn=lambda **_: {
                    "time_us": 10.0,
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                },
            )

            _run_engine_without_slow_finalization(engine)

            nodes = [node.to_dict(include_llm_io=True) for node in engine.db.nodes.values()]
            generated = [node for node in nodes if node.get("parent_ids")]
            self.assertEqual(len(generated), 1)
            self.assertIn("topo_dsl:\n```python", generated[0]["llm_input"])
            self.assertIn("Implement `construct_sketches()`", generated[0]["llm_input"])
            self.assertIn("# EVOLVE-BLOCK-START", generated[0]["llm_input"])
            self.assertIn("# EVOLVE-BLOCK-END", generated[0]["llm_input"])
            self.assertIn("def topology():", generated[0]["llm_input"])
            self.assertEqual(generated[0]["llm_output"], completion)
            self.assertIsInstance(generated[0]["token_usage"], dict)
            self.assertEqual(generated[0]["metrics"]["validity_status"], "ok")
            self.assertEqual(generated[0]["metrics"]["combined_score"], -10.0)
            self.assertEqual(generated[0]["score"], -10.0)

    def test_fake_smoke_runs_10_sequential_rounds_through_flow_sim_boundary(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            calls: list[tuple[str, Path, Path]] = []

            def flow_sim_fn(*, config_path: Path, sketch_path: Path, output_path: Path, **_: object) -> dict[str, object]:
                calls.append(("flow-sim-rs", config_path, sketch_path))
                return {
                    "time_us": 10.0 + len(calls),
                    "flow_count": 12,
                    "bottleneck_profile": _fake_bottleneck_profile(),
                }

            engine, _ = _build_engine(
                tmp_path,
                topo_path=topo_path,
                rounds=10,
                completions=[_fake_sketch_evolve_block() for _ in range(10)],
                flow_sim_fn=flow_sim_fn,
            )

            _run_engine_without_slow_finalization(engine)

            self.assertEqual(len(calls), 11)
            artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
            summary = json.loads((artifacts_dir / "summary.json").read_text())
            nodes = [node.to_dict(include_llm_io=False) for node in engine.db.nodes.values()]
            generated = [node for node in nodes if node.get("parent_ids")]
            self.assertEqual(len(generated), 10)
            self.assertEqual(summary["rounds_completed"], 10)
            self.assertEqual(summary["enumeration_status"], "skipped")
            self.assertEqual(len((artifacts_dir / "rounds.jsonl").read_text().splitlines()), 10)
            self.assertEqual(len((artifacts_dir / "llm_calls.jsonl").read_text().splitlines()), 20)
            self.assertTrue((artifacts_dir / "best_sketch.json").exists())

    def test_simpletes_runtime_prompt_marks_sketch_search_skipped_for_now(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_2host.py"
            _write_python_topodsl(topo_path)
            engine, llm = _build_engine(
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

            _run_engine_without_slow_finalization(engine)

            self.assertIn("Sketch enumeration: skipped", llm.prompts[0])
            self.assertIn("not compact DSL", llm.prompts[0])

    def test_proposal_prompt_uses_syccl_template_with_replacements(self) -> None:
        with self._tmpdir() as tmp_path:
            topo_path = tmp_path / "clos_derived.py"
            _write_derived_class_topodsl(topo_path)
            engine, llm = _build_engine(
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

            _run_engine_without_slow_finalization(engine)

            prompt = llm.prompts[0]
            self.assertIn("Optimize SyCCL direct-event sketch generation", prompt)
            self.assertIn("Implement `construct_sketches()`", prompt)
            self.assertIn("# EVOLVE-BLOCK-START", prompt)
            self.assertIn("# EVOLVE-BLOCK-END", prompt)
            self.assertIn("class ClosTopology(BaseTopology):", prompt)
            self.assertIn("The target collective is allgather", prompt)
            self.assertIn("rotated across all 32 roots", prompt)
            self.assertNotIn("$Collective", prompt)
            self.assertNotIn("$GPU_NUM", prompt)
            self.assertNotIn("topo_dsl:\n```python\n\n```", prompt)

    def test_sketch_parser_extracts_list_from_llm_text(self) -> None:
        text = (
            "Here is the compact SketchDSL:\n"
            '[{"step": 0, "layer": 1, "group": 0, "srcs": [0], "dsts": [1]}]\n'
            "This completes the root propagation."
        )

        sketch = parse_sketch_dsl(text)

        self.assertEqual(sketch, [{"step": 0, "layer": 1, "group": 0, "srcs": [0], "dsts": [1]}])

    def test_sketch_parser_extracts_construct_sketches_return_value(self) -> None:
        text = """
def construct_sketches():
    return [
        (0, 1, 0, 0, [1]),
        (1, 3, 0, [0, 1], [8, 16]),
    ]
"""

        sketch = parse_sketch_dsl(text)

        self.assertEqual(
            sketch,
            [
                {"step": 0, "layer": 1, "group": 0, "srcs": [0], "dsts": [1]},
                {"step": 1, "layer": 3, "group": 0, "srcs": [0, 1], "dsts": [8, 16]},
            ],
        )

    def test_sketch_parser_allows_construct_sketches_to_call_top_level_helpers(self) -> None:
        text = """
def tx(step, layer, group, srcs, dsts):
    return (step, layer, group, srcs, dsts)


def construct_sketches():
    return [
        tx(0, 1, 0, 0, [1]),
    ]
"""

        sketch = parse_sketch_dsl(text)

        self.assertEqual(sketch, [{"step": 0, "layer": 1, "group": 0, "srcs": [0], "dsts": [1]}])

    def test_flow_sim_runner_invokes_simulate_sketch_without_python_evaluator(self) -> None:
        with self._tmpdir() as tmp_path:
            seen: dict[str, object] = {}

            def fake_run(command: list[str], **kwargs: object) -> object:
                seen["command"] = command
                seen["kwargs"] = kwargs
                output = Path(command[command.index("--output") + 1])
                output.write_text(
                    json.dumps({"time_us": 1.0, "flow_count": 1, "bottleneck_profile": _fake_bottleneck_profile()}),
                    encoding="utf-8",
                )

                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return Completed()

            with patch("subprocess.run", fake_run):
                runner = FlowSimRunner("flow-sim-rs")
                result = runner.simulate_sketch(
                    config_path=tmp_path / "config.json",
                    sketch_path=tmp_path / "sketch.json",
                    output_path=tmp_path / "sim.json",
                )

            command = seen["command"]
            self.assertEqual(result["time_us"], 1.0)
            self.assertEqual(command[:2], ["flow-sim-rs", "simulate-sketch"])
            self.assertNotIn("scheme1_direct_events/evaluator.py", " ".join(command))

    def test_flow_sim_runner_rejects_output_without_bottleneck_profile(self) -> None:
        with self._tmpdir() as tmp_path:
            def fake_run(command: list[str], **kwargs: object) -> object:
                del kwargs
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps({"time_us": 1.0, "flow_count": 1, "links": []}), encoding="utf-8")

                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return Completed()

            with patch("subprocess.run", fake_run):
                runner = FlowSimRunner("flow-sim-rs")
                with self.assertRaisesRegex(RuntimeError, "bottleneck_profile"):
                    runner.simulate_sketch(
                        config_path=tmp_path / "config.json",
                        sketch_path=tmp_path / "sketch.json",
                        output_path=tmp_path / "sim.json",
                    )

    def _tmpdir(self):
        import tempfile

        return _TemporaryDirectoryPath(tempfile.TemporaryDirectory(prefix="syccl_two_agent_"))


class _TemporaryDirectoryPath:
    def __init__(self, inner):
        self._inner = inner

    def __enter__(self) -> Path:
        return Path(self._inner.__enter__())

    def __exit__(self, exc_type, exc, tb):
        return self._inner.__exit__(exc_type, exc, tb)


if __name__ == "__main__":
    unittest.main()
