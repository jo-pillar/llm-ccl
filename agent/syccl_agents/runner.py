from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from warnings import deprecated

from simpletes.node import EvolveBlockContext, extract_code_detailed
from simpletes.engine.syccl_two_agent import (
    CandidateInvalidError,
    FAILURE_COMPLETION_TIME,
    FAILURE_SCORE,
    MissingBottleneckProfileError,
    _artifact_relpath,
    _run_candidate_code,
)
from syccl_agents.config_render import (
    layer_group_summary,
    seed_sketch_hint,
    topology_summary,
    write_syccl_config,
)
from syccl_agents.flow_sim import FlowSimOutputError, FlowSimRunner, validate_flow_sim_result
from syccl_agents.llm import LLMResponse
from syccl_agents.record_agent import RecordAgent
from syccl_agents.records import (
    append_jsonl,
    stable_hash,
    validate_llm_call_record,
    validate_round_record,
)
from syccl_agents.prompts import load_init_program, render_proposal_prompt
from syccl_agents.sketch_dsl import write_compact_sketch
from syccl_agents.topodsl import load_topodsl


@dataclass(frozen=True)
class SycclTwoAgentConfig:
    topo_path: Path
    output_dir: Path
    rounds: int
    collective: str | None = None
    message_size: int | None = None
    flow_sim_bin: str | Path = "flow-sim-rs"
    flow_sim_timeout_s: float | None = None
    save_llm_io: bool = False


@dataclass(frozen=True)
class SycclTwoAgentResult:
    run_id: str
    rounds_completed: int
    best_time_us: float | None
    output_dir: Path
    enumeration_status: str


FlowSimFn = Callable[..., dict[str, Any]]


class SycclTwoAgentRunner:
    def __init__(
        self,
        config: SycclTwoAgentConfig,
        *,
        llm: Any,
        record_agent: RecordAgent | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.record_agent = record_agent or RecordAgent()
        self.run_id = f"syccl-two-agent-{uuid.uuid4().hex[:12]}"
    @deprecated(reason="This method is not intended to be used directly; ")
    def __run(self, *, flow_sim_fn: FlowSimFn | None = None) -> SycclTwoAgentResult:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        topo = load_topodsl(
            self.config.topo_path,
            collective=self.config.collective,
            message_size=self.config.message_size,
        )
        params = topo.params
        if self.config.collective is not None and self.config.collective != params.collective:
            raise ValueError(f"TopoDSL collective {params.collective} does not match {self.config.collective}")
        if self.config.message_size is not None and self.config.message_size != params.message_size:
            raise ValueError(f"TopoDSL message_size {params.message_size} does not match {self.config.message_size}")

        config_path = write_syccl_config(params, output_dir / "flow-sim-config.json")
        enumeration_status = "skipped"
        enumeration_note = "Sketch enumeration: skipped because syccl-sketch-search currently emits JSON, not compact DSL suitable for LLM prompt parsing."
        best_time: float | None = None
        best_sketch: list[dict[str, Any]] | None = None
        rounds_completed = 0
        flow_sim = FlowSimRunner(self.config.flow_sim_bin, timeout_s=self.config.flow_sim_timeout_s)
        evolve_context = EvolveBlockContext.from_program(_candidate_scaffold())

        for round_id in range(1, self.config.rounds + 1):
            prompt = self._build_prompt(
                topo_source=topo.prompt_source,
                topo_summary=topology_summary(params),
                layer_summary=layer_group_summary(params),
                seed_hint=seed_sketch_hint(params),
                enumeration_note=enumeration_note,
                collective=params.collective,
                message_size=params.message_size,
            )
            llm_response = self.llm.generate(prompt, agent_type="proposal", round_id=round_id)
            self._write_llm_record(round_id, prompt, llm_response)

            code_extract_reason: str | None = None
            candidate_code_path: Path | None = None
            try:
                candidate_code, code_extract_reason = extract_code_detailed(llm_response.text, evolve_context)
                if candidate_code is None:
                    raise CandidateInvalidError(code_extract_reason)
                candidate_code_path = output_dir / "rounds" / f"round-{round_id:04d}-candidate.py"
                candidate_code_path.parent.mkdir(parents=True, exist_ok=True)
                candidate_code_path.write_text(candidate_code, encoding="utf-8")
                try:
                    candidate = _run_candidate_code(candidate_code_path)
                except Exception as exc:
                    raise CandidateInvalidError(str(exc)) from exc
                sketch_path = write_compact_sketch(candidate, output_dir / "rounds" / f"round-{round_id:04d}-sketch.json")
                sim_output_path = output_dir / "rounds" / f"round-{round_id:04d}-flow-sim.json"
                sim_result = (
                    flow_sim_fn(config_path=config_path, sketch_path=sketch_path, output_path=sim_output_path)
                    if flow_sim_fn
                    else flow_sim.simulate_sketch(
                        config_path=config_path,
                        sketch_path=sketch_path,
                        output_path=sim_output_path,
                    )
                )
                sim_result = validate_flow_sim_result(sim_result, output_path=sim_output_path)
                validity_status = "ok"
                completion_time = float(sim_result["time_us"])
                expanded_events = int(sim_result["flow_count"])
                bottleneck_profile = _bottleneck_profile(sim_result, candidate)
            except (MissingBottleneckProfileError, FlowSimOutputError, OSError):
                raise
            except CandidateInvalidError as exc:
                candidate = []
                validity_status = f"invalid: {exc}"
                completion_time = FAILURE_COMPLETION_TIME
                expanded_events = 0
                bottleneck_profile = _invalid_bottleneck_profile(str(exc))

            best_so_far = math.isfinite(completion_time) and (best_time is None or completion_time < best_time)
            if best_so_far:
                best_time = completion_time
                best_sketch = candidate
                write_compact_sketch(candidate, output_dir / "best_sketch.json")

            self.record_agent.update(
                round_id=round_id,
                validity_status=validity_status,
                completion_time=completion_time,
                bottleneck_profile=bottleneck_profile,
                candidate_sketch=candidate,
                best_so_far=best_so_far,
            )
            self._write_round_record(
                topodsl_config_id=topo.config_id,
                round_id=round_id,
                prompt=prompt,
                llm_response=llm_response,
                candidate=candidate,
                collective=params.collective,
                message_size=params.message_size,
                validity_status=validity_status,
                expanded_events=expanded_events,
                combined_score=_score(completion_time),
                completion_time=completion_time,
                bottleneck_profile=bottleneck_profile,
                best_so_far=best_so_far,
                proposal_output=llm_response.raw_output or llm_response.text,
                code_extract_reason=code_extract_reason,
                candidate_code_path=_artifact_relpath(candidate_code_path, output_dir),
            )
            rounds_completed += 1

        summary = {
            "run_id": self.run_id,
            "rounds_completed": rounds_completed,
            "best_time_us": best_time,
            "best_sketch_stages": len(best_sketch or []),
            "enumeration_status": enumeration_status,
            "enumeration_note": enumeration_note,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return SycclTwoAgentResult(
            run_id=self.run_id,
            rounds_completed=rounds_completed,
            best_time_us=best_time,
            output_dir=output_dir,
            enumeration_status=enumeration_status,
        )

    def _build_prompt(
        self,
        *,
        topo_source: str,
        topo_summary: str,
        layer_summary: str,
        seed_hint: str,
        enumeration_note: str,
        collective: str,
        message_size: int,
    ) -> str:
        total_gpus = _total_gpus_from_summary(topo_summary)
        return render_proposal_prompt(
            topo_source=topo_source,
            collective=collective,
            gpu_num=total_gpus,
            message_size=message_size,
            topo_summary=topo_summary,
            layer_summary=layer_summary,
            seed_hint=seed_hint,
            enumeration_note=enumeration_note,
            record_summary=self.record_agent.summary,
            direction_hint=self.record_agent.direction_hint,
        )

    def _write_llm_record(self, round_id: int, prompt: str, response: LLMResponse) -> None:
        record = {
            "run_id": self.run_id,
            "agent_type": "proposal",
            "round_id": round_id,
            "model_name": response.model_name,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cached_input_tokens": response.cached_input_tokens,
            "reasoning_tokens": response.reasoning_tokens,
            "wall_clock_time_ms": response.wall_clock_time_ms,
            "api_cost_usd": response.api_cost_usd,
            "prompt_hash": stable_hash(prompt),
            "completion_hash": stable_hash(response.text),
        }
        if self.config.save_llm_io:
            prompt_rel = Path("llm_io") / f"round-{round_id:04d}-proposal-prompt.txt"
            completion_rel = Path("llm_io") / f"round-{round_id:04d}-proposal-completion.txt"
            raw_rel = Path("llm_io") / f"round-{round_id:04d}-proposal-raw-output.txt"
            prompt_path = self.config.output_dir / prompt_rel
            completion_path = self.config.output_dir / completion_rel
            raw_path = self.config.output_dir / raw_rel
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt, encoding="utf-8")
            completion_path.write_text(response.text, encoding="utf-8")
            raw_path.write_text(response.raw_output or response.text, encoding="utf-8")
            record.update(
                {
                    "prompt_path": prompt_rel.as_posix(),
                    "completion_path": completion_rel.as_posix(),
                    "raw_output_path": raw_rel.as_posix(),
                }
            )
        validate_llm_call_record(record)
        append_jsonl(self.config.output_dir / "llm_calls.jsonl", record)

    def _write_round_record(
        self,
        *,
        topodsl_config_id: str,
        round_id: int,
        prompt: str,
        llm_response: LLMResponse,
        candidate: list[dict[str, Any]],
        collective: str,
        message_size: int,
        validity_status: str,
        expanded_events: int,
        combined_score: float,
        completion_time: float | None,
        bottleneck_profile: dict[str, Any],
        best_so_far: bool,
        proposal_output: str,
        code_extract_reason: str | None,
        candidate_code_path: str | None,
    ) -> None:
        record = {
            "topodsl_config_id": topodsl_config_id,
            "collective": collective,
            "message_size": message_size,
            "round_id": round_id,
            "prompt_hash": stable_hash(prompt),
            "input_tokens": llm_response.input_tokens,
            "output_tokens": llm_response.output_tokens,
            "candidate_sketch": candidate,
            "validity_status": validity_status,
            "expanded_events": expanded_events,
            "combined_score": combined_score,
            "completion_time": completion_time,
            "algorithm_bandwidth": _algorithm_bandwidth(message_size, completion_time),
            "bottleneck_profile": bottleneck_profile,
            "best_so_far": best_so_far,
            "proposal_output": proposal_output,
            "code_extract_reason": code_extract_reason,
            "candidate_code_path": candidate_code_path,
        }
        validate_round_record(record)
        append_jsonl(self.config.output_dir / "rounds.jsonl", record)


def _bottleneck_profile(result: dict[str, Any], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    profile = result.get("bottleneck_profile")
    if isinstance(profile, dict):
        return profile
    links = result.get("links")
    link_note = " legacy links were present but cannot be mapped to sketch transmissions." if links else ""
    raise MissingBottleneckProfileError(
        "flow-sim result is missing required bottleneck_profile; update flow-sim-rs to emit "
        "sketch-level bottleneck_profile diagnostics or disable this optimizer path."
        f"{link_note}"
    )


def _invalid_bottleneck_profile(error: str) -> dict[str, Any]:
    return {
        "status": "invalid",
        "diagnosis": error,
        "critical_transmission": None,
        "critical_chain": [],
        "top_transmission_bottlenecks": [],
        "stage_pressure": [],
    }


def _algorithm_bandwidth(message_size: int | None, completion_time_us: float | None) -> float | None:
    if (
        message_size is None
        or completion_time_us is None
        or completion_time_us <= 0
        or not math.isfinite(completion_time_us)
    ):
        return None
    return float(message_size) / completion_time_us


def _score(completion_time_us: float | None) -> float:
    if completion_time_us is None or completion_time_us <= 0 or not math.isfinite(completion_time_us):
        return FAILURE_SCORE
    return -float(completion_time_us)


def _candidate_scaffold() -> str:
    return load_init_program()


def _total_gpus_from_summary(topo_summary: str) -> int:
    marker = "total_gpus="
    start = topo_summary.find(marker)
    if start < 0:
        raise ValueError(f"topology summary missing {marker!r}: {topo_summary}")
    start += len(marker)
    end = start
    while end < len(topo_summary) and topo_summary[end].isdigit():
        end += 1
    return int(topo_summary[start:end])
