from __future__ import annotations

import json
import math
import runpy
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from simpletes.engine.runtime import RuntimeBase
from simpletes.node import Node, Status, extract_code_detailed
from syccl_agents.config_render import (
    layer_group_summary,
    seed_sketch_hint,
    topology_summary,
    write_syccl_config,
)
from syccl_agents.flow_sim import FlowSimRunner
from syccl_agents.record_agent import RecordAgent
from syccl_agents.records import (
    append_jsonl,
    stable_hash,
    validate_llm_call_record,
    validate_round_record,
)
from syccl_agents.prompts import render_proposal_prompt, render_record_prompt
from syccl_agents.sketch_dsl import parse_sketch_dsl, write_compact_sketch
from syccl_agents.topodsl import load_topodsl


FlowSimFn = Callable[..., dict[str, Any]]
FAILURE_SCORE = -1_000_000_000_000.0
FAILURE_COMPLETION_TIME = float("inf")


@dataclass(frozen=True)
class AgentLLMResult:
    text: str
    raw_output: str
    token_usage: dict[str, int | None] | None
    model_name: str
    wall_clock_time_ms: int


class SycclTwoAgentRuntime(RuntimeBase):
    """SimpleTES runtime that replaces only the SyCCL search loop.

    This keeps SimpleTES initialization, LLM backend, NodeDatabase, checkpoint,
    and save_llm_io behavior, while replacing inspiration/multi-chain scheduling
    with the paper-style sequential proposal/record loop.
    """

    def __init__(
        self,
        *,
        topo_path: str | Path,
        collective: str | None = None,
        message_size: int | None = None,
        flow_sim_bin: str | Path,
        llm: Any | None = None,
        flow_sim_fn: FlowSimFn | None = None,
    ) -> None:
        self.topo_path = Path(topo_path)
        self.collective = collective
        self.message_size = message_size
        self.flow_sim_bin = flow_sim_bin
        self.llm = llm
        self.flow_sim_fn = flow_sim_fn
        self.record_agent = RecordAgent()
        self.record_trails: list[str] = []

    def decorate_init_info(self, base_info: str) -> str:
        return base_info + "\n[dim]SyCCL two-agent runtime:[/dim] [cyan]ON[/cyan]"

    async def run(self, engine) -> None:
        async with engine._db_lock:
            is_empty = len(engine.db.nodes) == 0
        if is_empty:
            await engine._initialize_from_scratch()

        topo = load_topodsl(
            self.topo_path,
            collective=self.collective,
            message_size=self.message_size,
        )
        if self.collective is not None and topo.params.collective != self.collective:
            raise ValueError(f"TopoDSL collective {topo.params.collective} does not match {self.collective}")
        if self.message_size is not None and topo.params.message_size != self.message_size:
            raise ValueError(f"TopoDSL message_size {topo.params.message_size} does not match {self.message_size}")
        self.collective = topo.params.collective
        self.message_size = topo.params.message_size

        artifacts_dir = Path(engine.checkpoint_dir) / "syccl_two_agent"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        config_path = write_syccl_config(topo.params, artifacts_dir / "flow-sim-config.json")
        flow_sim = FlowSimRunner(self.flow_sim_bin, timeout_s=engine.config.eval_timeout)
        proposal_llm = self.llm if self.llm is not None else engine.generator._llm
        record_llm = proposal_llm
        parent_id = engine.best_node_id
        best_time: float | None = None
        best_sketch: list[dict[str, Any]] | None = None
        rounds_completed = 0
        run_id = f"syccl-two-agent-{engine.instance_id}"
        enumeration_note = (
            "Sketch enumeration: skipped because syccl-sketch-search currently emits JSON, "
            "not compact DSL suitable for LLM prompt parsing."
        )

        for round_id in range(1, engine.config.max_generations + 1):
            prompt = self._build_prompt(
                topo_source=topo.prompt_source,
                topo_summary=topology_summary(topo.params),
                layer_summary=layer_group_summary(topo.params),
                seed_hint=seed_sketch_hint(topo.params),
                enumeration_note=enumeration_note,
            )
            llm_result = await _call_llm(
                proposal_llm,
                prompt,
                engine.instance_id,
                track_io=True,
                round_id=round_id,
                agent_type="proposal",
                model_name=engine.config.model,
            )
            self._write_llm_record(
                artifacts_dir=artifacts_dir,
                run_id=run_id,
                round_id=round_id,
                agent_type="proposal",
                prompt=prompt,
                result=llm_result,
                save_llm_io=engine.config.save_llm_io,
            )
            metrics, candidate = self._evaluate_candidate(
                round_id=round_id,
                text=llm_result.text,
                raw_output=llm_result.raw_output,
                evolve_context=engine._evolve_context,
                config_path=config_path,
                artifacts_dir=artifacts_dir,
                flow_sim=flow_sim,
                best_time=best_time,
            )
            record_prompt = self._build_record_prompt(
                round_id=round_id,
                candidate=candidate,
                metrics=metrics,
                current_summary=self._record_trail_history(),
                direction_hint=self.record_agent.direction_hint,
            )
            record_result = await _call_llm(
                record_llm,
                record_prompt,
                engine.instance_id,
                track_io=True,
                round_id=round_id,
                agent_type="record",
                model_name=engine.config.model,
            )
            self._write_llm_record(
                artifacts_dir=artifacts_dir,
                run_id=run_id,
                round_id=round_id,
                agent_type="record",
                prompt=record_prompt,
                result=record_result,
                save_llm_io=engine.config.save_llm_io,
            )
            self._apply_record_agent_output(record_result.text)
            metrics["record_summary"] = self.record_agent.summary
            metrics["direction_hint"] = self.record_agent.direction_hint

            if metrics.get("validity_status") == "ok":
                completion_time = metrics.get("completion_time")
                if completion_time is not None and math.isfinite(completion_time) and (best_time is None or completion_time < best_time):
                    best_time = float(completion_time)
                    best_sketch = candidate
                    write_compact_sketch(candidate, artifacts_dir / "best_sketch.json")

            node = Node(
                id=uuid.uuid4().hex,
                code=json.dumps(candidate, indent=2),
                parent_ids=[parent_id] if parent_id else [],
                gen_id=round_id - 1,
                chain_idx=0,
                metrics=metrics,
                score=metrics["combined_score"],
                status=Status.DONE,
            )
            if engine.config.save_llm_io:
                node.llm_input = prompt
                node.llm_output = llm_result.raw_output
                node.token_usage = llm_result.token_usage or {}

            self._write_round_record(
                artifacts_dir=artifacts_dir,
                topodsl_config_id=topo.config_id,
                round_id=round_id,
                prompt=prompt,
                llm_result=llm_result,
                candidate=candidate,
                metrics=metrics,
            )
            await engine._commit_node(node)
            parent_id = node.id
            async with engine._counter_lock:
                engine.generation_attempts += 1
            rounds_completed += 1

        summary = {
            "run_id": run_id,
            "rounds_completed": rounds_completed,
            "best_time_us": best_time,
            "best_sketch_stages": len(best_sketch or []),
            "enumeration_status": "skipped",
            "enumeration_note": enumeration_note,
        }
        (artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        await engine._write_checkpoint()
        await engine._finalize_run()

    def _evaluate_candidate(
        self,
        *,
        round_id: int,
        text: str,
        raw_output: str,
        evolve_context: Any,
        config_path: Path,
        artifacts_dir: Path,
        flow_sim: FlowSimRunner,
        best_time: float | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        candidate: list[dict[str, Any]] = []
        candidate_code_path: Path | None = None
        code_extract_reason: str | None = None
        try:
            candidate_code, code_extract_reason = extract_code_detailed(text, evolve_context)
            if candidate_code is None:
                raise ValueError(code_extract_reason)
            candidate_code_path = artifacts_dir / "rounds" / f"round-{round_id:04d}-candidate.py"
            candidate_code_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_code_path.write_text(candidate_code, encoding="utf-8")
            candidate = _run_candidate_code(candidate_code_path)
            sketch_path = write_compact_sketch(candidate, artifacts_dir / "rounds" / f"round-{round_id:04d}-sketch.json")
            sim_output_path = artifacts_dir / "rounds" / f"round-{round_id:04d}-flow-sim.json"
            sim_result = (
                self.flow_sim_fn(config_path=config_path, sketch_path=sketch_path, output_path=sim_output_path)
                if self.flow_sim_fn
                else flow_sim.simulate_sketch(
                    config_path=config_path,
                    sketch_path=sketch_path,
                    output_path=sim_output_path,
                )
            )
            completion_time = float(sim_result.get("time_us"))
            best_so_far = math.isfinite(completion_time) and (best_time is None or completion_time < best_time)
            bottlenecks = _bottleneck_links(sim_result)
            record = self.record_agent.update(
                round_id=round_id,
                validity_status="ok",
                completion_time=completion_time,
                bottleneck_links=bottlenecks,
                candidate_sketch=candidate,
                best_so_far=best_so_far,
            )
            return {
                "combined_score": _score(completion_time),
                "validity_status": "ok",
                "completion_time": completion_time,
                "algorithm_bandwidth": _algorithm_bandwidth(self.message_size, completion_time),
                "expanded_events": int(sim_result.get("flow_count", 0)),
                "critical_path": _critical_path(sim_result),
                "bottleneck_links": bottlenecks,
                "best_so_far": best_so_far,
                "record_summary": record.summary,
                "direction_hint": record.direction_hint,
                "prompt_hash": stable_hash(text),
                "enumeration_status": "skipped",
                "proposal_output": raw_output,
                "code_extract_reason": code_extract_reason,
                "candidate_code_path": _artifact_relpath(candidate_code_path, artifacts_dir),
            }, candidate
        except Exception as exc:
            record = self.record_agent.update(
                round_id=round_id,
                validity_status=f"invalid: {exc}",
                completion_time=FAILURE_COMPLETION_TIME,
                bottleneck_links=[],
                candidate_sketch=candidate,
                best_so_far=False,
            )
            return {
                "combined_score": _invalid_score(),
                "error": str(exc),
                "validity_status": f"invalid: {exc}",
                "completion_time": FAILURE_COMPLETION_TIME,
                "algorithm_bandwidth": None,
                "expanded_events": 0,
                "critical_path": str(exc),
                "bottleneck_links": [],
                "best_so_far": False,
                "record_summary": record.summary,
                "direction_hint": record.direction_hint,
                "enumeration_status": "skipped",
                "proposal_output": raw_output,
                "code_extract_reason": code_extract_reason,
                "candidate_code_path": _artifact_relpath(candidate_code_path, artifacts_dir),
            }, candidate

    def _write_round_record(
        self,
        *,
        artifacts_dir: Path,
        topodsl_config_id: str,
        round_id: int,
        prompt: str,
        llm_result: AgentLLMResult,
        candidate: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> None:
        usage = llm_result.token_usage or {}
        record = {
            "topodsl_config_id": topodsl_config_id,
            "collective": self.collective,
            "message_size": self.message_size,
            "round_id": round_id,
            "prompt_hash": stable_hash(prompt),
            "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
            "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
            "candidate_sketch": candidate,
            "validity_status": metrics.get("validity_status"),
            "expanded_events": metrics.get("expanded_events", 0),
            "combined_score": metrics.get("combined_score"),
            "completion_time": metrics.get("completion_time"),
            "algorithm_bandwidth": metrics.get("algorithm_bandwidth"),
            "critical_path": metrics.get("critical_path"),
            "bottleneck_links": metrics.get("bottleneck_links", []),
            "best_so_far": metrics.get("best_so_far", False),
            "proposal_output": metrics.get("proposal_output"),
            "code_extract_reason": metrics.get("code_extract_reason"),
            "candidate_code_path": metrics.get("candidate_code_path"),
        }
        validate_round_record(record)
        append_jsonl(artifacts_dir / "rounds.jsonl", record)

    def _write_llm_record(
        self,
        *,
        artifacts_dir: Path,
        run_id: str,
        round_id: int,
        agent_type: str,
        prompt: str,
        result: AgentLLMResult,
        save_llm_io: bool,
    ) -> None:
        usage = result.token_usage or {}
        record = {
            "run_id": run_id,
            "agent_type": agent_type,
            "round_id": round_id,
            "model_name": result.model_name,
            "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
            "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
            "cached_input_tokens": usage.get("cached_input_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "wall_clock_time_ms": result.wall_clock_time_ms,
            "api_cost_usd": None,
            "prompt_hash": stable_hash(prompt),
            "completion_hash": stable_hash(result.text),
        }
        if save_llm_io:
            llm_io_dir = artifacts_dir / "llm_io"
            llm_io_dir.mkdir(parents=True, exist_ok=True)
            prompt_rel = Path("llm_io") / f"round-{round_id:04d}-{agent_type}-prompt.txt"
            completion_rel = Path("llm_io") / f"round-{round_id:04d}-{agent_type}-completion.txt"
            raw_rel = Path("llm_io") / f"round-{round_id:04d}-{agent_type}-raw-output.txt"
            (artifacts_dir / prompt_rel).write_text(prompt, encoding="utf-8")
            (artifacts_dir / completion_rel).write_text(result.text, encoding="utf-8")
            (artifacts_dir / raw_rel).write_text(result.raw_output, encoding="utf-8")
            record.update(
                {
                    "prompt_path": prompt_rel.as_posix(),
                    "completion_path": completion_rel.as_posix(),
                    "raw_output_path": raw_rel.as_posix(),
                }
            )
        validate_llm_call_record(record)
        append_jsonl(artifacts_dir / "llm_calls.jsonl", record)

    def _build_record_prompt(
        self,
        *,
        round_id: int,
        candidate: list[dict[str, Any]],
        metrics: dict[str, Any],
        current_summary: str,
        direction_hint: str,
    ) -> str:
        return render_record_prompt(
            round_id=round_id,
            candidate=candidate,
            metrics=metrics,
            current_summary=current_summary,
            direction_hint=direction_hint,
        )

    def _apply_record_agent_output(self, text: str) -> None:
        parsed = _parse_json_object(text)
        if parsed:
            summary = parsed.get("summary")
            direction_hint = parsed.get("direction_hint")
            if isinstance(summary, str) and summary.strip():
                self._append_record_trail(summary)
            if isinstance(direction_hint, str) and direction_hint.strip():
                self.record_agent.direction_hint = direction_hint.strip()
            return

        trail = _extract_record_trail(text)
        if trail:
            self._append_record_trail(trail)

    def _append_record_trail(self, trail: str) -> None:
        self.record_trails.append(trail.strip())
        self.record_agent.summary = self._record_trail_history()

    def _record_trail_history(self) -> str:
        return "\n".join(self.record_trails[-6:])

    def _build_prompt(
        self,
        *,
        topo_source: str,
        topo_summary: str,
        layer_summary: str,
        seed_hint: str,
        enumeration_note: str,
    ) -> str:
        total_gpus = _total_gpus_from_summary(topo_summary)
        return render_proposal_prompt(
            topo_source=topo_source,
            collective=self.collective,
            gpu_num=total_gpus,
            message_size=self.message_size,
            topo_summary=topo_summary,
            layer_summary=layer_summary,
            seed_hint=seed_hint,
            enumeration_note=enumeration_note,
            record_summary=self.record_agent.summary,
            direction_hint=self.record_agent.direction_hint,
        )


async def _call_llm(
    llm: Any,
    prompt: str,
    instance_id: str,
    track_io: bool,
    round_id: int,
    agent_type: str,
    model_name: str,
) -> AgentLLMResult:
    started = time.perf_counter()
    if hasattr(llm, "generate") and not hasattr(llm, "generate_batch"):
        response = llm.generate(prompt, agent_type=agent_type, round_id=round_id)
        wall_ms = int((time.perf_counter() - started) * 1000)
        return AgentLLMResult(
            text=response.text,
            raw_output=getattr(response, "raw_output", None) or response.text,
            token_usage={
                "prompt_tokens": response.input_tokens,
                "completion_tokens": response.output_tokens,
                "cached_input_tokens": response.cached_input_tokens,
                "reasoning_tokens": response.reasoning_tokens,
            },
            model_name=getattr(response, "model_name", model_name),
            wall_clock_time_ms=getattr(response, "wall_clock_time_ms", wall_ms),
        )

    result = await llm.generate(prompt, instance_id=instance_id, track_io=track_io)
    wall_ms = int((time.perf_counter() - started) * 1000)
    return AgentLLMResult(
        text=result.text,
        raw_output=getattr(result, "raw_output", None) or result.text,
        token_usage=getattr(result, "token_usage", None),
        model_name=model_name,
        wall_clock_time_ms=wall_ms,
    )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    for idx, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_record_trail(text: str) -> str:
    lines = text.strip().splitlines()
    start = next((idx for idx, line in enumerate(lines) if line.strip().lower().startswith("trail id")), None)
    if start is None:
        return text.strip()
    trail_lines: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if trail_lines and stripped.lower().startswith("trail id"):
            break
        if stripped:
            trail_lines.append(line.rstrip())
    return "\n".join(trail_lines).strip()


def _run_candidate_code(path: Path) -> list[dict[str, Any]]:
    namespace = runpy.run_path(str(path), init_globals={"__builtins__": __builtins__})
    entrypoint = namespace.get("run_code") or namespace.get("construct_sketches")
    if not callable(entrypoint):
        raise ValueError("candidate program must define run_code() or construct_sketches()")
    raw = entrypoint()
    candidate = _unwrap_candidate_result(raw)
    return [_normalize_candidate_transmission(item, index) for index, item in enumerate(candidate)]


def _unwrap_candidate_result(raw: Any) -> list[Any]:
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        return raw[0]
    return parse_sketch_dsl(repr(raw))


def _normalize_candidate_transmission(item: Any, index: int) -> dict[str, Any]:
    return parse_sketch_dsl(repr([item]))[0]


def _artifact_relpath(path: Path | None, artifacts_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(artifacts_dir).as_posix()
    except ValueError:
        return path.as_posix()


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


def _score(completion_time: float | None) -> float:
    if completion_time is None or completion_time <= 0 or not math.isfinite(completion_time):
        return _invalid_score()
    return -float(completion_time)


def _invalid_score() -> float:
    return FAILURE_SCORE


def _algorithm_bandwidth(message_size: int | None, completion_time_us: float | None) -> float | None:
    if (
        message_size is None
        or completion_time_us is None
        or completion_time_us <= 0
        or not math.isfinite(completion_time_us)
    ):
        return None
    return float(message_size) / completion_time_us


def _bottleneck_links(result: dict[str, Any]) -> list[str]:
    links = result.get("links") or []
    normalized = []
    for link in links:
        wait = int(link.get("queue_wait_ns", 0))
        normalized.append((wait, f"{link.get('src')}->{link.get('dst')}:{wait}ns"))
    normalized.sort(reverse=True)
    return [text for _, text in normalized[:5]]


def _critical_path(result: dict[str, Any]) -> str:
    critical = result.get("critical_flow_id")
    if critical is not None:
        return f"critical_flow_id={critical}"
    return f"finish_time_ns={result.get('finish_time_ns', 'unknown')}"
