from __future__ import annotations

import json
import math
import runpy
import time
import uuid
import inspect
import asyncio
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from simpletes.evaluator import rich_print
from simpletes.engine.runtime import RuntimeBase
from simpletes.node import Node, Status, extract_code_detailed, validate_node_for_db
from simpletes.utils.log import format_log
from syccl_agents.config_render import (
    layer_group_summary,
    seed_sketch_hint,
    topology_summary,
    write_syccl_config,
)
from syccl_agents.flow_sim import FlowSimOutputError, FlowSimRunner, validate_flow_sim_result
from syccl_agents.record_agent import RecordAgent
from syccl_agents.records import (
    append_jsonl,
    stable_hash,
    validate_llm_call_record,
    validate_round_record,
)
from syccl_agents.prompts import render_proposal_prompt, render_record_prompt
from syccl_agents.sketch_dsl import parse_sketch_dsl, write_compact_sketch
from syccl_agents.topodsl import TopologyParams, load_topodsl


FlowSimFn = Callable[..., dict[str, Any]]
FAILURE_SCORE = -1_000_000_000_000.0
FAILURE_COMPLETION_TIME = float("inf")
MAX_RECORD_BACKLOG = 5
MAX_RECORD_ATTEMPTS = 2


class MissingBottleneckProfileError(RuntimeError):
    pass


class CandidateInvalidError(ValueError):
    pass


@dataclass(frozen=True)
class AgentLLMResult:
    text: str
    raw_output: str
    token_usage: dict[str, int | None] | None
    model_name: str
    wall_clock_time_ms: int


class RecordAgentOutputError(RuntimeError):
    def __init__(self, message: str, *, attempts: tuple[AgentLLMResult, ...] = (), prompt: str = "") -> None:
        super().__init__(message)
        self.attempts = attempts
        self.prompt = prompt


@dataclass(frozen=True)
class PendingRecordRequest:
    round_id: int
    prompt: str


@dataclass(frozen=True)
class CompletedRecord:
    request: PendingRecordRequest
    parsed: dict[str, str]
    attempts: tuple[AgentLLMResult, ...]


@dataclass(frozen=True)
class CandidateEvaluation:
    metrics: dict[str, Any]
    candidate: list[dict[str, Any]]
    candidate_code: str | None


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
        return (
            base_info
            + "\n[dim]SyCCL two-agent runtime:[/dim] [cyan]ON[/cyan]"
            + "\n[dim]Runtime note:[/dim] SyCCL overrides SimpleTES scheduler/evaluator loop; "
            "SimpleTES queues/policy settings are infrastructure only."
        )

    async def run(self, engine) -> None:
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

        async with engine._db_lock:
            is_empty = len(engine.db.nodes) == 0
        if is_empty:
            await self._initialize_syccl_root(
                engine=engine,
                config_path=config_path,
                artifacts_dir=artifacts_dir,
                flow_sim=flow_sim,
                topo_params=topo.params,
            )

        proposal_llm = self.llm if self.llm is not None else engine.generator._llm
        record_llm = proposal_llm
        root_node = await self._root_node(engine)
        root_parent_id = root_node.id if root_node is not None else engine.best_node_id
        initial_time = _metric_float(root_node.metrics, "completion_time") if root_node else None
        best_time: float | None = initial_time
        best_sketch: list[dict[str, Any]] | None = None
        if root_node is not None:
            initial_sketch = root_node.metrics.get("candidate_sketch") if root_node.metrics else None
            if isinstance(initial_sketch, list):
                best_sketch = initial_sketch
                write_compact_sketch(best_sketch, artifacts_dir / "best_sketch.json")
        best_source = "initial" if best_time is not None and math.isfinite(best_time) else None
        best_source_round = 0 if best_source == "initial" else None
        rounds_completed = 0
        run_id = f"syccl-two-agent-{engine.instance_id}"
        enumeration_note = (
            "Sketch enumeration: skipped because syccl-sketch-search currently emits JSON, "
            "not compact DSL suitable for LLM prompt parsing."
        )
        record_tasks: dict[int, asyncio.Task[CompletedRecord]] = {}
        completed_records: dict[int, CompletedRecord] = {}
        skipped_record_rounds: set[int] = set()
        next_record_round_to_apply = 1
        record_tasks_started = 0
        record_tasks_applied = 0
        record_tasks_cancelled = 0
        record_tasks_failed = 0
        proposal_wait_for_record_backlog_events = 0
        proposal_wait_for_record_backlog_ms_total = 0.0
        proposal_wait_for_record_backlog_ms_max = 0.0
        init_program_reference = await self._initial_program_reference(engine)

        for round_id in range(1, engine.config.max_generations + 1):
            await asyncio.sleep(0)
            next_record_round_to_apply, applied, failed, _retried = self._drain_completed_record_tasks(
                record_tasks=record_tasks,
                completed_records=completed_records,
                skipped_record_rounds=skipped_record_rounds,
                next_record_round_to_apply=next_record_round_to_apply,
                artifacts_dir=artifacts_dir,
                run_id=run_id,
                save_llm_io=engine.config.save_llm_io,
            )
            record_tasks_applied += applied
            record_tasks_failed += failed
            prompt = self._build_prompt(
                topo_source=topo.prompt_source,
                topo_summary=topology_summary(topo.params),
                layer_summary=layer_group_summary(topo.params),
                seed_hint=seed_sketch_hint(topo.params),
                enumeration_note=enumeration_note,
                init_program_reference=init_program_reference,
                evolve_context=engine._evolve_context,
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
            evaluation = self._evaluate_candidate(
                round_id=round_id,
                text=llm_result.text,
                raw_output=llm_result.raw_output,
                evolve_context=engine._evolve_context,
                config_path=config_path,
                artifacts_dir=artifacts_dir,
                flow_sim=flow_sim,
                best_time=best_time,
            )
            metrics = evaluation.metrics
            candidate = evaluation.candidate
            record_prompt = self._build_record_prompt(
                round_id=round_id,
                candidate=candidate,
                metrics=metrics,
                current_summary=self._record_trail_history(),
                direction_hint=self.record_agent.direction_hint,
            )
            next_record_round_to_apply, applied, failed, _retried, wait_ms = await self._wait_for_record_backlog_slot(
                record_tasks=record_tasks,
                completed_records=completed_records,
                skipped_record_rounds=skipped_record_rounds,
                next_record_round_to_apply=next_record_round_to_apply,
                artifacts_dir=artifacts_dir,
                run_id=run_id,
                save_llm_io=engine.config.save_llm_io,
            )
            record_tasks_applied += applied
            record_tasks_failed += failed
            if wait_ms > 0:
                proposal_wait_for_record_backlog_events += 1
                proposal_wait_for_record_backlog_ms_total += wait_ms
                proposal_wait_for_record_backlog_ms_max = max(proposal_wait_for_record_backlog_ms_max, wait_ms)
            next_record_round_to_apply, applied, failed, _retried = self._drain_completed_record_tasks(
                record_tasks=record_tasks,
                completed_records=completed_records,
                skipped_record_rounds=skipped_record_rounds,
                next_record_round_to_apply=next_record_round_to_apply,
                artifacts_dir=artifacts_dir,
                run_id=run_id,
                save_llm_io=engine.config.save_llm_io,
            )
            record_tasks_applied += applied
            record_tasks_failed += failed
            record_tasks[round_id] = asyncio.create_task(
                self._run_record_task(
                    record_llm=record_llm,
                    request=PendingRecordRequest(round_id=round_id, prompt=record_prompt),
                    instance_id=engine.instance_id,
                    model_name=engine.config.model,
                    artifacts_dir=artifacts_dir,
                )
            )
            await asyncio.sleep(0)
            record_tasks_started += 1

            if metrics.get("validity_status") == "ok":
                completion_time = metrics.get("completion_time")
                if completion_time is not None and math.isfinite(completion_time) and (best_time is None or completion_time < best_time):
                    best_time = float(completion_time)
                    best_sketch = candidate
                    best_source = "proposal"
                    best_source_round = round_id
                    write_compact_sketch(candidate, artifacts_dir / "best_sketch.json")

            node = Node(
                id=uuid.uuid4().hex,
                code=_node_code_from_evaluation(evaluation, raw_output=llm_result.raw_output),
                parent_ids=[root_parent_id] if root_parent_id else [],
                gen_id=round_id - 1,
                chain_idx=0,
                metrics=metrics,
                score=metrics["combined_score"],
                status=Status.DONE,
            )
            if engine.config.save_llm_io:
                node.llm_input = prompt
                node.llm_output = llm_result.raw_output
                node.token_usage = llm_result.token_usage

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
            async with engine._counter_lock:
                engine.generation_attempts += 1
            rounds_completed += 1

        next_record_round_to_apply, applied, failed, _retried = self._drain_completed_record_tasks(
            record_tasks=record_tasks,
            completed_records=completed_records,
            skipped_record_rounds=skipped_record_rounds,
            next_record_round_to_apply=next_record_round_to_apply,
            artifacts_dir=artifacts_dir,
            run_id=run_id,
            save_llm_io=engine.config.save_llm_io,
        )
        record_tasks_applied += applied
        record_tasks_failed += failed
        for round_id, task in record_tasks.items():
            if task.done() and task.exception() is not None:
                skipped_record_rounds.add(round_id)
                record_tasks_failed += 1
        for task in record_tasks.values():
            if not task.done():
                task.cancel()
                record_tasks_cancelled += 1
        if record_tasks:
            await asyncio.gather(*record_tasks.values(), return_exceptions=True)

        summary = {
            "run_id": run_id,
            "rounds_completed": rounds_completed,
            "best_time_us": best_time,
            "best_sketch_stages": len(best_sketch or []),
            "best_source": best_source,
            "best_source_round": best_source_round,
            "initial_time_us": initial_time,
            "initial_combined_score": root_node.score if root_node else None,
            "initial_validity_status": (root_node.metrics or {}).get("validity_status") if root_node else None,
            "enumeration_status": "skipped",
            "enumeration_note": enumeration_note,
            "record_tasks_started": record_tasks_started,
            "record_tasks_applied": record_tasks_applied,
            "record_tasks_cancelled": record_tasks_cancelled,
            "record_tasks_failed": record_tasks_failed,
            "record_tasks_retried": _count_record_retries(artifacts_dir),
            "max_record_backlog": MAX_RECORD_BACKLOG,
            "proposal_wait_for_record_backlog_events": proposal_wait_for_record_backlog_events,
            "proposal_wait_for_record_backlog_ms_total": round(proposal_wait_for_record_backlog_ms_total, 3),
            "proposal_wait_for_record_backlog_ms_max": round(proposal_wait_for_record_backlog_ms_max, 3),
        }
        (artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        await engine._finalize_run()

    async def _run_record_task(
        self,
        *,
        record_llm: Any,
        request: PendingRecordRequest,
        instance_id: str,
        model_name: str,
        artifacts_dir: Path,
    ) -> CompletedRecord:
        attempts: list[AgentLLMResult] = []
        for attempt in range(1, MAX_RECORD_ATTEMPTS + 1):
            result = await _call_llm(
                record_llm,
                request.prompt,
                instance_id,
                track_io=True,
                round_id=request.round_id,
                agent_type="record",
                model_name=model_name,
            )
            attempts.append(result)
            try:
                parsed = _parse_record_agent_output(result.text)
                return CompletedRecord(
                    request=request,
                    parsed=parsed,
                    attempts=tuple(attempts),
                )
            except RecordAgentOutputError as exc:
                _write_record_error(
                    artifacts_dir=artifacts_dir,
                    round_id=request.round_id,
                    attempt=attempt,
                    exc=exc,
                    will_retry=attempt < MAX_RECORD_ATTEMPTS,
                )
                if attempt >= MAX_RECORD_ATTEMPTS:
                    raise RecordAgentOutputError(str(exc), attempts=tuple(attempts), prompt=request.prompt) from exc
        raise AssertionError("unreachable record retry loop exit")

    async def _initialize_syccl_root(
        self,
        *,
        engine: Any,
        config_path: Path,
        artifacts_dir: Path,
        flow_sim: FlowSimRunner,
        topo_params: TopologyParams,
    ) -> None:
        init_code = Path(engine.config.init_program).read_text(encoding="utf-8")
        init_code_path = artifacts_dir / "initial-candidate.py"
        init_code_path.write_text(init_code, encoding="utf-8")
        candidate = _run_initial_candidate_code(init_code_path, topo_params)
        sketch_path = write_compact_sketch(candidate, artifacts_dir / "initial-sketch.json")
        sim_output_path = artifacts_dir / "initial-flow-sim.json"
        sim_result = (
            self.flow_sim_fn(config_path=config_path, sketch_path=sketch_path, output_path=sim_output_path)
            if self.flow_sim_fn
            else flow_sim.simulate_sketch(
                config_path=config_path,
                sketch_path=sketch_path,
                output_path=sim_output_path,
            )
        )
        sim_result = validate_flow_sim_result(sim_result, output_path=sim_output_path)
        completion_time = float(sim_result["time_us"])
        bottleneck_profile = _bottleneck_profile(sim_result, candidate)
        metrics = {
            "combined_score": _score(completion_time),
            "validity_status": "ok",
            "completion_time": completion_time,
            "algorithm_bandwidth": _algorithm_bandwidth(self.message_size, completion_time),
            "expanded_events": int(sim_result["flow_count"]),
            "bottleneck_profile": bottleneck_profile,
            "best_so_far": True,
            "candidate_sketch": candidate,
            "candidate_code_path": _artifact_relpath(init_code_path, artifacts_dir),
            "enumeration_status": "initial",
        }
        node = Node(
            id=uuid.uuid4().hex,
            code=init_code,
            parent_ids=[],
            gen_id=None,
            chain_idx=0,
            metrics=metrics,
            score=metrics["combined_score"],
            status=Status.DONE,
        )
        validate_node_for_db(node)
        async with engine._db_lock:
            engine.db.add(node)
            engine.best_score = node.score
            engine.best_node_id = node.id
            engine.completed_evaluations += 1
            for chain_idx in engine._chain_best_scores:
                engine._chain_best_scores[chain_idx] = node.score
        rich_print(
            engine._log(
                "",
                f"[bold]Initial SyCCL score:[/bold] [green]{node.score:.6f}[/green] "
                f"[dim](time_us={completion_time})[/dim]",
            )
        )

    async def _wait_for_record_backlog_slot(
        self,
        *,
        record_tasks: dict[int, asyncio.Task[CompletedRecord]],
        completed_records: dict[int, CompletedRecord],
        skipped_record_rounds: set[int],
        next_record_round_to_apply: int,
        artifacts_dir: Path,
        run_id: str,
        save_llm_io: bool,
    ) -> tuple[int, int, int, int, float]:
        total_applied = 0
        total_failed = 0
        total_retried = 0
        wait_started_at: float | None = None
        while len(record_tasks) + len(completed_records) >= MAX_RECORD_BACKLOG:
            if not record_tasks:
                break
            if wait_started_at is None:
                wait_started_at = time.time()
            done, _ = await asyncio.wait(record_tasks.values(), return_when=asyncio.FIRST_COMPLETED)
            del done
            next_record_round_to_apply, applied, failed, retried = self._drain_completed_record_tasks(
                record_tasks=record_tasks,
                completed_records=completed_records,
                skipped_record_rounds=skipped_record_rounds,
                next_record_round_to_apply=next_record_round_to_apply,
                artifacts_dir=artifacts_dir,
                run_id=run_id,
                save_llm_io=save_llm_io,
            )
            total_applied += applied
            total_failed += failed
            total_retried += retried
        wait_ms = 0.0 if wait_started_at is None else (time.time() - wait_started_at) * 1000.0
        return next_record_round_to_apply, total_applied, total_failed, total_retried, wait_ms

    def _drain_completed_record_tasks(
        self,
        *,
        record_tasks: dict[int, asyncio.Task[CompletedRecord]],
        completed_records: dict[int, CompletedRecord],
        skipped_record_rounds: set[int],
        next_record_round_to_apply: int,
        artifacts_dir: Path,
        run_id: str,
        save_llm_io: bool,
    ) -> tuple[int, int, int, int]:
        failed = 0
        for round_id, task in list(record_tasks.items()):
            if not task.done():
                continue
            del record_tasks[round_id]
            try:
                completed_records[round_id] = task.result()
            except asyncio.CancelledError:
                skipped_record_rounds.add(round_id)
                continue
            except Exception as exc:
                attempts = getattr(exc, "attempts", ())
                if attempts:
                    prompt = getattr(exc, "prompt", "")
                    for attempt_index, result in enumerate(attempts, start=1):
                        self._write_llm_record(
                            artifacts_dir=artifacts_dir,
                            run_id=run_id,
                            round_id=round_id,
                            agent_type="record",
                            prompt=prompt,
                            result=result,
                            save_llm_io=save_llm_io,
                            attempt=attempt_index,
                        )
                if not isinstance(exc, RecordAgentOutputError):
                    _write_record_error(
                        artifacts_dir=artifacts_dir,
                        round_id=round_id,
                        attempt=0,
                        exc=exc,
                        will_retry=False,
                    )
                skipped_record_rounds.add(round_id)
                failed += 1
                continue

        applied = 0
        retried = 0
        while next_record_round_to_apply in completed_records or next_record_round_to_apply in skipped_record_rounds:
            if next_record_round_to_apply in skipped_record_rounds:
                skipped_record_rounds.remove(next_record_round_to_apply)
                next_record_round_to_apply += 1
                continue
            completed = completed_records.pop(next_record_round_to_apply)
            for attempt_index, result in enumerate(completed.attempts, start=1):
                self._write_llm_record(
                    artifacts_dir=artifacts_dir,
                    run_id=run_id,
                    round_id=completed.request.round_id,
                    agent_type="record",
                    prompt=completed.request.prompt,
                    result=result,
                    save_llm_io=save_llm_io,
                    attempt=attempt_index,
                )
            self._apply_record_agent_output(completed.parsed)
            retried += max(0, len(completed.attempts) - 1)
            next_record_round_to_apply += 1
            applied += 1
        return next_record_round_to_apply, applied, failed, retried

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
    ) -> CandidateEvaluation:
        candidate: list[dict[str, Any]] = []
        candidate_code_path: Path | None = None
        code_extract_reason: str | None = None
        candidate_code: str | None = None
        try:
            candidate_code, code_extract_reason = extract_code_detailed(text, evolve_context)
            if candidate_code is None:
                raise CandidateInvalidError(code_extract_reason)
            candidate_code_path = artifacts_dir / "rounds" / f"round-{round_id:04d}-candidate.py"
            candidate_code_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_code_path.write_text(candidate_code, encoding="utf-8")
            try:
                candidate = _run_candidate_code(candidate_code_path)
            except Exception as exc:
                raise CandidateInvalidError(str(exc)) from exc
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
            sim_result = validate_flow_sim_result(sim_result, output_path=sim_output_path)
            completion_time = float(sim_result["time_us"])
            best_so_far = math.isfinite(completion_time) and (best_time is None or completion_time < best_time)
            bottleneck_profile = _bottleneck_profile(sim_result, candidate)
            metrics = {
                "combined_score": _score(completion_time),
                "validity_status": "ok",
                "completion_time": completion_time,
                "algorithm_bandwidth": _algorithm_bandwidth(self.message_size, completion_time),
                "expanded_events": int(sim_result["flow_count"]),
                "bottleneck_profile": bottleneck_profile,
                "best_so_far": best_so_far,
                "record_summary": self.record_agent.summary,
                "direction_hint": self.record_agent.direction_hint,
                "prompt_hash": stable_hash(text),
                "enumeration_status": "skipped",
                "proposal_output": raw_output,
                "code_extract_reason": code_extract_reason,
                "candidate_code_path": _artifact_relpath(candidate_code_path, artifacts_dir),
            }
            return CandidateEvaluation(metrics=metrics, candidate=candidate, candidate_code=candidate_code)
        except (MissingBottleneckProfileError, FlowSimOutputError, OSError):
            raise
        except CandidateInvalidError as exc:
            metrics = {
                "combined_score": _invalid_score(),
                "error": str(exc),
                "validity_status": f"invalid: {exc}",
                "completion_time": FAILURE_COMPLETION_TIME,
                "algorithm_bandwidth": None,
                "expanded_events": 0,
                "bottleneck_profile": _invalid_bottleneck_profile(str(exc)),
                "best_so_far": False,
                "record_summary": self.record_agent.summary,
                "direction_hint": self.record_agent.direction_hint,
                "enumeration_status": "skipped",
                "proposal_output": raw_output,
                "code_extract_reason": code_extract_reason,
                "candidate_code_path": _artifact_relpath(candidate_code_path, artifacts_dir),
            }
            return CandidateEvaluation(metrics=metrics, candidate=candidate, candidate_code=candidate_code)

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
        required_metrics = (
            "validity_status",
            "expanded_events",
            "combined_score",
            "completion_time",
            "algorithm_bandwidth",
            "bottleneck_profile",
            "best_so_far",
        )
        missing_metrics = [field for field in required_metrics if field not in metrics]
        if missing_metrics:
            raise ValueError(f"round metrics missing required fields: {missing_metrics}")
        record = {
            "topodsl_config_id": topodsl_config_id,
            "collective": self.collective,
            "message_size": self.message_size,
            "round_id": round_id,
            "prompt_hash": stable_hash(prompt),
            "input_tokens": _required_token(usage, "prompt_tokens", "input_tokens"),
            "output_tokens": _required_token(usage, "completion_tokens", "output_tokens"),
            "candidate_sketch": candidate,
            "validity_status": metrics["validity_status"],
            "expanded_events": metrics["expanded_events"],
            "combined_score": metrics["combined_score"],
            "completion_time": metrics["completion_time"],
            "algorithm_bandwidth": metrics["algorithm_bandwidth"],
            "bottleneck_profile": metrics["bottleneck_profile"],
            "best_so_far": metrics["best_so_far"],
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
        attempt: int | None = None,
    ) -> None:
        usage = result.token_usage or {}
        record = {
            "run_id": run_id,
            "agent_type": agent_type,
            "round_id": round_id,
            "model_name": result.model_name,
            "input_tokens": _required_token(usage, "prompt_tokens", "input_tokens"),
            "output_tokens": _required_token(usage, "completion_tokens", "output_tokens"),
            "cached_input_tokens": usage.get("cached_input_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "wall_clock_time_ms": result.wall_clock_time_ms,
            "api_cost_usd": None,
            "prompt_hash": stable_hash(prompt),
            "completion_hash": stable_hash(result.text),
        }
        if attempt is not None:
            record["attempt"] = attempt
        if save_llm_io:
            llm_io_dir = artifacts_dir / "llm_io"
            llm_io_dir.mkdir(parents=True, exist_ok=True)
            attempt_suffix = f"-attempt-{attempt:02d}" if attempt is not None else ""
            prompt_rel = Path("llm_io") / f"round-{round_id:04d}-{agent_type}{attempt_suffix}-prompt.txt"
            completion_rel = Path("llm_io") / f"round-{round_id:04d}-{agent_type}{attempt_suffix}-completion.txt"
            raw_rel = Path("llm_io") / f"round-{round_id:04d}-{agent_type}{attempt_suffix}-raw-output.txt"
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

    def _apply_record_agent_output(self, parsed: dict[str, str]) -> None:
        self._append_record_trail(parsed["summary"])
        direction_hint = parsed.get("direction_hint")
        if direction_hint:
            self.record_agent.direction_hint = direction_hint

    def _append_record_trail(self, trail: str) -> None:
        self.record_trails.append(trail.strip())
        self.record_agent.summary = self._record_trail_history()

    def _record_trail_history(self) -> str:
        return "\n".join(self.record_trails[-6:])

    async def _initial_program_reference(self, engine) -> str:
        async with engine._db_lock:
            root = next((node for node in engine.db.nodes.values() if not node.parent_ids), None)
            if root is None:
                return ""
            return _format_initial_program_reference(root)

    async def _root_node(self, engine) -> Node | None:
        async with engine._db_lock:
            return next((node for node in engine.db.nodes.values() if not node.parent_ids), None)

    def _build_prompt(
        self,
        *,
        topo_source: str,
        topo_summary: str,
        layer_summary: str,
        seed_hint: str,
        enumeration_note: str,
        init_program_reference: str,
        evolve_context: Any,
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
            init_program_reference=init_program_reference,
            evolve_scaffold=_format_evolve_scaffold(evolve_context),
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
        if inspect.isawaitable(response):
            response = await response
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


def _required_token(usage: dict[str, Any], *fields: str) -> int:
    for field in fields:
        value = usage.get(field)
        if value is not None:
            return value
    raise ValueError(f"LLM token usage missing any of {fields}")


def _metric_float(metrics: dict[str, Any] | None, field: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(field)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _node_code_from_evaluation(evaluation: CandidateEvaluation, *, raw_output: str) -> str:
    if evaluation.candidate_code and evaluation.candidate_code.strip():
        return evaluation.candidate_code
    reason = evaluation.metrics.get("code_extract_reason") or evaluation.metrics.get("validity_status") or "invalid proposal"
    return (
        "# Invalid SyCCL proposal; no executable evolve-block candidate was extracted.\n"
        f"# Reason: {reason}\n"
        "# Raw LLM output follows for checkpoint/debug inspection.\n"
        '"""\n'
        f"{raw_output.rstrip()}\n"
        '"""'
    )


def _format_evolve_scaffold(evolve_context: Any) -> str:
    return (
        "Generation instruction (must follow exactly):\n"
        f"1) Only the code between `{evolve_context.start_marker_line}` and "
        f"`{evolve_context.end_marker_line}` is extracted.\n"
        "2) The final program is reconstructed as EXACT_PREFIX + evolved_block + EXACT_SUFFIX.\n"
        "3) Keep marker lines exactly as written.\n"
        "4) Return one Python code block that includes both EVOLVE-BLOCK markers.\n\n"
        "EXACT_PREFIX (kept unchanged):\n"
        f"```python\n{evolve_context.prefix.rstrip(chr(10))}\n```\n\n"
        "EXACT_SUFFIX (kept unchanged):\n"
        f"```python\n{evolve_context.suffix.rstrip(chr(10))}\n```"
    )


def _format_initial_program_reference(node: Node) -> str:
    metrics_lines = []
    for key, value in (node.metrics or {}).items():
        if isinstance(value, float):
            metrics_lines.append(f"  {key}: {value:.6f}")
        else:
            metrics_lines.append(f"  {key}: {value}")
    metrics_text = "\n".join(metrics_lines)
    score = node.score if node.score is not None else float("-inf")
    return (
        "--- Inspiration 1 ---\n"
        f"Score: {score}\n"
        "Metrics:\n"
        f"{metrics_text}\n"
        "Code:\n"
        f"```python\n{node.code.rstrip()}\n```"
    )


def _parse_record_agent_output(text: str) -> dict[str, str]:
    try:
        from json_repair import loads as repair_json_loads
    except ModuleNotFoundError as exc:
        raise RecordAgentOutputError(
            "json_repair is required to parse record agent output; install the json-repair package"
        ) from exc
    try:
        parsed = repair_json_loads(text)
    except Exception as exc:
        raise RecordAgentOutputError(f"record agent output is not repairable JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RecordAgentOutputError("record agent output must repair to a JSON object")
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise RecordAgentOutputError("record agent output missing non-empty string field summary")
    direction_hint = parsed.get("direction_hint")
    if direction_hint is None:
        direction_hint = ""
    if not isinstance(direction_hint, str):
        raise RecordAgentOutputError("record agent output field direction_hint must be a string when present")
    return {"summary": summary.strip(), "direction_hint": direction_hint.strip()}


def _write_record_error(
    *,
    artifacts_dir: Path,
    round_id: int,
    attempt: int,
    exc: Exception,
    will_retry: bool,
) -> None:
    record = {
        "round_id": round_id,
        "attempt": attempt,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "will_retry": will_retry,
    }
    append_jsonl(artifacts_dir / "record_errors.jsonl", record)
    action = "retrying" if will_retry else "giving up"
    rich_print(
        format_log(
            "!",
            f"[yellow]Record agent output failed for round {round_id} attempt {attempt}; {action}: {type(exc).__name__}: {exc}[/yellow]",
        )
    )


def _count_record_retries(artifacts_dir: Path) -> int:
    errors_path = artifacts_dir / "record_errors.jsonl"
    if not errors_path.exists():
        return 0
    retries = 0
    for line in errors_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("will_retry") is True:
            retries += 1
    return retries


def _run_candidate_code(path: Path) -> list[dict[str, Any]]:
    namespace = runpy.run_path(str(path), init_globals={"__builtins__": __builtins__})
    entrypoint = namespace.get("run_code") or namespace.get("construct_sketches")
    if not callable(entrypoint):
        raise ValueError("candidate program must define run_code() or construct_sketches()")
    raw = entrypoint()
    candidate = _unwrap_candidate_result(raw)
    return [_normalize_candidate_transmission(item, index) for index, item in enumerate(candidate)]


def _run_initial_candidate_code(path: Path, topo_params: TopologyParams) -> list[dict[str, Any]]:
    namespace = runpy.run_path(str(path), init_globals={"__builtins__": __builtins__})
    construct = namespace.get("construct_sketches")
    entrypoint = namespace.get("run_code") or construct
    if not callable(entrypoint):
        raise ValueError("candidate program must define run_code() or construct_sketches()")
    if callable(construct) and _callable_accepts_kwargs(
        construct,
        {"ngpus", "root_gpu", "gpus_per_host", "hosts_per_leaf"},
    ):
        raw = construct(**_initial_construct_kwargs(topo_params))
    else:
        raw = entrypoint()
    candidate = _unwrap_candidate_result(raw)
    return [_normalize_candidate_transmission(item, index) for index, item in enumerate(candidate)]


def _initial_construct_kwargs(topo_params: TopologyParams) -> dict[str, int]:
    hosts_per_leaf = topo_params.hosts
    if topo_params.leaf_switches:
        hosts_per_leaf = max(1, topo_params.hosts // topo_params.leaf_switches)
    return {
        "ngpus": topo_params.hosts * topo_params.gpus_per_host,
        "root_gpu": 0,
        "gpus_per_host": topo_params.gpus_per_host,
        "hosts_per_leaf": hosts_per_leaf,
    }


def _callable_accepts_kwargs(fn: Callable[..., Any], names: set[str]) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    partial = functools.partial(fn)
    try:
        inspect.signature(partial).bind_partial(**{name: 0 for name in names if name in accepted})
    except TypeError:
        return False
    return names <= accepted


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
