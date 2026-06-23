from __future__ import annotations

import json
from pathlib import Path
from typing import Any


AGENT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = AGENT_ROOT / "prompt" / "syccl_two_agent"


def load_prompt(name: str) -> str:
    return (PROMPT_ROOT / name).read_text(encoding="utf-8")


def load_init_program() -> str:
    return load_prompt("init_program.py")


def render_proposal_base(
    *,
    topo_source: str,
    collective: str,
    gpu_num: int,
) -> str:
    template = load_prompt("proposal_base.txt")
    rendered = template.replace("$Collective", collective).replace("$GPU_NUM", str(gpu_num))
    empty_block = "topo_dsl:\n```python\n\n```"
    filled_block = "topo_dsl:\n```python\n" + topo_source.rstrip() + "\n```"
    if empty_block in rendered:
        return rendered.replace(empty_block, filled_block)
    return rendered.rstrip() + "\n\n" + filled_block


def render_proposal_dynamic_context(
    *,
    message_size: int,
    topo_summary: str,
    layer_summary: str,
    seed_hint: str,
    enumeration_note: str,
    record_summary: str,
    direction_hint: str,
    init_program_reference: str = "",
    evolve_scaffold: str = "",
) -> str:
    context = load_prompt("proposal_dynamic_context.txt").format(
        message_size=message_size,
        topo_summary=topo_summary,
        layer_summary=layer_summary,
        seed_hint=seed_hint,
        enumeration_note=enumeration_note,
        record_summary=record_summary or "none",
        direction_hint=direction_hint,
    )
    extra_sections = []
    if init_program_reference.strip():
        extra_sections.append(
            "Initial program reference (mirrors SimpleTES root-node inspiration):\n"
            + init_program_reference.strip()
        )
    if evolve_scaffold.strip():
        extra_sections.append(evolve_scaffold.strip())
    if extra_sections:
        return context.rstrip() + "\n\n" + "\n\n".join(extra_sections)
    return context


def render_proposal_prompt(
    *,
    topo_source: str,
    collective: str,
    gpu_num: int,
    message_size: int,
    topo_summary: str,
    layer_summary: str,
    seed_hint: str,
    enumeration_note: str,
    record_summary: str,
    direction_hint: str,
    init_program_reference: str = "",
    evolve_scaffold: str = "",
) -> str:
    base = render_proposal_base(
        topo_source=topo_source,
        collective=collective,
        gpu_num=gpu_num,
    )
    context = render_proposal_dynamic_context(
        message_size=message_size,
        topo_summary=topo_summary,
        layer_summary=layer_summary,
        seed_hint=seed_hint,
        enumeration_note=enumeration_note,
        record_summary=record_summary,
        direction_hint=direction_hint,
        init_program_reference=init_program_reference,
        evolve_scaffold=evolve_scaffold,
    )
    return base.rstrip() + "\n\n" + context


def render_record_prompt(
    *,
    round_id: int,
    candidate: list[dict[str, Any]],
    metrics: dict[str, Any],
    current_summary: str,
    direction_hint: str,
) -> str:
    trail_id = round_id - 1
    required_metrics = {
        "validity_status",
        "combined_score",
        "completion_time",
        "expanded_events",
        "bottleneck_profile",
        "best_so_far",
    }
    missing = sorted(field for field in required_metrics if field not in metrics or metrics[field] is None)
    if missing:
        raise ValueError(f"record prompt metrics missing required fields: {missing}")
    if not isinstance(metrics["bottleneck_profile"], dict):
        raise ValueError("record prompt metric bottleneck_profile must be a dict")
    payload = {
        "round_id": round_id,
        "trail_id": trail_id,
        "candidate_sketch": candidate,
        "validity_status": metrics["validity_status"],
        "combined_score": metrics["combined_score"],
        "completion_time": metrics["completion_time"],
        "expanded_events": metrics["expanded_events"],
        "bottleneck_profile": metrics["bottleneck_profile"],
        "best_so_far": metrics["best_so_far"],
        "proposal_output": metrics.get("proposal_output"),
        "code_extract_reason": metrics.get("code_extract_reason"),
        "candidate_code_path": metrics.get("candidate_code_path"),
        "current_summary": current_summary or "none",
        "current_direction_hint": direction_hint,
    }
    return load_prompt("record_agent.txt").format(
        trail_id=trail_id,
        proposal_output=str(metrics.get("proposal_output") or ""),
        payload_json=json.dumps(payload, indent=2),
    )
