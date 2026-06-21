from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROUND_FIELDS = {
    "topodsl_config_id",
    "collective",
    "message_size",
    "round_id",
    "prompt_hash",
    "input_tokens",
    "output_tokens",
    "candidate_sketch",
    "validity_status",
    "expanded_events",
    "combined_score",
    "completion_time",
    "algorithm_bandwidth",
    "critical_path",
    "bottleneck_links",
    "best_so_far",
}


LLM_CALL_FIELDS = {
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


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def validate_round_record(record: dict[str, Any]) -> None:
    missing = ROUND_FIELDS - set(record)
    if missing:
        raise ValueError(f"round record missing fields: {sorted(missing)}")


def validate_llm_call_record(record: dict[str, Any]) -> None:
    missing = LLM_CALL_FIELDS - set(record)
    if missing:
        raise ValueError(f"llm call record missing fields: {sorted(missing)}")
