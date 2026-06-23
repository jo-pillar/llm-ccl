from __future__ import annotations

import hashlib
import json
import math
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
    "bottleneck_profile",
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
    _require_str(record, "topodsl_config_id")
    _require_str(record, "collective")
    _require_int(record, "message_size", minimum=1)
    _require_int(record, "round_id", minimum=1)
    _require_str(record, "prompt_hash")
    _require_int(record, "input_tokens", minimum=0)
    _require_int(record, "output_tokens", minimum=0)
    if not isinstance(record["candidate_sketch"], list):
        raise ValueError("round record field candidate_sketch must be a list")
    _require_str(record, "validity_status")
    _require_int(record, "expanded_events", minimum=0)
    _require_number(record, "combined_score")
    _require_number(record, "completion_time", allow_inf=True)
    if record["algorithm_bandwidth"] is not None:
        _require_number(record, "algorithm_bandwidth")
    profile = record["bottleneck_profile"]
    if not isinstance(profile, dict):
        raise ValueError("round record field bottleneck_profile must be a dict")
    status = profile.get("status")
    if status not in {"ok", "invalid"}:
        raise ValueError(f"round record bottleneck_profile.status must be ok or invalid, got {status!r}")
    if not isinstance(record["best_so_far"], bool):
        raise ValueError("round record field best_so_far must be a bool")


def validate_llm_call_record(record: dict[str, Any]) -> None:
    missing = LLM_CALL_FIELDS - set(record)
    if missing:
        raise ValueError(f"llm call record missing fields: {sorted(missing)}")
    _require_str(record, "run_id")
    if record["agent_type"] not in {"proposal", "record"}:
        raise ValueError(f"llm call record agent_type must be proposal or record, got {record['agent_type']!r}")
    _require_int(record, "round_id", minimum=1)
    _require_str(record, "model_name")
    _require_int(record, "input_tokens", minimum=0)
    _require_int(record, "output_tokens", minimum=0)
    for field in ("cached_input_tokens", "reasoning_tokens"):
        if record[field] is not None:
            _require_int(record, field, minimum=0)
    _require_int(record, "wall_clock_time_ms", minimum=0)
    if record["api_cost_usd"] is not None:
        _require_number(record, "api_cost_usd")
    _require_str(record, "prompt_hash")
    _require_str(record, "completion_hash")
    if "attempt" in record:
        _require_int(record, "attempt", minimum=1)


def _require_str(record: dict[str, Any], field: str) -> None:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record field {field} must be a non-empty string")


def _require_int(record: dict[str, Any], field: str, *, minimum: int | None = None) -> None:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"record field {field} must be an int")
    if minimum is not None and value < minimum:
        raise ValueError(f"record field {field} must be >= {minimum}")


def _require_number(record: dict[str, Any], field: str, *, allow_inf: bool = False) -> None:
    value = record.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"record field {field} must be a number")
    numeric = float(value)
    if math.isnan(numeric):
        raise ValueError(f"record field {field} must not be NaN")
    if not allow_inf and not math.isfinite(numeric):
        raise ValueError(f"record field {field} must be finite")
