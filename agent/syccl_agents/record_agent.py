from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RecordUpdate:
    summary: str
    direction_hint: str


@dataclass
class RecordAgent:
    stagnation_rounds: int = 20
    summary: str = ""
    direction_hint: str = "try a more different transfer stage order or redistribute traffic away from recent bottlenecks"
    best_time: float | None = None
    stale_rounds: int = 0
    recent: list[str] = field(default_factory=list)

    def update(
        self,
        *,
        round_id: int,
        validity_status: str,
        completion_time: float | None,
        bottleneck_profile: dict[str, Any] | None,
        candidate_sketch: Any,
        best_so_far: bool,
    ) -> RecordUpdate:
        if best_so_far and completion_time is not None:
            self.best_time = completion_time
            self.stale_rounds = 0
        else:
            self.stale_rounds += 1

        status = "best" if best_so_far else validity_status
        bottlenecks = _profile_summary(bottleneck_profile)
        stages = len(candidate_sketch) if isinstance(candidate_sketch, list) else 0
        line = (
            f"round {round_id}: {status}, time={completion_time}, stages={stages}, "
            f"sketch_bottlenecks={bottlenecks}"
        )
        self.recent.append(line)
        self.recent = self.recent[:]
        self.summary = "\n".join(self.recent)

        if self.stale_rounds >= self.stagnation_rounds:
            self.direction_hint = (
                "Search stagnated; try a structurally different stage order or redistribute traffic away from "
                f"recent sketch transmission bottlenecks ({bottlenecks})."
            )
            self.stale_rounds = 0

        return RecordUpdate(summary=self.summary, direction_hint=self.direction_hint)


def _profile_summary(profile: dict[str, Any] | None) -> str:
    if not isinstance(profile, dict):
        return "none"
    top = profile.get("top_transmission_bottlenecks")
    if isinstance(top, list) and top:
        parts = []
        for item in top[:3]:
            if not isinstance(item, dict):
                continue
            index = item.get("transmission_index")
            step = item.get("step")
            layer = item.get("layer")
            group = item.get("group")
            delay = item.get("delay_ns")
            parts.append(f"tx{index} step {step} layer {layer} group {group}: {delay}ns")
        if parts:
            return ", ".join(parts)
    diagnosis = profile.get("diagnosis")
    if isinstance(diagnosis, str) and diagnosis.strip():
        return diagnosis.strip()
    status = profile.get("status")
    return str(status) if status is not None else "none"
