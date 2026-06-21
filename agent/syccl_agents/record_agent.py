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
        bottleneck_links: list[str],
        candidate_sketch: Any,
        best_so_far: bool,
    ) -> RecordUpdate:
        if best_so_far and completion_time is not None:
            self.best_time = completion_time
            self.stale_rounds = 0
        else:
            self.stale_rounds += 1

        status = "best" if best_so_far else validity_status
        bottlenecks = ", ".join(bottleneck_links[:3]) if bottleneck_links else "none"
        stages = len(candidate_sketch) if isinstance(candidate_sketch, list) else 0
        line = (
            f"round {round_id}: {status}, time={completion_time}, stages={stages}, "
            f"bottlenecks={bottlenecks}"
        )
        self.recent.append(line)
        self.recent = self.recent[:]
        self.summary = "\n".join(self.recent)

        if self.stale_rounds >= self.stagnation_rounds:
            self.direction_hint = (
                "Search stagnated; try a structurally different stage order or redistribute traffic away from "
                f"recent bottlenecks ({bottlenecks})."
            )
            self.stale_rounds = 0

        return RecordUpdate(summary=self.summary, direction_hint=self.direction_hint)

