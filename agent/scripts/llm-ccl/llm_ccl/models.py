from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=-]*$")
_COLLECTIVES = {"allgather", "alltoall", "allreduce", "broadcast"}


def _validate_slug(value: str, label: str) -> None:
    if not _SLUG.fullmatch(value):
        raise ValueError(f"{label} must be a path-safe slug: {value!r}")


@dataclass(frozen=True)
class LayerShape:
    layer_id: int
    group_num: int
    node_num: int

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if self.group_num <= 0 or self.node_num <= 0:
            raise ValueError("group_num and node_num must be positive")


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    gpu_count: int
    layers: tuple[LayerShape, ...]

    def __post_init__(self) -> None:
        _validate_slug(self.name, "scale name")
        if self.gpu_count <= 0:
            raise ValueError("gpu_count must be positive")
        layer_ids = [layer.layer_id for layer in self.layers]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("duplicate layer_id in scale")


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    scale: ScaleSpec
    collective: str
    total_message_size: int

    def __post_init__(self) -> None:
        _validate_slug(self.case_id, "case_id")
        if self.collective not in _COLLECTIVES:
            raise ValueError(f"unsupported collective: {self.collective}")
        if self.total_message_size <= 0 or self.total_message_size % self.scale.gpu_count:
            raise ValueError("total_message_size must be positive and divisible by gpu_count")

    @property
    def coll_byte(self) -> int:
        return self.total_message_size // self.scale.gpu_count


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    topology_template: Path
    instruction_template: Path
    initial_program: Path
    cases: tuple[CaseSpec, ...]

    def __post_init__(self) -> None:
        _validate_slug(self.name, "project name")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(f"duplicate case_id in project {self.name}")


def case_matrix(
    scales: tuple[ScaleSpec, ...],
    collectives: tuple[str, ...],
    total_message_sizes: tuple[int, ...],
) -> tuple[CaseSpec, ...]:
    return tuple(
        CaseSpec(
            case_id=f"{scale.name}-{collective}-{total_size}B",
            scale=scale,
            collective=collective,
            total_message_size=total_size,
        )
        for scale in scales
        for collective in collectives
        for total_size in total_message_sizes
    )

