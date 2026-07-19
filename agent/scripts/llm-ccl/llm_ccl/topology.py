from __future__ import annotations

import ast
import tempfile
from dataclasses import dataclass
from pathlib import Path

from syccl_agents.topodsl import TopoDSLSpec, load_topodsl

from .models import CaseSpec


_COLLECTIVE_ENUM = {
    "allgather": "CollectiveType.ALLGATHER",
    "alltoall": "CollectiveType.ALLTOALL",
    "allreduce": "CollectiveType.ALLREDUCE",
    "broadcast": "CollectiveType.BROADCAST",
}


@dataclass(frozen=True)
class RenderedTopology:
    source: str
    topo: TopoDSLSpec


def render_case_topology(case: CaseSpec, template_path: Path) -> RenderedTopology:
    source = template_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "topology"
        and isinstance(node.value, ast.Call)
    ]
    if len(assignments) != 1:
        raise ValueError("topology template must contain exactly one topology assignment")

    assignment = assignments[0]
    call = assignment.value
    scale_layers = {layer.layer_id: layer for layer in case.scale.layers}
    rendered_keywords: list[str] = []
    template_layer_ids: set[int] = set()
    for keyword in call.keywords:
        layer_call = keyword.value
        if keyword.arg is None or not isinstance(layer_call, ast.Call) or len(layer_call.args) < 2:
            raise ValueError("topology layer must be a named LayerSpec call")
        layer_id = ast.literal_eval(layer_call.args[0])
        template_layer_ids.add(layer_id)
        if layer_id not in scale_layers:
            raise ValueError(f"scale missing layer {layer_id}")
        shape = scale_layers[layer_id]
        link_source = ast.get_source_segment(source, layer_call.args[1])
        node_type = next((item for item in layer_call.keywords if item.arg == "node_type"), None)
        if link_source is None or node_type is None:
            raise ValueError(f"layer {layer_id} is missing LinkSpec or node_type")
        node_type_source = ast.get_source_segment(source, node_type.value)
        rendered_keywords.append(
            f"    {keyword.arg}=LayerSpec({layer_id}, {link_source}, "
            f"group_num={shape.group_num}, node_num={shape.node_num}, node_type={node_type_source}),"
        )
    if template_layer_ids != set(scale_layers):
        raise ValueError("scale and template layer IDs differ")

    constructor = ast.get_source_segment(source, call.func)
    rendered_assignment = "\n".join(
        [
            f"topology = {constructor}(",
            f"    {case.scale.gpu_count},",
            f"    {case.total_message_size},",
            f"    {_COLLECTIVE_ENUM[case.collective]},",
            *rendered_keywords,
            ")",
        ]
    )
    lines = source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: assignment.lineno - 1]) + assignment.col_offset
    end = sum(len(line) for line in lines[: assignment.end_lineno - 1]) + assignment.end_col_offset
    rendered_source = source[:start] + rendered_assignment + source[end:]

    with tempfile.TemporaryDirectory(prefix="llm-ccl-topology-") as tmp:
        rendered_path = Path(tmp) / template_path.name
        rendered_path.write_text(rendered_source, encoding="utf-8")
        topo = load_topodsl(rendered_path)
    gpu_count = topo.params.hosts * topo.params.gpus_per_host
    if gpu_count != case.scale.gpu_count:
        raise ValueError(f"rendered GPU count {gpu_count} != {case.scale.gpu_count}")
    if topo.params.message_size != case.coll_byte:
        raise ValueError("message size was not normalized exactly once")
    if topo.params.collective != case.collective:
        raise ValueError("rendered collective differs from case")
    return RenderedTopology(rendered_source, topo)

