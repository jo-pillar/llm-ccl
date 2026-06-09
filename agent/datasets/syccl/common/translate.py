from __future__ import annotations

from typing import Any

from .config import SycclConfigContext


def sketches_to_translated_schedule(
    sketches: list[dict[str, Any]],
    context: SycclConfigContext,
) -> dict[str, Any]:
  if len(sketches) != 1:
    raise ValueError("flow-sim evaluator expects exactly one sketch")

  graph = sketches[0]
  if context.collective == "allgather":
    events = [
        {
            "src_chunk": f"({root}, 0)",
            "sends": rotate_graph_sends(graph, root, context),
        }
        for root in range(context.ngpus)
    ]
  elif context.collective == "alltoall":
    events = [
        {
            "src_chunk": f"({src}, {dst})",
            "sends": rotate_path_sends(graph, src, dst, context),
        }
        for src in range(context.ngpus)
        for dst in range(context.ngpus)
    ]
  else:
    raise ValueError(f"flow-sim translator does not support collective {context.collective}")
  return {
      "coll_name": context.collective,
      "ngpus": context.ngpus,
      "chunk_size_byte": context.coll_byte,
      "algorithms": [
          {
              "final_schedule": {
                  "Schedule": {
                      "Events": events,
                  },
              },
          },
      ],
  }


def rotate_graph_sends(
    graph: dict[str, Any],
    root: int,
    context: SycclConfigContext,
) -> list[dict[str, Any]]:
  sends = []
  for node in sorted(graph.get("nodes", []), key=lambda item: (int(item["step"]), int(item["id"]))):
    pair = node.get("src_dest_pair", {})
    srcs = [rotate_gpu(int(src), root, context) for src in pair.get("srcs", [])]
    dsts = [rotate_gpu(int(dst), root, context) for dst in pair.get("dsts", [])]
    for src in srcs:
      for dst in dsts:
        sends.append({
            "src_gpu": src,
            "dst_gpu": dst,
            "epoch": int(node["step"]),
            "layer_used": int(node["layer"]),
            "copy": True,
            "reduce": False,
        })
  return sends


def rotate_path_sends(
    graph: dict[str, Any],
    src_root: int,
    dst_gpu: int,
    context: SycclConfigContext,
) -> list[dict[str, Any]]:
  if src_root == dst_gpu:
    return []
  target = unrotate_gpu(dst_gpu, src_root, context)
  parent_by_dst = {}
  node_by_dst = {}
  for node in graph.get("nodes", []):
    pair = node.get("src_dest_pair", {})
    srcs = [int(src) for src in pair.get("srcs", [])]
    dsts = [int(dst) for dst in pair.get("dsts", [])]
    if not srcs:
      continue
    for dst in dsts:
      parent_by_dst[dst] = srcs[0]
      node_by_dst[dst] = node

  path_nodes = []
  current = target
  visited = set()
  while current != 0:
    if current in visited:
      raise ValueError(f"cycle while tracing alltoall path to GPU {dst_gpu}")
    visited.add(current)
    if current not in parent_by_dst:
      raise ValueError(f"sketch has no path from root 0 to GPU {target}")
    parent = parent_by_dst[current]
    path_nodes.append((parent, current, node_by_dst[current]))
    current = parent
  path_nodes.reverse()

  sends = []
  for parent, child, node in path_nodes:
    sends.append({
        "src_gpu": rotate_gpu(parent, src_root, context),
        "dst_gpu": rotate_gpu(child, src_root, context),
        "epoch": int(node["step"]),
        "layer_used": int(node["layer"]),
        "copy": True,
        "reduce": False,
    })
  return sends


def rotate_gpu(gpu: int, root: int, context: SycclConfigContext) -> int:
  src_host = gpu // context.host_gpu_num
  src_local = gpu % context.host_gpu_num
  root_host = root // context.host_gpu_num
  root_local = root % context.host_gpu_num
  host = (src_host + root_host) % context.host_num
  local = (src_local + root_local) % context.host_gpu_num
  return host * context.host_gpu_num + local


def unrotate_gpu(gpu: int, root: int, context: SycclConfigContext) -> int:
  host = gpu // context.host_gpu_num
  local = gpu % context.host_gpu_num
  root_host = root // context.host_gpu_num
  root_local = root % context.host_gpu_num
  src_host = (host - root_host) % context.host_num
  src_local = (local - root_local) % context.host_gpu_num
  return src_host * context.host_gpu_num + src_local
