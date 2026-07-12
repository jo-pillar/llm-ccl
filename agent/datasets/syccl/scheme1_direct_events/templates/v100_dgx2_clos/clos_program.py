# EVOLVE-BLOCK-START
"""Initial double-ring Concrete SketchDSL for a multi-host DGX-2 Clos topology."""


def tx(step, layer, group, srcs, dsts):
  """Return one Concrete SketchDSL transmission."""
  return (step, layer, group, srcs, dsts)


def construct_sketches(ngpus: int = 64, root_gpu: int = 0, gpus_per_host: int = 16, hosts_per_leaf: int = 1):
  """Return a topology-aware single-root broadcast Concrete SketchDSL."""
  def ring_gpu(offset):
    return (root_gpu + offset) % ngpus

  def host_id(gpu):
    return gpu // gpus_per_host

  def leaf_id(gpu):
    return host_id(gpu) // hosts_per_leaf

  def nearest_layer_group(src, dst):
    if host_id(src) == host_id(dst):
      return 1, host_id(src)
    if leaf_id(src) == leaf_id(dst):
      return 3, leaf_id(src)
    return 4, 0

  sketch = []
  if ngpus <= 1:
    return sketch

  left_front = root_gpu
  right_front = root_gpu
  remaining = ngpus - 1
  step = 0
  while remaining > 0:
    left_dst = ring_gpu(step + 1)
    layer, group = nearest_layer_group(left_front, left_dst)
    sketch.append(tx(step, layer, group, left_front, left_dst))
    left_front = left_dst
    remaining -= 1

    if remaining > 0:
      right_dst = ring_gpu(-(step + 1))
      layer, group = nearest_layer_group(right_front, right_dst)
      sketch.append(tx(step, layer, group, right_front, right_dst))
      right_front = right_dst
      remaining -= 1
    step += 1

  return sketch


# EVOLVE-BLOCK-END


def run_code():
  return construct_sketches()
