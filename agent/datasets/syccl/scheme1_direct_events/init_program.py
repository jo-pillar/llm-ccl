# EVOLVE-BLOCK-START
"""Initial double-ring compact sketch DSL for SyCCL config-driven tasks."""

def tx(step, layer, group, srcs, dsts):
  """Return one compact DSL transmission.

  Fields are (step, layer, group, srcs, dsts). srcs/dsts may be either a
  single GPU id or a list of GPU ids.
  """
  return (step, layer, group, srcs, dsts)
def construct_sketches(ngpus: int = 1024,root_gpu: int = 0, gpus_per_host: int = 8, hosts_per_leaf: int = 2):
  """Return a single-root double-ring broadcast sketch in compact DSL form."""
  ngpus = ngpus
  root_gpu = root_gpu
  gpus_per_host = gpus_per_host
  hosts_per_leaf = hosts_per_leaf

  def ring_gpu(offset):
    return (root_gpu + offset) % ngpus

  def host_id(gpu):
    return gpu // gpus_per_host

  def leaf_id(gpu):
    return host_id(gpu) // hosts_per_leaf
  ## choose the nearest layer and group for src and dst
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
    if remaining >= 1:
      left_dst = ring_gpu(step + 1)
      layer, group = nearest_layer_group(left_front, left_dst)
      sketch.append(tx(step, layer, group, left_front, left_dst))
      left_front = left_dst
      remaining -= 1

    if remaining >= 1:
      right_dst = ring_gpu(-(step + 1))
      layer, group = nearest_layer_group(right_front, right_dst)
      sketch.append(tx(step, layer, group, right_front, right_dst))
      right_front = right_dst
      remaining -= 1

    step += 1

  return sketch


# EVOLVE-BLOCK-END


def run_code():
  """Entry point used by the SimpleTES evaluator."""
  return construct_sketches()
