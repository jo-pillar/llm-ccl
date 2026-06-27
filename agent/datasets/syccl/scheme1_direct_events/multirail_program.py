# EVOLVE-BLOCK-START
"""Multirail double-ring compact sketch DSL for SyCCL config-driven tasks."""

def tx(step, layer, group, srcs, dsts):
  """Return one compact DSL transmission.

  Fields are (step, layer, group, srcs, dsts). srcs/dsts may be either a
  single GPU id or a list of GPU ids.
  """
  return (step, layer, group, srcs, dsts)
def construct_sketches(ngpus: int = 512, root_gpu: int = 0, gpus_per_host: int = 8):
  """Return a single-root double-ring broadcast sketch for multirail topology."""
  if ngpus % gpus_per_host != 0:
    raise ValueError("ngpus must be divisible by gpus_per_host")
  if root_gpu != 0:
    raise ValueError("this multirail sketch assumes root_gpu=0")

  sketch = []
  if ngpus <= 1:
    return sketch

  host_num = ngpus // gpus_per_host

  def rail_gpu(host, rail):
    return host * gpus_per_host + rail

  # Step 0: fan out inside the root host so every rail has a local source.
  if gpus_per_host > 1:
    sketch.append(tx(0, 1, 0, root_gpu, list(range(1, gpus_per_host))))

  for rail in range(gpus_per_host):
    left_host = 0
    right_host = 0
    visited_hosts = {0}
    step = 1

    while len(visited_hosts) < host_num:
      next_left = left_host + 1
      if next_left < host_num and next_left not in visited_hosts:
        sketch.append(
            tx(step, 3, rail, rail_gpu(left_host, rail), rail_gpu(next_left, rail))
        )
        visited_hosts.add(next_left)
        left_host = next_left

      next_right = right_host - 1 if right_host > 0 else host_num - 1
      if next_right not in visited_hosts:
        sketch.append(
            tx(step, 3, rail, rail_gpu(right_host, rail), rail_gpu(next_right, rail))
        )
        visited_hosts.add(next_right)
        right_host = next_right

      step += 1

  return sketch


# EVOLVE-BLOCK-END


def run_code():
  """Entry point used by the SimpleTES evaluator."""
  return construct_sketches()
