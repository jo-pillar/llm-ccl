# EVOLVE-BLOCK-START
"""Initial compact sketch DSL for SyCCL config-driven tasks."""

import os


HOST_NUM = int(os.environ.get("SYCCL_TASK_HOST_NUM", "4"))
HOST_GPU_NUM = int(os.environ.get("SYCCL_TASK_HOST_GPU_NUM", "8"))
TOPOLOGY = os.environ.get("SYCCL_TASK_TOPOLOGY", "clos").lower()
CROSS_LAYER = int(os.environ.get("SYCCL_TASK_CROSS_LAYER", "4"))
CROSS_GROUP = int(os.environ.get("SYCCL_TASK_CROSS_GROUP", "0"))
NGPUS = int(os.environ.get("SYCCL_TASK_NGPUS", str(HOST_NUM * HOST_GPU_NUM)))
ROOT_GPU = 0


def host_gpus(host_id):
  base = host_id * HOST_GPU_NUM
  return list(range(base, base + HOST_GPU_NUM))


def tx(step, layer, group, srcs, dsts):
  """Return one compact DSL transmission.

  Fields are (step, layer, group, srcs, dsts).  srcs/dsts may be either a
  single GPU id or a list of GPU ids.
  """
  return (step, layer, group, srcs, dsts)


def construct_sketches():
  """Return candidate single-root broadcast sketches in compact DSL form."""
  sketch = []
  if HOST_GPU_NUM > 1:
    sketch.append(tx(0, 1, 0, ROOT_GPU, host_gpus(0)[1:]))

  remote_roots = [host * HOST_GPU_NUM for host in range(1, HOST_NUM)]
  if remote_roots:
    sketch.append(tx(0, CROSS_LAYER, CROSS_GROUP, ROOT_GPU, remote_roots))

  for host in range(1, HOST_NUM):
    gpus = host_gpus(host)
    if len(gpus) > 1:
      sketch.append(tx(1, 1, host, gpus[0], gpus[1:]))

  return [sketch]


# EVOLVE-BLOCK-END


def run_code():
  """Entry point used by the SimpleTES evaluator."""
  return construct_sketches()
