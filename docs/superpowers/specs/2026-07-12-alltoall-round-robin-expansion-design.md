# Alltoall Round-Robin Expansion Design

## Goal

Keep the Origin-SyCCL-selected Alltoall root sketch and every recovered path
unchanged, but replace the coarse `epoch = sketch.step` mapping with a
deterministic host-pair schedule that gives SyCCL resim enough ordering
information to avoid large equal-priority link queues.

## Scope

- Change only FlowSim-rs Alltoall sketch expansion.
- Preserve AllGather expansion exactly.
- Preserve every Alltoall `(src_chunk, chunk_index, src_gpu, dst_gpu, layer)`
  send and every per-chunk path dependency.
- Do not invoke Origin-SyCCL solve. Evaluate only with existing archived MILP
  schedules and the fixed H800 `synthesize resim` binary.

## Temporal Allocator

Let `H` be the host count and `G` the GPUs per host. For an Alltoall chunk with
original source GPU `src_root` and final destination GPU `dst_gpu`:

- `src_host = src_root / G`
- `src_local = src_root % G`
- `dst_host = dst_gpu / G`
- `dst_local = dst_gpu % G`

For a remote destination, define the nonzero cyclic host offset:

```text
host_offset = (dst_host + H - src_host) % H
round = src_local * (H - 1) + (host_offset - 1)
```

Every send on that chunk's recovered path receives `epoch = round`, including
the source-local relay and the following cross-host send. For fixed
`src_local` and `host_offset`, the mapping `src_host -> dst_host` is a
permutation, so each source rail and destination rail participates in at most
one remote transfer in that round.

Same-host final sends are assigned after all remote rounds:

```text
epoch = G * (H - 1) + dst_local
```

This keeps remote-path preparation on the critical path and prevents local
completion traffic from winning priority over remote traffic.

## Required Invariants

1. The total send count and the multiset of route edges are unchanged.
2. Every two-hop chunk retains first-hop-before-second-hop path order.
3. For cross-host hops, each `(epoch, src_gpu)` occurs at most once.
4. For cross-host hops, each `(epoch, dst_gpu)` occurs at most once.
5. Expansion is deterministic and independent of hash-map iteration order.
6. AllGather output remains byte-for-byte equivalent at the schedule level.

## Verification

First add focused Rust tests on a small multirail Alltoall fixture. Then use the
FlowSim-rs `translate-sketch` command, which exports the SyCCL schedule without
running FlowSim's unrelated performance model. Replay the existing 32-host
Alltoall cases with the same root sketches, configs, and H800 resimulator.
Compare new deterministic time against the already archived MILP resim time;
no solve command is permitted.
