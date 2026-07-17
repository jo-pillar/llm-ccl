# Alltoall Round-Robin Expansion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace FlowSim-rs's two-bucket Alltoall epoch expansion with a deterministic host-pair round-robin allocator while preserving all routes and sends.

**Architecture:** `sketches_to_translated_schedule` continues to recover one path per `(src,dst)`. A focused epoch helper derives a remote round from source-local GPU and cyclic host offset, while same-host sends are placed after remote rounds. `rotate_path_sends` applies the computed chunk epoch to every hop, preserving path order and sketch provenance.

**Tech Stack:** Rust 2021, Cargo tests, FlowSim-rs debug CLI, Origin-SyCCL H800 `synthesize resim`.

---

### Task 1: Lock Down Temporal Invariants

**Files:**
- Modify: `agent/datasets/syccl/scheme1_direct_events/flow-sim-rs/tests/simulator_tests.rs`

- [ ] Add a small 3-host, 2-GPU multirail Alltoall sketch fixture.
- [ ] Assert the complete non-epoch send tuple multiset and send count.
- [ ] Assert remote `(epoch, src_gpu)` and `(epoch, dst_gpu)` uniqueness.
- [ ] Assert two-hop sends share a round and retain path order.
- [ ] Assert independent AllGather and Alltoall expansions are byte-deterministic.
- [ ] Run the focused tests and confirm they fail under `epoch = node.step`.

### Task 2: Implement Host-Pair Round-Robin Epochs

**Files:**
- Modify: `agent/datasets/syccl/scheme1_direct_events/flow-sim-rs/src/sketch.rs`

- [ ] Add a pure helper that computes the Alltoall chunk epoch.
- [ ] Pass the final destination-derived epoch into `rotate_path_sends`.
- [ ] Keep AllGather expansion unchanged.
- [ ] Run focused and full FlowSim-rs tests.

### Task 3: Replay Existing H800 Alltoall Cases

**Files:**
- Create: `experiments/h800_sketch_sufficiency/results/20260712-alltoall-round-robin/**`

- [ ] Build the FlowSim-rs debug binary.
- [ ] Re-expand only the existing 32-host Alltoall root sketches with
  `flow-sim-rs translate-sketch --config <config> --sketch <root> --output <schedule>`.
- [ ] Run only H800 `synthesize resim`; never run solve.
- [ ] Record schedule hashes, epoch statistics, resim times, and ratios against archived MILP times.
- [ ] If the first allocator remains materially weaker, inspect concrete queue-order evidence and revise one variable at a time.

### Task 4: Verify And Report

- [ ] Run the full Rust test suite and the H800 ablation unit tests.
- [ ] Confirm no command contains `solve`.
- [ ] Report per-size ratios, the worst case, and whether Alltoall reaches the 0.95 non-inferiority threshold.

No git commit is created because the working tree already contains unrelated untracked experiment artifacts and the user requested an experimental implementation, not repository-history changes.
