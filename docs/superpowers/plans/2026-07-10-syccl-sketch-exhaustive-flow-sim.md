# SyCCL Sketch Exhaustive Flow-Sim Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `syccl-sketch-search` experiment command that exhaustively generates and evaluates sketches for the V100 DGX-2 flow-sim comparison bundle.

**Architecture:** Keep the existing sketch search API as the generator. Add a focused experiment driver module that reads the bundle manifest, creates resumable per-case result directories under `syccl-sketch-search`, invokes the bundled `flow-sim-rs batch-sketch`, and writes per-case plus aggregate summaries.

**Tech Stack:** Rust 2021, clap, serde/serde_json, std process management, existing `flow-sim-rs` CLI.

---

### Task 1: Manifest And Output Planning

**Files:**
- Create: `syccl-sketch-search/src/experiment.rs`
- Modify: `syccl-sketch-search/src/lib.rs`
- Test: `syccl-sketch-search/tests/experiment_driver.rs`

- [ ] Write failing tests for manifest filtering and case output paths.
- [ ] Run `cargo test --test experiment_driver` and confirm the missing module/API failure.
- [ ] Implement manifest parsing, case filters, and output layout planning.
- [ ] Run `cargo test --test experiment_driver`.

### Task 2: Case Execution

**Files:**
- Modify: `syccl-sketch-search/src/experiment.rs`
- Test: `syccl-sketch-search/tests/experiment_driver.rs`

- [ ] Write failing tests for resumable sketch generation and flow-sim command construction.
- [ ] Run the focused test and confirm the expected failure.
- [ ] Implement per-case execution: write `sketches/candidate-sketch.json`, `search-profile.json`, run `batch-sketch`, and write `case-summary.json`.
- [ ] Run the focused test.

### Task 3: CLI And Verification

**Files:**
- Modify: `syccl-sketch-search/src/main.rs`
- Modify: `syccl-sketch-search/README.md`
- Test: `syccl-sketch-search/tests/experiment_driver.rs`

- [ ] Write failing CLI smoke test for the new experiment subcommand help or dry-run behavior.
- [ ] Implement the CLI subcommand with defaults: `case-jobs=4`, `flow-sim-threads=8`, full manifest by default, and optional filters.
- [ ] Run `cargo test`.
- [ ] Run a single-case verification against `v100-dgx2-64gpu-allgather-1024B-prune=small`.
