# Lightweight Agent Flow-Sim Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a lightweight SyCCL SimpleTES agent into top-level `agent/` and switch its evaluator from SyCCL `resim` to the Rust flow simulator.

**Architecture:** Keep SimpleTES core Python code intact, but move SyCCL-specific config/sketch/translation/evaluation code behind small modules under `agent/datasets/syccl/common/`. The evaluator writes a translated schedule, calls `Flow-Simulator/flow-sim-rs`, and returns SimpleTES-compatible metrics.

**Tech Stack:** Python 3.11+, pytest/unittest, Rust 2021, cargo tests.

---

### Task 1: Create Lightweight Agent Skeleton

**Files:**
- Create: `agent/README.md`
- Copy: `agent/simpletes/`, `agent/scripts/run_syccl_simpletes.py`, `agent/datasets/syccl/`, `agent/tests/test_syccl_*.py`, `agent/pyproject.toml`, `agent/uv.lock`, `agent/LICENSE`, `agent/main.py`, `agent/sitecustomize.py`

- [ ] Copy only SyCCL-relevant agent files from `syccl/agent`, excluding `.git`, `.venv`, caches, best results, and unrelated datasets.
- [ ] Update tests that assume `ROOT / "agent"` from the old nested location.
- [ ] Run `uv run pytest tests/test_syccl_dynamic_config.py tests/test_syccl_simpletes_runner.py -q` from `agent/`.

### Task 2: Add Config, Sketch, and Translator Modules

**Files:**
- Create: `agent/datasets/syccl/common/config.py`
- Create: `agent/datasets/syccl/common/sketch.py`
- Create: `agent/datasets/syccl/common/translate.py`
- Test: `agent/tests/test_syccl_flow_translate.py`

- [ ] Write failing tests for translating the initial 2-host sketch into a `candidate-translated.json` shape accepted by `flow-sim-rs`.
- [ ] Move config/layer-group and sketch validation logic out of the old evaluator without changing behavior.
- [ ] Implement allgather expansion from a single-root compact tree into one event group per root chunk.
- [ ] Run the new translator tests and existing dynamic-config tests.

### Task 3: Switch Evaluator to Flow Simulator

**Files:**
- Modify: `agent/datasets/syccl/scheme1_direct_events/evaluator.py`
- Test: `agent/tests/test_syccl_flow_evaluator.py`

- [ ] Write failing tests proving the evaluator calls a configurable flow-sim binary instead of SyCCL `synthesize resim`.
- [ ] Implement `SYCCL_FLOW_SIM_BIN`, `SYCCL_FLOW_SIM_ROOT`, and `SYCCL_EVALUATOR_BACKEND=flow-sim` defaults.
- [ ] Preserve failure metrics and actionable feedback categories.
- [ ] Run SyCCL agent tests.

### Task 4: Extend Rust Flow Simulator for Agent Inputs

**Files:**
- Modify: `Flow-Simulator/flow-sim-rs/src/config.rs`
- Modify: `Flow-Simulator/flow-sim-rs/src/topology.rs`
- Modify: `Flow-Simulator/flow-sim-rs/src/schedule.rs`
- Test: `Flow-Simulator/flow-sim-rs/tests/simulator_tests.rs`

- [ ] Add tests for Clos config parsing and non-one-NIC-per-GPU configs.
- [ ] Support host-local routes and Clos/pod cross-host routes enough for agent-generated translated schedules.
- [ ] Keep existing multirail behavior intact.
- [ ] Run `cargo test`.

### Task 5: End-to-End Fake LLM Workflow and Docs

**Files:**
- Create: `agent/tests/test_syccl_end_to_end_fake_llm.py`
- Modify: `agent/README.md`

- [ ] Add an end-to-end test that runs the initial program through the flow-sim evaluator with a small config.
- [ ] Document directory layout, required env vars, and example commands.
- [ ] Run Python tests and Rust tests.
