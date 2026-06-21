# Python SyCCL Two-Agent Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Python two-agent SyCCL workflow that can pass the minimal allgather end-to-end smoke test while leaving the legacy SimpleTES flow available.

**Architecture:** Keep `agent/simpletes/` and the old SyCCL SimpleTES runner intact. Add an isolated `agent/syccl_agents/` package that reads user-editable TopoDSL, maps supported Clos/MultiRail parameters onto existing SyCCL/flow-sim config JSON, skips `syccl-sketch-search` for now because it emits JSON rather than LLM-friendly compact DSL, uses `flow-sim-rs simulate-sketch` for deterministic evaluation, and writes the experiment-plan raw records.

**Tech Stack:** Python 3.11+, pytest, subprocess-bound external Rust CLIs, existing SimpleTES LLM backend for real LLM calls.

---

## File Structure

- Create `agent/syccl_agents/topodsl.py`: load Python-code or text TopoDSL, preserve prompt source, normalize parameters, compute config id.
- Create `agent/syccl_agents/config_render.py`: render target SyCCL/flow-sim configs for `clos` and `multirail`.
- Create `agent/syccl_agents/sketch_dsl.py`: extract and normalize proposal SketchDSL into compact sketch JSON accepted by flow-sim.
- Create `agent/syccl_agents/flow_sim.py`: subprocess wrapper for `flow-sim-rs simulate-sketch`.
- Create `agent/syccl_agents/llm.py`: fake backend and SimpleTES backend adapter.
- Create `agent/syccl_agents/record_agent.py`: deterministic summary and direction hint updates.
- Create `agent/syccl_agents/records.py`: JSONL records with required experiment fields.
- Create `agent/syccl_agents/runner.py`: sequential proposal/record loop.
- Create `agent/syccl_agents/__init__.py`: package marker.
- Create `agent/scripts/run_syccl_two_agent.py`: CLI entry point.
- Create `agent/tests/test_syccl_two_agent.py`: minimal behavior and fake E2E tests.

## Tasks

- [ ] Write failing tests for TopoDSL prompt preservation and parameter extraction.
- [ ] Implement minimal TopoDSL loader and config renderer.
- [ ] Write failing tests for record-agent summary/direction and required raw record fields.
- [ ] Implement record-agent and record writers.
- [ ] Write failing fake E2E test for 10 sequential proposal rounds through the flow-sim wrapper, with sketch enumeration explicitly skipped.
- [ ] Implement runner, fake LLM, SketchDSL parser, and subprocess wrappers.
- [ ] Add CLI and verify targeted tests.
- [ ] Run a real LLM E2E with `/home/antl/wzd/llm-ccl/agent/env.toml` if the local model endpoint and binaries are reachable.
