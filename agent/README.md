# Lightweight SyCCL Agent

## Legacy SimpleTES Path

This directory keeps the old SimpleTES-based SyCCL agent for reference and
backward compatibility. The active LLM-CCL agent mainline is `../agent-rs`.
New experiments should use the Rust runner; these commands are legacy and must
not be used for the new two-agent feedback loop.

This directory contains the extracted SimpleTES agent pieces needed for SyCCL
algorithm search. The evaluator no longer calls SyCCL `resim` directly. It
validates compact sketches, translates them into `candidate-translated.json`,
and evaluates AllGather/AllToAll candidates with the Rust flow-level simulator
in `../Flow-Simulator/flow-sim-rs`.

## Layout

- `main.py`: SimpleTES CLI entry point.
- `simpletes/`: generic generation, policy, checkpoint, and LLM runtime code.
- `scripts/run_syccl_simpletes.py`: SyCCL launcher that selects a manifest entry,
  writes the task instruction, and starts `main.py`.
- `datasets/syccl/scheme1_direct_events/`: SyCCL task entry points used by
  SimpleTES.
- `datasets/syccl/common/config.py`: reads SyCCL configs and derives layer/group
  memberships for prompts and validation.
- `datasets/syccl/common/sketch.py`: validates the LLM compact sketch DSL and
  builds the single-root sketch graph.
- `datasets/syccl/common/translate.py`: expands the sketch graph into the
  translated schedule shape accepted by `flow-sim-rs`.
- `tests/`: focused SyCCL runner, translator, evaluator, and fake-LLM workflow
  tests.

## Key Environment Variables

- `SYCCL_BASE_CONFIG`: SyCCL config JSON for the active task.
- `SYCCL_REPO_ROOT`: optional path to the SyCCL checkout used for configs.
- `SYCCL_FLOW_SIM_ROOT`: optional path to `Flow-Simulator/flow-sim-rs`.
- `SYCCL_FLOW_SIM_BIN`: flow simulator binary. Defaults to
  `../Flow-Simulator/flow-sim-rs/target/release/flow-sim-rs`.
- `SYCCL_EVAL_ARTIFACT_DIR`: directory for `candidate-sketch.json`,
  `candidate-translated.json`, flow-sim output, logs, and metrics.

## Build Flow Simulator

```bash
cd ../Flow-Simulator/flow-sim-rs
cargo build --release
```

## Run Tests

```bash
cd agent
python -m pytest tests/test_syccl_*.py -q

cd ../Flow-Simulator/flow-sim-rs
cargo test
```

## Fake LLM Workflow

The test `tests/test_syccl_end_to_end_fake_llm.py` creates a small `run_code()`
program that returns a compact sketch, then runs the full evaluator path:

```text
run_code() -> sketch validation -> candidate-translated.json -> flow-sim-rs simulate -> metrics
```

The metrics returned to SimpleTES include `combined_score`, `validity`,
`best_time_us`, flow/channel counts, queue wait, and critical-link diagnostics.
AllGather expands one rotated broadcast tree per root chunk. AllToAll treats the
same compact tree as a routing template and emits one source/destination chunk
event for each GPU pair.
