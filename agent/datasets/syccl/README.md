# SyCCL SimpleTES Integration

This directory contains the SimpleTES task wrapper for exploring SyCCL collective
communication schedules with LLM-generated compact sketches.

The active task is:

```text
agent/datasets/syccl/scheme1_direct_events/
  init_program.py
  evaluator.py
  syccl_sketch_4host_clos.txt
```

`scheme1_direct_events` evolves Python code that returns a compact single-root
broadcast sketch. The evaluator validates the sketch, writes
`candidate-sketch.json`, runs `flow-sim-rs batch-sketch`, and scores the
candidate with:

```text
combined_score = coll.byte / simulated_time_us
```

Higher score is better. For a fixed config, this means minimizing flow-sim-rs
`time_us`.

## What Was Added

The SyCCL SimpleTES integration adds a config-driven task runner around the
generic SimpleTES engine.

Main additions:

- A SyCCL dataset evaluator at
  `agent/datasets/syccl/scheme1_direct_events/evaluator.py`.
- A compact-sketch seed program at
  `agent/datasets/syccl/scheme1_direct_events/init_program.py`.
- A manifest/config generator at `scripts/syccl_config_matrix.py`.
- A SimpleTES launcher at `agent/scripts/run_syccl_simpletes.py`.
- Regression tests for dynamic SyCCL configs, runner command construction, and
  failure feedback under `agent/tests/` and `scripts/test_*`.

The evaluator now derives task prompt content from the active SyCCL JSON config:

- topology type: Clos, multirail, or host-local
- collective: `allgather` or `alltoall`
- number of hosts, GPUs, NICs, and total GPUs
- allowed layer/group memberships
- collective-specific expansion guidance

This means the same SimpleTES task can be reused across cases such as
`clos-4host`, `clos-128gpu`, `multirail-4host`, and `multirail-512gpu`.

The evaluator also returns structured failure metadata for invalid sketches:

```json
{
  "failure_category": "dependency_error",
  "failure_feedback": "GPU 16 is used as a source at step 1, but ..."
}
```

Current categories include dependency errors, duplicate destinations, incomplete
coverage, topology group errors, DSL schema errors, flow-sim timeouts,
flow-sim runtime errors, and flow-sim output errors.

Important limitation: SimpleTES's built-in failure-pattern prompt currently
summarizes `metrics["error"]`. It records failure history, but it does not yet
use `failure_feedback` as the primary text shown to the model.

## Prerequisites

From the repository root:

```bash
cd /home/antl/wzd/syccl
```

Build flow-sim-rs first. The evaluator expects:

```text
/home/antl/wzd/llm-ccl/Flow-Simulator/flow-sim-rs/target/release/flow-sim-rs
```

Override the binary path with `FLOW_SIM_BIN` if needed. A typical build is:

```bash
cd /home/antl/wzd/llm-ccl/Flow-Simulator/flow-sim-rs
cargo build --release
```

Install the SimpleTES Python environment:

```bash
cd /home/antl/wzd/syccl/agent
uv sync
```

Set the model endpoint in:

```text
agent/env.toml
```

Example:

```toml
model = "deepseek/deepseek-v4-flash"
api_base = "https://api.deepseek.com"
api_key = "..."
```

`agent/scripts/run_syccl_simpletes.py` reads this file and forwards `model`,
`api_base`, and `api_key` to `main.py`.

## Prepare TopoDSL Inputs

The SimpleTES launcher now takes three user inputs:

```text
TOPODSL=/path/to/topology.py
--init-program /path/to/ring_init.py
--instruction /path/to/prompt_templete.txt
```

`TOPODSL` is mandatory and must point to a TopoDSL file. The launcher parses it,
derives the GPU count, collective, message size, and topology parameters, then
writes the flow-sim config used by the evaluator. Supported topology families
are `clos` and `multirail`.

The init program must expose one of these functions:

```python
def construct_sketches(GPU_NUM):
    ...

def build_initial_sketch(GPU_NUM):
    ...
```

The instruction file is a template. The launcher fills placeholders such as
`${GPU_NUM}`, `${Collective}`, `${TOPOLOGY}`, `${MESSAGE_SIZE}`,
`${TOPOLOGY_SUMMARY}`, and `${LAYER_GROUP_SUMMARY}`.

## Run SimpleTES On One SyCCL Case

Use `agent/scripts/run_syccl_simpletes.py` from the `agent/` directory.

Run one TopoDSL task:

```bash
cd /home/antl/wzd/syccl/agent

TOPODSL=/home/antl/wzd/llm-ccl/agent/examples/topologies/multirail_topo.py \
uv run python scripts/run_syccl_simpletes.py \
  --init-program datasets/syccl/scheme1_direct_events/init_program.py \
  --instruction datasets/syccl/scheme1_direct_events/prompt_templete.txt \
  --condition full \
  --max-generations 10000 \
  --save-llm-io
```

`--condition full` maps to the main SimpleTES search setting:

- `--selector balance`
- `--num-chains 4`
- `--k-candidates 4`
- `--include-construction`
- failure patterns enabled unless explicitly disabled

`--save-llm-io` stores the full prompt and raw model output in checkpoint
`nodes.json`, which is useful when inspecting whether failure feedback was
actually included.

Run an ablation:

```bash
cd /home/antl/wzd/syccl/agent

TOPODSL=/home/antl/wzd/llm-ccl/agent/examples/topologies/multirail_topo.py \
uv run python scripts/run_syccl_simpletes.py \
  --init-program datasets/syccl/scheme1_direct_events/init_program.py \
  --instruction datasets/syccl/scheme1_direct_events/prompt_templete.txt \
  --condition ablation \
  --max-generations 10000 \
  --save-llm-io
```

`--condition ablation` maps to:

- `--num-inspirations 0`
- `--num-chains 1`
- `--k-candidates 1`
- `--disable-reflection`
- `--disable-failure-patterns`

Use it as a lower-feedback comparison point.

Preview the exact command and SyCCL environment without starting the run:

```bash
cd /home/antl/wzd/syccl/agent

TOPODSL=/home/antl/wzd/llm-ccl/agent/examples/topologies/multirail_topo.py \
uv run python scripts/run_syccl_simpletes.py \
  --init-program datasets/syccl/scheme1_direct_events/init_program.py \
  --instruction datasets/syccl/scheme1_direct_events/prompt_templete.txt \
  --condition full \
  --max-generations 10000 \
  --save-llm-io \
  --dry-run
```

## Output Layout

By default SimpleTES output is written under:

```text
/home/antl/mntdisk/syccl-llm-scheme1-direct-events/simpletes/
  <case_id>/
    <collective>/
      <size>/
        <condition>/
          instructions/syccl_instruction.txt
          eval_artifacts/scheme1_direct_events/eval_.../
          checkpoints/<date>/instance-.../
```

Important files:

```text
instructions/syccl_instruction.txt
  The exact task instruction generated from the active SyCCL config.

eval_artifacts/scheme1_direct_events/eval_.../program.py
  The candidate program evaluated in that attempt.

eval_artifacts/scheme1_direct_events/eval_.../metrics.json
  Evaluator metrics, including combined_score, validity, eval_s,
  bottleneck_profile, failure_category, and failure_feedback when the candidate
  fails.

checkpoints/<date>/instance-.../run.log
  SimpleTES runtime log.

checkpoints/<date>/instance-.../db_state_*/nodes.json
  Archive of evaluated candidates. Includes llm_input/llm_output when
  --save-llm-io was enabled.

checkpoints/<date>/instance-.../db_state_*/best_program.py
  Best evolved program in the latest checkpoint.
```

To inspect whether failure patterns were included in prompts:

```bash
grep -R "FAILURE PATTERNS" \
  /home/antl/mntdisk/syccl-llm-scheme1-direct-events/simpletes/*/*/*/full/checkpoints \
  | head
```

To inspect structured SyCCL feedback:

```bash
grep -R "\"failure_category\"" \
  /home/antl/mntdisk/syccl-llm-scheme1-direct-events/simpletes/*/*/*/full/eval_artifacts \
  | head
```

## What Is Fed To The Model

Each SimpleTES prompt contains these sections:

1. The generated SyCCL task instruction.
2. The EVOLVE-BLOCK extraction rules.
3. The fixed `EXACT_PREFIX` and `EXACT_SUFFIX` from `init_program.py`.
4. Optional available packages.
5. Optional `GLOBAL_BEST_CONSTRUCTION` summary when `--include-construction`
   is enabled and a valid chain-best construction exists.
6. Optional policy context for selectors that provide it.
7. Sampled inspirations: successful prior programs with score, metrics,
   optional reflection, and full code.
8. Optional failure patterns: common `metrics["error"]` summaries for the
   current chain.

Failed candidates are stored in the node database, but they are scored as
`-inf` by SimpleTES when `metrics["error"]` is present. They are therefore not
selected as normal inspirations. Their errors can still contribute to the
failure-pattern summary in later prompts for the same chain.

The first prompt has no failure patterns because no failures have happened yet.
Failure patterns also disappear in ablation mode, static prompt mode
(`--num-inspirations 0`), or after chain-local restart clears the chain's error
history.

## Failure Feedback Notes

The SyCCL evaluator's structured feedback is written to `metrics.json` and
`nodes.json`. Valid candidates include a `bottleneck_profile` from flow-sim-rs
with `critical_flow_chain` information from the simulator. Invalid candidates
include `failure_category` and `failure_feedback`.

## Common Pitfalls

- `flow-sim-rs` is missing: build flow-sim-rs or set `FLOW_SIM_BIN`.
- No failure feedback appears in prompts: check that the run is `--condition full`,
  not ablation, and inspect prompts after at least one batch has completed.
- No `llm_input` in `nodes.json`: rerun with `--save-llm-io`.
- Generated configs are not found: regenerate the manifest and configs with
  `scripts/syccl_config_matrix.py`.
- Private model endpoint goes through a proxy: the runner adds private API hosts
  to `NO_PROXY`, but check `agent/env.toml` and your shell proxy variables if
  requests still fail.
