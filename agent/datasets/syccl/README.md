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
broadcast sketch. The evaluator validates the sketch, expands it into SyCCL
direct events, runs `build/synthesize ... resim --sketch`, and scores the
candidate with:

```text
combined_score = -best_time_us
```

Higher score is better.

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
coverage, topology group errors, DSL schema errors, SyCCL timeouts, SyCCL runtime
errors, and `syccl_expand_incompatible`.

Important limitation: SimpleTES's built-in failure-pattern prompt currently
summarizes `metrics["error"]`. It records failure history, but it does not yet
use `failure_feedback` as the primary text shown to the model.

## Prerequisites

From the repository root:

```bash
cd /home/antl/wzd/syccl
```

Build SyCCL first. The evaluator expects:

```text
/home/antl/wzd/syccl/build/synthesize
```

A typical build is:

```bash
mkdir -p build
cd build
cmake .. -DSCIP_SUITE_DIR=/path/to/SCIP -DSCIP_PP_DIR=/path/to/SCIPpp
make -j
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

## Generate SyCCL Experiment Configs

Use `scripts/syccl_config_matrix.py` from the repository root. The generated
manifest is the handoff file used by both SyCCL batch runs and SimpleTES runs.

Available cases:

```text
clos-4host          4 hosts x 8 GPUs = 32 GPUs
clos-128gpu         16 hosts x 8 GPUs = 128 GPUs
multirail-4host     4 hosts x 8 GPUs = 32 GPUs
multirail-512gpu    64 hosts x 8 GPUs = 512 GPUs
```

Available collectives:

```text
allgather
alltoall
```

Generate one 512-GPU multirail allgather config:

```bash
cd /home/antl/wzd/syccl

python3 scripts/syccl_config_matrix.py \
  --output-root /home/antl/mntdisk/syccl-llm-scheme1-direct-events \
  --manifest /home/antl/mntdisk/syccl-llm-scheme1-direct-events/manifest-multirail512-ag-4k.json \
  --case multirail-512gpu \
  --collective allgather \
  --coll-bytes 4K
```

Generate several message sizes for the same case:

```bash
cd /home/antl/wzd/syccl

python3 scripts/syccl_config_matrix.py \
  --output-root /home/antl/mntdisk/syccl-llm-scheme1-direct-events \
  --manifest /home/antl/mntdisk/syccl-llm-scheme1-direct-events/manifest-multirail512-ag.json \
  --case multirail-512gpu \
  --collective allgather \
  --coll-bytes 4K,512K,1M,16M
```

Generate a full matrix for selected cases and collectives:

```bash
cd /home/antl/wzd/syccl

python3 scripts/syccl_config_matrix.py \
  --output-root /home/antl/mntdisk/syccl-llm-scheme1-direct-events \
  --manifest /home/antl/mntdisk/syccl-llm-scheme1-direct-events/manifest-selected.json \
  --case clos-128gpu \
  --case multirail-512gpu \
  --collective allgather \
  --collective alltoall \
  --coll-bytes 4K,512K,1M,16M
```

Use `--dry-run` to preview entries without writing configs:

```bash
python3 scripts/syccl_config_matrix.py \
  --case multirail-512gpu \
  --collective allgather \
  --coll-bytes 4K \
  --dry-run
```

## Optional: Run Baseline SyCCL Solves

The SimpleTES evaluator uses direct resimulation of generated sketches, so a
baseline solve is not required before running SimpleTES. It can still be useful
for comparison.

Run every entry in a manifest:

```bash
cd /home/antl/wzd/syccl

python3 scripts/runexp.py \
  --manifest /home/antl/mntdisk/syccl-llm-scheme1-direct-events/manifest-selected.json \
  --solver /home/antl/wzd/syccl/build/synthesize
```

Run only one entry:

```bash
python3 scripts/runexp.py \
  --manifest /home/antl/mntdisk/syccl-llm-scheme1-direct-events/manifest-selected.json \
  --solver /home/antl/wzd/syccl/build/synthesize \
  --case multirail-512gpu \
  --collective allgather \
  --coll-byte 4K
```

## Run SimpleTES On One SyCCL Case

Use `agent/scripts/run_syccl_simpletes.py` from the `agent/` directory.

Run 512-GPU multirail allgather, 4 KiB:

```bash
cd /home/antl/wzd/syccl/agent

uv run python scripts/run_syccl_simpletes.py \
  --manifest /home/antl/mntdisk/syccl-llm-scheme1-direct-events/manifest-multirail512-ag-4k.json \
  --case multirail-512gpu \
  --collective allgather \
  --coll-byte 4K \
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

uv run python scripts/run_syccl_simpletes.py \
  --manifest /home/antl/mntdisk/syccl-llm-scheme1-direct-events/manifest-multirail512-ag-4k.json \
  --case multirail-512gpu \
  --collective allgather \
  --coll-byte 4K \
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

uv run python scripts/run_syccl_simpletes.py \
  --manifest /home/antl/mntdisk/syccl-llm-scheme1-direct-events/manifest-multirail512-ag-4k.json \
  --case multirail-512gpu \
  --collective allgather \
  --coll-byte 4K \
  --condition full \
  --max-generations 10000 \
  --save-llm-io \
  --dry-run
```

## Run Several Message Sizes

After generating a manifest with multiple `--coll-bytes`, run one SimpleTES
search per size:

```bash
cd /home/antl/wzd/syccl/agent

for s in 4K 512K 1M 16M; do
  uv run python scripts/run_syccl_simpletes.py \
    --manifest /home/antl/mntdisk/syccl-llm-scheme1-direct-events/manifest-multirail512-ag.json \
    --case multirail-512gpu \
    --collective allgather \
    --coll-byte "$s" \
    --condition full \
    --max-generations 10000 \
    --save-llm-io
done
```

For alltoall, generate the manifest with `--collective alltoall`, then run:

```bash
cd /home/antl/wzd/syccl/agent

uv run python scripts/run_syccl_simpletes.py \
  --manifest /home/antl/mntdisk/syccl-llm-scheme1-direct-events/manifest-multirail512-a2a-4k.json \
  --case multirail-512gpu \
  --collective alltoall \
  --coll-byte 4K \
  --condition full \
  --max-generations 10000 \
  --save-llm-io
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
  Evaluator metrics, including validity, best_time_us, failure_category, and
  failure_feedback when the candidate fails.

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
`nodes.json`. For example, `syccl_expand_incompatible` means the compact sketch
could not be expanded by SyCCL's topology-symmetry mapping, often because a
transmission used an over-broad layer/group or produced mapped src/dst set size
changes.

Do not treat `syccl_expand_incompatible` as a harmless warning unless the
evaluator can still produce a valid `best_time_us`. The safer experiment is to
keep the candidate invalid while improving the prompt feedback text shown to the
model.

## Common Pitfalls

- `build/synthesize` is missing: build SyCCL before running SimpleTES.
- No failure feedback appears in prompts: check that the run is `--condition full`,
  not ablation, and inspect prompts after at least one batch has completed.
- No `llm_input` in `nodes.json`: rerun with `--save-llm-io`.
- Generated configs are not found: regenerate the manifest and configs with
  `scripts/syccl_config_matrix.py`.
- Private model endpoint goes through a proxy: the runner adds private API hosts
  to `NO_PROXY`, but check `agent/env.toml` and your shell proxy variables if
  requests still fail.
