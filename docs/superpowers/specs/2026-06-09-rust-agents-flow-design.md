# Rust Agents Flow Design

## Goal

Replace the active LLM-CCL agent workflow with a Rust implementation while keeping
the existing SimpleTES Python implementation available as a legacy path. The new
default workflow must not invoke SimpleTES inspiration selection, multi-chain
exploration, Python code evolution, or `agent/main.py`.

The acceptance target is an end-to-end run for one topology scale, one message
size, and `allgather`, with exactly 10 LLM interaction rounds and a final
generated algorithm candidate evaluated by the deterministic toolchain.

## Non-Goals

- Do not delete `agent/simpletes/` in this phase.
- Do not rewrite unrelated SyCCL, flow simulator, or paper-writing code.
- Do not keep the SimpleTES inspiration pool or multi-chain policy in the new
  mainline.
- Do not make the LLM responsible for full event schedules, dependencies, or
  link-level timing.

## Entry Points

The new workflow will live in a Rust runner, tentatively `agent-rs`, with a CLI
similar to:

```bash
cargo run --manifest-path agent-rs/Cargo.toml -- \
  run \
  --topo examples/topologies/clos_4host.py \
  --collective allgather \
  --message-size 4096 \
  --rounds 10 \
  --output /tmp/llm-ccl-run
```

SimpleTES remains callable only through its old explicit commands. New docs and
tests will use the Rust runner as the default path.

## TopoDSL Format

TopoDSL should be easy for users to edit directly. The runner will support two
forms:

1. Python topology spec files.
   - A Python file exposes a small function, for example `build_topology()`,
     returning JSON-serializable topology data.
   - This is the preferred path for parameterized topologies and scale-down
     generation.
   - Rust executes the Python file in a controlled subprocess and reads one JSON
     object from stdout.

2. Plain text topology spec files.
   - A simple line-oriented format for quick edits.
   - The parser accepts key-value fields for hosts, GPUs per host, collective,
     message size, layers, groups, bandwidth, and latency.
   - The parsed result is normalized into the same internal topology model used
     by Python specs.

Both forms compile into one canonical internal `TopologySpec` and then into the
SyCCL/flow-sim config shape. The canonical topology object includes a stable
hash used as `topodsl_config_id` in experiment records.

## Architecture

The new Rust implementation has five bounded components:

- `topodsl`: load Python or text topology specs, normalize them, generate
  scale-down variants, and emit SyCCL/flow-sim compatible configs.
- `sketch_search`: call `syccl-sketch-search` as a Rust library for exhaustive
  small-scale root-sketch enumeration. This module is the meaning of "sketch
  exhaustive search" in this workflow.
- `evaluator`: validate candidate SketchDSL, expand root sketches for
  `allgather`, call the flow simulator, and collect completion-time and
  bottleneck metrics.
- `agents`: implement the two-agent loop from `cclpaper/usenix2019_v3.1_zh.tex`:
  proposal agent and record agent.
- `records`: persist raw per-round records and raw per-LLM-call records required
  by `cclpaper/experiment_plan.md`.

The first implementation should favor clear interfaces and narrow behavior over
generalizing to every collective. `allgather` is the required acceptance path;
`alltoall` can remain a later extension unless an existing helper is cheap to
reuse without increasing risk.

## Data Flow

1. Load TopoDSL from Python or plain text.
2. Compile it into the canonical topology model and the concrete target config.
3. Run a topology sanity check before any LLM calls.
4. Find a smaller same-family topology where exhaustive sketch search finishes
   within the configured budget.
5. Call `syccl-sketch-search` on that small topology and evaluate the enumerated
   sketches through the deterministic pipeline.
6. Ask the record agent to summarize the small-scale exploration records.
7. For each of 10 target-scale rounds:
   - Build the proposal prompt from fixed topology context, small-scale records,
     target-scale history, static errors, bottlenecks, and the latest direction
     hint.
   - Ask the proposal agent for one or more SketchDSL candidates.
   - Parse and normalize each candidate.
   - Reject invalid candidates before simulation with field-level feedback.
   - Expand valid candidates and evaluate them with flow-sim.
   - Append round records and update best-so-far state.
   - Ask the record agent to summarize this round, update history, and, if
     needed, emit a new exploration direction.
8. Return the best sketch, translated schedule artifact, metrics, and run
   records.

## Agent Responsibilities

### Proposal Agent

The proposal agent only generates root-level SketchDSL. It receives:

- target topology summary and layer/group semantics;
- collective type and message size;
- SketchDSL syntax constraints;
- small-scale enumeration records;
- target-scale candidate archive summary;
- recent static errors and bottleneck reports;
- the current direction hint from the record agent.

It must not output full allgather event schedules or dependency fields.

### Record Agent

The record agent owns history shaping, not candidate generation. It receives
deterministic feedback and produces:

- a concise per-round record summary;
- an updated rolling history summary when context grows too large;
- a new direction hint when search stagnates or candidates are too similar;
- structured failure feedback for invalid candidates;
- raw LLM call metadata for cost accounting.

The record agent summary update and new-direction hint are explicit test targets.

## Records

The run directory should include at least:

- `rounds.jsonl`: one record per search round.
- `llm_calls.jsonl`: one record per official LLM call.
- `best_sketch.json`: best root sketch.
- `best_translated.json`: expanded schedule or simulator input artifact.
- `summary.json`: run-level aggregate metrics and token totals.

`rounds.jsonl` must include the experiment-plan fields:

- `topodsl_config_id`
- `collective`
- `message_size`
- `round_id`
- `prompt_hash`
- `input_tokens`
- `output_tokens`
- `candidate_sketch`
- `validity_status`
- `expanded_events`
- `completion_time`
- `algorithm_bandwidth`
- `critical_path`
- `bottleneck_links`
- `best_so_far`

`llm_calls.jsonl` must include:

- `run_id`
- `agent_type`
- `round_id`
- `model_name`
- `input_tokens`
- `output_tokens`
- `cached_input_tokens`
- `reasoning_tokens`
- `wall_clock_time_ms`
- `api_cost_usd`
- `prompt_hash`
- `completion_hash`

When an API does not expose a token field, the value is recorded as `null` or
`N/A` rather than estimated from chat history.

## LLM Backend

The Rust runner should use a narrow OpenAI-compatible chat-completions adapter
first because the existing environment already relies on model/api-base/api-key
configuration. The adapter records request and response hashes, token usage when
available, and call latency. A fake LLM backend is required for deterministic
tests.

The adapter must keep prompts and completions available in an optional debug mode
but should always record hashes and token counts for experiment reproducibility.

## Error Handling

- TopoDSL parse or execution errors stop before LLM calls.
- Topology sanity-check failures stop before sketch search.
- SketchDSL parse, schema, topology group, propagation, and coverage errors are
  converted into record-agent feedback and do not invoke flow-sim.
- Flow-sim failures are recorded as evaluation failures with logs and finite
  penalty metrics.
- LLM failures are recorded in `llm_calls.jsonl`; the run stops if a required
  agent call cannot be completed after configured retries.

## Testing Strategy

The implementation should be test-driven. Required focused tests:

- Python TopoDSL compiles into canonical topology and SyCCL/flow-sim config.
- Plain-text TopoDSL compiles into the same canonical shape for a small Clos
  topology.
- The Rust runner invokes `syccl-sketch-search` for small-scale enumeration.
- Fake proposal agent plus fake record agent run exactly 10 target rounds.
- A valid allgather candidate for one small topology and one message size
  reaches flow-sim and produces positive completion time.
- Record agent summary update preserves best structures and repeated failures.
- Record agent new-direction hint triggers after stagnation or duplicate-like
  candidates.
- `rounds.jsonl` and `llm_calls.jsonl` contain the raw exploration fields from
  the experiment plan.
- Legacy SimpleTES tests remain runnable but are not part of the new default
  acceptance path.

## Migration Plan

1. Add the Rust runner and tests without modifying `agent/simpletes/`.
2. Add docs that mark SimpleTES commands as legacy and point default usage to the
   Rust runner.
3. Once the Rust runner is stable, remove SimpleTES in a later cleanup phase.

## Open Decisions

- Exact Rust crate location: `agent-rs/` is preferred for isolation; placing it
  under `agent/` is acceptable if repository conventions require it.
- Exact plain-text TopoDSL grammar can stay intentionally small for the first
  allgather acceptance case.
- `alltoall` support should not block the first implementation unless reuse from
  existing translator code is straightforward.
