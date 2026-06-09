# LLM-CCL Rust Agent Runner

`agent-rs` is the active Rust mainline for the LLM-CCL two-agent search loop.
The older SimpleTES implementation remains under `../agent` as a legacy path.

## Fake End-to-End Run

```bash
cargo run --manifest-path agent-rs/Cargo.toml -- \
  run \
  --topo agent-rs/examples/topologies/clos_4host.py \
  --collective allgather \
  --message-size 4096 \
  --rounds 10 \
  --output ./result/llm-ccl-agent-smoke \
  --fake-llm
```

The fake run executes 10 proposal rounds, evaluates each generated compact
SketchDSL candidate through `Flow-Simulator/flow-sim-rs`, and writes experiment
records without invoking SimpleTES.

## TopoDSL Input

The LLM prompt receives the user-editable TopoDSL source as Python code. The
fixture is:

```text
agent-rs/examples/topologies/clos_4host.py
```

Plain text `key=value` TopoDSL is also accepted for equivalent parameters:

```text
agent-rs/examples/topologies/clos_4host.txt
```

TopoDSL is not passed directly to the simulator as a topology generator. The
runner reads supported `clos` or `multirail` parameters and adjusts the existing
`flow-sim-rs` config fields for hosts, GPUs per host, NICs, switch counts,
bandwidths, latencies, collective, and message size.

## Output Artifacts

Each run writes:

- `rounds.jsonl`: per-round proposal, validation, expansion, and simulation
  metrics required by `cclpaper/experiment_plan.md`.
- `llm_calls.jsonl`: proposal-agent token, hash, model, timing, and cost fields.
- `best_sketch.json`: best compact SketchDSL candidate found so far.
- `best_translated.json`: expanded flow-sim schedule summary for the best round.
- `summary.json`: aggregate run summary including `simpletes_invoked=false`.
