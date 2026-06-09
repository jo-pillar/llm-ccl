# Rust Agents Flow Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the new Rust LLM-CCL agent mainline, with SimpleTES kept as legacy, and pass the minimal fake-LLM allgather end-to-end acceptance tests from the approved design.

**Architecture:** Add an isolated `agent-rs` crate. It reads Python-code or plain-text TopoDSL into canonical topology parameters, maps those parameters into existing `multirail`/`clos` flow-sim config JSON, calls `syccl-sketch-search` for small-scale enumeration, runs proposal/record agents through pluggable fake/OpenAI-compatible backends, evaluates root sketches through `flow_sim_rs::sketch` and `flow_sim_rs::simulator`, and writes experiment records.

**Tech Stack:** Rust 2021, `cargo test`, `assert_cmd`, `tempfile`, `serde_json`, path dependencies on `syccl-sketch-search` and `flow-sim-rs`. Existing Python SimpleTES code remains untouched except documentation marking it legacy.

---

## File Structure

- Create `agent-rs/Cargo.toml`: Rust package definition and path dependencies.
- Create `agent-rs/src/lib.rs`: module exports for tests.
- Create `agent-rs/src/main.rs`: CLI entry point.
- Create `agent-rs/src/topodsl.rs`: Python-code/text TopoDSL loading, canonical `TopologyParams`, stable hash, scale-down parameters.
- Create `agent-rs/src/config_render.rs`: render adjusted SyCCL/flow-sim config JSON for `clos` and `multirail`.
- Create `agent-rs/src/sketch_dsl.rs`: parse proposal-agent SketchDSL JSON or Python-list-like compact transmissions into canonical sketch JSON.
- Create `agent-rs/src/evaluator.rs`: validate and evaluate sketches through `flow_sim_rs` library APIs.
- Create `agent-rs/src/llm.rs`: `LlmBackend` trait, fake backend, OpenAI-compatible backend scaffold.
- Create `agent-rs/src/agents.rs`: proposal prompt builder, proposal call, record agent summary/direction logic.
- Create `agent-rs/src/records.rs`: `rounds.jsonl`, `llm_calls.jsonl`, `summary.json`, best artifacts.
- Create `agent-rs/src/run.rs`: end-to-end orchestration.
- Create `agent-rs/examples/topologies/clos_4host.py`: prompt-facing Python TopoDSL fixture.
- Create `agent-rs/examples/topologies/clos_4host.txt`: equivalent text TopoDSL fixture.
- Create `agent-rs/tests/topodsl.rs`: TopoDSL parsing/config mapping tests.
- Create `agent-rs/tests/sketch_eval.rs`: sketch parse/evaluate/search tests.
- Create `agent-rs/tests/records.rs`: record agent and required raw fields tests.
- Create `agent-rs/tests/e2e_fake_llm.rs`: minimal 10-round fake-LLM acceptance test.
- Modify `agent/README.md`: mark SimpleTES as legacy and point to `agent-rs`.

## Task 1: Crate Skeleton and CLI

**Files:**
- Create: `agent-rs/Cargo.toml`
- Create: `agent-rs/src/lib.rs`
- Create: `agent-rs/src/main.rs`
- Test: `agent-rs/tests/cli.rs`

- [ ] **Step 1: Write the failing CLI smoke test**

```rust
use assert_cmd::Command;

#[test]
fn cli_prints_help() {
    Command::cargo_bin("llm-ccl-agent")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicates::str::contains("LLM-CCL Rust agent runner"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd agent-rs && cargo test --test cli cli_prints_help`

Expected: FAIL because `agent-rs` and binary do not exist.

- [ ] **Step 3: Create minimal crate**

`agent-rs/Cargo.toml`:

```toml
[package]
name = "llm-ccl-agent"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1.0"
clap = { version = "4.5", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
sha2 = "0.10"
hex = "0.4"
syccl-sketch-search = { path = "../syccl-sketch-search" }
flow-sim-rs = { path = "../Flow-Simulator/flow-sim-rs" }

[dev-dependencies]
assert_cmd = "2.0"
predicates = "3.1"
tempfile = "3.10"
```

`agent-rs/src/lib.rs`:

```rust
// Module exports are added as implementation tasks create them.
```

`agent-rs/src/main.rs`:

```rust
use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(about = "LLM-CCL Rust agent runner")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Run {
        #[arg(long)]
        topo: std::path::PathBuf,
        #[arg(long, default_value = "allgather")]
        collective: String,
        #[arg(long)]
        message_size: u64,
        #[arg(long, default_value_t = 10)]
        rounds: usize,
        #[arg(long)]
        output: std::path::PathBuf,
    },
}

fn main() -> Result<()> {
    let _cli = Cli::parse();
    Ok(())
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd agent-rs && cargo test --test cli cli_prints_help`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent-rs/Cargo.toml agent-rs/src/lib.rs agent-rs/src/main.rs agent-rs/tests/cli.rs
git commit -m "feat: add rust agent crate skeleton"
```

## Task 2: TopoDSL Parameters and Flow-Sim Config Mapping

**Files:**
- Modify: `agent-rs/src/lib.rs`
- Create: `agent-rs/src/topodsl.rs`
- Create: `agent-rs/src/config_render.rs`
- Create: `agent-rs/examples/topologies/clos_4host.py`
- Create: `agent-rs/examples/topologies/clos_4host.txt`
- Test: `agent-rs/tests/topodsl.rs`

- [ ] **Step 1: Write failing test for Python-code TopoDSL prompt context**

```rust
use llm_ccl_agent::topodsl::load_topodsl;

#[test]
fn topodsl_python_code_is_preserved_for_prompt() {
    let spec = load_topodsl("examples/topologies/clos_4host.py").unwrap();

    assert_eq!(spec.params.family, "clos");
    assert!(spec.prompt_source.contains("def topology()"));
    assert!(!spec.prompt_source.trim_start().starts_with('{'));
    assert_eq!(spec.params.hosts, 4);
    assert_eq!(spec.params.gpus_per_host, 2);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd agent-rs && cargo test --test topodsl topodsl_python_code_is_preserved_for_prompt`

Expected: FAIL because `topodsl` does not exist.

- [ ] **Step 3: Implement minimal Python-code loader**

Use a controlled Python subprocess that imports the file and prints `json.dumps(topology())`.

`agent-rs/examples/topologies/clos_4host.py`:

```python
def topology():
    return {
        "family": "clos",
        "hosts": 4,
        "gpus_per_host": 2,
        "nics_per_host": 1,
        "leaf_switches": 2,
        "spine_switches": 1,
        "collective": "allgather",
        "message_size": 4096,
        "host_bw_mbpus": 0.15,
        "host_lat_us": 3.0,
        "nic_bw_mbpus": 0.0455,
        "nic_lat_us": 0.0,
        "leaf_bw_mbpus": 0.0455,
        "leaf_lat_us": 10.0,
        "spine_bw_mbpus": 0.36,
        "spine_lat_us": 10.0,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd agent-rs && cargo test --test topodsl topodsl_python_code_is_preserved_for_prompt`

Expected: PASS.

- [ ] **Step 5: Write failing test for text TopoDSL equivalence**

```rust
use llm_ccl_agent::topodsl::load_topodsl;

#[test]
fn text_topodsl_maps_to_same_params_shape() {
    let py = load_topodsl("examples/topologies/clos_4host.py").unwrap();
    let txt = load_topodsl("examples/topologies/clos_4host.txt").unwrap();

    assert_eq!(txt.params.family, py.params.family);
    assert_eq!(txt.params.hosts, py.params.hosts);
    assert_eq!(txt.params.gpus_per_host, py.params.gpus_per_host);
    assert_eq!(txt.params.leaf_switches, py.params.leaf_switches);
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd agent-rs && cargo test --test topodsl text_topodsl_maps_to_same_params_shape`

Expected: FAIL because text parser does not exist.

- [ ] **Step 7: Implement small line-oriented text parser**

`agent-rs/examples/topologies/clos_4host.txt`:

```text
family=clos
hosts=4
gpus_per_host=2
nics_per_host=1
leaf_switches=2
spine_switches=1
collective=allgather
message_size=4096
host_bw_mbpus=0.15
host_lat_us=3
nic_bw_mbpus=0.0455
nic_lat_us=0
leaf_bw_mbpus=0.0455
leaf_lat_us=10
spine_bw_mbpus=0.36
spine_lat_us=10
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd agent-rs && cargo test --test topodsl text_topodsl_maps_to_same_params_shape`

Expected: PASS.

- [ ] **Step 9: Write failing test for flow-sim config parameter update**

```rust
use flow_sim_rs::config::{parse_config, TopologyKind};
use llm_ccl_agent::{config_render::render_flow_sim_config, topodsl::load_topodsl};

#[test]
fn topodsl_params_update_flow_sim_config() {
    let spec = load_topodsl("examples/topologies/clos_4host.py").unwrap();
    let json = render_flow_sim_config(&spec.params, "allgather", 8192).unwrap();
    let cfg = parse_config(json.as_bytes()).unwrap();

    assert_eq!(cfg.topology, TopologyKind::Clos);
    assert_eq!(cfg.hosts.host_num, 4);
    assert_eq!(cfg.hosts.gpus_per_host, 2);
    assert_eq!(cfg.coll_name, "allgather");
    assert_eq!(cfg.coll_bytes, 8192);
}
```

- [ ] **Step 10: Run test to verify it fails**

Run: `cd agent-rs && cargo test --test topodsl topodsl_params_update_flow_sim_config`

Expected: FAIL because `config_render` does not exist.

- [ ] **Step 11: Implement `render_flow_sim_config` for `clos` and `multirail`**

Keep scope to current flow-sim config fields:

- `coll.name`, `coll.byte`
- `hosts.host_num`, `hosts.host_gpu_num`, `hosts.host_nic_num`
- `topo` switch shape for `multirail` or `clos`
- `link_spec` bandwidth/latency fields

- [ ] **Step 12: Run test to verify it passes**

Run: `cd agent-rs && cargo test --test topodsl`

Expected: PASS.

- [ ] **Step 13: Commit**

```bash
git add agent-rs/src/topodsl.rs agent-rs/src/config_render.rs agent-rs/src/lib.rs agent-rs/examples/topologies/clos_4host.py agent-rs/examples/topologies/clos_4host.txt agent-rs/tests/topodsl.rs
git commit -m "feat: add topodsl parameter mapping"
```

## Task 3: Sketch Search and Flow-Sim Evaluation

**Files:**
- Modify: `agent-rs/src/lib.rs`
- Create: `agent-rs/src/sketch_dsl.rs`
- Create: `agent-rs/src/evaluator.rs`
- Test: `agent-rs/tests/sketch_eval.rs`

- [ ] **Step 1: Write failing test that `syccl-sketch-search` is invoked**

```rust
use llm_ccl_agent::{
    config_render::render_flow_sim_config,
    evaluator::enumerate_small_scale_sketches,
    topodsl::load_topodsl,
};

#[test]
fn invokes_syccl_sketch_search_for_small_scale_enumeration() {
    let spec = load_topodsl("examples/topologies/clos_4host.py").unwrap();
    let config_json = render_flow_sim_config(&spec.params, "allgather", 4096).unwrap();
    let sketches = enumerate_small_scale_sketches(&config_json, 2).unwrap();

    assert!(!sketches.is_empty());
    assert!(sketches.len() <= 2);
    assert_eq!(sketches[0].src_gpu, 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd agent-rs && cargo test --test sketch_eval invokes_syccl_sketch_search_for_small_scale_enumeration`

Expected: FAIL because evaluator module does not exist.

- [ ] **Step 3: Implement `enumerate_small_scale_sketches`**

Parse `config_json` into `syccl_sketch_search::SycclConfig` and call:

```rust
syccl_sketch_search::search_sketches(
    &config,
    syccl_sketch_search::SearchOptions {
        limit: Some(limit),
        parallelism: 1,
    },
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd agent-rs && cargo test --test sketch_eval invokes_syccl_sketch_search_for_small_scale_enumeration`

Expected: PASS.

- [ ] **Step 5: Write failing test for compact SketchDSL evaluation**

```rust
use llm_ccl_agent::{
    config_render::render_flow_sim_config,
    evaluator::evaluate_sketch_json,
    topodsl::load_topodsl,
};

#[test]
fn valid_allgather_candidate_reaches_flow_sim() {
    let spec = load_topodsl("examples/topologies/clos_4host.py").unwrap();
    let config_json = render_flow_sim_config(&spec.params, "allgather", 4096).unwrap();
    let sketch_json = r#"[
      [
        [0, 1, 0, 0, 1],
        [0, 4, 0, 0, [2, 4, 6]],
        [1, 1, 1, 2, 3],
        [1, 1, 2, 4, 5],
        [1, 1, 3, 6, 7]
      ]
    ]"#;

    let metrics = evaluate_sketch_json(&config_json, sketch_json).unwrap();

    assert_eq!(metrics.validity_status, "valid");
    assert!(metrics.completion_time_us > 0.0);
    assert!(metrics.expanded_events > 0);
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd agent-rs && cargo test --test sketch_eval valid_allgather_candidate_reaches_flow_sim`

Expected: FAIL because `evaluate_sketch_json` does not exist.

- [ ] **Step 7: Implement `evaluate_sketch_json` using flow-sim library APIs**

Use:

```rust
let config = flow_sim_rs::config::parse_config(config_json.as_bytes())?;
let sketches = flow_sim_rs::sketch::parse_compact_sketches(sketch_json.as_bytes(), &config)?;
let schedule = flow_sim_rs::sketch::sketches_to_translated_schedule(&sketches, &config)?;
let result = flow_sim_rs::simulator::simulate_case(&config, &schedule)?;
```

Return `EvaluationMetrics` with:

- `validity_status`
- `completion_time_us`
- `expanded_events = schedule.sends.len()`
- `algorithm_bandwidth = message_size / completion_time`
- `critical_path`
- `bottleneck_links`

- [ ] **Step 8: Run test to verify it passes**

Run: `cd agent-rs && cargo test --test sketch_eval`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add agent-rs/src/sketch_dsl.rs agent-rs/src/evaluator.rs agent-rs/src/lib.rs agent-rs/tests/sketch_eval.rs
git commit -m "feat: evaluate sketches through flow simulator"
```

## Task 4: LLM Backends and Prompt Construction

**Files:**
- Modify: `agent-rs/src/lib.rs`
- Create: `agent-rs/src/llm.rs`
- Create: `agent-rs/src/agents.rs`
- Test: `agent-rs/tests/agents.rs`

- [ ] **Step 1: Write failing test for proposal prompt using Python TopoDSL text**

```rust
use llm_ccl_agent::{
    agents::build_proposal_prompt,
    topodsl::load_topodsl,
};

#[test]
fn topodsl_python_code_is_prompt_context() {
    let spec = load_topodsl("examples/topologies/clos_4host.py").unwrap();
    let prompt = build_proposal_prompt(&spec, "allgather", 4096, "try fewer cross-host sends", &[]);

    assert!(prompt.contains("def topology()"));
    assert!(prompt.contains("try fewer cross-host sends"));
    assert!(!prompt.contains("\"family\": \"clos\""));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd agent-rs && cargo test --test agents topodsl_python_code_is_prompt_context`

Expected: FAIL because `agents` module does not exist.

- [ ] **Step 3: Implement prompt builder**

Prompt sections:

- fixed task instruction;
- Python-code TopoDSL source;
- derived layer/group summary;
- compact SketchDSL output rules;
- small-scale exploration summaries;
- current history summary;
- record-agent direction hint.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd agent-rs && cargo test --test agents topodsl_python_code_is_prompt_context`

Expected: PASS.

- [ ] **Step 5: Write failing fake LLM backend test**

```rust
use llm_ccl_agent::llm::{FakeLlmBackend, LlmBackend};

#[test]
fn fake_llm_returns_scripted_calls_with_token_metadata() {
    let mut fake = FakeLlmBackend::new(vec!["[[[0,1,0,0,1]]]".to_string()]);
    let call = fake.complete("proposal", 0, "prompt text").unwrap();

    assert_eq!(call.agent_type, "proposal");
    assert_eq!(call.round_id, 0);
    assert!(call.input_tokens.unwrap() > 0);
    assert!(call.output_tokens.unwrap() > 0);
    assert_eq!(call.completion, "[[[0,1,0,0,1]]]");
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd agent-rs && cargo test --test agents fake_llm_returns_scripted_calls_with_token_metadata`

Expected: FAIL because `llm` module does not exist.

- [ ] **Step 7: Implement `LlmBackend` and `FakeLlmBackend`**

Trait:

```rust
pub trait LlmBackend {
    fn complete(&mut self, agent_type: &str, round_id: usize, prompt: &str) -> anyhow::Result<LlmCall>;
}
```

`LlmCall` includes prompt/completion text, hashes, model name, token metadata, and wall-clock time.

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd agent-rs && cargo test --test agents`

Expected: PASS.

- [ ] **Step 9: Add OpenAI-compatible backend scaffold**

Implement configuration and a clear `unimplemented!("real backend wiring")` boundary only if the fake backend is enough for acceptance. If wiring real HTTP now, use a small blocking client and record token usage when provider returns it. Do not block acceptance on live network tests.

- [ ] **Step 10: Commit**

```bash
git add agent-rs/src/llm.rs agent-rs/src/agents.rs agent-rs/src/lib.rs agent-rs/tests/agents.rs
git commit -m "feat: add agent prompts and fake llm backend"
```

## Task 5: Record Agent and Experiment Records

**Files:**
- Modify: `agent-rs/src/lib.rs`
- Modify: `agent-rs/src/agents.rs`
- Create: `agent-rs/src/records.rs`
- Test: `agent-rs/tests/records.rs`

- [ ] **Step 1: Write failing record-agent summary/direction test**

```rust
use llm_ccl_agent::agents::{RecordAgentState, RoundOutcome};

#[test]
fn record_agent_summary_and_direction() {
    let mut state = RecordAgentState::default();
    for round_id in 0..3 {
        state.observe(RoundOutcome {
            round_id,
            sketch_fingerprint: "same".to_string(),
            validity_status: "valid".to_string(),
            completion_time_us: Some(10.0),
            best_so_far: round_id == 0,
            bottleneck_links: vec!["0->2 queue_wait=10".to_string()],
            failure_feedback: None,
        });
    }

    assert!(state.summary().contains("best"));
    assert!(state.direction_hint().contains("change"));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd agent-rs && cargo test --test records record_agent_summary_and_direction`

Expected: FAIL because record state does not exist.

- [ ] **Step 3: Implement deterministic record-agent state**

Keep this logic deterministic for tests:

- preserve best structures;
- count repeated failures;
- detect stagnation after 3 non-improving or duplicate-like outcomes;
- emit a direction hint that asks for changed cross-domain order, relay distribution, or concurrency.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd agent-rs && cargo test --test records record_agent_summary_and_direction`

Expected: PASS.

- [ ] **Step 5: Write failing required-fields record test**

```rust
use llm_ccl_agent::records::{RunRecorder, RoundRecord, LlmCallRecord};
use serde_json::Value;
use tempfile::tempdir;

#[test]
fn experiment_plan_raw_records_have_required_fields() {
    let dir = tempdir().unwrap();
    let mut recorder = RunRecorder::new(dir.path(), "run-a").unwrap();
    recorder.append_round(&RoundRecord::fixture()).unwrap();
    recorder.append_llm_call(&LlmCallRecord::fixture()).unwrap();
    recorder.finish_summary().unwrap();

    let round: Value = serde_json::from_str(&std::fs::read_to_string(dir.path().join("rounds.jsonl")).unwrap()).unwrap();
    for field in ["topodsl_config_id", "collective", "message_size", "round_id", "prompt_hash", "input_tokens", "output_tokens", "candidate_sketch", "validity_status", "expanded_events", "completion_time", "algorithm_bandwidth", "critical_path", "bottleneck_links", "best_so_far"] {
        assert!(round.get(field).is_some(), "missing {field}");
    }

    let call: Value = serde_json::from_str(&std::fs::read_to_string(dir.path().join("llm_calls.jsonl")).unwrap()).unwrap();
    for field in ["run_id", "agent_type", "round_id", "model_name", "input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens", "wall_clock_time_ms", "api_cost_usd", "prompt_hash", "completion_hash"] {
        assert!(call.get(field).is_some(), "missing {field}");
    }
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd agent-rs && cargo test --test records experiment_plan_raw_records_have_required_fields`

Expected: FAIL because recorder does not exist.

- [ ] **Step 7: Implement `RunRecorder`**

Write:

- `rounds.jsonl`
- `llm_calls.jsonl`
- `best_sketch.json`
- `best_translated.json`
- `summary.json`

Use append-only JSONL for per-round/per-call data.

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd agent-rs && cargo test --test records`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add agent-rs/src/records.rs agent-rs/src/agents.rs agent-rs/src/lib.rs agent-rs/tests/records.rs
git commit -m "feat: record agent history and experiment records"
```

## Task 6: End-to-End Runner

**Files:**
- Modify: `agent-rs/src/lib.rs`
- Modify: `agent-rs/src/main.rs`
- Create: `agent-rs/src/run.rs`
- Test: `agent-rs/tests/e2e_fake_llm.rs`

- [ ] **Step 1: Write failing minimal fake-LLM acceptance test**

```rust
use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

#[test]
fn minimal_allgather_fake_llm_10_rounds() {
    let out = tempdir().unwrap();
    Command::cargo_bin("llm-ccl-agent")
        .unwrap()
        .args([
            "run",
            "--topo",
            "examples/topologies/clos_4host.py",
            "--collective",
            "allgather",
            "--message-size",
            "4096",
            "--rounds",
            "10",
            "--output",
            out.path().to_str().unwrap(),
            "--fake-llm",
        ])
        .assert()
        .success();

    let rounds = std::fs::read_to_string(out.path().join("rounds.jsonl")).unwrap();
    assert_eq!(rounds.lines().count(), 10);
    assert!(out.path().join("llm_calls.jsonl").exists());
    assert!(out.path().join("best_sketch.json").exists());
    assert!(out.path().join("best_translated.json").exists());
    assert!(out.path().join("summary.json").exists());

    let summary: Value = serde_json::from_str(&std::fs::read_to_string(out.path().join("summary.json")).unwrap()).unwrap();
    assert!(summary["best_completion_time_us"].as_f64().unwrap() > 0.0);
    assert_eq!(summary["simpletes_invoked"], false);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd agent-rs && cargo test --test e2e_fake_llm minimal_allgather_fake_llm_10_rounds`

Expected: FAIL because runner is not implemented.

- [ ] **Step 3: Implement `run_fake_llm` orchestration**

Flow:

1. Load TopoDSL.
2. Render target config.
3. Run a simple sanity check with `flow_sim_rs::config::parse_config`.
4. Enumerate small-scale sketches with limit 2.
5. Evaluate enumerated sketch records.
6. Build 10 proposal prompts.
7. Use fake proposal outputs that include at least one valid allgather compact sketch.
8. Evaluate each round.
9. Update record-agent state.
10. Write all artifacts and summary.

No SimpleTES import, process command, or path should appear anywhere in this flow.

- [ ] **Step 4: Add `--fake-llm` CLI flag and call `run::run`**

Extend `Commands::Run` with:

```rust
#[arg(long)]
fake_llm: bool,
```

For now, require `--fake-llm` for tests and return a clear error for live LLM mode if the real backend is not wired.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd agent-rs && cargo test --test e2e_fake_llm minimal_allgather_fake_llm_10_rounds`

Expected: PASS.

- [ ] **Step 6: Run full agent-rs test suite**

Run: `cd agent-rs && cargo test`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add agent-rs/src/run.rs agent-rs/src/main.rs agent-rs/src/lib.rs agent-rs/tests/e2e_fake_llm.rs
git commit -m "feat: run minimal rust agent loop"
```

## Task 7: Documentation and Legacy Boundary

**Files:**
- Modify: `agent/README.md`
- Create: `agent-rs/README.md`
- Test: no new test; verify docs and CLI commands.

- [ ] **Step 1: Update `agent/README.md`**

Add a short top section:

```markdown
## Legacy SimpleTES Path

This directory keeps the old SimpleTES-based SyCCL agent for reference and
backward compatibility. The active LLM-CCL agent mainline is `../agent-rs`.
New experiments should use the Rust runner; these commands are legacy and must
not be used for the new two-agent feedback loop.
```

- [ ] **Step 2: Create `agent-rs/README.md`**

Include:

- CLI command for fake end-to-end run.
- Python TopoDSL example path.
- Note that the LLM prompt receives Python-code TopoDSL.
- Note that flow-sim receives adjusted `clos`/`multirail` config parameters, not raw TopoDSL.
- Output artifacts list.

- [ ] **Step 3: Verify docs commands**

Run: `cd agent-rs && cargo run -- --help`

Expected: help text prints successfully.

- [ ] **Step 4: Commit**

```bash
git add agent/README.md agent-rs/README.md
git commit -m "docs: document rust agent runner"
```

## Task 8: Full Verification

**Files:**
- No intended file changes.

- [ ] **Step 1: Run `agent-rs` tests**

Run: `cd agent-rs && cargo test`

Expected: PASS.

- [ ] **Step 2: Run dependency crate tests touched by integration**

Run: `cd syccl-sketch-search && cargo test`

Expected: PASS.

Run: `cd Flow-Simulator/flow-sim-rs && cargo test compact_sketch_expands_and_simulates_without_translated_file`

Expected: PASS.

- [ ] **Step 3: Run fake CLI manually**

Run:

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

Expected:

- exit code 0;
- `result/llm-ccl-agent-smoke/rounds.jsonl` has 10 lines;
- `summary.json` reports positive `best_completion_time_us`;
- no SimpleTES command is run.

- [ ] **Step 4: Check git status**

Run: `git status --short`

Expected: only intentional tracked changes are present; untracked pre-existing directories remain ignored by the implementation commits.

- [ ] **Step 5: Final commit if verification produced doc or fixture adjustments**

```bash
git add <intentional files>
git commit -m "test: verify rust agent acceptance flow"
```
