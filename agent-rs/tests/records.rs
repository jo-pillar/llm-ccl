use llm_ccl_agent::{
    agents::{RecordAgentState, RoundOutcome},
    records::{LlmCallRecord, RoundRecord, RunRecorder},
};
use serde_json::Value;
use tempfile::tempdir;

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

#[test]
fn experiment_plan_raw_records_have_required_fields() {
    let dir = tempdir().unwrap();
    let mut recorder = RunRecorder::new(dir.path(), "run-a").unwrap();
    recorder.append_round(&RoundRecord::fixture()).unwrap();
    recorder.append_llm_call(&LlmCallRecord::fixture()).unwrap();
    recorder.finish_summary().unwrap();

    let round: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("rounds.jsonl")).unwrap())
            .unwrap();
    for field in [
        "topodsl_config_id",
        "collective",
        "message_size",
        "round_id",
        "prompt_hash",
        "input_tokens",
        "output_tokens",
        "candidate_sketch",
        "validity_status",
        "expanded_events",
        "completion_time",
        "algorithm_bandwidth",
        "critical_path",
        "bottleneck_links",
        "best_so_far",
    ] {
        assert!(round.get(field).is_some(), "missing {field}");
    }

    let call: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("llm_calls.jsonl")).unwrap())
            .unwrap();
    for field in [
        "run_id",
        "agent_type",
        "round_id",
        "model_name",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "wall_clock_time_ms",
        "api_cost_usd",
        "prompt_hash",
        "completion_hash",
    ] {
        assert!(call.get(field).is_some(), "missing {field}");
    }
}
