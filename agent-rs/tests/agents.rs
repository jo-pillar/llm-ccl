use llm_ccl_agent::{
    agents::build_proposal_prompt,
    llm::{FakeLlmBackend, LlmBackend},
    topodsl::load_topodsl,
};

#[test]
fn topodsl_python_code_is_prompt_context() {
    let spec = load_topodsl("examples/topologies/clos_4host.py").unwrap();
    let prompt = build_proposal_prompt(
        &spec,
        "allgather",
        4096,
        "try fewer cross-host sends",
        &[],
    );

    assert!(prompt.contains("[TopoDSL Python Code]"));
    assert!(prompt.contains("```python"));
    assert!(prompt.contains("def topology()"));
    assert!(prompt.contains("try fewer cross-host sends"));
    assert!(!prompt.contains("[Topology JSON]"));
    assert!(!prompt.contains("topology_json"));
}

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
