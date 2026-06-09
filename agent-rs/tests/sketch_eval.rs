use llm_ccl_agent::{
    config_render::render_flow_sim_config,
    evaluator::{enumerate_small_scale_sketches, evaluate_sketch_json},
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
