use llm_ccl_agent::topodsl::load_topodsl;
use flow_sim_rs::config::{parse_config, TopologyKind};
use llm_ccl_agent::config_render::render_flow_sim_config;

#[test]
fn topodsl_python_code_is_preserved_for_prompt() {
    let spec = load_topodsl("examples/topologies/clos_4host.py").unwrap();

    assert_eq!(spec.params.family, "clos");
    assert!(spec.prompt_source.contains("def topology()"));
    assert!(!spec.prompt_source.trim_start().starts_with('{'));
    assert_eq!(spec.params.hosts, 4);
    assert_eq!(spec.params.gpus_per_host, 2);
}

#[test]
fn text_topodsl_maps_to_same_params_shape() {
    let py = load_topodsl("examples/topologies/clos_4host.py").unwrap();
    let txt = load_topodsl("examples/topologies/clos_4host.txt").unwrap();

    assert_eq!(txt.params.family, py.params.family);
    assert_eq!(txt.params.hosts, py.params.hosts);
    assert_eq!(txt.params.gpus_per_host, py.params.gpus_per_host);
    assert_eq!(txt.params.leaf_switches, py.params.leaf_switches);
}

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
