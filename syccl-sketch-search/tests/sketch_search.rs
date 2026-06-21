use std::collections::BTreeSet;

use assert_cmd::Command;
use syccl_sketch_search::{search_sketches, topology::Topology, SearchOptions, SycclConfig};

fn parse_config(json: &str) -> SycclConfig {
    serde_json::from_str(json).expect("valid config")
}

fn tiny_single_host_config() -> &'static str {
    r#"{
      "coll": {
        "name": "allgather",
        "byte": 4096,
        "root_sender": -1,
        "root_receiver": -1
      },
      "hosts": {
        "host_num": 1,
        "host_gpu_num": 4,
        "host_nic_num": 0,
        "host_links": "nvswitch"
      },
      "topo": [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"}
      ],
      "host_links": {
        "nvlink": [[1,2,3], [0,2,3], [0,1,3], [0,1,2]]
      },
      "prune": {
        "max_comb_layer_num": 1,
        "allow_unequal_source_per_group": false,
        "ignored_layers": [0],
        "must_contain_layers": [1],
        "layer_comm_ordering": {},
        "layer_comm_uplimit": {},
        "all_groups_used": false,
        "limit_num_steps": 3,
        "allow_source_group_special": true
      },
      "sketch": {
        "customize_sketch": false,
        "use_sketch_input": false,
        "save_sketch": false,
        "sketch_path": ""
      }
    }"#
}

fn tiny_clos_config() -> &'static str {
    r#"{
      "coll": {
        "name": "allgather",
        "byte": 4096,
        "root_sender": -1,
        "root_receiver": -1
      },
      "hosts": {
        "host_num": 4,
        "host_gpu_num": 4,
        "host_nic_num": 2,
        "host_links": "nvswitch"
      },
      "topo": [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {"layer_id": 3, "type": "switch", "switch_topo": "pod", "switch_num": 2, "link_spec": "netlink_leaf"},
        {"layer_id": 4, "type": "switch", "switch_topo": "pod", "switch_num": 1, "link_spec": "netlink_spine"}
      ],
      "host_links": {
        "nvlink": [[1,2,3], [0,2,3], [0,1,3], [0,1,2]]
      },
      "prune": {
        "max_comb_layer_num": 3,
        "allow_unequal_source_per_group": false,
        "ignored_layers": [0, 2],
        "must_contain_layers": [1, 3, 4],
        "layer_comm_ordering": {},
        "layer_comm_uplimit": {"4": 8},
        "all_groups_used": false,
        "limit_num_steps": 2,
        "allow_source_group_special": true
      },
      "sketch": {
        "customize_sketch": false,
        "use_sketch_input": false,
        "save_sketch": false,
        "sketch_path": ""
      }
    }"#
}

#[test]
fn builds_layer_groups_from_syccl_config_subset() {
    let config = parse_config(tiny_single_host_config());
    let topo = Topology::from_syccl_config(&config).expect("topology");

    assert_eq!(topo.num_gpus(), 4);
    assert_eq!(topo.layers().len(), 2);
    assert_eq!(topo.layers()[0].groups.len(), 4);
    assert_eq!(topo.layers()[1].groups.len(), 1);
    assert_eq!(
        topo.layers()[1].groups[0].connected_gpus,
        BTreeSet::from([0, 1, 2, 3])
    );
}

#[test]
fn generates_syccl_compatible_single_root_sketch_json() {
    let config = parse_config(tiny_single_host_config());
    let (sketches, stats) = search_sketches(
        &config,
        SearchOptions {
            limit: Some(4),
            ..Default::default()
        },
    )
    .expect("search succeeds");

    assert!(!sketches.is_empty());
    assert!(sketches.len() <= 4);
    assert!(stats.layer_combinations >= 1);
    assert!(stats.group_configurations >= 1);
    assert!(stats.schedules >= sketches.len());

    let first = serde_json::to_value(&sketches[0]).expect("json");
    assert_eq!(first["ngpus"], 4);
    assert_eq!(first["src_gpu"], 0);
    let nodes = first["nodes"].as_array().expect("nodes array");
    assert!(!nodes.is_empty());

    let mut reached = BTreeSet::from([0]);
    for node in nodes {
        assert!(node["id"].is_number());
        assert!(node["step"].is_number());
        assert_eq!(node["layer"], 1);
        assert_eq!(node["group"], 0);
        let srcs = node["src_dest_pair"]["srcs"]
            .as_array()
            .expect("srcs array");
        let dsts = node["src_dest_pair"]["dsts"]
            .as_array()
            .expect("dsts array");
        assert!(!srcs.is_empty());
        assert!(!dsts.is_empty());
        for dst in dsts {
            reached.insert(dst.as_i64().expect("integer dst") as i32);
        }
    }
    assert_eq!(reached, BTreeSet::from([0, 1, 2, 3]));
}

#[test]
fn cli_writes_limited_pretty_json_output() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_path = temp.path().join("config.json");
    let output_path = temp.path().join("sketches.json");
    std::fs::write(&config_path, tiny_single_host_config()).expect("write config");

    Command::cargo_bin("syccl-sketch-search")
        .expect("binary")
        .args([
            "--config",
            config_path.to_str().expect("config path"),
            "--output",
            output_path.to_str().expect("output path"),
            "--limit",
            "2",
            "--pretty",
        ])
        .assert()
        .success();

    let output = std::fs::read_to_string(output_path).expect("read output");
    assert!(output.contains('\n'));
    let sketches: serde_json::Value = serde_json::from_str(&output).expect("json output");
    let sketches = sketches.as_array().expect("array output");
    assert!(!sketches.is_empty());
    assert!(sketches.len() <= 2);
}

#[test]
fn cli_profiles_without_emitting_sketch_json() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_path = temp.path().join("config.json");
    std::fs::write(&config_path, tiny_single_host_config()).expect("write config");

    let assert = Command::cargo_bin("syccl-sketch-search")
        .expect("binary")
        .args([
            "--config",
            config_path.to_str().expect("config path"),
            "--profile",
            "--no-output",
        ])
        .assert()
        .success()
        .stdout("");
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).expect("utf8 stderr");
    assert!(stderr.contains("\"total_ms\""), "{stderr}");
}

#[test]
fn cli_prints_compact_dsl_to_terminal_without_replacing_json_output() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config_path = temp.path().join("config.json");
    let output_path = temp.path().join("sketches.json");
    std::fs::write(&config_path, tiny_single_host_config()).expect("write config");

    let assert = Command::cargo_bin("syccl-sketch-search")
        .expect("binary")
        .args([
            "--config",
            config_path.to_str().expect("config path"),
            "--output",
            output_path.to_str().expect("output path"),
            "--limit",
            "1",
            "--print-compact-dsl",
        ])
        .assert()
        .success()
        .stdout("");

    let stderr = String::from_utf8(assert.get_output().stderr.clone()).expect("utf8 stderr");
    assert!(stderr.contains("compact_sketches = ["), "{stderr}");
    assert!(stderr.contains("(0, 1, 0, 0,"), "{stderr}");

    let output = std::fs::read_to_string(output_path).expect("read output");
    let sketches: serde_json::Value = serde_json::from_str(&output).expect("json output");
    assert!(sketches.as_array().is_some());
}

#[test]
fn source_group_special_pruning_does_not_underflow_on_clos_config() {
    let config = parse_config(tiny_clos_config());
    let (sketches, _stats) = search_sketches(
        &config,
        SearchOptions {
            limit: Some(1),
            ..Default::default()
        },
    )
    .expect("clos-like search does not panic");

    assert_eq!(sketches.len(), 1);
}
