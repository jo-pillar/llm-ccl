use std::fs;
use std::path::PathBuf;
use std::process::Command;

use flow_sim_rs::{
    batch::{build_manifest, run_batch, BenchmarkCase, BenchmarkManifest},
    batch_sketch::{build_sketch_manifest, run_sketch_batch},
    compare::compare_manifest,
    config::{parse_config, spec_to_bps, spec_to_ns},
    schedule::{build_flows, parse_all_translated_schedules, parse_translated_schedule},
    simulator::{simulate_case, LinkKey, RouteKind},
    sketch::{parse_compact_sketches, sketches_to_translated_schedule},
    topology::Topology,
};
use tempfile::tempdir;

fn fixture_config() -> &'static str {
    r#"{
      "coll": {"name": "allgather", "byte": 4096, "root_sender": -1, "root_receiver": -1},
      "hosts": {"host_num": 2, "host_gpu_num": 2, "host_nic_num": 2, "host_links": "nvswitch"},
      "topo": [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {"layer_id": 3, "type": "switch", "switch_topo": "multirail", "switch_num": 2, "link_spec": "netlink"}
      ],
      "link_spec": {
        "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
        "nvlink": {"bw_mbpus": 0.15, "lat_us": 3},
        "link_nic": {"bw_mbpus": 0.0455, "lat_us": 0},
        "netlink": {"bw_mbpus": 0.0455, "lat_us": 10}
      }
    }"#
}

fn clos_config() -> &'static str {
    r#"{
      "coll": {"name": "allgather", "byte": 4096, "root_sender": -1, "root_receiver": -1},
      "hosts": {"host_num": 4, "host_gpu_num": 2, "host_nic_num": 1, "host_links": "nvswitch"},
      "topo": [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {"layer_id": 3, "type": "switch", "switch_topo": "pod", "switch_num": 2, "link_spec": "netlink_leaf"},
        {"layer_id": 4, "type": "switch", "switch_topo": "pod", "switch_num": 1, "link_spec": "netlink_spine"}
      ],
      "link_spec": {
        "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
        "nvlink": {"bw_mbpus": 0.15, "lat_us": 3},
        "link_nic": {"bw_mbpus": 0.0455, "lat_us": 0},
        "netlink_leaf": {"bw_mbpus": 0.0455, "lat_us": 10},
        "netlink_spine": {"bw_mbpus": 0.36, "lat_us": 10}
      }
    }"#
}

fn multirail_shared_nic_config() -> &'static str {
    r#"{
      "coll": {"name": "allgather", "byte": 4096, "root_sender": -1, "root_receiver": -1},
      "hosts": {"host_num": 2, "host_gpu_num": 4, "host_nic_num": 2, "host_links": "nvswitch"},
      "topo": [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {"layer_id": 3, "type": "switch", "switch_topo": "multirail", "switch_num": 2, "link_spec": "netlink"}
      ],
      "link_spec": {
        "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
        "nvlink": {"bw_mbpus": 0.15, "lat_us": 3},
        "link_nic": {"bw_mbpus": 0.0455, "lat_us": 0},
        "netlink": {"bw_mbpus": 0.0455, "lat_us": 10}
      }
    }"#
}

fn alltoall_config() -> &'static str {
    fixture_config()
        .replace("\"allgather\"", "\"alltoall\"")
        .leak()
}

fn alltoall_translated() -> &'static str {
    r#"{
      "coll_name": "alltoall",
      "ngpus": 4,
      "chunk_size_byte": 4096,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 1)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false}
          ]},
          {"src_chunk": "(0, 2)", "sends": [
            {"src_gpu": 0, "dst_gpu": 2, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false}
          ]},
          {"src_chunk": "(1, 0)", "sends": [
            {"src_gpu": 1, "dst_gpu": 0, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

fn multi_algorithm_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 4,
      "chunk_size_byte": 4096,
      "algorithms": [
        {
          "final_schedule": {
            "Time": 11.5,
            "Schedule": {"Events": [
              {"src_chunk": "(0, 0)", "sends": [
                {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false}
              ]}
            ]}
          }
        },
        {
          "final_schedule": {
            "Time": 22.5,
            "Schedule": {"Events": [
              {"src_chunk": "(0, 0)", "sends": [
                {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false},
                {"src_gpu": 1, "dst_gpu": 3, "epoch": 1, "layer_used": 1, "copy": true, "reduce": false}
              ]}
            ]}
          }
        }
      ]
    }"#
}

#[test]
fn parses_all_translated_algorithms_in_input_order() {
    let schedules =
        parse_all_translated_schedules(multi_algorithm_translated().as_bytes()).unwrap();

    assert_eq!(schedules.len(), 2);
    assert_eq!(schedules[0].algorithm_index, 0);
    assert_eq!(schedules[0].syccl_time_us, Some(11.5));
    assert_eq!(schedules[0].schedule.sends.len(), 1);
    assert_eq!(schedules[1].algorithm_index, 1);
    assert_eq!(schedules[1].syccl_time_us, Some(22.5));
    assert_eq!(schedules[1].schedule.sends.len(), 2);
}

#[test]
fn simulate_cli_writes_all_algorithm_results_in_input_order() {
    let root = tempdir().unwrap();
    let config_path = root.path().join("config.json");
    let translated_path = root.path().join("translated.json");
    let output_path = root.path().join("all-results.json");
    fs::write(&config_path, fixture_config()).unwrap();
    fs::write(&translated_path, multi_algorithm_translated()).unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_flow-sim-rs"))
        .arg("simulate")
        .arg("--config")
        .arg(&config_path)
        .arg("--translated")
        .arg(&translated_path)
        .arg("--output")
        .arg(&output_path)
        .status()
        .unwrap();

    assert!(status.success());
    let output: serde_json::Value =
        serde_json::from_slice(&fs::read(&output_path).unwrap()).unwrap();
    let solutions = output["solutions"].as_array().unwrap();
    assert_eq!(solutions.len(), 2);
    assert_eq!(solutions[0]["solution_index"], 0);
    assert_eq!(solutions[0]["syccl_time_us"], 11.5);
    assert!(!solutions[0].as_object().unwrap().contains_key("flow_count"));
    assert!(!solutions[0]
        .as_object()
        .unwrap()
        .contains_key("channel_count"));
    assert_eq!(solutions[1]["solution_index"], 1);
    assert_eq!(solutions[1]["syccl_time_us"], 22.5);
    assert!(!solutions[1].as_object().unwrap().contains_key("flow_count"));
    assert!(!solutions[1]
        .as_object()
        .unwrap()
        .contains_key("channel_count"));
}

#[test]
fn simulate_cli_can_select_one_algorithm_by_original_index() {
    let root = tempdir().unwrap();
    let config_path = root.path().join("config.json");
    let translated_path = root.path().join("translated.json");
    let output_path = root.path().join("one-result.json");
    fs::write(&config_path, fixture_config()).unwrap();
    fs::write(&translated_path, multi_algorithm_translated()).unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_flow-sim-rs"))
        .arg("simulate")
        .arg("--config")
        .arg(&config_path)
        .arg("--translated")
        .arg(&translated_path)
        .arg("--output")
        .arg(&output_path)
        .arg("--algorithm-index")
        .arg("1")
        .status()
        .unwrap();

    assert!(status.success());
    let output: serde_json::Value =
        serde_json::from_slice(&fs::read(&output_path).unwrap()).unwrap();
    let solutions = output["solutions"].as_array().unwrap();
    assert_eq!(solutions.len(), 1);
    assert_eq!(solutions[0]["solution_index"], 1);
    assert_eq!(solutions[0]["syccl_time_us"], 22.5);
    assert!(!solutions[0].as_object().unwrap().contains_key("flow_count"));
    assert!(!solutions[0]
        .as_object()
        .unwrap()
        .contains_key("channel_count"));
}

#[test]
fn simulate_sketch_reports_bottlenecks_at_sketch_transmission_level() {
    let root = tempdir().unwrap();
    let config_path = root.path().join("config.json");
    let sketch_path = root.path().join("sketch.json");
    let output_path = root.path().join("sketch-profile.json");
    fs::write(&config_path, clos_config()).unwrap();
    fs::write(
        &sketch_path,
        r#"[
          {"step": 0, "layer": 1, "group": 0, "srcs": [0], "dsts": [1]},
          {"step": 1, "layer": 3, "group": 0, "srcs": [0], "dsts": [2]},
          {"step": 2, "layer": 1, "group": 1, "srcs": [2], "dsts": [3]},
          {"step": 3, "layer": 4, "group": 0, "srcs": [2], "dsts": [4]},
          {"step": 4, "layer": 1, "group": 2, "srcs": [4], "dsts": [5]},
          {"step": 5, "layer": 3, "group": 1, "srcs": [4], "dsts": [6]},
          {"step": 6, "layer": 1, "group": 3, "srcs": [6], "dsts": [7]}
        ]"#,
    )
    .unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_flow-sim-rs"))
        .arg("simulate-sketch")
        .arg("--config")
        .arg(&config_path)
        .arg("--sketch")
        .arg(&sketch_path)
        .arg("--output")
        .arg(&output_path)
        .status()
        .unwrap();

    assert!(status.success());
    let output: serde_json::Value =
        serde_json::from_slice(&fs::read(&output_path).unwrap()).unwrap();
    let output_object = output.as_object().unwrap();
    assert_eq!(output_object.len(), 2);
    assert!(output_object.contains_key("time_us"));
    assert!(output_object.contains_key("bottleneck_profile"));
    let profile = output["bottleneck_profile"].as_object().unwrap();
    assert_eq!(profile.len(), 1);
    let critical_flow_chain = &profile["critical_flow_chain"];
    assert_eq!(
        critical_flow_chain["collective_duration_ns"],
        critical_flow_chain["collective_finish_ns"]
    );

    let chain = critical_flow_chain["chain"].as_array().unwrap();
    assert!(!chain.is_empty());
    assert_eq!(
        chain.last().unwrap()["flow_id"],
        critical_flow_chain["latest_flow_id"]
    );
    assert!(chain[0]["previous_bottleneck_flow_id"].is_null());
    assert!(chain[0]["previous_bottleneck_flow_end_time"].is_null());
    assert_eq!(chain[0]["current_flow_start_time"], 0);

    for window in chain.windows(2) {
        assert_eq!(
            window[1]["previous_bottleneck_flow_id"],
            window[0]["flow_id"]
        );
        assert_eq!(
            window[1]["current_flow_start_time"],
            window[1]["previous_bottleneck_flow_end_time"]
        );
    }

    for entry in chain {
        assert!(entry["flow_id"].is_u64());
        assert!(entry["chunk"]["src_chunk"].is_u64());
        assert!(entry["chunk"]["chunk_index"].is_u64());
        assert!(entry["chunk"]["src_gpu"].is_u64());
        assert!(entry["chunk"]["dst_gpu"].is_u64());
        assert!(
            entry["sketch_transmission"]["transmission_index"]
                .as_u64()
                .unwrap()
                < 7
        );
        assert!(entry["sketch_transmission"]["step"].is_u64());
        assert!(entry["sketch_transmission"]["layer"].is_i64());
        assert!(entry["sketch_transmission"]["group"].is_u64());
        assert!(entry["sketch_transmission"]["srcs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|gpu| gpu.is_u64()));
        assert!(entry["sketch_transmission"]["dsts"]
            .as_array()
            .unwrap()
            .iter()
            .all(|gpu| gpu.is_u64()));
        assert!(entry["propagation_delay_ns"].as_u64().unwrap() >= 3_010);
        assert!(entry["current_flow_duration_ns"].as_u64().unwrap() > 0);
        assert!(entry["transmission_time_ns"]["link_busy_ns"].is_u64());
        assert!(entry["transmission_time_ns"]["link_wait_ns"].is_u64());
    }
}

#[test]
fn compare_reports_absolute_ratio_against_cached_simai_time() {
    let root = tempdir().unwrap();
    let case_dir = root.path().join("eval_a");
    fs::create_dir_all(&case_dir).unwrap();
    fs::write(
        case_dir.join("candidate-translated.json"),
        fixture_translated(),
    )
    .unwrap();

    let (_cfg_dir, cfg_path) = write_temp_file("config.json", fixture_config());
    let output_dir = tempdir().unwrap();
    let mut manifest = build_manifest(root.path(), &cfg_path, output_dir.path()).unwrap();
    let outputs = run_batch(&manifest).unwrap();
    manifest.cases[0].simai_time_us = Some(outputs[0].result.time_us * 2.0);
    let manifest_path = output_dir.path().join("benchmarks.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let report = compare_manifest(&manifest_path, 5.0).unwrap();

    assert_eq!(report.cases.len(), 1);
    assert!((report.cases[0].abs_ratio.unwrap() - 2.0).abs() < 1e-9);
    assert_eq!(report.cases[0].passes_abs_ratio, Some(true));
    assert_eq!(report.abs_ratio_case_total, 1);
    assert_eq!(report.abs_ratio_passed, 1);
    assert_eq!(report.abs_ratio_failed, 0);
    assert_eq!(report.abs_ratio_pass_rate, Some(1.0));
    assert_eq!(report.passes_abs_ratio_requirement, Some(true));
    assert_eq!(report.passes_trend_requirement, None);
    assert_eq!(report.passes_alignment_requirement, None);
}

#[test]
fn compare_reports_duplicate_schedule_groups_separately_from_unique_trends() {
    let root = tempdir().unwrap();
    let case_a = root.path().join("eval_a");
    let case_b = root.path().join("eval_b");
    fs::create_dir_all(&case_a).unwrap();
    fs::create_dir_all(&case_b).unwrap();
    fs::write(
        case_a.join("candidate-translated.json"),
        fixture_translated(),
    )
    .unwrap();
    fs::write(
        case_b.join("candidate-translated.json"),
        fixture_translated(),
    )
    .unwrap();

    let (_cfg_dir, cfg_path) = write_temp_file("config.json", fixture_config());
    let output_dir = tempdir().unwrap();
    let mut manifest = build_manifest(root.path(), &cfg_path, output_dir.path()).unwrap();
    let outputs = run_batch(&manifest).unwrap();
    manifest.cases[0].simai_time_us = Some(outputs[0].result.time_us);
    manifest.cases[1].simai_time_us = Some(outputs[1].result.time_us + 0.5);
    let manifest_path = output_dir.path().join("benchmarks.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let report = compare_manifest(&manifest_path, 5.0).unwrap();

    assert_eq!(report.duplicate_schedule_groups.len(), 1);
    assert_eq!(report.duplicate_schedule_groups[0].case_count, 2);
    assert!((report.duplicate_schedule_groups[0].simai_spread_us.unwrap() - 0.5).abs() < 1e-9);
    assert_eq!(report.unique_schedule_total, 1);
    assert_eq!(report.unique_pairwise_total, 0);
}

#[test]
fn compare_reports_tolerance_filtered_trend_stats() {
    let root = tempdir().unwrap();
    let cfg_path = root.path().join("config.json");
    fs::write(&cfg_path, fixture_config()).unwrap();
    let input_a = root.path().join("a-candidate-translated.json");
    let input_b = root.path().join("b-candidate-translated.json");
    let input_c = root.path().join("c-candidate-translated.json");
    fs::write(&input_a, fixture_translated()).unwrap();
    fs::write(
        &input_b,
        fixture_translated().replace("epoch\": 1", "epoch\": 2"),
    )
    .unwrap();
    fs::write(
        &input_c,
        fixture_translated().replace("epoch\": 1", "epoch\": 3"),
    )
    .unwrap();

    let output_a = root.path().join("a.json");
    let output_b = root.path().join("b.json");
    let output_c = root.path().join("c.json");
    write_case_output(&output_a, "a", &cfg_path, &input_a, 10.0);
    write_case_output(&output_b, "b", &cfg_path, &input_b, 20.0);
    write_case_output(&output_c, "c", &cfg_path, &input_c, 30.0);

    let manifest = BenchmarkManifest {
        cases: vec![
            BenchmarkCase {
                name: "a".to_string(),
                config: cfg_path.clone(),
                translated: input_a,
                rust_output: output_a,
                simai_time_us: Some(10.0),
                simai_end_to_end_csv: None,
            },
            BenchmarkCase {
                name: "b".to_string(),
                config: cfg_path.clone(),
                translated: input_b,
                rust_output: output_b,
                simai_time_us: Some(10.1),
                simai_end_to_end_csv: None,
            },
            BenchmarkCase {
                name: "c".to_string(),
                config: cfg_path,
                translated: input_c,
                rust_output: output_c,
                simai_time_us: Some(30.0),
                simai_end_to_end_csv: None,
            },
        ],
    };

    let manifest_path = root.path().join("benchmarks.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let report = compare_manifest(&manifest_path, 5.0).unwrap();

    assert_eq!(report.trend_tolerance_relative, 0.05);
    assert_eq!(report.tolerant_unique_pairwise_total, 2);
    assert_eq!(report.tolerant_unique_pairwise_inconsistent, 0);
    assert_eq!(report.tolerant_unique_pairwise_consistency_rate, Some(1.0));
    assert_eq!(report.trend_consistency_requirement, 0.95);
    assert_eq!(report.passes_trend_requirement, Some(true));
    assert_eq!(report.passes_alignment_requirement, Some(true));
}

#[test]
fn compare_filters_trends_within_five_percent_of_simai_time() {
    let root = tempdir().unwrap();
    let cfg_path = root.path().join("config.json");
    fs::write(&cfg_path, fixture_config()).unwrap();
    let input_a = root.path().join("a-candidate-translated.json");
    let input_b = root.path().join("b-candidate-translated.json");
    let input_c = root.path().join("c-candidate-translated.json");
    fs::write(&input_a, fixture_translated()).unwrap();
    fs::write(
        &input_b,
        fixture_translated().replace("epoch\": 1", "epoch\": 2"),
    )
    .unwrap();
    fs::write(
        &input_c,
        fixture_translated().replace("epoch\": 1", "epoch\": 3"),
    )
    .unwrap();

    let output_a = root.path().join("a.json");
    let output_b = root.path().join("b.json");
    let output_c = root.path().join("c.json");
    write_case_output(&output_a, "a", &cfg_path, &input_a, 100.0);
    write_case_output(&output_b, "b", &cfg_path, &input_b, 90.0);
    write_case_output(&output_c, "c", &cfg_path, &input_c, 130.0);

    let manifest = BenchmarkManifest {
        cases: vec![
            BenchmarkCase {
                name: "a".to_string(),
                config: cfg_path.clone(),
                translated: input_a,
                rust_output: output_a,
                simai_time_us: Some(100.0),
                simai_end_to_end_csv: None,
            },
            BenchmarkCase {
                name: "b".to_string(),
                config: cfg_path.clone(),
                translated: input_b,
                rust_output: output_b,
                simai_time_us: Some(104.9),
                simai_end_to_end_csv: None,
            },
            BenchmarkCase {
                name: "c".to_string(),
                config: cfg_path,
                translated: input_c,
                rust_output: output_c,
                simai_time_us: Some(120.0),
                simai_end_to_end_csv: None,
            },
        ],
    };

    let manifest_path = root.path().join("benchmarks.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let report = compare_manifest(&manifest_path, 5.0).unwrap();

    assert_eq!(report.trend_tolerance_relative, 0.05);
    assert_eq!(report.tolerant_unique_pairwise_total, 2);
    assert_eq!(report.tolerant_unique_pairwise_inconsistent, 0);
}

#[test]
fn compare_accepts_abs_ratio_requirement_when_at_least_95_percent_pass() {
    let root = tempdir().unwrap();
    let cfg_path = root.path().join("config.json");
    fs::write(&cfg_path, fixture_config()).unwrap();
    let mut cases = Vec::new();
    for idx in 0..20 {
        let input = root
            .path()
            .join(format!("case_{idx}-candidate-translated.json"));
        fs::write(
            &input,
            fixture_translated().replace("epoch\": 1", &format!("epoch\": {idx}")),
        )
        .unwrap();
        let output = root.path().join(format!("case_{idx}.json"));
        write_case_output(
            &output,
            &format!("case_{idx}"),
            &cfg_path,
            &input,
            10.0 + idx as f64,
        );
        cases.push(BenchmarkCase {
            name: format!("case_{idx}"),
            config: cfg_path.clone(),
            translated: input,
            rust_output: output,
            simai_time_us: Some(if idx == 19 { 1000.0 } else { 10.0 + idx as f64 }),
            simai_end_to_end_csv: None,
        });
    }

    let manifest = BenchmarkManifest { cases };
    let manifest_path = root.path().join("benchmarks.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let report = compare_manifest(&manifest_path, 5.0).unwrap();

    assert_eq!(report.abs_ratio_requirement, 0.95);
    assert_eq!(report.abs_ratio_case_total, 20);
    assert_eq!(report.abs_ratio_passed, 19);
    assert_eq!(report.abs_ratio_failed, 1);
    assert_eq!(report.abs_ratio_pass_rate, Some(0.95));
    assert_eq!(report.passes_abs_ratio_requirement, Some(true));
}

fn write_case_output(
    path: &std::path::Path,
    name: &str,
    config: &std::path::Path,
    translated: &std::path::Path,
    time_us: f64,
) {
    let output = serde_json::json!({
        "name": name,
        "config": config,
        "translated": translated,
        "time_us": time_us
    });
    fs::write(path, serde_json::to_string_pretty(&output).unwrap()).unwrap();
}

fn fixture_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 4,
      "chunk_size_byte": 4096,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false},
            {"src_gpu": 0, "dst_gpu": 2, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false},
            {"src_gpu": 1, "dst_gpu": 3, "epoch": 1, "layer_used": 3, "copy": true, "reduce": false}
          ]},
          {"src_chunk": "(2, 0)", "sends": [
            {"src_gpu": 2, "dst_gpu": 0, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

#[test]
fn compact_sketch_expands_and_simulates_without_translated_file() {
    let config = parse_config(fixture_config().as_bytes()).unwrap();
    let sketches = parse_compact_sketches(
        r#"[
          [
            [0, 1, 0, 0, 1],
            [0, 3, 0, 0, 2],
            [1, 1, 1, 2, 3]
          ]
        ]"#
        .as_bytes(),
        &config,
    )
    .unwrap();

    let schedule = sketches_to_translated_schedule(&sketches, &config).unwrap();
    let result = simulate_case(&config, &schedule).unwrap();

    assert_eq!(schedule.ngpus, 4);
    assert_eq!(schedule.sends.len(), 12);
    assert!(result.time_us > 0.0);
}

#[test]
fn compact_sketch_maps_multiple_sources_to_destination_chunks() {
    let config = parse_config(clos_config().as_bytes()).unwrap();
    let sketches = parse_compact_sketches(
        r#"[
          [0, 3, 0, [0], [1, 2, 3]],
          [1, 4, 0, [0, 1, 2, 3], [4, 5, 6, 7]]
        ]"#
        .as_bytes(),
        &config,
    )
    .unwrap();

    let schedule = sketches_to_translated_schedule(&sketches, &config).unwrap();
    let root0_sends: Vec<_> = schedule
        .sends
        .iter()
        .filter(|send| send.src_chunk == 0)
        .map(|send| (send.epoch, send.src_gpu, send.dst_gpu))
        .collect();

    assert_eq!(
        root0_sends,
        vec![
            (0, 0, 1),
            (0, 0, 2),
            (0, 0, 3),
            (1, 0, 4),
            (1, 1, 5),
            (1, 2, 6),
            (1, 3, 7),
        ]
    );
    assert_eq!(schedule.sends.len(), 56);
}

#[test]
fn compact_sketch_rejects_unbalanced_multi_source_destinations() {
    let config = parse_config(clos_config().as_bytes()).unwrap();
    let error = parse_compact_sketches(
        r#"[
          [0, 4, 0, [0, 1], [2, 3, 4]]
        ]"#
        .as_bytes(),
        &config,
    )
    .unwrap_err();

    assert!(error
        .to_string()
        .contains("dsts length 3 must be divisible by srcs length 2"));
}

fn queue_config() -> &'static str {
    r#"{
      "coll": {"name": "allgather", "byte": 4096, "root_sender": -1, "root_receiver": -1},
      "hosts": {"host_num": 3, "host_gpu_num": 2, "host_nic_num": 2, "host_links": "nvswitch"},
      "topo": [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {"layer_id": 3, "type": "switch", "switch_topo": "multirail", "switch_num": 2, "link_spec": "netlink"}
      ],
      "link_spec": {
        "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
        "nvlink": {"bw_mbpus": 0.15, "lat_us": 3},
        "link_nic": {"bw_mbpus": 0.0455, "lat_us": 0},
        "netlink": {"bw_mbpus": 0.0455, "lat_us": 10}
      }
    }"#
}

fn queue_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 6,
      "chunk_size_byte": 4096,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 2, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false},
            {"src_gpu": 0, "dst_gpu": 4, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

fn slow_host_config() -> &'static str {
    r#"{
      "coll": {"name": "allgather", "byte": 4096, "root_sender": -1, "root_receiver": -1},
      "hosts": {"host_num": 1, "host_gpu_num": 4, "host_nic_num": 4, "host_links": "nvswitch"},
      "topo": [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {"layer_id": 3, "type": "switch", "switch_topo": "multirail", "switch_num": 4, "link_spec": "netlink"}
      ],
      "link_spec": {
        "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
        "nvlink": {"bw_mbpus": 0.15, "lat_us": 10.5},
        "link_nic": {"bw_mbpus": 0.0455, "lat_us": 0},
        "netlink": {"bw_mbpus": 0.0455, "lat_us": 21.5}
      }
    }"#
}

fn source_fanout_config() -> &'static str {
    r#"{
      "coll": {"name": "allgather", "byte": 4096, "root_sender": -1, "root_receiver": -1},
      "hosts": {"host_num": 4, "host_gpu_num": 1, "host_nic_num": 1, "host_links": "nvswitch"},
      "topo": [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {"layer_id": 3, "type": "switch", "switch_topo": "multirail", "switch_num": 1, "link_spec": "netlink"}
      ],
      "link_spec": {
        "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
        "nvlink": {"bw_mbpus": 10, "lat_us": 0},
        "link_nic": {"bw_mbpus": 0.0455, "lat_us": 0},
        "netlink": {"bw_mbpus": 0.0455, "lat_us": 10}
      }
    }"#
}

fn mixed_staging_fanout_config() -> &'static str {
    r#"{
      "coll": {"name": "allgather", "byte": 4096, "root_sender": -1, "root_receiver": -1},
      "hosts": {"host_num": 4, "host_gpu_num": 4, "host_nic_num": 4, "host_links": "nvswitch"},
      "topo": [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {"layer_id": 3, "type": "switch", "switch_topo": "multirail", "switch_num": 4, "link_spec": "netlink"}
      ],
      "link_spec": {
        "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
        "nvlink": {"bw_mbpus": 10, "lat_us": 0},
        "link_nic": {"bw_mbpus": 0.0455, "lat_us": 0},
        "netlink": {"bw_mbpus": 0.0455, "lat_us": 10}
      }
    }"#
}

fn source_epoch_config() -> &'static str {
    r#"{
      "coll": {"name": "allgather", "byte": 4096, "root_sender": -1, "root_receiver": -1},
      "hosts": {"host_num": 3, "host_gpu_num": 1, "host_nic_num": 1, "host_links": "nvswitch"},
      "topo": [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {"layer_id": 3, "type": "switch", "switch_topo": "multirail", "switch_num": 1, "link_spec": "netlink"}
      ],
      "link_spec": {
        "memcpy": {"bw_mbpus": 10, "lat_us": 0.01},
        "nvlink": {"bw_mbpus": 10, "lat_us": 0},
        "link_nic": {"bw_mbpus": 0.0455, "lat_us": 0},
        "netlink": {"bw_mbpus": 0.0455, "lat_us": 10}
      }
    }"#
}

fn source_epoch_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 3,
      "chunk_size_byte": 4096,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false},
            {"src_gpu": 0, "dst_gpu": 2, "epoch": 1, "layer_used": 3, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

fn bidirectional_cross_host_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 3,
      "chunk_size_byte": 4096,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false}
          ]},
          {"src_chunk": "(1, 0)", "sends": [
            {"src_gpu": 1, "dst_gpu": 0, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

fn single_cross_host_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 3,
      "chunk_size_byte": 4096,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

fn large_single_local_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 4,
      "chunk_size_byte": 16777216,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

fn large_cross_source_fanout_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 4,
      "chunk_size_byte": 16777216,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false},
            {"src_gpu": 0, "dst_gpu": 2, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false},
            {"src_gpu": 0, "dst_gpu": 3, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

fn mixed_staging_cross_fanout_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 16,
      "chunk_size_byte": 16777216,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false},
            {"src_gpu": 0, "dst_gpu": 2, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false},
            {"src_gpu": 0, "dst_gpu": 3, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false},
            {"src_gpu": 0, "dst_gpu": 4, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false},
            {"src_gpu": 0, "dst_gpu": 8, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false},
            {"src_gpu": 0, "dst_gpu": 12, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

fn multi_epoch_cross_chain_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 3,
      "chunk_size_byte": 4096,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false},
            {"src_gpu": 1, "dst_gpu": 2, "epoch": 1, "layer_used": 3, "copy": true, "reduce": false},
            {"src_gpu": 2, "dst_gpu": 0, "epoch": 2, "layer_used": 3, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

fn sparse_epoch_timestamp_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 3,
      "chunk_size_byte": 4096,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false},
            {"src_gpu": 1, "dst_gpu": 2, "epoch": 19224, "layer_used": 3, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

fn local_chain_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 4,
      "chunk_size_byte": 4096,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false},
            {"src_gpu": 1, "dst_gpu": 2, "epoch": 1, "layer_used": 1, "copy": true, "reduce": false},
            {"src_gpu": 2, "dst_gpu": 3, "epoch": 2, "layer_used": 1, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

fn packetized_same_link_fanout_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 4,
      "chunk_size_byte": 18000000,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false},
            {"src_gpu": 0, "dst_gpu": 2, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false},
            {"src_gpu": 1, "dst_gpu": 3, "epoch": 1, "layer_used": 1, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

fn ack_preempts_packetized_fanout_translated() -> &'static str {
    r#"{
      "coll_name": "allgather",
      "ngpus": 4,
      "chunk_size_byte": 18000000,
      "algorithms": [{
        "final_schedule": {"Schedule": {"Events": [
          {"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false},
            {"src_gpu": 0, "dst_gpu": 2, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false},
            {"src_gpu": 0, "dst_gpu": 3, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false}
          ]},
          {"src_chunk": "(1, 0)", "sends": [
            {"src_gpu": 1, "dst_gpu": 0, "epoch": 0, "layer_used": 1, "copy": true, "reduce": false}
          ]}
        ]}}
      }]
    }"#
}

fn ack_backpressure_translated(backlog_flows: usize) -> String {
    let mut events = Vec::new();
    events.push(
        r#"{"src_chunk": "(0, 0)", "sends": [
            {"src_gpu": 0, "dst_gpu": 1, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false},
            {"src_gpu": 1, "dst_gpu": 2, "epoch": 1, "layer_used": 3, "copy": true, "reduce": false}
        ]}"#
        .to_string(),
    );
    for chunk in 0..backlog_flows {
        events.push(format!(
            r#"{{"src_chunk": "(1, {chunk})", "sends": [
                {{"src_gpu": 1, "dst_gpu": 0, "epoch": 0, "layer_used": 3, "copy": true, "reduce": false}}
            ]}}"#
        ));
    }
    format!(
        r#"{{
          "coll_name": "allgather",
          "ngpus": 3,
          "chunk_size_byte": 4096,
          "algorithms": [{{
            "final_schedule": {{"Schedule": {{"Events": [
              {}
            ]}}}}
          }}]
        }}"#,
        events.join(",\n")
    )
}

fn fast_local_config() -> &'static str {
    r#"{
      "coll": {"name": "allgather", "byte": 4096, "root_sender": -1, "root_receiver": -1},
      "hosts": {"host_num": 1, "host_gpu_num": 4, "host_nic_num": 4, "host_links": "nvswitch"},
      "topo": [
        {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
        {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
        {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
        {"layer_id": 3, "type": "switch", "switch_topo": "multirail", "switch_num": 4, "link_spec": "netlink"}
      ],
      "link_spec": {
        "memcpy": {"bw_mbpus": 10, "lat_us": 0},
        "nvlink": {"bw_mbpus": 10, "lat_us": 0},
        "link_nic": {"bw_mbpus": 10, "lat_us": 0},
        "netlink": {"bw_mbpus": 10, "lat_us": 0}
      }
    }"#
}

fn write_temp_file(name: &str, contents: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    let path = dir.path().join(name);
    fs::write(&path, contents).unwrap();
    (dir, path)
}

#[test]
fn parses_multirail_config_and_converts_units() {
    let cfg = parse_config(fixture_config().as_bytes()).unwrap();

    assert_eq!(cfg.coll_name, "allgather");
    assert_eq!(cfg.coll_bytes, 4096);
    assert_eq!(cfg.hosts.host_num, 2);
    assert_eq!(cfg.hosts.gpus_per_host, 2);
    assert_eq!(cfg.hosts.nics_per_host, 2);
    assert_eq!(cfg.rail_count, 2);
    assert_eq!(spec_to_bps(0.0455), 381681664000);
    assert_eq!(spec_to_ns(10.0), 10_000);
}

#[test]
fn parses_clos_config_and_routes_cross_host_via_leaf_spine() {
    let cfg = parse_config(clos_config().as_bytes()).unwrap();
    assert_eq!(cfg.hosts.host_num, 4);
    assert_eq!(cfg.hosts.gpus_per_host, 2);
    assert_eq!(cfg.hosts.nics_per_host, 1);
    assert_eq!(cfg.rail_count, 2);

    let topo = Topology::from_config(&cfg).unwrap();
    let route = topo.route(0, 6).unwrap();
    assert_eq!(route.kind, RouteKind::CrossHost);
    assert!(route.links.len() >= 6, "route={:?}", route.links);
}

#[test]
fn multirail_allows_fewer_nics_than_gpus_per_host() {
    let cfg = parse_config(multirail_shared_nic_config().as_bytes()).unwrap();
    assert_eq!(cfg.hosts.gpus_per_host, 4);
    assert_eq!(cfg.hosts.nics_per_host, 2);
    assert_eq!(cfg.rail_count, 2);

    let topo = Topology::from_config(&cfg).unwrap();
    let route = topo.route(3, 7).unwrap();
    assert_eq!(route.kind, RouteKind::CrossHost);
    assert!(!route.links.is_empty());
}

#[test]
fn parses_translated_schedule_and_builds_simai_style_dependencies() {
    let schedule = parse_translated_schedule(fixture_translated().as_bytes()).unwrap();
    let flows = build_flows(&schedule, 4096).unwrap();

    assert_eq!(schedule.ngpus, 4);
    assert_eq!(schedule.sends.len(), 4);
    assert_eq!(flows.channel_count, 2);
    assert_eq!(flows.flows.len(), 4);

    let root_to_one = flows
        .flows
        .iter()
        .find(|flow| flow.channel_id == 0 && flow.src == 0 && flow.dst == 1)
        .unwrap();
    let root_to_two = flows
        .flows
        .iter()
        .find(|flow| flow.channel_id == 0 && flow.src == 0 && flow.dst == 2)
        .unwrap();
    let one_to_three = flows
        .flows
        .iter()
        .find(|flow| flow.channel_id == 0 && flow.src == 1 && flow.dst == 3)
        .unwrap();

    assert!(root_to_one.recv_parents.is_empty());
    assert!(root_to_one.send_parents.is_empty());
    assert!(root_to_two.recv_parents.is_empty());
    assert!(root_to_two.send_parents.is_empty());
    assert_eq!(one_to_three.recv_parents, vec![root_to_one.id]);
    assert!(one_to_three.send_parents.is_empty());
}

#[test]
fn source_fanout_from_existing_chunk_does_not_depend_on_epoch() {
    let schedule = parse_translated_schedule(source_epoch_translated().as_bytes()).unwrap();
    let flows = build_flows(&schedule, 4096).unwrap();

    let root_to_one = flows
        .flows
        .iter()
        .find(|flow| flow.src == 0 && flow.dst == 1)
        .unwrap();
    let root_to_two = flows
        .flows
        .iter()
        .find(|flow| flow.src == 0 && flow.dst == 2)
        .unwrap();

    assert!(root_to_one.recv_parents.is_empty());
    assert!(root_to_one.send_parents.is_empty());
    assert!(root_to_two.recv_parents.is_empty());
    assert!(root_to_two.send_parents.is_empty());
}

#[test]
fn parses_and_simulates_alltoall_translated_chunks() {
    let cfg = parse_config(alltoall_config().as_bytes()).unwrap();
    let schedule = parse_translated_schedule(alltoall_translated().as_bytes()).unwrap();
    let flows = build_flows(&schedule, 4096).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert_eq!(cfg.coll_name, "alltoall");
    assert_eq!(schedule.sends.len(), 3);
    assert_eq!(flows.channel_count, 3);
    assert!(result.time_us > 0.0);
}

#[test]
fn constructs_same_host_and_cross_host_routes() {
    let cfg = parse_config(fixture_config().as_bytes()).unwrap();
    let topo = Topology::from_config(&cfg).unwrap();

    let same_host = topo.route(0, 1).unwrap();
    assert_eq!(same_host.kind, RouteKind::SameHost);
    assert_eq!(
        same_host.links,
        vec![LinkKey::new(0, 4), LinkKey::new(4, 1)]
    );

    let cross_host = topo.route(0, 2).unwrap();
    assert_eq!(cross_host.kind, RouteKind::CrossHost);
    assert_eq!(
        cross_host.links,
        vec![
            LinkKey::new(0, 6),
            LinkKey::new(6, 10),
            LinkKey::new(10, 8),
            LinkKey::new(8, 2),
        ]
    );
}

#[test]
fn queueing_on_shared_links_increases_finish_time_and_records_wait() {
    let cfg = parse_config(queue_config().as_bytes()).unwrap();
    let schedule = parse_translated_schedule(queue_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us > 0.0);
    let chain = &result.bottleneck_profile.critical_flow_chain.chain;
    assert!(chain
        .iter()
        .any(|entry| entry.transmission_time_ns.link_wait_ns > 0));
}

#[test]
fn link_latency_delays_arrival_but_not_next_serialization_slot() {
    let cfg = parse_config(queue_config().as_bytes()).unwrap();
    let schedule = parse_translated_schedule(queue_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us > 0.0);
}

#[test]
fn source_epoch_dependency_releases_at_sender_serialization_completion() {
    let cfg = parse_config(source_epoch_config().as_bytes()).unwrap();
    let schedule = parse_translated_schedule(source_epoch_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us < 60.0, "time_us={}", result.time_us);
}

#[test]
fn receive_completion_follows_data_arrival_before_ack_return() {
    let cfg = parse_config(source_epoch_config().as_bytes()).unwrap();
    let schedule = parse_translated_schedule(single_cross_host_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us < 30.0, "time_us={}", result.time_us);
}

#[test]
fn large_same_host_data_pipelines_between_host_links() {
    let cfg = parse_config(slow_host_config().as_bytes()).unwrap();
    let schedule = parse_translated_schedule(large_single_local_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us < 200.0, "time_us={}", result.time_us);
}

#[test]
fn source_cross_epoch_fanout_adds_serial_pressure() {
    let cfg = parse_config(mixed_staging_fanout_config().as_bytes()).unwrap();
    let fanout =
        parse_translated_schedule(mixed_staging_cross_fanout_translated().as_bytes()).unwrap();

    let fanout_result = simulate_case(&cfg, &fanout).unwrap();

    assert!(fanout_result.time_us > 0.0);
}

#[test]
fn direct_cross_dominant_fanout_uses_explicit_nic_queue_pressure() {
    let cfg = parse_config(source_fanout_config().as_bytes()).unwrap();
    let fanout =
        parse_translated_schedule(large_cross_source_fanout_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &fanout).unwrap();

    assert!(result.time_us > 0.0);
}

#[test]
fn epoch_labels_are_not_scaled_into_network_round_latency() {
    let cfg = parse_config(source_epoch_config().as_bytes()).unwrap();
    let schedule =
        parse_translated_schedule(multi_epoch_cross_chain_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us > 0.0);
}

#[test]
fn sparse_epoch_timestamps_do_not_add_global_barrier() {
    let cfg = parse_config(source_epoch_config().as_bytes()).unwrap();
    let schedule =
        parse_translated_schedule(sparse_epoch_timestamp_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us < 1_000.0, "time_us={}", result.time_us);
}

#[test]
fn rdma_ack_is_serialized_on_reverse_links() {
    let cfg = parse_config(source_epoch_config().as_bytes()).unwrap();
    let schedule = parse_translated_schedule(single_cross_host_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us > 0.0);
}

#[test]
fn rdma_ack_bypasses_queued_data_packets() {
    let cfg = parse_config(source_epoch_config().as_bytes()).unwrap();
    let schedule =
        parse_translated_schedule(bidirectional_cross_host_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us < 50.0, "time_us={}", result.time_us);
}

#[test]
fn rdma_ack_queue_is_checked_before_data_qps() {
    let cfg = parse_config(source_epoch_config().as_bytes()).unwrap();
    let translated = ack_backpressure_translated(512);
    let schedule = parse_translated_schedule(translated.as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us < 150.0, "time_us={}", result.time_us);
}

#[test]
fn usual_routes_pay_endpoint_injection_delay_per_flow() {
    let cfg = parse_config(fast_local_config().as_bytes()).unwrap();
    let schedule = parse_translated_schedule(local_chain_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us > 0.03, "time_us={}", result.time_us);
}

#[test]
fn dependent_flows_pay_simai_send_start_latency_per_flow() {
    let cfg = parse_config(fast_local_config().as_bytes()).unwrap();
    let schedule = parse_translated_schedule(local_chain_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us > 9.0, "time_us={}", result.time_us);
    assert!(result.time_us < 10.0, "time_us={}", result.time_us);
}

#[test]
fn same_link_fanout_uses_packetized_round_robin_completion() {
    let cfg = parse_config(fast_local_config().as_bytes()).unwrap();
    let schedule =
        parse_translated_schedule(packetized_same_link_fanout_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us > 9.3, "time_us={}", result.time_us);
    assert!(result.time_us < 10.0, "time_us={}", result.time_us);
}

#[test]
fn ack_preemption_preserves_packetized_data_progress() {
    let cfg = parse_config(fast_local_config().as_bytes()).unwrap();
    let schedule =
        parse_translated_schedule(ack_preempts_packetized_fanout_translated().as_bytes()).unwrap();
    let result = simulate_case(&cfg, &schedule).unwrap();

    assert!(result.time_us < 13.0, "time_us={}", result.time_us);
}

#[test]
fn batch_manifest_discovers_candidate_translated_files_and_writes_outputs() {
    let root = tempdir().unwrap();
    let case_a = root.path().join("eval_a");
    let case_b = root.path().join("eval_b");
    fs::create_dir_all(&case_a).unwrap();
    fs::create_dir_all(&case_b).unwrap();
    fs::write(
        case_a.join("candidate-translated.json"),
        fixture_translated(),
    )
    .unwrap();
    fs::write(
        case_b.join("candidate-translated.json"),
        fixture_translated(),
    )
    .unwrap();

    let (_cfg_dir, cfg_path) = write_temp_file("config.json", fixture_config());
    let output_dir = tempdir().unwrap();
    let manifest = build_manifest(root.path(), &cfg_path, output_dir.path()).unwrap();

    assert_eq!(manifest.cases.len(), 2);
    run_batch(&manifest).unwrap();

    let result_files = fs::read_dir(output_dir.path())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("json"))
        .count();
    assert_eq!(result_files, 2);
}

#[test]
fn sketch_batch_splits_multiple_sketches_from_each_candidate_file() {
    let root = tempdir().unwrap();
    let case_a = root.path().join("eval_a");
    let case_b = root.path().join("eval_b");
    fs::create_dir_all(&case_a).unwrap();
    fs::create_dir_all(&case_b).unwrap();
    let sketch_a = r#"[
      [
        [0, 1, 0, 0, 1],
        [0, 3, 0, 0, 2],
        [1, 1, 1, 2, 3]
      ],
      [
        [0, 3, 0, 0, 2],
        [1, 1, 0, 0, 1],
        [1, 1, 1, 2, 3]
      ]
    ]"#;
    let sketch_b = r#"{
      "ngpus": 4,
      "src_gpu": 0,
      "nodes": [
        {
          "id": 0,
          "step": 0,
          "layer": 1,
          "group": 0,
          "src_dest_pair": {"srcs": [0], "dsts": [1]},
          "deps": [],
          "next": []
        },
        {
          "id": 1,
          "step": 0,
          "layer": 3,
          "group": 0,
          "src_dest_pair": {"srcs": [0], "dsts": [2]},
          "deps": [],
          "next": [2]
        },
        {
          "id": 2,
          "step": 1,
          "layer": 1,
          "group": 1,
          "src_dest_pair": {"srcs": [2], "dsts": [3]},
          "deps": [1],
          "next": []
        }
      ]
    }"#;
    fs::write(case_a.join("candidate-sketch.json"), sketch_a).unwrap();
    fs::write(case_b.join("candidate-sketch.json"), sketch_b).unwrap();

    let (_cfg_dir, cfg_path) = write_temp_file("config.json", fixture_config());
    let output_dir = tempdir().unwrap();
    let manifest = build_sketch_manifest(root.path(), &cfg_path, output_dir.path()).unwrap();

    assert_eq!(manifest.cases.len(), 3);
    assert_eq!(manifest.cases[0].name, "eval_a-sketch-000");
    assert_eq!(manifest.cases[1].name, "eval_a-sketch-001");
    assert_eq!(manifest.cases[2].name, "eval_b");

    let outputs = run_sketch_batch(&manifest).unwrap();
    assert_eq!(outputs.len(), 3);
    assert!(outputs.iter().all(|output| output.result.time_us > 0.0));
    assert!(output_dir.path().join("eval_a-sketch-000.json").exists());
    assert!(output_dir.path().join("eval_a-sketch-001.json").exists());
    assert!(output_dir.path().join("eval_b.json").exists());
}

#[test]
fn sketch_batch_accepts_direct_sketch_file_input() {
    let root = tempdir().unwrap();
    let sketch_path = root.path().join("sketch-one.json");
    fs::write(
        &sketch_path,
        r#"[
          [
            [0, 1, 0, 0, 1],
            [0, 3, 0, 0, 2],
            [1, 1, 1, 2, 3]
          ],
          [
            [0, 3, 0, 0, 2],
            [1, 1, 0, 0, 1],
            [1, 1, 1, 2, 3]
          ]
        ]"#,
    )
    .unwrap();

    let (_cfg_dir, cfg_path) = write_temp_file("config.json", fixture_config());
    let output_dir = tempdir().unwrap();
    let manifest = build_sketch_manifest(&sketch_path, &cfg_path, output_dir.path()).unwrap();

    assert_eq!(manifest.cases.len(), 2);
    assert_eq!(manifest.cases[0].name, "sketch-one-sketch-000");
    assert_eq!(manifest.cases[1].name, "sketch-one-sketch-001");

    let outputs = run_sketch_batch(&manifest).unwrap();
    assert_eq!(outputs.len(), 2);
    assert!(output_dir
        .path()
        .join("sketch-one-sketch-000.json")
        .exists());
    assert!(output_dir
        .path()
        .join("sketch-one-sketch-001.json")
        .exists());
}
