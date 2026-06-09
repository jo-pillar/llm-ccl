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

    let summary: Value =
        serde_json::from_str(&std::fs::read_to_string(out.path().join("summary.json")).unwrap())
            .unwrap();
    assert!(summary["best_completion_time_us"].as_f64().unwrap() > 0.0);
    assert_eq!(summary["simpletes_invoked"], false);
}
