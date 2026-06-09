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
