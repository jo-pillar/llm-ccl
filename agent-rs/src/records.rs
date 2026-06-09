use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::llm::LlmCall;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoundRecord {
    pub topodsl_config_id: String,
    pub collective: String,
    pub message_size: u64,
    pub round_id: usize,
    pub prompt_hash: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub candidate_sketch: String,
    pub validity_status: String,
    pub expanded_events: usize,
    pub completion_time: f64,
    pub algorithm_bandwidth: f64,
    pub critical_path: String,
    pub bottleneck_links: Vec<String>,
    pub best_so_far: bool,
}

impl RoundRecord {
    pub fn fixture() -> Self {
        Self {
            topodsl_config_id: "topodsl-fixture".to_string(),
            collective: "allgather".to_string(),
            message_size: 4096,
            round_id: 0,
            prompt_hash: "prompt-hash".to_string(),
            input_tokens: 12,
            output_tokens: 5,
            candidate_sketch: "[]".to_string(),
            validity_status: "valid".to_string(),
            expanded_events: 1,
            completion_time: 10.0,
            algorithm_bandwidth: 409.6,
            critical_path: "flow:0".to_string(),
            bottleneck_links: vec!["0->1 queue_wait_ns=0 busy_ns=1".to_string()],
            best_so_far: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmCallRecord {
    pub run_id: String,
    pub agent_type: String,
    pub round_id: usize,
    pub model_name: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub reasoning_tokens: u64,
    pub wall_clock_time_ms: u64,
    pub api_cost_usd: f64,
    pub prompt_hash: String,
    pub completion_hash: String,
}

impl LlmCallRecord {
    pub fn fixture() -> Self {
        Self {
            run_id: "run-a".to_string(),
            agent_type: "proposal".to_string(),
            round_id: 0,
            model_name: "fake-llm".to_string(),
            input_tokens: 12,
            output_tokens: 5,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
            wall_clock_time_ms: 0,
            api_cost_usd: 0.0,
            prompt_hash: "prompt-hash".to_string(),
            completion_hash: "completion-hash".to_string(),
        }
    }

    pub fn from_llm_call(run_id: &str, call: &LlmCall) -> Self {
        Self {
            run_id: run_id.to_string(),
            agent_type: call.agent_type.clone(),
            round_id: call.round_id,
            model_name: call.model_name.clone(),
            input_tokens: call.input_tokens.unwrap_or_default(),
            output_tokens: call.output_tokens.unwrap_or_default(),
            cached_input_tokens: call.cached_input_tokens.unwrap_or_default(),
            reasoning_tokens: call.reasoning_tokens.unwrap_or_default(),
            wall_clock_time_ms: call.wall_clock_time_ms.unwrap_or_default(),
            api_cost_usd: call.api_cost_usd.unwrap_or_default(),
            prompt_hash: call.prompt_hash.clone(),
            completion_hash: call.completion_hash.clone(),
        }
    }
}

#[derive(Debug)]
pub struct RunRecorder {
    output_dir: PathBuf,
    run_id: String,
    round_count: usize,
    llm_call_count: usize,
    best_round: Option<RoundRecord>,
    best_translated_json: Option<String>,
}

impl RunRecorder {
    pub fn new(output_dir: impl AsRef<Path>, run_id: impl Into<String>) -> Result<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        fs::create_dir_all(&output_dir)?;
        fs::write(output_dir.join("rounds.jsonl"), "")?;
        fs::write(output_dir.join("llm_calls.jsonl"), "")?;
        Ok(Self {
            output_dir,
            run_id: run_id.into(),
            round_count: 0,
            llm_call_count: 0,
            best_round: None,
            best_translated_json: None,
        })
    }

    pub fn append_round(&mut self, record: &RoundRecord) -> Result<()> {
        append_jsonl(&self.output_dir.join("rounds.jsonl"), record)?;
        self.round_count += 1;
        if record.best_so_far {
            self.best_round = Some(record.clone());
        }
        Ok(())
    }

    pub fn append_llm_call(&mut self, record: &LlmCallRecord) -> Result<()> {
        append_jsonl(&self.output_dir.join("llm_calls.jsonl"), record)?;
        self.llm_call_count += 1;
        Ok(())
    }

    pub fn set_best_translated_json(&mut self, translated_json: impl Into<String>) {
        self.best_translated_json = Some(translated_json.into());
    }

    pub fn finish_summary(&self) -> Result<()> {
        let best_completion_time_us = self
            .best_round
            .as_ref()
            .map(|record| record.completion_time)
            .unwrap_or_default();
        let summary = json!({
            "run_id": self.run_id,
            "rounds": self.round_count,
            "llm_calls": self.llm_call_count,
            "best_completion_time_us": best_completion_time_us,
            "simpletes_invoked": false,
        });
        fs::write(
            self.output_dir.join("summary.json"),
            serde_json::to_string_pretty(&summary)?,
        )?;
        fs::write(
            self.output_dir.join("best_sketch.json"),
            self.best_round
                .as_ref()
                .map(|record| record.candidate_sketch.as_str())
                .unwrap_or("null"),
        )?;
        fs::write(
            self.output_dir.join("best_translated.json"),
            self.best_translated_json.as_deref().unwrap_or("null"),
        )?;
        Ok(())
    }
}

fn append_jsonl<T>(path: &Path, value: &T) -> Result<()>
where
    T: Serialize,
{
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}
