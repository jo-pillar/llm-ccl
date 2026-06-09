use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlmCall {
    pub agent_type: String,
    pub round_id: usize,
    pub model_name: String,
    pub prompt: String,
    pub completion: String,
    pub prompt_hash: String,
    pub completion_hash: String,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub wall_clock_time_ms: Option<u64>,
    pub api_cost_usd: Option<f64>,
}

pub trait LlmBackend {
    fn complete(&mut self, agent_type: &str, round_id: usize, prompt: &str) -> Result<LlmCall>;
}

#[derive(Debug, Clone)]
pub struct FakeLlmBackend {
    completions: Vec<String>,
    next_completion: usize,
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleBackend {
    model_name: String,
}

impl OpenAiCompatibleBackend {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
        }
    }
}

impl LlmBackend for OpenAiCompatibleBackend {
    fn complete(&mut self, _agent_type: &str, _round_id: usize, _prompt: &str) -> Result<LlmCall> {
        Err(anyhow!(
            "OpenAI-compatible backend for model {} is not wired yet; use --fake-llm",
            self.model_name
        ))
    }
}

impl FakeLlmBackend {
    pub fn new(completions: Vec<String>) -> Self {
        Self {
            completions,
            next_completion: 0,
        }
    }
}

impl LlmBackend for FakeLlmBackend {
    fn complete(&mut self, agent_type: &str, round_id: usize, prompt: &str) -> Result<LlmCall> {
        let Some(completion) = self.completions.get(self.next_completion).cloned() else {
            return Err(anyhow!("fake LLM backend has no scripted completion left"));
        };
        self.next_completion += 1;

        Ok(LlmCall {
            agent_type: agent_type.to_string(),
            round_id,
            model_name: "fake-llm".to_string(),
            prompt: prompt.to_string(),
            completion_hash: stable_hash(&completion),
            completion,
            prompt_hash: stable_hash(prompt),
            input_tokens: Some(rough_token_count(prompt)),
            output_tokens: Some(rough_token_count(
                self.completions[self.next_completion - 1].as_str(),
            )),
            cached_input_tokens: Some(0),
            reasoning_tokens: Some(0),
            wall_clock_time_ms: Some(0),
            api_cost_usd: Some(0.0),
        })
    }
}

fn stable_hash(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    hex::encode(digest)
}

fn rough_token_count(text: &str) -> u64 {
    text.split_whitespace().count().max(1) as u64
}
