use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use serde_json::json;

use crate::{
    agents::{build_proposal_prompt, RecordAgentState, RoundOutcome},
    config_render::render_flow_sim_config,
    evaluator::{enumerate_small_scale_sketches, EvaluationMetrics},
    llm::{FakeLlmBackend, LlmBackend},
    records::{LlmCallRecord, RoundRecord, RunRecorder},
    topodsl::load_topodsl,
};

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub topo: PathBuf,
    pub collective: String,
    pub message_size: u64,
    pub rounds: usize,
    pub output: PathBuf,
    pub fake_llm: bool,
}

pub fn run(options: RunOptions) -> Result<()> {
    if !options.fake_llm {
        return Err(anyhow!(
            "live LLM backend is not wired yet; rerun with --fake-llm"
        ));
    }
    run_fake_llm(options)
}

fn run_fake_llm(options: RunOptions) -> Result<()> {
    let spec = load_topodsl(&options.topo)?;
    let config_json =
        render_flow_sim_config(&spec.params, &options.collective, options.message_size)?;
    flow_sim_rs::config::parse_config(config_json.as_bytes())
        .context("rendered flow-sim config is invalid")?;

    let run_id = format!("{}-{}-{}", spec.topodsl_config_id, options.collective, options.message_size);
    let mut recorder = RunRecorder::new(&options.output, &run_id)?;
    let small_scale_records = summarize_small_scale_search(&config_json)?;
    let mut exploration_records = small_scale_records;
    let mut record_agent = RecordAgentState::default();
    let mut llm = FakeLlmBackend::new(fake_completions(options.rounds));
    let mut best_completion_time_us = f64::INFINITY;

    for round_id in 0..options.rounds {
        let direction_hint = record_agent.direction_hint();
        let prompt = build_proposal_prompt(
            &spec,
            &options.collective,
            options.message_size,
            &direction_hint,
            &exploration_records,
        );
        let call = llm.complete("proposal", round_id, &prompt)?;
        let (metrics, translated_json) =
            evaluate_candidate_with_translated(&config_json, &call.completion)?;
        let best_so_far = metrics.completion_time_us < best_completion_time_us;
        if best_so_far {
            best_completion_time_us = metrics.completion_time_us;
            recorder.set_best_translated_json(translated_json);
        }

        let round = RoundRecord {
            topodsl_config_id: spec.topodsl_config_id.clone(),
            collective: options.collective.clone(),
            message_size: options.message_size,
            round_id,
            prompt_hash: call.prompt_hash.clone(),
            input_tokens: call.input_tokens.unwrap_or_default(),
            output_tokens: call.output_tokens.unwrap_or_default(),
            candidate_sketch: call.completion.clone(),
            validity_status: metrics.validity_status.clone(),
            expanded_events: metrics.expanded_events,
            completion_time: metrics.completion_time_us,
            algorithm_bandwidth: metrics.algorithm_bandwidth,
            critical_path: metrics.critical_path.clone(),
            bottleneck_links: metrics.bottleneck_links.clone(),
            best_so_far,
        };
        recorder.append_llm_call(&LlmCallRecord::from_llm_call(&run_id, &call))?;
        recorder.append_round(&round)?;

        record_agent.observe(RoundOutcome {
            round_id,
            sketch_fingerprint: call.completion_hash,
            validity_status: metrics.validity_status,
            completion_time_us: Some(metrics.completion_time_us),
            best_so_far,
            bottleneck_links: metrics.bottleneck_links,
            failure_feedback: None,
        });
        exploration_records.push(record_agent.summary());
    }

    recorder.finish_summary()?;
    Ok(())
}

fn summarize_small_scale_search(config_json: &str) -> Result<Vec<String>> {
    let sketches = enumerate_small_scale_sketches(config_json, 2)?;
    Ok(sketches
        .iter()
        .enumerate()
        .map(|(idx, sketch)| {
            format!(
                "small-scale sketch {idx}: root={} nodes={}",
                sketch.src_gpu,
                sketch.nodes.len()
            )
        })
        .collect())
}

fn fake_completions(rounds: usize) -> Vec<String> {
    let valid_allgather_sketch = r#"[
      [
        [0, 1, 0, 0, 1],
        [0, 4, 0, 0, [2, 4, 6]],
        [1, 1, 1, 2, 3],
        [1, 1, 2, 4, 5],
        [1, 1, 3, 6, 7]
      ]
    ]"#;
    (0..rounds)
        .map(|_| valid_allgather_sketch.to_string())
        .collect()
}

fn evaluate_candidate_with_translated(
    config_json: &str,
    sketch_json: &str,
) -> Result<(EvaluationMetrics, String)> {
    let config = flow_sim_rs::config::parse_config(config_json.as_bytes())
        .context("failed to parse flow-sim config")?;
    let sketches = flow_sim_rs::sketch::parse_compact_sketches(sketch_json.as_bytes(), &config)
        .context("failed to parse compact sketch")?;
    let schedule = flow_sim_rs::sketch::sketches_to_translated_schedule(&sketches, &config)
        .context("failed to expand sketch")?;
    let result = flow_sim_rs::simulator::simulate_case(&config, &schedule)
        .context("flow-sim simulation failed")?;
    let algorithm_bandwidth = if result.time_us > 0.0 {
        config.coll_bytes as f64 / result.time_us
    } else {
        0.0
    };
    let metrics = EvaluationMetrics {
        validity_status: "valid".to_string(),
        completion_time_us: result.time_us,
        expanded_events: schedule.sends.len(),
        algorithm_bandwidth,
        critical_path: result
            .critical_flow_id
            .map(|id| format!("flow:{id}"))
            .unwrap_or_default(),
        bottleneck_links: result
            .links
            .iter()
            .max_by_key(|link| link.queue_wait_ns.saturating_add(link.busy_ns))
            .map(|link| {
                vec![format!(
                    "{}->{} queue_wait_ns={} busy_ns={}",
                    link.src, link.dst, link.queue_wait_ns, link.busy_ns
                )]
            })
            .unwrap_or_default(),
    };
    let translated_json = json!({
        "coll_name": schedule.coll_name,
        "ngpus": schedule.ngpus,
        "chunk_size_byte": schedule.chunk_size_byte,
        "sends": schedule.sends.iter().map(|send| json!({
            "src_chunk": send.src_chunk,
            "chunk_index": send.chunk_index,
            "src_gpu": send.src_gpu,
            "dst_gpu": send.dst_gpu,
            "epoch": send.epoch,
            "layer_used": send.layer_used,
            "order": send.order,
        })).collect::<Vec<_>>(),
    });
    Ok((metrics, serde_json::to_string_pretty(&translated_json)?))
}
