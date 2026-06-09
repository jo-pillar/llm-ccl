use anyhow::{Context, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationMetrics {
    pub validity_status: String,
    pub completion_time_us: f64,
    pub expanded_events: usize,
    pub algorithm_bandwidth: f64,
    pub critical_path: String,
    pub bottleneck_links: Vec<String>,
}

pub fn enumerate_small_scale_sketches(
    config_json: &str,
    limit: usize,
) -> Result<Vec<syccl_sketch_search::SketchGraph>> {
    let config: syccl_sketch_search::SycclConfig =
        serde_json::from_str(config_json).context("failed to parse sketch-search config")?;
    let (sketches, _stats) = syccl_sketch_search::search_sketches(
        &config,
        syccl_sketch_search::SearchOptions {
            limit: Some(limit),
            parallelism: 1,
        },
    )?;
    Ok(sketches)
}

pub fn evaluate_sketch_json(config_json: &str, sketch_json: &str) -> Result<EvaluationMetrics> {
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
    Ok(EvaluationMetrics {
        validity_status: "valid".to_string(),
        completion_time_us: result.time_us,
        expanded_events: schedule.sends.len(),
        algorithm_bandwidth,
        critical_path: result
            .critical_flow_id
            .map(|id| format!("flow:{id}"))
            .unwrap_or_default(),
        bottleneck_links: summarize_bottleneck_links(&result.links),
    })
}

fn summarize_bottleneck_links(links: &[flow_sim_rs::simulator::LinkSummary]) -> Vec<String> {
    let Some(link) = links
        .iter()
        .max_by_key(|link| link.queue_wait_ns.saturating_add(link.busy_ns))
    else {
        return Vec::new();
    };
    vec![format!(
        "{}->{} queue_wait_ns={} busy_ns={}",
        link.src, link.dst, link.queue_wait_ns, link.busy_ns
    )]
}
