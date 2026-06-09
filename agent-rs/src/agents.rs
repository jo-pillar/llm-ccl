use crate::topodsl::TopologySpec;

#[derive(Debug, Clone, PartialEq)]
pub struct RoundOutcome {
    pub round_id: usize,
    pub sketch_fingerprint: String,
    pub validity_status: String,
    pub completion_time_us: Option<f64>,
    pub best_so_far: bool,
    pub bottleneck_links: Vec<String>,
    pub failure_feedback: Option<String>,
}

#[derive(Debug, Default, Clone)]
pub struct RecordAgentState {
    outcomes: Vec<RoundOutcome>,
    best_round_id: Option<usize>,
    best_completion_time_us: Option<f64>,
}

impl RecordAgentState {
    pub fn observe(&mut self, outcome: RoundOutcome) {
        if outcome.best_so_far {
            self.best_round_id = Some(outcome.round_id);
            self.best_completion_time_us = outcome.completion_time_us;
        }
        self.outcomes.push(outcome);
    }

    pub fn summary(&self) -> String {
        let best = match (self.best_round_id, self.best_completion_time_us) {
            (Some(round_id), Some(time)) => {
                format!("best round={round_id} completion_time_us={time:.3}")
            }
            (Some(round_id), None) => format!("best round={round_id} completion_time_us=unknown"),
            (None, _) => "best unavailable".to_string(),
        };
        let recent_bottleneck = self
            .outcomes
            .last()
            .and_then(|outcome| outcome.bottleneck_links.first())
            .cloned()
            .unwrap_or_else(|| "none".to_string());
        format!(
            "{best}; observed_rounds={}; recent_bottleneck={recent_bottleneck}",
            self.outcomes.len()
        )
    }

    pub fn direction_hint(&self) -> String {
        if self.is_stagnating() {
            return "change the cross-host order, relay distribution, or concurrency pattern"
                .to_string();
        }
        "continue exploring variants around the current best sketch".to_string()
    }

    fn is_stagnating(&self) -> bool {
        if self.outcomes.len() < 3 {
            return false;
        }
        let recent = &self.outcomes[self.outcomes.len() - 3..];
        let duplicate_sketch = recent
            .windows(2)
            .all(|pair| pair[0].sketch_fingerprint == pair[1].sketch_fingerprint);
        let no_recent_improvement = recent.iter().all(|outcome| !outcome.best_so_far);
        duplicate_sketch || no_recent_improvement
    }
}

pub fn build_proposal_prompt(
    spec: &TopologySpec,
    collective: &str,
    message_size: u64,
    direction_hint: &str,
    exploration_records: &[String],
) -> String {
    let records = if exploration_records.is_empty() {
        "none yet".to_string()
    } else {
        exploration_records.join("\n")
    };
    format!(
        "\
You are the LLM-CCL proposal agent.

Generate one root-level SketchDSL candidate for {collective} with message_size={message_size}.
Do not output expanded schedules, dependencies, or link-level events.

[TopoDSL Python Code]
```python
{}
```

[Derived Topology Parameters]
family={}
hosts={}
gpus_per_host={}
nics_per_host={}

[SketchDSL Rules]
Return only compact transmissions shaped as:
[[step, layer, group, srcs, dsts], ...]
GPU 0 starts with the root chunk. Every other GPU must be reached exactly once.

[Exploration Records]
{records}

[Direction Hint]
{direction_hint}
",
        spec.prompt_source,
        spec.params.family,
        spec.params.hosts,
        spec.params.gpus_per_host,
        spec.params.nics_per_host,
    )
}
