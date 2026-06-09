use crate::topodsl::TopologySpec;

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
