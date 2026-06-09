use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

pub fn canonicalize_sketch_dsl(proposal: &str) -> Result<String> {
    let raw = extract_code_block(proposal).unwrap_or(proposal).trim();
    let normalized = parse_json_or_python_literal(raw)?;
    let canonical = normalize_candidate_shape(normalized)?;
    serde_json::to_string_pretty(&canonical).map_err(Into::into)
}

fn extract_code_block(text: &str) -> Option<&str> {
    let start = text.find("```")?;
    let after_ticks = &text[start + 3..];
    let content_start = after_ticks
        .find('\n')
        .map(|idx| idx + 1)
        .unwrap_or_default();
    let after_lang = &after_ticks[content_start..];
    let end = after_lang.find("```")?;
    Some(&after_lang[..end])
}

fn parse_json_or_python_literal(raw: &str) -> Result<Value> {
    if let Ok(value) = serde_json::from_str(raw) {
        return Ok(value);
    }

    let script = r#"
import ast
import json
import sys

value = ast.literal_eval(sys.stdin.read())
print(json.dumps(value, sort_keys=True))
"#;
    let mut child = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("failed to start python3 for SketchDSL normalization")?;
    {
        use std::io::Write;
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("failed to open python3 stdin"))?;
        stdin.write_all(raw.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "failed to parse SketchDSL as JSON or Python literal: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    serde_json::from_slice(&output.stdout).context("python3 emitted invalid SketchDSL JSON")
}

fn normalize_candidate_shape(value: Value) -> Result<Value> {
    let Value::Array(items) = value else {
        bail!("SketchDSL must be an array of transmissions or candidate arrays");
    };
    if items.is_empty() {
        bail!("SketchDSL must not be empty");
    }

    if items.iter().all(looks_like_transmission) {
        return Ok(Value::Array(vec![Value::Array(
            items
                .into_iter()
                .enumerate()
                .map(|(idx, item)| normalize_transmission(item, idx))
                .collect::<Result<Vec<_>>>()?,
        )]));
    }

    let mut candidates = Vec::with_capacity(items.len());
    for (candidate_idx, candidate) in items.into_iter().enumerate() {
        let Value::Array(transmissions) = candidate else {
            bail!("SketchDSL candidate {candidate_idx} must be an array");
        };
        let normalized = transmissions
            .into_iter()
            .enumerate()
            .map(|(idx, item)| normalize_transmission(item, idx))
            .collect::<Result<Vec<_>>>()?;
        candidates.push(Value::Array(normalized));
    }
    Ok(Value::Array(candidates))
}

fn normalize_transmission(value: Value, index: usize) -> Result<Value> {
    match value {
        Value::Array(values) if values.len() == 5 => Ok(Value::Array(values)),
        Value::Object(map) => {
            let step = map
                .get("step")
                .cloned()
                .ok_or_else(|| anyhow!("transmission {index} missing step"))?;
            let layer = map
                .get("layer")
                .cloned()
                .ok_or_else(|| anyhow!("transmission {index} missing layer"))?;
            let group = map
                .get("group")
                .cloned()
                .ok_or_else(|| anyhow!("transmission {index} missing group"))?;
            let srcs = map
                .get("srcs")
                .or_else(|| map.get("src"))
                .cloned()
                .ok_or_else(|| anyhow!("transmission {index} missing srcs/src"))?;
            let dsts = map
                .get("dsts")
                .or_else(|| map.get("dst"))
                .cloned()
                .ok_or_else(|| anyhow!("transmission {index} missing dsts/dst"))?;
            Ok(Value::Array(vec![step, layer, group, srcs, dsts]))
        }
        _ => bail!("transmission {index} must be a dict or [step, layer, group, srcs, dsts]"),
    }
}

fn looks_like_transmission(value: &Value) -> bool {
    match value {
        Value::Array(items) => {
            items.len() == 5 && is_scalar(&items[0]) && is_scalar(&items[1]) && is_scalar(&items[2])
        }
        Value::Object(map) => {
            map.contains_key("step")
                && map.contains_key("layer")
                && map.contains_key("group")
                && (map.contains_key("srcs") || map.contains_key("src"))
                && (map.contains_key("dsts") || map.contains_key("dst"))
        }
        _ => false,
    }
}

fn is_scalar(value: &Value) -> bool {
    matches!(value, Value::Number(_) | Value::String(_))
}
