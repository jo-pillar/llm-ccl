use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TopologyParams {
    pub family: String,
    pub hosts: usize,
    pub gpus_per_host: usize,
    pub nics_per_host: usize,
    #[serde(default)]
    pub leaf_switches: Option<usize>,
    #[serde(default)]
    pub spine_switches: Option<usize>,
    #[serde(default)]
    pub rails: Option<usize>,
    #[serde(default = "default_collective")]
    pub collective: String,
    #[serde(default = "default_message_size")]
    pub message_size: u64,
    #[serde(default = "default_host_bw")]
    pub host_bw_mbpus: f64,
    #[serde(default = "default_host_lat")]
    pub host_lat_us: f64,
    #[serde(default = "default_nic_bw")]
    pub nic_bw_mbpus: f64,
    #[serde(default)]
    pub nic_lat_us: f64,
    #[serde(default = "default_net_bw")]
    pub leaf_bw_mbpus: f64,
    #[serde(default = "default_net_lat")]
    pub leaf_lat_us: f64,
    #[serde(default = "default_net_bw")]
    pub spine_bw_mbpus: f64,
    #[serde(default = "default_net_lat")]
    pub spine_lat_us: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TopologySpec {
    pub path: PathBuf,
    pub prompt_source: String,
    pub params: TopologyParams,
    pub topodsl_config_id: String,
}

pub fn load_topodsl(path: impl AsRef<Path>) -> Result<TopologySpec> {
    let path = path.as_ref();
    let prompt_source = fs::read_to_string(path)
        .with_context(|| format!("failed to read TopoDSL {}", path.display()))?;
    let params = match path.extension().and_then(|ext| ext.to_str()) {
        Some("py") => load_python_params(path)?,
        Some("txt") => load_text_params(&prompt_source)?,
        _ => anyhow::bail!("unsupported TopoDSL extension for {}", path.display()),
    };
    let topodsl_config_id = stable_id(&params)?;
    Ok(TopologySpec {
        path: path.to_path_buf(),
        prompt_source,
        params,
        topodsl_config_id,
    })
}

fn load_text_params(source: &str) -> Result<TopologyParams> {
    let mut fields = BTreeMap::new();
    for (idx, raw_line) in source.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| anyhow!("invalid TopoDSL text line {}: {line}", idx + 1))?;
        fields.insert(key.trim().to_string(), value.trim().to_string());
    }

    Ok(TopologyParams {
        family: required_string(&fields, "family")?,
        hosts: required_parse(&fields, "hosts")?,
        gpus_per_host: required_parse(&fields, "gpus_per_host")?,
        nics_per_host: required_parse(&fields, "nics_per_host")?,
        leaf_switches: optional_parse(&fields, "leaf_switches")?,
        spine_switches: optional_parse(&fields, "spine_switches")?,
        rails: optional_parse(&fields, "rails")?,
        collective: fields
            .get("collective")
            .cloned()
            .unwrap_or_else(default_collective),
        message_size: optional_parse(&fields, "message_size")?.unwrap_or_else(default_message_size),
        host_bw_mbpus: optional_parse(&fields, "host_bw_mbpus")?.unwrap_or_else(default_host_bw),
        host_lat_us: optional_parse(&fields, "host_lat_us")?.unwrap_or_else(default_host_lat),
        nic_bw_mbpus: optional_parse(&fields, "nic_bw_mbpus")?.unwrap_or_else(default_nic_bw),
        nic_lat_us: optional_parse(&fields, "nic_lat_us")?.unwrap_or_default(),
        leaf_bw_mbpus: optional_parse(&fields, "leaf_bw_mbpus")?.unwrap_or_else(default_net_bw),
        leaf_lat_us: optional_parse(&fields, "leaf_lat_us")?.unwrap_or_else(default_net_lat),
        spine_bw_mbpus: optional_parse(&fields, "spine_bw_mbpus")?.unwrap_or_else(default_net_bw),
        spine_lat_us: optional_parse(&fields, "spine_lat_us")?.unwrap_or_else(default_net_lat),
    })
}

fn load_python_params(path: &Path) -> Result<TopologyParams> {
    let script = r#"
import importlib.util
import json
import sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("llm_ccl_topodsl", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if not hasattr(module, "topology"):
    raise SystemExit("TopoDSL python file must define topology()")
print(json.dumps(module.topology(), sort_keys=True))
"#;
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(path)
        .output()
        .with_context(|| format!("failed to execute python TopoDSL {}", path.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "python TopoDSL {} failed: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("failed to parse topology() output from {}", path.display()))
}

fn required_string(fields: &BTreeMap<String, String>, key: &str) -> Result<String> {
    fields
        .get(key)
        .cloned()
        .ok_or_else(|| anyhow!("missing required TopoDSL field {key}"))
}

fn required_parse<T>(fields: &BTreeMap<String, String>, key: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = fields
        .get(key)
        .ok_or_else(|| anyhow!("missing required TopoDSL field {key}"))?;
    value
        .parse()
        .map_err(|err| anyhow!("invalid TopoDSL field {key}={value}: {err}"))
}

fn optional_parse<T>(fields: &BTreeMap<String, String>, key: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let Some(value) = fields.get(key) else {
        return Ok(None);
    };
    value
        .parse()
        .map(Some)
        .map_err(|err| anyhow!("invalid TopoDSL field {key}={value}: {err}"))
}

fn stable_id(params: &TopologyParams) -> Result<String> {
    let json = serde_json::to_vec(params)?;
    let digest = Sha256::digest(json);
    Ok(format!("topodsl-{}", hex::encode(&digest[..8])))
}

fn default_collective() -> String {
    "allgather".to_string()
}

fn default_message_size() -> u64 {
    4096
}

fn default_host_bw() -> f64 {
    0.15
}

fn default_host_lat() -> f64 {
    3.0
}

fn default_nic_bw() -> f64 {
    0.0455
}

fn default_net_bw() -> f64 {
    0.0455
}

fn default_net_lat() -> f64 {
    10.0
}
