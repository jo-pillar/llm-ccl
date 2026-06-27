use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::config::parse_config;
use crate::schedule::parse_translated_schedule;
use crate::simulator::{simulate_case, SimulationResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkCase {
    pub name: String,
    pub config: PathBuf,
    pub translated: PathBuf,
    pub rust_output: PathBuf,
    pub simai_time_us: Option<f64>,
    pub simai_end_to_end_csv: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkManifest {
    pub cases: Vec<BenchmarkCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseOutput {
    pub name: String,
    pub config: PathBuf,
    pub translated: PathBuf,
    #[serde(flatten)]
    pub result: SimulationResult,
}

pub fn build_manifest(
    input_dir: &Path,
    config: &Path,
    output_dir: &Path,
) -> Result<BenchmarkManifest> {
    let mut candidates = Vec::new();
    for entry in WalkDir::new(input_dir).min_depth(1).max_depth(2) {
        let entry = entry?;
        if entry.file_type().is_file() && entry.file_name() == "candidate-translated.json" {
            candidates.push(entry.path().to_path_buf());
        }
    }
    candidates.sort();

    let cases = candidates
        .into_iter()
        .map(|translated| {
            let name = translated
                .parent()
                .and_then(Path::file_name)
                .and_then(|name| name.to_str())
                .unwrap_or("candidate")
                .to_string();
            BenchmarkCase {
                rust_output: output_dir.join(format!("{name}.json")),
                name,
                config: config.to_path_buf(),
                translated,
                simai_time_us: None,
                simai_end_to_end_csv: None,
            }
        })
        .collect();

    Ok(BenchmarkManifest { cases })
}

pub fn write_manifest(manifest: &BenchmarkManifest, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    serde_json::to_writer_pretty(file, manifest)?;
    Ok(())
}

pub fn read_manifest(path: &Path) -> Result<BenchmarkManifest> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    serde_json::from_reader(file).context("failed to parse benchmark manifest")
}

pub fn run_batch(manifest: &BenchmarkManifest) -> Result<Vec<CaseOutput>> {
    let mut outputs = manifest
        .cases
        .par_iter()
        .enumerate()
        .map(|(index, case)| -> Result<(usize, CaseOutput)> {
            let output = run_case(case)?;
            if let Some(parent) = case.rust_output.parent() {
                fs::create_dir_all(parent)?;
            }
            let file = File::create(&case.rust_output)
                .with_context(|| format!("failed to create {}", case.rust_output.display()))?;
            serde_json::to_writer_pretty(file, &output)?;
            Ok((index, output))
        })
        .collect::<Result<Vec<_>>>()?;
    outputs.sort_by_key(|(index, _)| *index);
    Ok(outputs.into_iter().map(|(_, output)| output).collect())
}

pub fn run_case(case: &BenchmarkCase) -> Result<CaseOutput> {
    let config_file = File::open(&case.config)
        .with_context(|| format!("failed to open {}", case.config.display()))?;
    let translated_file = File::open(&case.translated)
        .with_context(|| format!("failed to open {}", case.translated.display()))?;
    let config = parse_config(config_file)?;
    let schedule = parse_translated_schedule(translated_file)?;
    let result = simulate_case(&config, &schedule)?;
    Ok(CaseOutput {
        name: case.name.clone(),
        config: case.config.clone(),
        translated: case.translated.clone(),
        result,
    })
}

pub fn write_summary_csv(outputs: &[CaseOutput], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["name", "time_us", "translated"])?;
    let mut sorted = outputs.to_vec();
    sorted.sort_by(|a, b| a.result.time_us.total_cmp(&b.result.time_us));
    for output in sorted {
        writer.write_record([
            output.name,
            format!("{:.6}", output.result.time_us),
            output.translated.display().to_string(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}
