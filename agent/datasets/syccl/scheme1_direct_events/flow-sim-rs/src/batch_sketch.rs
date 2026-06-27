use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::batch::CaseOutput;
use crate::config::parse_config;
use crate::simulator::simulate_case;
use crate::sketch::{parse_compact_sketch_candidates, sketches_to_translated_schedule};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SketchBenchmarkCase {
    pub name: String,
    pub config: PathBuf,
    pub sketch: PathBuf,
    pub sketch_index: usize,
    pub rust_output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SketchBenchmarkManifest {
    pub cases: Vec<SketchBenchmarkCase>,
}

pub fn build_sketch_manifest(
    input_dir: &Path,
    config: &Path,
    output_dir: &Path,
) -> Result<SketchBenchmarkManifest> {
    let config_file =
        File::open(config).with_context(|| format!("failed to open {}", config.display()))?;
    let config_data = parse_config(config_file)?;
    let mut sketches = Vec::new();
    if input_dir.is_file() {
        sketches.push(input_dir.to_path_buf());
    } else {
        for entry in WalkDir::new(input_dir).min_depth(1).max_depth(2) {
            let entry = entry?;
            if entry.file_type().is_file() && entry.file_name() == "candidate-sketch.json" {
                sketches.push(entry.path().to_path_buf());
            }
        }
    }
    sketches.sort();

    let mut cases = Vec::new();
    for sketch_path in sketches {
        let sketch_file = File::open(&sketch_path)
            .with_context(|| format!("failed to open {}", sketch_path.display()))?;
        let sketch_count = parse_compact_sketch_candidates(sketch_file, &config_data)?.len();
        let base_name = case_base_name(&sketch_path, input_dir.is_file());
        for sketch_index in 0..sketch_count {
            let name = if sketch_count == 1 {
                base_name.to_string()
            } else {
                format!("{base_name}-sketch-{sketch_index:03}")
            };
            cases.push(SketchBenchmarkCase {
                rust_output: output_dir.join(format!("{name}.json")),
                name,
                config: config.to_path_buf(),
                sketch: sketch_path.clone(),
                sketch_index,
            });
        }
    }

    Ok(SketchBenchmarkManifest { cases })
}

fn case_base_name(sketch_path: &Path, direct_file_input: bool) -> String {
    if direct_file_input {
        return sketch_path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("candidate")
            .to_string();
    }
    sketch_path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .unwrap_or("candidate")
        .to_string()
}

pub fn write_sketch_manifest(manifest: &SketchBenchmarkManifest, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    serde_json::to_writer_pretty(file, manifest)?;
    Ok(())
}

pub fn run_sketch_batch(manifest: &SketchBenchmarkManifest) -> Result<Vec<CaseOutput>> {
    let mut outputs = manifest
        .cases
        .par_iter()
        .enumerate()
        .map(|(index, case)| -> Result<(usize, CaseOutput)> {
            let output = run_sketch_case(case)?;
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

pub fn run_sketch_case(case: &SketchBenchmarkCase) -> Result<CaseOutput> {
    let config_file = File::open(&case.config)
        .with_context(|| format!("failed to open {}", case.config.display()))?;
    let sketch_file = File::open(&case.sketch)
        .with_context(|| format!("failed to open {}", case.sketch.display()))?;
    let config = parse_config(config_file)?;
    let sketches = parse_compact_sketch_candidates(sketch_file, &config)?;
    let graph = sketches.get(case.sketch_index).with_context(|| {
        format!(
            "sketch index {} out of range for {} sketches in {}",
            case.sketch_index,
            sketches.len(),
            case.sketch.display()
        )
    })?;
    let schedule = sketches_to_translated_schedule(std::slice::from_ref(graph), &config)?;
    let result = simulate_case(&config, &schedule)?;
    Ok(CaseOutput {
        name: case.name.clone(),
        config: case.config.clone(),
        translated: case.sketch.clone(),
        result,
    })
}
