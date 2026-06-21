use std::{fs, path::PathBuf};

use anyhow::Context;
use clap::Parser;
use serde::Serialize;
use syccl_sketch_search::{
    format_compact_dsl, search_sketches_profiled, SearchOptions, SycclConfig,
};

#[derive(Debug, Parser)]
#[command(about = "Generate SyCCL single-root sketch candidates")]
struct Args {
    #[arg(short, long)]
    config: PathBuf,

    #[arg(short, long)]
    output: Option<PathBuf>,

    #[arg(long)]
    limit: Option<usize>,

    #[arg(long)]
    pretty: bool,

    #[arg(long)]
    profile: bool,

    #[arg(long)]
    no_output: bool,

    #[arg(long)]
    threads: Option<usize>,

    #[arg(long)]
    print_compact_dsl: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let raw = fs::read_to_string(&args.config)
        .with_context(|| format!("failed to read {}", args.config.display()))?;
    let config: SycclConfig = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {}", args.config.display()))?;
    anyhow::ensure!(
        !(args.no_output && args.output.is_some()),
        "--no-output cannot be combined with --output"
    );
    let parallelism = args
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let result = search_sketches_profiled(
        &config,
        SearchOptions {
            limit: args.limit,
            parallelism,
        },
    )?;

    let json_ms = if args.no_output {
        0.0
    } else {
        let start = std::time::Instant::now();
        let output = if args.pretty {
            serde_json::to_string_pretty(&result.sketches)?
        } else {
            serde_json::to_string(&result.sketches)?
        };
        let json_ms = start.elapsed().as_secs_f64() * 1000.0;
        if let Some(path) = args.output {
            fs::write(&path, output)
                .with_context(|| format!("failed to write {}", path.display()))?;
        } else {
            println!("{output}");
        }
        json_ms
    };

    if args.print_compact_dsl {
        eprint!("{}", format_compact_dsl(&result.sketches));
    }

    if args.profile {
        let profile = ProfileOutput {
            sketches: result.sketches.len(),
            layer_combinations: result.stats.layer_combinations,
            group_configurations: result.stats.group_configurations,
            schedules: result.stats.schedules,
            threads: parallelism,
            layer_ms: result.timings.layer_ms,
            group_ms: result.timings.group_ms,
            step_ms: result.timings.step_ms,
            graph_ms: result.timings.graph_ms,
            search_total_ms: result.timings.total_ms,
            json_ms,
            total_ms: result.timings.total_ms + json_ms,
        };
        eprintln!("{}", serde_json::to_string(&profile)?);
    } else {
        eprintln!(
            "generated {} sketches (layers={}, group_configs={}, schedules={})",
            result.sketches.len(),
            result.stats.layer_combinations,
            result.stats.group_configurations,
            result.stats.schedules
        );
    }
    Ok(())
}

#[derive(Serialize)]
struct ProfileOutput {
    sketches: usize,
    layer_combinations: usize,
    group_configurations: usize,
    schedules: usize,
    threads: usize,
    layer_ms: f64,
    group_ms: f64,
    step_ms: f64,
    graph_ms: f64,
    search_total_ms: f64,
    json_ms: f64,
    total_ms: f64,
}
