use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use flow_sim_rs::batch::{
    build_manifest, read_manifest, run_batch, write_manifest, write_summary_csv,
};
use flow_sim_rs::batch_sketch::{build_sketch_manifest, run_sketch_batch, write_sketch_manifest};
use flow_sim_rs::compare::{compare_manifest, write_compare_report};
use flow_sim_rs::config::parse_config;
use flow_sim_rs::schedule::{
    parse_all_translated_schedules, parse_translated_schedule, to_syccl_resim_input,
};
use flow_sim_rs::simulator::{simulate_case, SimulationResult};
use flow_sim_rs::sketch::{parse_compact_sketches, sketches_to_translated_schedule};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Fast Rust flow-level simulator for SYCCL translated schedules"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Simulate {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        translated: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        algorithm_index: Option<usize>,
    },
    SimulateSketch {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        sketch: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        dump_translated: Option<PathBuf>,
    },
    Manifest {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        input_dir: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
    },
    Batch {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        input_dir: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long)]
        summary: Option<PathBuf>,
    },
    BatchSketch {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        input_dir: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long)]
        summary: Option<PathBuf>,
    },
    Compare {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 5.0)]
        abs_ratio_limit: f64,
    },
    Stats {
        #[arg(long)]
        translated: PathBuf,
    },
    CacheSimai {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        simai_root: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        limit: usize,
        #[arg(long, default_value = "A100")]
        gpu_type: String,
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Simulate {
            config,
            translated,
            output,
            algorithm_index,
        } => simulate_cmd(&config, &translated, &output, algorithm_index),
        Commands::SimulateSketch {
            config,
            sketch,
            output,
            dump_translated,
        } => simulate_sketch_cmd(&config, &sketch, &output, dump_translated.as_deref()),
        Commands::Manifest {
            config,
            input_dir,
            output_dir,
            manifest,
        } => {
            let manifest_data = build_manifest(&input_dir, &config, &output_dir)?;
            write_manifest(&manifest_data, &manifest)?;
            println!(
                "wrote {} cases to {}",
                manifest_data.cases.len(),
                manifest.display()
            );
            Ok(())
        }
        Commands::Batch {
            config,
            input_dir,
            output_dir,
            manifest,
            summary,
        } => {
            let manifest_data = build_manifest(&input_dir, &config, &output_dir)?;
            if let Some(path) = &manifest {
                write_manifest(&manifest_data, path)?;
            }
            let outputs = run_batch(&manifest_data)?;
            if let Some(path) = &summary {
                write_summary_csv(&outputs, path)?;
            }
            println!("simulated {} cases", outputs.len());
            Ok(())
        }
        Commands::BatchSketch {
            config,
            input_dir,
            output_dir,
            manifest,
            summary,
        } => {
            let manifest_data = build_sketch_manifest(&input_dir, &config, &output_dir)?;
            if let Some(path) = &manifest {
                write_sketch_manifest(&manifest_data, path)?;
            }
            let outputs = run_sketch_batch(&manifest_data)?;
            if let Some(path) = &summary {
                write_summary_csv(&outputs, path)?;
            }
            println!("simulated {} sketch cases", outputs.len());
            Ok(())
        }
        Commands::Compare {
            manifest,
            output,
            abs_ratio_limit,
        } => {
            let report = compare_manifest(&manifest, abs_ratio_limit)?;
            write_compare_report(&report, &output)?;
            println!(
                "wrote compare report with {} cases to {}",
                report.cases.len(),
                output.display()
            );
            Ok(())
        }
        Commands::Stats { translated } => {
            let translated_file = File::open(&translated)
                .with_context(|| format!("failed to open {}", translated.display()))?;
            let schedule = parse_translated_schedule(translated_file)?;
            println!(
                "ngpus={} sends={} chunk_size_byte={}",
                schedule.ngpus,
                schedule.sends.len(),
                schedule.chunk_size_byte.unwrap_or(0)
            );
            Ok(())
        }
        Commands::CacheSimai {
            manifest,
            simai_root,
            output_dir,
            limit,
            gpu_type,
            dry_run,
        } => cache_simai_cmd(
            &manifest,
            &simai_root,
            &output_dir,
            limit,
            &gpu_type,
            dry_run,
        ),
    }
}

#[derive(Debug, Serialize)]
struct SimulateOutput {
    solutions: Vec<SimulateSolutionOutput>,
}

#[derive(Debug, Serialize)]
struct SimulateSolutionOutput {
    solution_index: usize,
    syccl_time_us: Option<f64>,
    rust_time_us: f64,
    result: SimulationResult,
}

fn simulate_cmd(
    config: &Path,
    translated: &Path,
    output: &Path,
    algorithm_index: Option<usize>,
) -> Result<()> {
    let config_file =
        File::open(config).with_context(|| format!("failed to open {}", config.display()))?;
    let translated_file = File::open(translated)
        .with_context(|| format!("failed to open {}", translated.display()))?;
    let config_data = parse_config(config_file)?;
    let schedules = parse_all_translated_schedules(translated_file)?;
    let selected: Vec<_> = if let Some(index) = algorithm_index {
        let schedule = schedules
            .into_iter()
            .find(|schedule| schedule.algorithm_index == index)
            .ok_or_else(|| anyhow!("algorithm_index {} is out of range", index))?;
        vec![schedule]
    } else {
        schedules
    };
    let mut solutions = Vec::with_capacity(selected.len());
    for schedule in selected {
        let result = simulate_case(&config_data, &schedule.schedule)?;
        solutions.push(SimulateSolutionOutput {
            solution_index: schedule.algorithm_index,
            syccl_time_us: schedule.syccl_time_us,
            rust_time_us: result.time_us,
            result,
        });
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let file =
        File::create(output).with_context(|| format!("failed to create {}", output.display()))?;
    let output_data = SimulateOutput { solutions };
    serde_json::to_writer_pretty(file, &output_data)?;
    println!(
        "simulated {} solutions output={}",
        output_data.solutions.len(),
        output.display()
    );
    println!("index,syccl_time_us,rust_time_us");
    for solution in &output_data.solutions {
        println!(
            "{},{},{:.6}",
            solution.solution_index,
            solution
                .syccl_time_us
                .map(|time| format!("{:.6}", time))
                .unwrap_or_else(|| "null".to_string()),
            solution.rust_time_us
        );
    }
    Ok(())
}

fn simulate_sketch_cmd(
    config: &Path,
    sketch: &Path,
    output: &Path,
    dump_translated: Option<&Path>,
) -> Result<()> {
    let config_file =
        File::open(config).with_context(|| format!("failed to open {}", config.display()))?;
    let sketch_file =
        File::open(sketch).with_context(|| format!("failed to open {}", sketch.display()))?;
    let config_data = parse_config(config_file)?;
    let sketches = parse_compact_sketches(sketch_file, &config_data)?;
    let schedule = sketches_to_translated_schedule(&sketches, &config_data)?;
    if let Some(translated_path) = dump_translated {
        if let Some(parent) = translated_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(translated_path)
            .with_context(|| format!("failed to create {}", translated_path.display()))?;
        serde_json::to_writer_pretty(file, &to_syccl_resim_input(&schedule, &config_data)?)?;
    }
    let result = simulate_case(&config_data, &schedule)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let file =
        File::create(output).with_context(|| format!("failed to create {}", output.display()))?;
    serde_json::to_writer_pretty(file, &result)?;
    println!("time_us={:.6} output={}", result.time_us, output.display());
    Ok(())
}

fn cache_simai_cmd(
    manifest_path: &Path,
    simai_root: &Path,
    output_dir: &Path,
    limit: usize,
    gpu_type: &str,
    dry_run: bool,
) -> Result<()> {
    let manifest = read_manifest(manifest_path)?;
    fs::create_dir_all(output_dir)?;
    let selected = manifest.cases.iter().take(limit);
    let mut commands_log = String::new();
    for case in selected {
        let case_dir = output_dir.join(&case.name);
        fs::create_dir_all(&case_dir)?;
        let topofile = case_dir.join("topofile");
        let sys_file = case_dir.join("syccl-sys.txt");
        let workload = case_dir.join("workload-allgather.txt");
        let simai_conf = case_dir.join("simai.conf");
        write_simai_inputs(case, &sys_file, &workload, &simai_conf)?;

        let topo_script =
            simai_root.join("astra-sim-alibabacloud/inputs/topo/gen_Syccl_Multirail_Topo.py");
        let sim_dir = simai_root
            .join("astra-sim-alibabacloud/extern/network_backend/ns3-interface/simulation");
        let ns3_bin = sim_dir.join("build/scratch/ns3.36.1-AstraSimNetwork-debug");

        let topo_cmd = format!(
            "python3 {} -c {} -o {} -gt {}",
            topo_script.display(),
            case.config.display(),
            topofile.display(),
            gpu_type
        );
        let sim_cmd = format!(
            "cd {} && AS_SEND_LAT=3 AS_NVLS_ENABLE=1 {} -t 1 -w {} -s {} -n {} -c {}",
            sim_dir.display(),
            ns3_bin.display(),
            workload.display(),
            sys_file.display(),
            topofile.display(),
            simai_conf.display()
        );
        commands_log.push_str(&topo_cmd);
        commands_log.push('\n');
        commands_log.push_str(&sim_cmd);
        commands_log.push('\n');

        if dry_run {
            continue;
        }
        run_shell(&topo_cmd, Path::new("."))?;
        run_shell(
            &format!(
                "AS_SEND_LAT=3 AS_NVLS_ENABLE=1 {} -t 1 -w {} -s {} -n {} -c {}",
                ns3_bin.display(),
                workload.display(),
                sys_file.display(),
                topofile.display(),
                simai_conf.display()
            ),
            &sim_dir,
        )?;
    }
    let mut log = File::create(output_dir.join("commands.log"))?;
    log.write_all(commands_log.as_bytes())?;
    println!(
        "{} SimAI command set(s) {} under {}",
        limit.min(manifest.cases.len()),
        if dry_run { "prepared" } else { "run" },
        output_dir.display()
    );
    Ok(())
}

fn write_simai_inputs(
    case: &flow_sim_rs::batch::BenchmarkCase,
    sys_file: &Path,
    workload: &Path,
    simai_conf: &Path,
) -> Result<()> {
    let config_file = File::open(&case.config)
        .with_context(|| format!("failed to open {}", case.config.display()))?;
    let config = parse_config(config_file)?;
    fs::write(
        sys_file,
        format!(
            "all-gather-implementation: sycclFlowModel\nsyccl-translated-path: {}\nsyccl-flow-size: {}\n",
            case.translated.display(),
            config.coll_bytes
        ),
    )?;
    fs::write(
        workload,
        format!(
            "HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 512 ep: 1 pp: 1 vpp: 1 ga: 1 all_gpus: 512 checkpoints: 0 checkpoint_initiates: 0\n1\nsyccl_ag -1 1 ALLGATHER {} 1 NONE 0 1 NONE 0 1\n",
            config.coll_bytes
        ),
    )?;
    fs::write(
        simai_conf,
        format!(
            "FCT_OUTPUT_FILE {}\nOUTPUT_TO_FILE 1\n",
            simai_conf
                .parent()
                .ok_or_else(|| anyhow!("simai conf missing parent"))?
                .join("flow_fct.txt")
                .display()
        ),
    )?;
    Ok(())
}

fn run_shell(command: &str, cwd: &Path) -> Result<()> {
    let status = Command::new("bash")
        .arg("-lc")
        .arg(command)
        .current_dir(cwd)
        .status()
        .with_context(|| format!("failed to run {}", command))?;
    if !status.success() {
        return Err(anyhow!("command failed with {}: {}", status, command));
    }
    Ok(())
}
