use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(about = "LLM-CCL Rust agent runner")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Run {
        #[arg(long)]
        topo: std::path::PathBuf,
        #[arg(long, default_value = "allgather")]
        collective: String,
        #[arg(long)]
        message_size: u64,
        #[arg(long, default_value_t = 10)]
        rounds: usize,
        #[arg(long)]
        output: std::path::PathBuf,
    },
}

fn main() -> Result<()> {
    let _cli = Cli::parse();
    Ok(())
}
