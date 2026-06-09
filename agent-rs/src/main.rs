use anyhow::Result;
use clap::{Parser, Subcommand};
use llm_ccl_agent::run::{self, RunOptions};

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
        #[arg(long)]
        fake_llm: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(Commands::Run {
        topo,
        collective,
        message_size,
        rounds,
        output,
        fake_llm,
    }) = cli.command
    {
        run::run(RunOptions {
            topo,
            collective,
            message_size,
            rounds,
            output,
            fake_llm,
        })?;
    }
    Ok(())
}
