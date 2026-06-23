#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

from simpletes.config import EngineConfig
from simpletes.engine import SimpleTESEngine
from simpletes.engine.syccl_two_agent import SycclTwoAgentRuntime
from syccl_agents.prompts import load_init_program, load_prompt


AGENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_TOML = AGENT_ROOT / "env.toml"


def load_env_toml(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if tomllib is None:
        raise RuntimeError("tomllib is required to load --env-toml on this Python version")
    env_path = Path(path).expanduser()
    if not env_path.exists():
        raise FileNotFoundError(f"env.toml not found: {env_path}")
    data = tomllib.loads(env_path.read_text(encoding="utf-8"))
    return {key: str(value) for key, value in data.items() if key in {"model", "api_base", "api_key"} and value}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SyCCL two-agent search on SimpleTES infrastructure")
    parser.add_argument("--topo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--collective", default=None)
    parser.add_argument("--message-size", type=int, default=None)
    parser.add_argument("--flow-sim-bin", default="flow-sim-rs")
    parser.add_argument("--env-toml", default=str(DEFAULT_ENV_TOML))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=393216)
    parser.add_argument("--timeout", type=float, default=3000.0)
    parser.add_argument(
        "--save-llm-io",
        action="store_true",
        help="Enable SimpleTES native checkpoint storage of full LLM input/output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    task_dir = output_dir / "simpletes_task"
    init_program, evaluator, instruction = _write_task_files(task_dir)
    env = load_env_toml(args.env_toml)
    config = EngineConfig(
        init_program=str(init_program),
        evaluator_path=str(evaluator),
        instruction_path=str(instruction),
        max_generations=args.rounds,
        init_eval_repeats=1,
        output_path=str(output_dir / "checkpoints"),
        save_llm_io=args.save_llm_io,
        log_interval=1024,
        db_show_interval=1,
        num_inspirations=0,
        num_chains=1,
        k_candidates=1,
        gen_concurrency=1,
        eval_concurrency=1,
        model=env.get("model", "gemini/gemini-2.0-flash"),
        api_base=env.get("api_base"),
        api_key=env.get("api_key"),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    runtime = SycclTwoAgentRuntime(
        topo_path=args.topo,
        collective=args.collective,
        message_size=args.message_size,
        flow_sim_bin=args.flow_sim_bin,
    )
    engine = SimpleTESEngine(config, runtime=runtime)
    asyncio.run(engine.run())
    for line in _format_final_output(engine):
        print(line)
    return 0


def _format_final_output(engine: SimpleTESEngine) -> list[str]:
    lines = [
        f"instance_id={engine.instance_id}",
        f"checkpoint_dir={engine.checkpoint_dir}",
        f"best_score={engine.best_score}",
    ]
    summary_path = Path(engine.checkpoint_dir) / "syccl_two_agent" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                f"syccl_best_time_us={summary.get('best_time_us')}",
                f"syccl_best_source={summary.get('best_source')}",
                f"syccl_best_source_round={summary.get('best_source_round')}",
                f"syccl_summary_json={summary_path}",
            ]
        )
    return lines


def _write_task_files(task_dir: Path) -> tuple[Path, Path, Path]:
    task_dir.mkdir(parents=True, exist_ok=True)
    init_program = task_dir / "init_program.py"
    evaluator = task_dir / "evaluator.py"
    instruction = task_dir / "instruction.txt"
    init_program.write_text(
        load_init_program(),
        encoding="utf-8",
    )
    evaluator.write_text(
        """
def evaluate(program_path):
    return {"combined_score": -1000000000000.0}
""".lstrip(),
        encoding="utf-8",
    )
    instruction.write_text(
        load_prompt("proposal_base.txt"),
        encoding="utf-8",
    )
    return init_program, evaluator, instruction


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
