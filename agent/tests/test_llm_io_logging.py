import asyncio
import contextlib
import io
import logging
import unittest
from pathlib import Path

from simpletes.config import EngineConfig
from simpletes.generator import GenerationTask, Generator
from simpletes.llm.types import LLMResult
from simpletes.node import EvolveBlockContext


class FakeLLM:
  def __init__(self) -> None:
    self.closed = False

  async def generate_batch(self, prompt: str, n: int, instance_id: str = "", track_io: bool = False):
    return [
        LLMResult(
            text=(
                "```python\n"
                "# EVOLVE-BLOCK-START\n"
                "def run_code():\n"
                "  return 2\n"
                "# EVOLVE-BLOCK-END\n"
                "```"
            ),
            prompt=prompt,
            raw_output="raw model output",
        )
        for _ in range(n)
    ]

  def close(self) -> None:
    self.closed = True


class LLMIOLoggingTest(unittest.TestCase):
  def test_generator_logs_prompt_and_model_output_at_debug_level(self):
    config = EngineConfig(
        init_program="init_program.py",
        evaluator_path="evaluator.py",
        instruction_path="instruction.txt",
        output_path="out",
    )
    evolve_context = EvolveBlockContext.from_program(
        "# prefix\n"
        "# EVOLVE-BLOCK-START\n"
        "def run_code():\n"
        "  return 1\n"
        "# EVOLVE-BLOCK-END\n"
        "# suffix\n"
    )
    generator = Generator(config, instruction="instruction", evolve_context=evolve_context)
    generator._llm = FakeLLM()
    task = GenerationTask(
        prompt="prompt sent to model",
        inspiration_ids=[],
        k=1,
        chain_idx=0,
        gen_id=7,
    )

    with self.assertLogs("simpletes.llm_io", level="DEBUG") as captured:
      results = asyncio.run(generator.generate(task, "instance-1", track_io=True))

    self.assertTrue(results[0].success)
    logs = "\n".join(captured.output)
    self.assertIn("gen_id=7", logs)
    self.assertIn("prompt sent to model", logs)
    self.assertIn("raw model output", logs)

  def test_generator_logging_uses_current_stdout_for_run_log_tee(self):
    config = EngineConfig(
        init_program="init_program.py",
        evaluator_path="evaluator.py",
        instruction_path="instruction.txt",
        output_path="out",
    )
    evolve_context = EvolveBlockContext.from_program(
        "# EVOLVE-BLOCK-START\n"
        "def run_code():\n"
        "  return 1\n"
        "# EVOLVE-BLOCK-END\n"
    )
    generator = Generator(config, instruction="instruction", evolve_context=evolve_context)
    generator._llm = FakeLLM()
    task = GenerationTask(
        prompt="tee prompt",
        inspiration_ids=[],
        k=1,
        chain_idx=0,
        gen_id=8,
    )
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
      asyncio.run(generator.generate(task, "instance-2", track_io=True))

    output = stdout.getvalue()
    self.assertIn("tee prompt", output)
    self.assertIn("raw model output", output)


if __name__ == "__main__":
  unittest.main()
