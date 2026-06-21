from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from simpletes.config import EngineConfig
from simpletes.llm import create_llm_client


@dataclass
class LLMResponse:
    text: str
    model_name: str
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    reasoning_tokens: int | None
    wall_clock_time_ms: int
    api_cost_usd: float | None = None
    raw_output: str | None = None


class FakeLLMBackend:
    def __init__(self, completions: list[str], model_name: str = "fake-llm") -> None:
        self._completions = list(completions)
        self.model_name = model_name
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, agent_type: str, round_id: int) -> LLMResponse:
        del agent_type, round_id
        if not self._completions:
            raise RuntimeError("FakeLLMBackend has no remaining completions")
        self.prompts.append(prompt)
        text = self._completions.pop(0)
        return LLMResponse(
            text=text,
            model_name=self.model_name,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            cached_input_tokens=None,
            reasoning_tokens=None,
            wall_clock_time_ms=0,
            api_cost_usd=None,
            raw_output=text,
        )


class SimpleTESLLMBackend:
    def __init__(self, config: EngineConfig) -> None:
        self._client = create_llm_client(config)
        self.model_name = config.model
        self.save_llm_io = config.save_llm_io

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def generate(self, prompt: str, *, agent_type: str, round_id: int) -> LLMResponse:
        import asyncio

        del agent_type, round_id
        start = time.perf_counter()
        result = asyncio.run(
            self._client.generate(
                prompt,
                instance_id="syccl-two-agent",
                track_io=self.save_llm_io,
            )
        )
        wall_ms = int((time.perf_counter() - start) * 1000)
        usage: dict[str, Any] = result.token_usage or {}
        return LLMResponse(
            text=result.text,
            model_name=self.model_name,
            input_tokens=usage.get("prompt_tokens") or usage.get("input_tokens"),
            output_tokens=usage.get("completion_tokens") or usage.get("output_tokens"),
            cached_input_tokens=usage.get("cached_input_tokens"),
            reasoning_tokens=usage.get("reasoning_tokens"),
            wall_clock_time_ms=wall_ms,
            api_cost_usd=None,
            raw_output=result.raw_output or result.text,
        )
