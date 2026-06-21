from __future__ import annotations

from simpletes.llm.litellm_client import LLMClient


def _client(model: str, **kwargs) -> LLMClient:
    return LLMClient(
        model=model,
        temperature=0.7,
        max_tokens=1024,
        pool_size=0,
        **kwargs,
    )


def test_deepseek_reasoner_uses_high_thinking_without_responses_api() -> None:
    client = _client("deepseek/deepseek-reasoner", reasoning_effort="medium")

    assert client._provider_model == "deepseek/deepseek-reasoner"
    assert client._common_call_kwargs()["thinking"] == {"type": "enabled"}
    assert client._common_call_kwargs()["reasoning_effort"] == "high"


def test_deepseek_chat_does_not_enable_reasoning() -> None:
    client = _client("deepseek/deepseek-chat", reasoning_effort="high")

    kwargs = client._common_call_kwargs()

    assert "thinking" not in kwargs
    assert "reasoning_effort" not in kwargs


def test_openai_reasoning_model_uses_responses_api_with_detailed_summary() -> None:
    client = _client("openai/gpt-5", reasoning_effort="medium")

    kwargs = client._common_call_kwargs()

    assert client._provider_model == "openai/responses/gpt-5"
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["reasoning"] == {"effort": "high", "summary": "detailed"}


def test_openai_reasoning_model_has_stable_prompt_cache_key() -> None:
    client = _client("openai/gpt-5")

    assert client._common_call_kwargs()["prompt_cache_key"] == "simpletes-openai-gpt-5"


def test_build_messages_keeps_prompt_as_single_stable_user_message() -> None:
    client = _client("deepseek/deepseek-reasoner")

    assert client._build_messages("Task\nDynamic suffix") == [
        {"role": "user", "content": "Task\nDynamic suffix"}
    ]
