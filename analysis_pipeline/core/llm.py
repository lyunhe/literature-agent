from __future__ import annotations

from typing import Any

from analysis_pipeline.core.common import (
    build_client,
    extract_chat_message_text,
    is_deepseek_request,
    resolve_llm_config,
)


class Message:
    def __init__(self, content: str) -> None:
        self.content = content


class Choice:
    def __init__(self, content: str) -> None:
        self.message = Message(content)


class Response:
    def __init__(self, content: str) -> None:
        self.choices = [Choice(content)]


def llm_request(
    messages: list[dict[str, Any]],
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
) -> Response:
    config = resolve_llm_config()
    client = build_client(config)
    selected_model = model or config.model
    request_kwargs: dict[str, Any] = {
        "model": selected_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools is not None:
        request_kwargs["tools"] = tools
        request_kwargs["tool_choice"] = tool_choice
    if is_deepseek_request(config, selected_model):
        model_name = selected_model.lower()
        if config.reasoning_effort and "flash" not in model_name:
            request_kwargs["reasoning_effort"] = config.reasoning_effort
        if config.enable_thinking and "flash" not in model_name:
            request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    response = client.chat.completions.create(**request_kwargs)
    return Response(extract_chat_message_text(response))
