import os
import json
import yaml
import base64
from .paths import resolve_library_path

# Load config
_env_path = os.path.join(os.path.dirname(__file__), "..", "env.yaml")
with open(_env_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

api_key = os.getenv(
    "OPENAI_API_KEY",
    cfg.get("api_keys", {}).get("openai_key", "")
)
base_url = os.getenv(
    "OPENAI_API_BASE_URL",
    cfg.get("openai", {}).get("base_url", "")
)
model = cfg.get("openai", {}).get("model", "claude-sonnet-4-6")
temperature = cfg.get("llm", {}).get("temperature", 0.6)
max_tokens = cfg.get("llm", {}).get("max_tokens", 4096)
timeout = cfg.get("llm", {}).get("time_out", 600)
max_retries = cfg.get("llm", {}).get("max_retries", 5)

if not api_key:
    raise ValueError(
        "API key not found. Set OPENAI_API_KEY environment variable "
        "or api_keys.openai_key in env.yaml"
    )

is_claude = model.startswith("claude-")

if is_claude:
    import anthropic

    anthropic_client = anthropic.Anthropic(
        api_key=api_key,
        base_url=base_url.rstrip("/") if base_url else "https://api.anthropic.com",
        timeout=timeout,
        max_retries=max_retries,
    )
else:
    import openai

    openai_client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=timeout,
        max_retries=max_retries,
    )


class ClaudeMsg:
    """Duck-typed message object compatible with OpenAI's .message interface."""
    def __init__(self, content, role="assistant", tool_calls=None):
        self.content = content
        self.role = role
        self._tool_calls = tool_calls
        self.tool_calls = tool_calls

    def __getitem__(self, key):
        return getattr(self, key)


class ClaudeChoice:
    def __init__(self, stop_reason, message):
        self.finish_reason = stop_reason
        self.message = message


class ClaudeResp:
    def __init__(self, stop_reason, content, tool_calls_out):
        msg = ClaudeMsg(content=content, tool_calls=tool_calls_out if tool_calls_out else None)
        self.choices = [ClaudeChoice(stop_reason, msg)]


def llm_request(
    messages: list,
    model: str = model,
    max_tokens: int = max_tokens,
    temperature: float = temperature,
    tools: list = None,
    tool_choice: str = "auto"
):
    """
    Unified LLM request — routes to Claude or OpenAI based on model name.
    Claude returns a duck-typed object mimicking OpenAI's response structure:
    { choices: [{ finish_reason, message }] }
    """
    use_claude = model.startswith("claude-")

    if use_claude:
        # Build system prompt and conversation messages
        system_text = None
        claude_msgs = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                role = "user" if msg["role"] in ("user", "tool") else "assistant"
                content = msg["content"]
                if isinstance(content, list):
                    converted = []
                    for block in content:
                        if hasattr(block, 'type'):
                            if block.type == "text":
                                converted.append({"type": "text", "text": block.text})
                            elif block.type == "tool_use":
                                converted.append({
                                    "type": "tool_use",
                                    "id": block.id,
                                    "name": block.name,
                                    "input": block.input
                                })
                            elif block.type == "tool_result":
                                converted.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.tool_use_id,
                                    "content": block.text
                                })
                    content = converted
                claude_msgs.append({"role": role, "content": content})

        # Convert tools to Claude format
        claude_tools = None
        if tools:
            claude_tools = []
            for tool in tools:
                f = tool.get("function", {})
                claude_tools.append({
                    "name": f.get("name"),
                    "description": f.get("description", ""),
                    "input_schema": f.get("parameters", {"type": "object", "properties": {}})
                })

        resp = anthropic_client.messages.create(
            model=model,
            system=system_text,
            messages=claude_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=claude_tools,
        )

        # Parse Claude response
        content = resp.content
        stop_reason = resp.stop_reason

        tool_calls_out = []
        text_parts = []
        if isinstance(content, list):
            for block in content:
                if hasattr(block, 'type'):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        raw_input = block.input
                        tool_calls_out.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": raw_input if isinstance(raw_input, str) else json.dumps(raw_input)
                            }
                        })

        text = "\n".join(text_parts) if text_parts else ""

        # Wrap tool_calls in duck-typed objects
        wrapped_tool_calls = None
        if tool_calls_out:
            wrapped_tool_calls = []
            for tc in tool_calls_out:
                fname = tc["function"]["name"]
                fargs = tc["function"]["arguments"]

                class Fn:
                    name = fname
                    arguments = fargs

                class TcObj:
                    id = tc["id"]
                    type = "function"
                    function = Fn()

                wrapped_tool_calls.append(TcObj())

        return ClaudeResp(stop_reason, text, wrapped_tool_calls)
    else:
        params = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if tools is not None:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        return openai_client.chat.completions.create(**params)


def analyze_pdf(
    pdf_path: str,
    prompt: str,
    model: str = model,
    max_output_tokens: int = max_tokens,
) -> str:
    """Send a local PDF to a Responses-compatible model for analysis."""
    if model.startswith("claude-"):
        raise ValueError("PDF analysis currently uses OpenAI-compatible Responses API, not Claude routing.")
    resolved_pdf = resolve_library_path(pdf_path)
    if not resolved_pdf:
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pdf_path = str(resolved_pdf)

    with open(pdf_path, "rb") as f:
        encoded_pdf = base64.b64encode(f.read()).decode("ascii")

    resp = openai_client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": os.path.basename(pdf_path),
                        "file_data": f"data:application/pdf;base64,{encoded_pdf}",
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_output_tokens=max_output_tokens,
    )
    return resp.output_text


def analyze_pdfs(
    pdf_paths: list[str],
    prompt: str,
    model: str = model,
    max_output_tokens: int = max_tokens,
) -> str:
    """Send one or more local PDFs to a Responses-compatible model."""
    if model.startswith("claude-"):
        raise ValueError("PDF analysis currently uses OpenAI-compatible Responses API, not Claude routing.")

    content = []
    for pdf_path in pdf_paths:
        resolved_pdf = resolve_library_path(pdf_path)
        if not resolved_pdf:
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        pdf_path = str(resolved_pdf)
        with open(pdf_path, "rb") as f:
            encoded_pdf = base64.b64encode(f.read()).decode("ascii")
        content.append({
            "type": "input_file",
            "filename": os.path.basename(pdf_path),
            "file_data": f"data:application/pdf;base64,{encoded_pdf}",
        })
    content.append({
        "type": "input_text",
        "text": prompt,
    })

    resp = openai_client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=max_output_tokens,
    )
    return resp.output_text
