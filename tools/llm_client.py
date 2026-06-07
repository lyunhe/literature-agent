from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class LLMError(RuntimeError):
    pass


def call_openai_json(
    *,
    prompt: str,
    schema: dict[str, Any],
    schema_name: str,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    codex_config = load_codex_api_config()
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or codex_config.get("api_key")
    if not api_key:
        raise LLMError("OPENAI_API_KEY is not set and no Codex auth key was found.")
    model = model or os.environ.get("OPENAI_MODEL") or codex_config.get("model") or "gpt-4o-mini"
    base_url = os.environ.get("OPENAI_BASE_URL") or codex_config.get("base_url") or DEFAULT_OPENAI_BASE_URL
    responses_url = base_url.rstrip("/") + "/responses"
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": "You are a precise research reproducibility auditor. Return schema-valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }
    request = urllib.request.Request(
        responses_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"OpenAI API HTTP {exc.code} at {responses_url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"OpenAI API request failed: {exc}") from exc

    data = json.loads(body)
    text = extract_response_text(data)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        out = Path("runs/last_llm_response.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        raise LLMError(f"Could not parse model JSON output. Raw response saved to {out}") from exc


def call_openai_text(
    *,
    prompt: str,
    system: str = "You are a careful research reproduction assistant.",
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    codex_config = load_codex_api_config()
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or codex_config.get("api_key")
    if not api_key:
        raise LLMError("OPENAI_API_KEY is not set and no Codex auth key was found.")
    model = model or os.environ.get("OPENAI_MODEL") or codex_config.get("model") or "gpt-4o-mini"
    base_url = os.environ.get("OPENAI_BASE_URL") or codex_config.get("base_url") or DEFAULT_OPENAI_BASE_URL
    responses_url = base_url.rstrip("/") + "/responses"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        responses_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"OpenAI API HTTP {exc.code} at {responses_url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"OpenAI API request failed: {exc}") from exc
    return extract_response_text(json.loads(body))


def extract_response_text(response: dict[str, Any]) -> str:
    if "output_text" in response:
        return response["output_text"]
    for output in response.get("output", []):
        if output.get("type") != "message":
            continue
        for item in output.get("content", []):
            if item.get("type") in {"output_text", "text"} and "text" in item:
                return item["text"]
            if "refusal" in item:
                raise LLMError(f"Model refusal: {item['refusal']}")
    raise LLMError("Could not find text output in OpenAI response.")


def load_codex_api_config() -> dict[str, str]:
    """Read Codex app API settings without requiring shell env exports."""
    result: dict[str, str] = {}
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    config_path = codex_home / "config.toml"
    auth_path = codex_home / "auth.json"

    if config_path.exists():
        try:
            config = load_toml_like(config_path)
            model = config.get("model")
            if isinstance(model, str):
                result["model"] = model
            provider_name = config.get("model_provider")
            providers = config.get("model_providers", {})
            if isinstance(provider_name, str) and isinstance(providers, dict):
                provider = providers.get(provider_name, {})
                if isinstance(provider, dict):
                    base_url = provider.get("base_url")
                    if isinstance(base_url, str):
                        result["base_url"] = base_url
        except Exception:
            pass

    if auth_path.exists():
        try:
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
            api_key = auth.get("OPENAI_API_KEY")
            if isinstance(api_key, str) and api_key:
                result["api_key"] = api_key
        except Exception:
            pass

    return result


def load_toml_like(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # type: ignore

        return tomllib.loads(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        pass

    root: dict[str, Any] = {}
    current: dict[str, Any] = root
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = root
            for part in line[1:-1].split("."):
                current = current.setdefault(part, {})
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        current[key.strip()] = parse_toml_scalar(value.strip())
    return root


def parse_toml_scalar(value: str) -> Any:
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value in {"true", "false"}:
        return value == "true"
    return value


def render_prompt(template_path: str | Path, **values: str) -> str:
    text = Path(template_path).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text
