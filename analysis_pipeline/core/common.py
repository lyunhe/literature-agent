from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_FLASH_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_DIR = PROJECT_ROOT / "input_pdfs"
DEFAULT_SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system" / "default_system_prompt.txt"


def default_system_prompt() -> str:
    return DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def find_dotenv_candidates() -> list[Path]:
    return [
        path
        for path in [
            PROJECT_ROOT / ".env",
            PROJECT_ROOT.parent / ".env",
            PROJECT_ROOT.parent / "单篇文献总结" / ".env",
        ]
        if path.exists()
    ]


def load_env_files() -> None:
    for env_path in find_dotenv_candidates():
        load_dotenv(env_path, override=False)
    load_dotenv(override=False)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return default


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    flash_model: str
    system_prompt: str
    reasoning_effort: str | None
    enable_thinking: bool


def resolve_llm_config(require_key: bool = True) -> LLMConfig:
    load_env_files()
    api_key = first_env("LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
    base_url = first_env("LLM_BASE_URL", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL", default=DEFAULT_BASE_URL)
    model = first_env("LLM_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL", default=DEFAULT_MODEL)
    flash_model = first_env("LLM_FLASH_MODEL", "OPENAI_FLASH_MODEL", "DEEPSEEK_FLASH_MODEL", default=DEFAULT_FLASH_MODEL)
    system_prompt = first_env("LLM_SYSTEM_PROMPT", "DEEPSEEK_SYSTEM_PROMPT") or default_system_prompt()
    raw_reasoning_effort = first_env("LLM_REASONING_EFFORT", "DEEPSEEK_REASONING_EFFORT")
    reasoning_effort = raw_reasoning_effort.strip() if raw_reasoning_effort is not None else None
    enable_thinking = env_flag("LLM_ENABLE_THINKING", default=env_flag("DEEPSEEK_ENABLE_THINKING", default=False))
    if require_key and not api_key:
        raise RuntimeError("未找到 LLM_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY，请检查 .env 文件。")
    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        flash_model=flash_model,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
    )


def build_client(config: LLMConfig | None = None) -> OpenAI:
    resolved = config or resolve_llm_config()
    return OpenAI(api_key=resolved.api_key, base_url=resolved.base_url, timeout=90.0)


def is_deepseek_request(config: LLMConfig, model: str) -> bool:
    model_name = model.lower()
    base_url = config.base_url.lower()
    return model_name.startswith("deepseek-") or "api.deepseek.com" in base_url


def extract_chat_message_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError("模型响应中没有 choices。")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("模型响应中没有 message。")
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
    raise RuntimeError("模型未返回可解析的文本内容。")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_output_stem(name: str, max_base_len: int = 64) -> str:
    normalized = re.sub(r"\s+", "_", name.strip())
    normalized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("._")
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    base = normalized[:max_base_len].rstrip("._")
    if not base:
        base = "item"
    return f"{base}_{digest}"


def safe_plain_stem(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "pdf"


def extract_text_from_pdf(pdf_path: Path, add_page_mark: bool = True) -> str:
    text_parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            if add_page_mark:
                text_parts.append(f"\n==================== 第 {page_num} 页 ====================\n")
            text_parts.append(page.get_text("text"))
    return clean_text("".join(text_parts))


def extract_pdf_metadata(pdf_path: Path, candidate_id: str) -> dict[str, Any]:
    title = pdf_path.stem
    authors: list[str] = []
    try:
        with fitz.open(pdf_path) as doc:
            meta = doc.metadata or {}
            raw_title = str(meta.get("title") or "").strip()
            if raw_title:
                title = raw_title
            author_text = str(meta.get("author") or "").strip()
            if author_text:
                authors = [part.strip() for part in re.split(r"[;,]", author_text) if part.strip()]
    except Exception:
        pass
    return {
        "candidate_id": candidate_id,
        "title": title,
        "title_cn": "",
        "abstract": "",
        "authors": authors,
        "year": "",
        "venue": "",
        "doi": "",
        "source": "local_pdf",
        "concepts": [],
        "cited_by_count": 0,
        "_pdf_path": str(pdf_path.resolve()),
    }


def trim_text_for_prompt(text: str, max_chars: int = 120000) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n\n[... text truncated ...]\n\n" + tail


def extract_json_text(response_text: str) -> str:
    text = response_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    starts = [idx for idx in [text.find("{"), text.find("[")] if idx != -1]
    if not starts:
        raise ValueError("API 返回中未找到 JSON 内容。")
    start = min(starts)
    stack: list[str] = []
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            opener = stack.pop()
            if (opener, ch) not in {("{", "}"), ("[", "]")}:
                raise ValueError("API 返回的 JSON 括号不匹配。")
            if not stack:
                return text[start : i + 1].strip()
    raise ValueError("API 返回中存在未闭合 JSON。")


def sanitize_json_text(json_text: str) -> str:
    repaired: list[str] = []
    in_string = False
    escape = False
    for ch in json_text:
        code = ord(ch)
        if in_string:
            if escape:
                repaired.append(ch)
                escape = False
            elif ch == "\\":
                repaired.append(ch)
                escape = True
            elif ch == '"':
                repaired.append(ch)
                in_string = False
            elif ch == "\n":
                repaired.append("\\n")
            elif ch == "\r":
                repaired.append("\\r")
            elif ch == "\t":
                repaired.append("\\t")
            elif code < 32:
                repaired.append(f"\\u{code:04x}")
            else:
                repaired.append(ch)
            continue
        if ch == '"':
            in_string = True
            repaired.append(ch)
        elif code >= 32 or ch in "\n\r\t":
            repaired.append(ch)
    return "".join(repaired)


def _chat_request_text(client: OpenAI, config: LLMConfig, model: str, prompt: str) -> str:
    if is_deepseek_request(config, model):
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        model_name = model.lower()
        if config.reasoning_effort and "flash" not in model_name:
            request_kwargs["reasoning_effort"] = config.reasoning_effort
        if config.enable_thinking and "flash" not in model_name:
            request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        response = client.chat.completions.create(**request_kwargs)
        return extract_chat_message_text(response).strip()
    response = client.responses.create(model=model, input=prompt)
    return response.output_text.strip()


def call_api_text(
    client: OpenAI,
    model: str,
    prompt: str,
    retries: int = 3,
    sleep_base: int = 2,
) -> str:
    config = resolve_llm_config()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return _chat_request_text(client, config, model, prompt)
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            wait_seconds = sleep_base * attempt
            print(f"API 调用失败，第 {attempt} 次重试后等待 {wait_seconds} 秒：{exc}")
            time.sleep(wait_seconds)
    raise RuntimeError(f"API 调用失败：{last_error}") from last_error


def call_api_json(
    client: OpenAI,
    model: str,
    prompt: str,
    retries: int = 3,
    sleep_base: int = 2,
) -> Any:
    response_text = ""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response_text = call_api_text(client, model, prompt, retries=1)
            json_text = extract_json_text(response_text)
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                return json.loads(sanitize_json_text(json_text))
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            wait_seconds = sleep_base * attempt
            print(f"API JSON 调用失败，第 {attempt} 次重试后等待 {wait_seconds} 秒：{exc}")
            time.sleep(wait_seconds)
    raise RuntimeError(f"API JSON 调用失败：{last_error}") from last_error


class TimeRecorder:
    def __init__(self) -> None:
        self.run_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.run_start = time.perf_counter()
        self.records: list[dict[str, Any]] = []

    @contextmanager
    def track(self, stage: str, item: str = ""):
        start = time.perf_counter()
        wall_start = time.strftime("%Y-%m-%d %H:%M:%S")
        status = "ok"
        error = ""
        try:
            yield
        except Exception as exc:
            status = "error"
            error = str(exc)
            raise
        finally:
            self.records.append(
                {
                    "stage": stage,
                    "item": item,
                    "status": status,
                    "started_at": wall_start,
                    "elapsed_seconds": round(time.perf_counter() - start, 3),
                    "error": error,
                }
            )

    def payload(self) -> dict[str, Any]:
        total = round(time.perf_counter() - self.run_start, 3)
        by_stage: dict[str, float] = {}
        for record in self.records:
            by_stage[record["stage"]] = round(
                by_stage.get(record["stage"], 0.0) + record["elapsed_seconds"], 3
            )
        return {
            "run_started_at": self.run_started_at,
            "run_finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_elapsed_seconds": total,
            "stage_elapsed_seconds": by_stage,
            "records": self.records,
        }

    def save(self, time_dir: Path) -> None:
        ensure_dir(time_dir)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        payload = self.payload()
        save_json(time_dir / f"run_{stamp}.json", payload)
        csv_path = time_dir / f"run_{stamp}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            fieldnames = ["stage", "item", "status", "started_at", "elapsed_seconds", "error"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        save_json(time_dir / "latest_time_record.json", payload)

