import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = BASE_DIR / "文献"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
TEXT_DIRNAME = "txt_output"
SINGLE_DIRNAME = "single_paper_structures"
DIRECTION_DIRNAME = "directions"
DIRECTION_SCHEMA_DIRNAME = "direction_schemas"
DIRECTION_RECORD_DIRNAME = "direction_records"
COMPARISON_DIRNAME = "comparisons"
TIME_DIRNAME = "time_records"


def find_dotenv_candidates() -> list[Path]:
    candidates = [
        BASE_DIR / ".env",
        BASE_DIR.parent / ".env",
        BASE_DIR.parent / "单篇文献总结" / ".env",
    ]
    return [path for path in candidates if path.exists()]


def load_env_files() -> None:
    for env_path in find_dotenv_candidates():
        load_dotenv(env_path, override=False)
    load_dotenv(override=False)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    system_prompt: str
    reasoning_effort: str | None
    enable_thinking: bool


def resolve_llm_config() -> LLMConfig:
    load_env_files()
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    model = os.getenv("DEEPSEEK_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    system_prompt = os.getenv("DEEPSEEK_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
    reasoning_effort = os.getenv("DEEPSEEK_REASONING_EFFORT")
    enable_thinking = env_flag("DEEPSEEK_ENABLE_THINKING", default=True)
    if not api_key:
        raise RuntimeError("未找到 DEEPSEEK_API_KEY，请检查 .env 文件。")

    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
    )


def build_client(config: LLMConfig | None = None) -> OpenAI:
    resolved = config or resolve_llm_config()
    return OpenAI(api_key=resolved.api_key, base_url=resolved.base_url)


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
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)

    raise RuntimeError("模型未返回可解析的文本内容。")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def extract_text_from_pdf(pdf_path: Path, add_page_mark: bool = True) -> str:
    text_parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            if add_page_mark:
                text_parts.append(f"\n==================== 第 {page_num} 页 ====================\n")
            text_parts.append(page_text)
    return clean_text("".join(text_parts))


def trim_text_for_prompt(text: str, max_chars: int = 120000) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return (
        head
        + "\n\n[... 文本过长，已截断中间部分，仅保留前后文用于抽取 ...]\n\n"
        + tail
    )


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_json_text(response_text: str) -> str:
    text = response_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    first_obj = text.find("{")
    first_arr = text.find("[")
    starts = [idx for idx in [first_obj, first_arr] if idx != -1]
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
                continue
            if ch == "\\":
                repaired.append(ch)
                escape = True
                continue
            if ch == '"':
                repaired.append(ch)
                in_string = False
                continue
            if ch == "\n":
                repaired.append("\\n")
                continue
            if ch == "\r":
                repaired.append("\\r")
                continue
            if ch == "\t":
                repaired.append("\\t")
                continue
            if code < 32:
                repaired.append(f"\\u{code:04x}")
                continue
            repaired.append(ch)
            continue

        if ch == '"':
            in_string = True
            repaired.append(ch)
            continue
        if code < 32 and ch not in "\n\r\t":
            continue
        repaired.append(ch)

    return "".join(repaired)


def call_api_json(
    client: OpenAI,
    model: str,
    prompt: str,
    retries: int = 3,
    sleep_base: int = 2,
) -> Any:
    config = resolve_llm_config()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
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
                response_text = extract_chat_message_text(response).strip()
            else:
                response = client.responses.create(model=model, input=prompt)
                response_text = response.output_text.strip()

            json_text = extract_json_text(response_text)
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                repaired_json_text = sanitize_json_text(json_text)
                return json.loads(repaired_json_text)
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            wait_seconds = sleep_base * attempt
            print(f"API 调用失败，第 {attempt} 次重试后等待 {wait_seconds} 秒：{exc}")
            time.sleep(wait_seconds)
    raise RuntimeError(f"API 调用失败：{last_error}") from last_error


class TimeRecorder:
    def __init__(self) -> None:
        self.run_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.run_start = time.perf_counter()
        self.records: list[dict[str, Any]] = []

    @contextmanager
    def track(self, stage: str, item: str = ""):
        start = time.perf_counter()
        wall_start = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            yield
            status = "ok"
            error = ""
        except Exception as exc:
            status = "error"
            error = str(exc)
            raise
        finally:
            end = time.perf_counter()
            self.records.append(
                {
                    "stage": stage,
                    "item": item,
                    "status": status,
                    "started_at": wall_start,
                    "elapsed_seconds": round(end - start, 3),
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
        print(f"耗时记录：{time_dir / f'run_{stamp}.json'}")
        print(f"耗时表格：{csv_path}")


def compact_single_structure(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": paper.get("paper_id"),
        "title": paper.get("bibliography", {}).get("title"),
        "year": paper.get("bibliography", {}).get("year"),
        "paper_position": paper.get("paper_position", {}),
        "problem_context": paper.get("problem_context", {}),
        "task_object": paper.get("task_object", {}),
        "inputs": paper.get("inputs", {}),
        "methods": paper.get("methods", {}),
        "outputs": paper.get("outputs", {}),
        "evaluation": paper.get("evaluation", {}),
        "direction_hint": paper.get("direction_hint", {}),
    }


def load_single_structures_from_dir(single_dir: Path) -> list[dict[str, Any]]:
    if not single_dir.exists():
        raise FileNotFoundError(f"单篇结构化目录不存在：{single_dir}")
    json_files = sorted(
        [path for path in single_dir.glob("*.json") if path.is_file()],
        key=lambda path: path.name.lower(),
    )
    if not json_files:
        raise FileNotFoundError(f"单篇结构化目录中没有 JSON 文件：{single_dir}")

    singles: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for json_path in json_files:
        single = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(single, dict):
            raise ValueError(f"单篇结构化文件不是 JSON 对象：{json_path}")
        paper_id = str(single.get("paper_id", "")).strip()
        if not paper_id:
            raise ValueError(f"单篇结构化文件缺少 paper_id：{json_path}")
        if paper_id in seen_ids:
            raise ValueError(f"单篇结构化结果中存在重复 paper_id：{paper_id}")
        seen_ids.add(paper_id)
        singles.append(single)
    return singles


def build_single_paper_prompt(paper_name: str, paper_text: str, topic: str) -> str:
    return f"""
你是“单篇文献自适应结构化助手”。

任务：不要套统一全局模板。请根据这一篇论文自身内容，生成一份单篇定制结构化 JSON。

研究主题：{topic}
论文文件名：{paper_name}

要求：
1. 只能依据论文文本，不允许猜测。
2. 大结构保持一致，但每个结构块内部的关键词、变量、方法、输出要来自本文。
3. 如果该论文不是当前主题主线，也要如实说明它的真实任务，并解释与主题的关系。
4. 区分 unknown 与 not_applicable：
   - unknown：论文应该有此信息，但文本中找不到。
   - not_applicable：该字段对这篇论文不适用。
5. 数组字段如果已有有效值，不要额外加入 unknown。
6. 输出必须是 JSON，不要附加解释文字。

输出 JSON 结构：
{{
  "paper_id": "优先使用 DOI；没有 DOI 则给出短 ID",
  "bibliography": {{
    "title": "",
    "year": null,
    "venue": "unknown",
    "doi": "unknown"
  }},
  "paper_position": {{
    "doc_type": "journal_article/review/framework_or_system_paper/dataset_paper/unknown",
    "relevance_to_topic": "high/medium/low/exclude",
    "inclusion_decision": "include/borderline/exclude",
    "relevance_reason": "",
    "role_in_review": "main_sample/background/boundary_reference/excluded"
  }},
  "problem_context": {{
    "background": "",
    "research_gap": "",
    "problem_to_solve": "",
    "why_it_matters": ""
  }},
  "task_object": {{
    "application_object": [],
    "research_task": "",
    "prediction_or_modeling_target": [],
    "spatial_temporal_scope": "",
    "forecast_horizon_or_time_scale": []
  }},
  "inputs": {{
    "data_sources": [],
    "input_variables": [],
    "input_modalities": [],
    "input_construction": [],
    "preprocessing": []
  }},
  "methods": {{
    "method_family": [],
    "core_models": [],
    "key_mechanisms": [],
    "physical_or_domain_knowledge": [],
    "training_or_optimization": []
  }},
  "outputs": {{
    "model_outputs": [],
    "computed_quantities": [],
    "output_type": "",
    "usable_for": []
  }},
  "evaluation": {{
    "datasets": [],
    "metrics": [],
    "baselines": [],
    "experiments": [],
    "key_results": [],
    "reproducibility": {{
      "code_available": "true/false/unknown",
      "data_available": "true/false/unknown",
      "notes": ""
    }}
  }},
  "conclusions": {{
    "main_findings": [],
    "advantages": [],
    "limitations": [],
    "future_work": []
  }},
  "direction_hint": {{
    "candidate_direction": "简短研究方向名，不要写未来工作建议",
    "direction_basis": "problem_background/method_principle/application_object/output_type/mixed",
    "reason": "",
    "keywords": []
  }}
}}

论文文本如下：
```text
{trim_text_for_prompt(paper_text, max_chars=120000)}
```
""".strip()


def build_direction_discovery_prompt(topic: str, single_structures: list[dict[str, Any]]) -> str:
    compact = [compact_single_structure(item) for item in single_structures]
    papers_json = json.dumps(compact, ensure_ascii=False, indent=2)
    return f"""
你是“多文献研究方向划分助手”。

任务：根据单篇文献自适应结构化结果，划分研究方向。不要简单按关键词聚类，要根据共同问题背景、共同任务、共同方法基础、共同应用对象或输出目标来解释方向。

研究主题：{topic}

要求：
1. 每篇文献必须分配到且只分配到一个 primary_direction。
2. 可以设置 boundary/background/excluded 方向，但必须说明原因。
3. 每个方向要说明共同点、纳入标准、排除标准、代表文献。
4. 方向之间允许有交叉关键词，但方向定义必须有清晰差别。
5. `included_paper_ids`、`borderline_paper_ids`、`representative_paper_ids` 只能服务该方向，不能把同一篇论文放入多个方向。
6. 如果一篇论文是方法可借鉴但任务不对齐，只能二选一：要么放入“边界方法参考”方向，要么放入“排除”方向，不能重复放。
7. `paper_assignments` 是最终归属依据；directions 中的文献列表必须与 `paper_assignments` 保持一致。
8. 不要输出空方向；每个 direction 必须至少有一篇文献在 included_paper_ids 或 borderline_paper_ids 中。
9. 输出必须是 JSON，不要附加解释文字。

输出 JSON 结构：
{{
  "topic": "{topic}",
  "direction_design_principle": "",
  "directions": [
    {{
      "direction_id": "D1",
      "direction_name": "",
      "direction_definition": "",
      "common_problem_background": [],
      "common_research_task": [],
      "shared_method_basis": [],
      "shared_inputs": [],
      "shared_outputs": [],
      "included_paper_ids": [],
      "borderline_paper_ids": [],
      "representative_paper_ids": [],
      "inclusion_rule": "",
      "exclusion_rule": "",
      "why_this_direction_is_separate": ""
    }}
  ],
  "paper_assignments": [
    {{
      "paper_id": "",
      "title": "",
      "primary_direction_id": "",
      "primary_direction_name": "",
      "assignment_reason": "",
      "role_in_direction": "main/background/boundary/excluded"
    }}
  ],
  "cross_direction_summary": {{
    "shared_commonalities": [],
    "major_differences": []
  }}
}}

单篇结构化结果：
```json
{papers_json}
```
""".strip()


def build_direction_schema_prompt(
    topic: str,
    direction: dict[str, Any],
    papers: list[dict[str, Any]],
) -> str:
    direction_json = json.dumps(direction, ensure_ascii=False, indent=2)
    papers_json = json.dumps([compact_single_structure(p) for p in papers], ensure_ascii=False, indent=2)
    return f"""
你是“方向级结构化模板设计助手”。

任务：只针对一个研究方向，基于该方向内文献生成专用结构化模板。模板要能容纳方向内论文的共同内容，也要能表达方向内差异。不要为了兼容其他方向而扩大字段。

研究主题：{topic}

方向信息：
```json
{direction_json}
```

该方向内文献：
```json
{papers_json}
```

要求：
1. 字段必须服务该方向，不要追求全局通用。
2. 明确共同字段和差异字段。
3. 对输入、方法、输出、评价分别给出方向内专用字段。
4. 给出 unknown、not_applicable、空数组的使用规则。
5. 输出必须是 JSON，不要附加解释文字。

输出 JSON 结构：
{{
  "direction_id": "",
  "direction_name": "",
  "schema_goal": "",
  "commonality_fields": [],
  "difference_fields": [],
  "field_spec": [
    {{
      "name": "",
      "path": "",
      "type": "string/number/boolean/array/object",
      "required": true,
      "description": "",
      "allowed_values_or_examples": []
    }}
  ],
  "direction_taxonomy": {{
    "problem_types": [],
    "input_types": [],
    "method_types": [],
    "output_types": [],
    "evaluation_types": []
  }},
  "json_template": {{}},
  "normalization_rules": []
}}
""".strip()


def build_direction_record_prompt(
    topic: str,
    direction: dict[str, Any],
    direction_schema: dict[str, Any],
    papers: list[dict[str, Any]],
) -> str:
    direction_json = json.dumps(direction, ensure_ascii=False, indent=2)
    schema_json = json.dumps(direction_schema, ensure_ascii=False, indent=2)
    papers_json = json.dumps(papers, ensure_ascii=False, indent=2)
    return f"""
你是“方向内文献规整助手”。

任务：根据某一方向的专用 schema，把该方向内的单篇文献结构化结果规整成可比较的方向级 records。

研究主题：{topic}

方向信息：
```json
{direction_json}
```

方向专用 schema：
```json
{schema_json}
```

单篇结构化结果：
```json
{papers_json}
```

要求：
1. 只处理该方向内文献。
2. 保留单篇文献的真实信息，不要编造。
3. 同方向内要便于比较：共同点用共同字段，差异点用差异字段。
4. 如果已有有效值，不要在数组中混入 unknown。
5. 输出必须是 JSON，不要附加解释文字。

输出 JSON 结构：
{{
  "direction_id": "",
  "direction_name": "",
  "records": [],
  "within_direction_comparison": {{
    "common_problem_background": [],
    "common_research_task": [],
    "common_method_or_principle": [],
    "main_differences": {{
      "inputs": [],
      "methods": [],
      "outputs": [],
      "evaluation": [],
      "conclusions": []
    }}
  }}
}}
""".strip()


def build_cross_direction_comparison_prompt(
    topic: str,
    direction_mapping: dict[str, Any],
    direction_records: list[dict[str, Any]],
) -> str:
    mapping_json = json.dumps(direction_mapping, ensure_ascii=False, indent=2)
    records_json = json.dumps(direction_records, ensure_ascii=False, indent=2)
    return f"""
你是“跨方向综述比较助手”。

任务：比较不同研究方向，只总结大体共同点和主要差异，不做过细字段级比较。

研究主题：{topic}

方向划分：
```json
{mapping_json}
```

各方向规整结果：
```json
{records_json}
```

要求：
1. 跨方向比较要概括，不强行字段对齐。
2. 共同点围绕问题背景、研究目标、AI/数据驱动方法、可靠性/可解释性/可用性。
3. 差异围绕方向视角：输入融合、时空依赖、物理约束、不确定性表达、系统框架、边界物理建模等。
4. 输出必须是 JSON，不要附加解释文字。

输出 JSON 结构：
{{
  "topic": "{topic}",
  "cross_direction_commonalities": [],
  "cross_direction_differences": [
    {{
      "direction_id": "",
      "direction_name": "",
      "main_focus": "",
      "what_is_common_with_others": [],
      "what_is_different": [],
      "role_in_review": ""
    }}
  ],
  "suggested_visualizations": [],
  "suggested_review_structure": []
}}
""".strip()


def build_corpus_synthesis_prompt(topic: str, single_structures: list[dict[str, Any]]) -> str:
    singles_json = json.dumps(single_structures, ensure_ascii=False, indent=2)
    return f"""
你是“多文献综合结构化助手”。

任务：基于全部单篇结构化结果，一次性完成：
1) 方向划分 direction_mapping
2) 方向专用 schema direction_schemas
3) 方向内规整 records direction_records
4) 跨方向比较 cross_direction_comparison

研究主题：{topic}

硬性约束：
1. 每篇论文必须且只能有一个 primary_direction。
2. 同一篇论文不能同时出现在多个方向的 included_paper_ids / borderline_paper_ids / representative_paper_ids。
3. 方向集合不能有空方向；每个方向至少要有 included 或 borderline 论文。
4. direction_mapping、direction_schemas、direction_records 三者中的 direction_id 必须一致可对齐。
5. 只能依据输入数据，不要编造信息。
6. 输出必须是 JSON，不要附加任何解释文字。

输出 JSON 结构：
{{
  "direction_mapping": {{
    "topic": "{topic}",
    "direction_design_principle": "",
    "directions": [
      {{
        "direction_id": "D1",
        "direction_name": "",
        "direction_definition": "",
        "common_problem_background": [],
        "common_research_task": [],
        "shared_method_basis": [],
        "shared_inputs": [],
        "shared_outputs": [],
        "included_paper_ids": [],
        "borderline_paper_ids": [],
        "representative_paper_ids": [],
        "inclusion_rule": "",
        "exclusion_rule": "",
        "why_this_direction_is_separate": ""
      }}
    ],
    "paper_assignments": [
      {{
        "paper_id": "",
        "title": "",
        "primary_direction_id": "",
        "primary_direction_name": "",
        "assignment_reason": "",
        "role_in_direction": "main/background/boundary/excluded"
      }}
    ],
    "cross_direction_summary": {{
      "shared_commonalities": [],
      "major_differences": []
    }}
  }},
  "direction_schemas": [
    {{
      "direction_id": "",
      "direction_name": "",
      "schema_goal": "",
      "commonality_fields": [],
      "difference_fields": [],
      "field_spec": [
        {{
          "name": "",
          "path": "",
          "type": "string/number/boolean/array/object",
          "required": true,
          "description": "",
          "allowed_values_or_examples": []
        }}
      ],
      "direction_taxonomy": {{
        "problem_types": [],
        "input_types": [],
        "method_types": [],
        "output_types": [],
        "evaluation_types": []
      }},
      "json_template": {{}},
      "normalization_rules": []
    }}
  ],
  "direction_records": [
    {{
      "direction_id": "",
      "direction_name": "",
      "records": [],
      "within_direction_comparison": {{
        "common_problem_background": [],
        "common_research_task": [],
        "common_method_or_principle": [],
        "main_differences": {{
          "inputs": [],
          "methods": [],
          "outputs": [],
          "evaluation": [],
          "conclusions": []
        }}
      }}
    }}
  ],
  "cross_direction_comparison": {{
    "topic": "{topic}",
    "cross_direction_commonalities": [],
    "cross_direction_differences": [
      {{
        "direction_id": "",
        "direction_name": "",
        "main_focus": "",
        "what_is_common_with_others": [],
        "what_is_different": [],
        "role_in_review": ""
      }}
    ],
    "suggested_visualizations": [],
    "suggested_review_structure": []
  }}
}}

单篇结构化结果：
```json
{singles_json}
```
""".strip()


def build_corpus_repair_prompt(
    topic: str,
    single_structures: list[dict[str, Any]],
    candidate_output: dict[str, Any],
    errors: list[str],
) -> str:
    singles_json = json.dumps(single_structures, ensure_ascii=False, indent=2)
    candidate_json = json.dumps(candidate_output, ensure_ascii=False, indent=2)
    err_text = "\n".join(f"- {err}" for err in errors[:30])
    return f"""
你是“多文献综合结构化修复助手”。

任务：修复一个不满足约束的 corpus_synthesis 结果。只修复错误，不要丢失已有有效信息。

研究主题：{topic}

发现的问题：
{err_text}

硬性约束：
1. 每篇论文必须且只能有一个 primary_direction。
2. 同一篇论文不能同时出现在多个方向的 included_paper_ids / borderline_paper_ids / representative_paper_ids。
3. 方向集合不能有空方向；每个方向至少要有 included 或 borderline 论文。
4. direction_mapping、direction_schemas、direction_records 三者中的 direction_id 必须一致可对齐。
5. 输出必须是 JSON，不要附加任何解释文字。

单篇结构化输入（不可违反）：
```json
{singles_json}
```

待修复输出：
```json
{candidate_json}
```
""".strip()


def validate_corpus_synthesis_output(
    corpus: dict[str, Any],
    singles: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if not isinstance(corpus, dict):
        return ["corpus_synthesis 输出不是 JSON 对象"]

    mapping = corpus.get("direction_mapping")
    schemas = corpus.get("direction_schemas")
    records = corpus.get("direction_records")
    comparison = corpus.get("cross_direction_comparison")

    if not isinstance(mapping, dict):
        errors.append("缺少 direction_mapping 对象")
    if not isinstance(schemas, list):
        errors.append("direction_schemas 必须是数组")
    if not isinstance(records, list):
        errors.append("direction_records 必须是数组")
    if not isinstance(comparison, dict):
        errors.append("缺少 cross_direction_comparison 对象")
    if errors:
        return errors

    single_ids = [str(item.get("paper_id")) for item in singles if item.get("paper_id")]
    expected_ids = set(single_ids)
    if len(expected_ids) != len(single_ids):
        errors.append("single_paper_structures 中存在重复或缺失 paper_id")

    directions = mapping.get("directions")
    assignments = mapping.get("paper_assignments")
    if not isinstance(directions, list):
        errors.append("direction_mapping.directions 必须是数组")
        return errors
    if not isinstance(assignments, list):
        errors.append("direction_mapping.paper_assignments 必须是数组")
        return errors

    direction_ids: set[str] = set()
    membership_map: dict[str, set[str]] = {}
    for idx, direction in enumerate(directions):
        if not isinstance(direction, dict):
            errors.append(f"directions[{idx}] 不是对象")
            continue
        direction_id = str(direction.get("direction_id", "")).strip()
        if not direction_id:
            errors.append(f"directions[{idx}] 缺少 direction_id")
            continue
        if direction_id in direction_ids:
            errors.append(f"direction_id 重复：{direction_id}")
        direction_ids.add(direction_id)

        buckets: set[str] = set()
        for key in ["included_paper_ids", "borderline_paper_ids", "representative_paper_ids"]:
            values = direction.get(key, [])
            if not isinstance(values, list):
                errors.append(f"{direction_id}.{key} 必须是数组")
                continue
            for value in values:
                pid = str(value).strip()
                if not pid:
                    continue
                buckets.add(pid)
                membership_map.setdefault(pid, set()).add(direction_id)
                if expected_ids and pid not in expected_ids:
                    errors.append(f"{direction_id}.{key} 包含未知 paper_id：{pid}")
        if not (direction.get("included_paper_ids") or direction.get("borderline_paper_ids")):
            errors.append(f"{direction_id} 是空方向（included/borderline 均为空）")
        if not buckets:
            errors.append(f"{direction_id} 未包含任何论文")

    assignment_map: dict[str, str] = {}
    for idx, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            errors.append(f"paper_assignments[{idx}] 不是对象")
            continue
        pid = str(assignment.get("paper_id", "")).strip()
        direction_id = str(assignment.get("primary_direction_id", "")).strip()
        if not pid:
            errors.append(f"paper_assignments[{idx}] 缺少 paper_id")
            continue
        if pid in assignment_map:
            errors.append(f"paper_assignments 出现重复 paper_id：{pid}")
        assignment_map[pid] = direction_id
        if expected_ids and pid not in expected_ids:
            errors.append(f"paper_assignments 包含未知 paper_id：{pid}")
        if not direction_id:
            errors.append(f"paper_assignments[{idx}] 缺少 primary_direction_id")
        elif direction_id not in direction_ids:
            errors.append(f"paper_assignments[{idx}] 指向未知方向：{direction_id}")

    for pid in expected_ids:
        if pid not in assignment_map:
            errors.append(f"缺少论文归属：{pid}")
    for pid in assignment_map:
        if expected_ids and pid not in expected_ids:
            errors.append(f"多余论文归属：{pid}")

    for pid, direction_group in membership_map.items():
        if len(direction_group) > 1:
            errors.append(f"论文出现在多个方向列表：{pid} -> {sorted(direction_group)}")
        if pid in assignment_map and direction_group and assignment_map[pid] not in direction_group:
            errors.append(f"论文归属与方向列表不一致：{pid} 归属 {assignment_map[pid]} 列表 {sorted(direction_group)}")

    schema_ids: set[str] = set()
    for idx, schema in enumerate(schemas):
        if not isinstance(schema, dict):
            errors.append(f"direction_schemas[{idx}] 不是对象")
            continue
        direction_id = str(schema.get("direction_id", "")).strip()
        if not direction_id:
            errors.append(f"direction_schemas[{idx}] 缺少 direction_id")
            continue
        if direction_id in schema_ids:
            errors.append(f"direction_schemas 出现重复方向：{direction_id}")
        schema_ids.add(direction_id)
        if direction_ids and direction_id not in direction_ids:
            errors.append(f"direction_schemas 包含未知方向：{direction_id}")

    record_ids: set[str] = set()
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"direction_records[{idx}] 不是对象")
            continue
        direction_id = str(record.get("direction_id", "")).strip()
        if not direction_id:
            errors.append(f"direction_records[{idx}] 缺少 direction_id")
            continue
        if direction_id in record_ids:
            errors.append(f"direction_records 出现重复方向：{direction_id}")
        record_ids.add(direction_id)
        if direction_ids and direction_id not in direction_ids:
            errors.append(f"direction_records 包含未知方向：{direction_id}")
        if not isinstance(record.get("records", []), list):
            errors.append(f"{direction_id}.records 必须是数组")

    for direction_id in direction_ids:
        if direction_id not in schema_ids:
            errors.append(f"缺少方向 schema：{direction_id}")
        if direction_id not in record_ids:
            errors.append(f"缺少方向 records：{direction_id}")

    differences = comparison.get("cross_direction_differences")
    if not isinstance(differences, list):
        errors.append("cross_direction_comparison.cross_direction_differences 必须是数组")

    return errors


def synthesize_corpus_structure(
    client: OpenAI,
    model: str,
    topic: str,
    singles: list[dict[str, Any]],
    output_dir: Path,
    overwrite: bool,
    timer: TimeRecorder,
    max_repair_rounds: int = 2,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    output_path = output_dir / "corpus_synthesis.json"
    if output_path.exists() and not overwrite:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        cached_errors = validate_corpus_synthesis_output(cached, singles)
        if not cached_errors:
            print(f"复用已有 corpus_synthesis：{output_path}")
            return cached
        print("已有 corpus_synthesis 校验失败，将重新生成。")
        for err in cached_errors[:10]:
            print(f"- {err}")

    with timer.track("corpus_synthesis", "all_papers"):
        print("正在执行 v2.1 corpus_synthesis（合并后处理）")
        corpus = call_api_json(
            client=client,
            model=model,
            prompt=build_corpus_synthesis_prompt(topic, singles),
        )
        errors = validate_corpus_synthesis_output(corpus, singles)

        round_id = 0
        while errors and round_id < max_repair_rounds:
            round_id += 1
            print(f"corpus_synthesis 校验失败，进入修复轮次 {round_id}/{max_repair_rounds}")
            for err in errors[:8]:
                print(f"- {err}")
            corpus = call_api_json(
                client=client,
                model=model,
                prompt=build_corpus_repair_prompt(topic, singles, corpus, errors),
            )
            errors = validate_corpus_synthesis_output(corpus, singles)

        if errors:
            raise RuntimeError("corpus_synthesis 输出校验失败：" + " | ".join(errors[:20]))

        save_json(output_path, corpus)
        print(f"已生成 corpus_synthesis：{output_path}")
    return corpus


def materialize_corpus_outputs(
    corpus: dict[str, Any],
    direction_dir: Path,
    schema_dir: Path,
    record_dir: Path,
    comparison_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ensure_dir(direction_dir)
    ensure_dir(schema_dir)
    ensure_dir(record_dir)
    ensure_dir(comparison_dir)

    direction_mapping = corpus.get("direction_mapping", {})
    direction_schemas = corpus.get("direction_schemas", [])
    direction_records = corpus.get("direction_records", [])
    cross_direction_comparison = corpus.get("cross_direction_comparison", {})

    save_json(direction_dir / "direction_mapping.json", direction_mapping)
    save_json(comparison_dir / "cross_direction_comparison.json", cross_direction_comparison)

    for schema in direction_schemas:
        direction_id = str(schema.get("direction_id", "direction"))
        direction_name = str(schema.get("direction_name", direction_id))
        stem = safe_output_stem(f"{direction_id}_{direction_name}", max_base_len=80)
        save_json(schema_dir / f"{stem}.json", schema)

    for record in direction_records:
        direction_id = str(record.get("direction_id", "direction"))
        direction_name = str(record.get("direction_name", direction_id))
        stem = safe_output_stem(f"{direction_id}_{direction_name}", max_base_len=80)
        save_json(record_dir / f"{stem}.json", record)

    return direction_mapping, direction_schemas, direction_records, cross_direction_comparison


def select_pdf_files(pdf_dir: Path, filenames: list[str] | None, max_papers: int | None) -> list[Path]:
    if filenames:
        pdf_files = [pdf_dir / name for name in filenames]
    else:
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
    pdf_files = [path for path in pdf_files if path.exists()]
    if max_papers is not None:
        pdf_files = pdf_files[:max_papers]
    return pdf_files


def convert_pdfs_to_txt(
    pdf_files: list[Path],
    txt_dir: Path,
    overwrite: bool,
    timer: TimeRecorder,
) -> list[Path]:
    ensure_dir(txt_dir)
    txt_files: list[Path] = []
    for pdf_path in pdf_files:
        txt_path = txt_dir / f"{safe_output_stem(pdf_path.stem)}.txt"
        txt_files.append(txt_path)
        if txt_path.exists() and not overwrite:
            print(f"跳过已有 TXT：{txt_path.name}")
            continue
        with timer.track("pdf_to_txt", pdf_path.name):
            print(f"正在提取 PDF 文本：{pdf_path.name}")
            text = extract_text_from_pdf(pdf_path, add_page_mark=True)
            txt_path.write_text(text + "\n", encoding="utf-8")
            print(f"已生成 TXT：{txt_path.name}")
    return txt_files


def load_txt_map(txt_files: list[Path]) -> dict[str, str]:
    txt_map: dict[str, str] = {}
    for txt_path in txt_files:
        txt_map[txt_path.stem] = clean_text(txt_path.read_text(encoding="utf-8", errors="ignore"))
    return txt_map


def discover_single_paper_structures(
    client: OpenAI,
    model: str,
    topic: str,
    txt_map: dict[str, str],
    single_dir: Path,
    overwrite: bool,
    timer: TimeRecorder,
) -> list[dict[str, Any]]:
    ensure_dir(single_dir)
    singles: list[dict[str, Any]] = []
    for paper_name, paper_text in txt_map.items():
        output_path = single_dir / f"{safe_output_stem(paper_name)}.json"
        if output_path.exists() and not overwrite:
            single = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            with timer.track("single_paper_structure", paper_name):
                print(f"正在生成单篇自适应结构：{paper_name}")
                single = call_api_json(
                    client=client,
                    model=model,
                    prompt=build_single_paper_prompt(paper_name, paper_text, topic),
                )
                save_json(output_path, single)
        singles.append(single)
    return singles


def discover_directions(
    client: OpenAI,
    model: str,
    topic: str,
    singles: list[dict[str, Any]],
    direction_dir: Path,
    overwrite: bool,
    timer: TimeRecorder,
) -> dict[str, Any]:
    ensure_dir(direction_dir)
    output_path = direction_dir / "direction_mapping.json"
    if output_path.exists() and not overwrite:
        return json.loads(output_path.read_text(encoding="utf-8"))
    with timer.track("direction_discovery", "all_papers"):
        print("正在根据单篇结构识别研究方向")
        mapping = call_api_json(
            client=client,
            model=model,
            prompt=build_direction_discovery_prompt(topic, singles),
        )
        save_json(output_path, mapping)
    return mapping


def get_assigned_paper_ids(direction: dict[str, Any], mapping: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ["included_paper_ids", "borderline_paper_ids", "representative_paper_ids"]:
        values = direction.get(key)
        if isinstance(values, list):
            ids.extend(str(v) for v in values if v)
    direction_id = direction.get("direction_id")
    for assignment in mapping.get("paper_assignments", []):
        if assignment.get("primary_direction_id") == direction_id and assignment.get("paper_id"):
            ids.append(str(assignment["paper_id"]))
    seen: set[str] = set()
    unique = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def papers_for_direction(
    direction: dict[str, Any],
    mapping: dict[str, Any],
    singles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    wanted = get_assigned_paper_ids(direction, mapping)
    by_id = {str(p.get("paper_id")): p for p in singles if p.get("paper_id")}
    papers = [by_id[pid] for pid in wanted if pid in by_id]
    if papers:
        return papers

    direction_id = direction.get("direction_id")
    assigned_titles = {
        a.get("title")
        for a in mapping.get("paper_assignments", [])
        if a.get("primary_direction_id") == direction_id
    }
    return [p for p in singles if p.get("bibliography", {}).get("title") in assigned_titles]


def generate_direction_schemas_and_records(
    client: OpenAI,
    model: str,
    topic: str,
    singles: list[dict[str, Any]],
    mapping: dict[str, Any],
    schema_dir: Path,
    record_dir: Path,
    overwrite: bool,
    timer: TimeRecorder,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ensure_dir(schema_dir)
    ensure_dir(record_dir)
    schemas: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for direction in mapping.get("directions", []):
        direction_id = str(direction.get("direction_id", "direction"))
        direction_name = str(direction.get("direction_name", direction_id))
        stem = safe_output_stem(f"{direction_id}_{direction_name}", max_base_len=80)
        direction_papers = papers_for_direction(direction, mapping, singles)
        if not direction_papers:
            raise RuntimeError(f"方向缺少可用于生成模板/规整的论文：{direction_id} {direction_name}")

        schema_path = schema_dir / f"{stem}.json"
        if schema_path.exists() and not overwrite:
            direction_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        else:
            with timer.track("direction_schema", direction_name):
                print(f"正在生成方向模板：{direction_id} {direction_name}")
                direction_schema = call_api_json(
                    client=client,
                    model=model,
                    prompt=build_direction_schema_prompt(topic, direction, direction_papers),
                )
                save_json(schema_path, direction_schema)
        schemas.append(direction_schema)

        record_path = record_dir / f"{stem}.json"
        if record_path.exists() and not overwrite:
            direction_record = json.loads(record_path.read_text(encoding="utf-8"))
        else:
            with timer.track("direction_record_normalization", direction_name):
                print(f"正在规整方向内文献：{direction_id} {direction_name}")
                direction_record = call_api_json(
                    client=client,
                    model=model,
                    prompt=build_direction_record_prompt(
                        topic,
                        direction,
                        direction_schema,
                        direction_papers,
                    ),
                )
                save_json(record_path, direction_record)
        records.append(direction_record)

    return schemas, records


def generate_cross_direction_comparison(
    client: OpenAI,
    model: str,
    topic: str,
    mapping: dict[str, Any],
    direction_records: list[dict[str, Any]],
    comparison_dir: Path,
    overwrite: bool,
    timer: TimeRecorder,
) -> dict[str, Any]:
    ensure_dir(comparison_dir)
    output_path = comparison_dir / "cross_direction_comparison.json"
    if output_path.exists() and not overwrite:
        return json.loads(output_path.read_text(encoding="utf-8"))
    with timer.track("cross_direction_comparison", "all_directions"):
        print("正在生成跨方向比较")
        comparison = call_api_json(
            client=client,
            model=model,
            prompt=build_cross_direction_comparison_prompt(topic, mapping, direction_records),
        )
        save_json(output_path, comparison)
    return comparison


def build_bundle(
    topic: str,
    singles: list[dict[str, Any]],
    direction_mapping: dict[str, Any],
    direction_schemas: list[dict[str, Any]],
    direction_records: list[dict[str, Any]],
    cross_direction_comparison: dict[str, Any],
    input_source: str,
    pipeline_version: str = "v2_direction_adaptive",
) -> dict[str, Any]:
    return {
        "topic": topic,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_version": pipeline_version,
        "input_source": input_source,
        "single_paper_structures": singles,
        "direction_mapping": direction_mapping,
        "direction_schemas": direction_schemas,
        "direction_records": direction_records,
        "cross_direction_comparison": cross_direction_comparison,
    }


def build_single_only_bundle(
    topic: str,
    singles: list[dict[str, Any]],
    input_source: str,
) -> dict[str, Any]:
    return {
        "topic": topic,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_version": "v2_single_paper_only",
        "input_source": input_source,
        "single_paper_structures": singles,
    }


def parse_args() -> argparse.Namespace:
    load_env_files()
    default_model = os.getenv("DEEPSEEK_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    parser = argparse.ArgumentParser(
        description="多文献结构化输出 v2/v2.1：单篇自适应结构 -> 后处理综合（默认 v2.1 合并调用）"
    )
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--single-structures-dir",
        type=Path,
        default=None,
        help="直接读取已有单篇结构化 JSON 目录，跳过 PDF->TXT->单篇抽取",
    )
    parser.add_argument(
        "--topic",
        default="系留风筝系统控制 / Airborne Wind Energy (AWE) kite guidance, path optimization, and flight control",
    )
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--file", action="append", default=None, help="指定 PDF 文件名，可重复传入")
    parser.add_argument("--max-papers", type=int, default=None, help="只处理前 N 篇 PDF，用于试跑")
    parser.add_argument("--single-only", action="store_true", help="只生成单篇自适应结构化结果，不进入方向识别")
    parser.add_argument("--legacy-staged", action="store_true", help="使用 v2 分步后处理流程（方向识别 -> 模板 -> 规整 -> 比较）")
    parser.add_argument("--no-fallback-staged", action="store_true", help="v2.1 合并后处理失败时不自动回退到分步流程")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有中间结果和输出结果")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timer = TimeRecorder()
    output_dir = ensure_dir(args.output_dir)
    txt_dir = output_dir / TEXT_DIRNAME
    single_dir = output_dir / SINGLE_DIRNAME
    direction_dir = output_dir / DIRECTION_DIRNAME
    direction_schema_dir = output_dir / DIRECTION_SCHEMA_DIRNAME
    direction_record_dir = output_dir / DIRECTION_RECORD_DIRNAME
    comparison_dir = output_dir / COMPARISON_DIRNAME
    time_dir = BASE_DIR / TIME_DIRNAME

    try:
        if args.single_structures_dir:
            input_source = f"single_structures_dir:{args.single_structures_dir}"
            print(f"正在读取已有单篇结构化结果：{args.single_structures_dir}")
            singles = load_single_structures_from_dir(args.single_structures_dir)
        else:
            input_source = f"pdf_pipeline:{args.pdf_dir}"
            with timer.track("build_client", "llm_client"):
                client = build_client()
            pdf_files = select_pdf_files(args.pdf_dir, args.file, args.max_papers)
            if not pdf_files:
                print(f"未在目录中找到 PDF：{args.pdf_dir}")
                return

            print(f"待处理 PDF 数量：{len(pdf_files)}")
            for pdf_path in pdf_files:
                print(f"- {pdf_path.name}")

            txt_files = convert_pdfs_to_txt(pdf_files, txt_dir, args.overwrite, timer)
            txt_map = load_txt_map(txt_files)

            singles = discover_single_paper_structures(
                client=client,
                model=args.model,
                topic=args.topic,
                txt_map=txt_map,
                single_dir=single_dir,
                overwrite=args.overwrite,
                timer=timer,
            )

        single_source_dir = args.single_structures_dir or single_dir

        if args.single_only:
            single_bundle = build_single_only_bundle(args.topic, singles, input_source)
            save_json(output_dir / "single_paper_structures_bundle.json", single_bundle)
            print("\n单篇结构化流程完成。")
            print(f"单篇结构输入来源：{input_source}")
            if input_source.startswith("pdf_pipeline:"):
                print(f"TXT 输出目录：{txt_dir}")
            print(f"单篇结构目录：{single_source_dir}")
            print(f"单篇结构聚合结果：{output_dir / 'single_paper_structures_bundle.json'}")
            return

        if args.single_structures_dir:
            with timer.track("build_client", "llm_client"):
                client = build_client()

        bundle_version = "v2_direction_adaptive"
        if args.legacy_staged:
            print("使用 v2 分步后处理流程。")
            direction_mapping = discover_directions(
                client=client,
                model=args.model,
                topic=args.topic,
                singles=singles,
                direction_dir=direction_dir,
                overwrite=args.overwrite,
                timer=timer,
            )
            direction_schemas, direction_records = generate_direction_schemas_and_records(
                client=client,
                model=args.model,
                topic=args.topic,
                singles=singles,
                mapping=direction_mapping,
                schema_dir=direction_schema_dir,
                record_dir=direction_record_dir,
                overwrite=args.overwrite,
                timer=timer,
            )
            cross_direction_comparison = generate_cross_direction_comparison(
                client=client,
                model=args.model,
                topic=args.topic,
                mapping=direction_mapping,
                direction_records=direction_records,
                comparison_dir=comparison_dir,
                overwrite=args.overwrite,
                timer=timer,
            )
        else:
            bundle_version = "v2.1_corpus_synthesis"
            try:
                corpus = synthesize_corpus_structure(
                    client=client,
                    model=args.model,
                    topic=args.topic,
                    singles=singles,
                    output_dir=output_dir,
                    overwrite=args.overwrite,
                    timer=timer,
                )
                direction_mapping, direction_schemas, direction_records, cross_direction_comparison = materialize_corpus_outputs(
                    corpus=corpus,
                    direction_dir=direction_dir,
                    schema_dir=direction_schema_dir,
                    record_dir=direction_record_dir,
                    comparison_dir=comparison_dir,
                )
            except Exception as exc:
                if args.no_fallback_staged:
                    raise
                print(f"v2.1 合并后处理失败，自动回退到 v2 分步流程：{exc}")
                bundle_version = "v2_direction_adaptive_fallback"
                direction_mapping = discover_directions(
                    client=client,
                    model=args.model,
                    topic=args.topic,
                    singles=singles,
                    direction_dir=direction_dir,
                    overwrite=args.overwrite,
                    timer=timer,
                )
                direction_schemas, direction_records = generate_direction_schemas_and_records(
                    client=client,
                    model=args.model,
                    topic=args.topic,
                    singles=singles,
                    mapping=direction_mapping,
                    schema_dir=direction_schema_dir,
                    record_dir=direction_record_dir,
                    overwrite=args.overwrite,
                    timer=timer,
                )
                cross_direction_comparison = generate_cross_direction_comparison(
                    client=client,
                    model=args.model,
                    topic=args.topic,
                    mapping=direction_mapping,
                    direction_records=direction_records,
                    comparison_dir=comparison_dir,
                    overwrite=args.overwrite,
                    timer=timer,
                )

        bundle = build_bundle(
            topic=args.topic,
            singles=singles,
            direction_mapping=direction_mapping,
            direction_schemas=direction_schemas,
            direction_records=direction_records,
            cross_direction_comparison=cross_direction_comparison,
            input_source=input_source,
            pipeline_version=bundle_version,
        )
        save_json(output_dir / "adaptive_structured_output_bundle.json", bundle)

        print("\n流程完成。")
        print(f"单篇结构输入来源：{input_source}")
        if input_source.startswith("pdf_pipeline:"):
            print(f"TXT 输出目录：{txt_dir}")
        print(f"单篇结构目录：{single_source_dir}")
        print(f"方向划分：{direction_dir / 'direction_mapping.json'}")
        print(f"方向模板目录：{direction_schema_dir}")
        print(f"方向规整目录：{direction_record_dir}")
        print(f"跨方向比较：{comparison_dir / 'cross_direction_comparison.json'}")
        print(f"聚合结果：{output_dir / 'adaptive_structured_output_bundle.json'}")
    finally:
        timer.save(time_dir)


if __name__ == "__main__":
    main()
