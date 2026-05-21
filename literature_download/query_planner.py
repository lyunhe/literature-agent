from __future__ import annotations

import json
import os
import re
from typing import Any

from analysis_pipeline.prompt_loader import render_prompt


def get_flash_model() -> str:
    return os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash")


def _extract_json(text: str) -> Any:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    starts = [idx for idx in [raw.find("{"), raw.find("[")] if idx != -1]
    if not starts:
        raise ValueError("LLM response does not contain JSON.")
    start = min(starts)
    stack: list[str] = []
    in_string = False
    escape = False
    for index, ch in enumerate(raw[start:], start=start):
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
                raise ValueError("LLM response JSON brackets are mismatched.")
            if not stack:
                return json.loads(raw[start : index + 1])
    raise ValueError("LLM response JSON is incomplete.")


def _clean_list(values: Any, limit: int = 30) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = re.sub(r"\s+", " ", text).lower()
        if key and key not in seen:
            result.append(text)
            seen.add(key)
        if len(result) >= limit:
            break
    return result


def _fallback_queries(topic: str, max_queries: int) -> list[dict[str, str]]:
    query = re.sub(r"\s+", " ", topic.strip())
    return [
        {
            "query_id": "Q1",
            "level": "broad",
            "query": query,
            "source_hint": "all",
            "rationale_cn": "AI 检索计划生成失败时使用原始主题。",
        }
    ][:max_queries]


def normalize_query_plan(plan: Any, topic: str, max_queries: int) -> dict[str, Any]:
    if not isinstance(plan, dict):
        plan = {}
    queries = plan.get("executable_queries") or plan.get("queries") or []
    normalized_queries: list[dict[str, str]] = []
    seen: set[str] = set()
    if isinstance(queries, list):
        for index, row in enumerate(queries, start=1):
            if isinstance(row, dict):
                query = str(row.get("query") or row.get("search_query") or "").strip()
                level = str(row.get("level") or "broad").strip()
                source_hint = str(row.get("source_hint") or "all").strip().lower()
                rationale = str(row.get("rationale_cn") or row.get("rationale") or "").strip()
            else:
                query = str(row or "").strip()
                level = "broad"
                source_hint = "all"
                rationale = ""
            key = re.sub(r"\s+", " ", query).lower()
            if not key or key in seen:
                continue
            normalized_queries.append(
                {
                    "query_id": f"Q{len(normalized_queries) + 1}",
                    "level": level,
                    "query": query,
                    "source_hint": source_hint or "all",
                    "rationale_cn": rationale,
                }
            )
            seen.add(key)
            if len(normalized_queries) >= max_queries:
                break
    if not normalized_queries:
        normalized_queries = _fallback_queries(topic, max_queries)

    return {
        "topic": str(plan.get("topic") or topic),
        "domain_terms": _clean_list(plan.get("domain_terms")),
        "problem_terms": _clean_list(plan.get("problem_terms")),
        "method_terms": _clean_list(plan.get("method_terms")),
        "venue_anchor_terms": _clean_list(plan.get("venue_anchor_terms")),
        "seed_terms": _clean_list(plan.get("seed_terms")),
        "negative_terms": _clean_list(plan.get("negative_terms")),
        "executable_queries": normalized_queries,
        "user_iteration_questions": _clean_list(plan.get("user_iteration_questions"), limit=8),
    }


def build_query_plan(
    topic: str,
    max_queries: int = 8,
    seed_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    topic = topic.strip()
    if not topic:
        return normalize_query_plan({}, topic, max_queries)
    seed_context = seed_context or {"seed_papers": [], "seed_terms": [], "seed_venues": []}
    prompt = render_prompt(
        "query_plan_multilevel",
        topic=topic,
        max_queries=max_queries,
        seed_context_json=seed_context,
    )
    try:
        from backend.llm_client import llm_request

        resp = llm_request(
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=get_flash_model(),
            temperature=0.2,
            max_tokens=1600,
        )
        payload = _extract_json(resp.choices[0].message.content)
        return normalize_query_plan(payload, topic, max_queries)
    except Exception as exc:
        print(f"[检索计划警告] 多层级检索计划生成失败，使用原始主题：{exc}")
        return normalize_query_plan({}, topic, max_queries)


def flatten_query_plan(plan: dict[str, Any]) -> list[str]:
    return [
        str(row.get("query") or "").strip()
        for row in plan.get("executable_queries", [])
        if str(row.get("query") or "").strip()
    ]
