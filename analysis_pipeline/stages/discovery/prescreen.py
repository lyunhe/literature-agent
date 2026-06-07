"""Download-before-PDF candidate screening for literature search results.

This module only uses metadata available before PDF download: title, abstract,
venue, source, year, DOI/URL, concepts, and citation counts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from analysis_pipeline.core.llm import llm_request
from analysis_pipeline.core.prompts import load_prompt, render_prompt
from analysis_pipeline.stages.discovery.paper_table import translate_titles


def get_flash_model() -> str:
    return (
        os.getenv("LLM_FLASH_MODEL")
        or os.getenv("OPENAI_FLASH_MODEL")
        or os.getenv("DEEPSEEK_FLASH_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or "deepseek-v4-flash"
    )


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_venue(text: str) -> str:
    """Normalize venue names for exact CSV matching after punctuation cleanup."""
    value = (text or "").lower()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def clamp_score(value: Any, default: float = 5.0) -> float:
    return max(0.0, min(10.0, safe_float(value, default)))


def candidate_id_for_paper(paper: dict[str, Any], index: int) -> str:
    for key in ("doi", "arxiv_id", "ieee_id", "openalex_id", "url", "title"):
        value = str(paper.get(key) or "").strip()
        if value:
            digest = hashlib.sha1(value.lower().encode("utf-8")).hexdigest()[:8]
            return f"P{index + 1:03d}_{digest}"
    return f"P{index + 1:03d}"


def with_candidate_ids(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, paper in enumerate(papers):
        item = dict(paper)
        candidate_id = str(item.get("candidate_id") or candidate_id_for_paper(item, index))
        if candidate_id in seen:
            candidate_id = f"{candidate_id}_{index + 1}"
        item["candidate_id"] = candidate_id
        seen.add(candidate_id)
        result.append(item)
    return result


def load_journal_levels(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Load local journal level CSV as an exact normalized-name lookup."""
    if not path:
        return {}
    csv_path = Path(path)
    if not csv_path.exists():
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            venue = str(row.get("venue") or "").strip()
            aliases = str(row.get("aliases") or "").strip()
            score = clamp_score(row.get("score"), default=0.0)
            level = str(row.get("level") or "").strip()
            payload = {
                "venue": venue,
                "level": level,
                "score": score,
                "source_path": str(csv_path.resolve()),
            }
            names = [venue]
            if aliases:
                names.extend(part.strip() for part in re.split(r"[|;/]", aliases) if part.strip())
            for name in names:
                key = normalize_venue(name)
                if key:
                    lookup[key] = payload
    return lookup


def journal_score_for_paper(
    paper: dict[str, Any],
    journal_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    venue = str(paper.get("venue") or "").strip()
    key = normalize_venue(venue)
    if not key or key not in journal_lookup:
        return {
            "journal_level_score": None,
            "journal_level": "",
            "journal_matched_venue": "",
            "journal_score_available": False,
        }
    row = journal_lookup[key]
    return {
        "journal_level_score": row["score"],
        "journal_level": row.get("level", ""),
        "journal_matched_venue": row.get("venue", venue),
        "journal_score_available": True,
    }


def final_score(relevance_score: float, journal_level_score: Any) -> float:
    if journal_level_score is None:
        return round(relevance_score, 3)
    return round(0.7 * relevance_score + 0.3 * clamp_score(journal_level_score), 3)


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

    raise ValueError("LLM response JSON is incomplete or truncated.")


def _paper_payload(
    papers: list[dict[str, Any]],
    abstract_limit: int = 500,
    text_excerpt_limit: int = 3500,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for paper in papers:
        payload.append(
            {
                "candidate_id": paper.get("candidate_id", ""),
                "title": paper.get("title", ""),
                "title_cn": paper.get("title_cn", ""),
                "abstract": str(paper.get("abstract", ""))[:abstract_limit],
                "authors": paper.get("authors", ""),
                "year": paper.get("year"),
                "venue": paper.get("venue", ""),
                "source": paper.get("source", ""),
                "doi": paper.get("doi", ""),
                "url": paper.get("url", ""),
                "oa_url": paper.get("oa_url", ""),
                "concepts": paper.get("concepts", []),
                "pdf_text_excerpt": str(paper.get("pdf_text_excerpt", ""))[:text_excerpt_limit],
                "cited_by_count": paper.get("cited_by_count", 0),
            }
        )
    return payload


def _paper_text_for_direction(paper: dict[str, Any]) -> str:
    parts = [
        paper.get("title", ""),
        paper.get("title_cn", ""),
        paper.get("abstract", ""),
        paper.get("pdf_text_excerpt", ""),
        " ".join(str(item) for item in paper.get("concepts", []) or []),
    ]
    return " ".join(str(part or "").lower() for part in parts)


def _topic_relevance_guard(paper: dict[str, Any]) -> tuple[bool, str, float]:
    text = _paper_text_for_direction(paper)
    title_and_concepts = " ".join(
        str(part or "").lower()
        for part in [
            paper.get("title", ""),
            paper.get("title_cn", ""),
            " ".join(str(item) for item in paper.get("concepts", []) or []),
        ]
    )
    abstract = str(paper.get("abstract") or "").lower()
    irrelevant_terms = [
        "dark energy",
        "dark-energy",
        "synoptic survey telescope",
        "large synoptic survey",
        "lsst",
        "astronom",
        "cosmolog",
        "telescope",
    ]
    storage_terms = ["energy storage", "battery", "bess", "storage system", "储能", "电池"]
    market_terms = [
        "electricity market",
        "power market",
        "energy market",
        "day-ahead",
        "real-time market",
        "bidding",
        "bid",
        "pricing",
        "auction",
        "aggregator",
        "arbitrage",
        "market clearing",
        "电力市场",
        "现货市场",
        "竞标",
        "报价",
        "定价",
        "聚合商",
        "套利",
    ]
    operation_terms = [
        "grid",
        "dispatch",
        "scheduling",
        "distributed control",
        "privacy-preserving",
        "sharing",
        "renewable",
        "frequency regulation",
        "电网",
        "调度",
        "控制",
        "隐私",
        "共享",
        "新能源",
        "频率调节",
    ]
    if any(term in text for term in irrelevant_terms):
        return False, "命中天文学/暗能量/望远镜等无关主题词", 0.0
    has_storage = any(term in text for term in storage_terms)
    has_market = any(term in text for term in market_terms)
    has_operation = any(term in text for term in operation_terms)
    storage_focus = any(term in title_and_concepts for term in storage_terms) or sum(abstract.count(term) for term in storage_terms) >= 2
    market_focus = any(term in title_and_concepts for term in market_terms) or sum(abstract.count(term) for term in market_terms) >= 2
    if has_storage and has_market:
        if storage_focus and market_focus:
            return True, "储能与电力市场均为题名、概念或摘要中的明确焦点", 1.0
        return False, "仅零散提及储能或电力市场，未体现为论文核心焦点", 0.0
    if has_storage and has_operation and storage_focus:
        return True, "储能为明确焦点，并命中运行控制/电网集成主题词", 0.8
    return False, "未命中储能/电池主体与电力市场或电网运行核心约束", 0.0


def _heuristic_direction_id(paper: dict[str, Any], *, force_assign_all: bool = False) -> str | None:
    text = _paper_text_for_direction(paper)
    ok, _, _ = _topic_relevance_guard(paper)
    if not ok and not force_assign_all:
        return None
    category_terms = {
        "D1": [
            "reinforcement learning",
            "deep reinforcement",
            "policy optimization",
            "q-learning",
            "learning-based",
            "bid learning",
            "predict-then-bid",
            "neural",
            "large language model",
            "ai-agent",
            "machine learning",
            "强化学习",
            "深度学习",
            "学习型",
            "预测后报价",
            "智能体",
        ],
        "D2": [
            "stochastic",
            "robust",
            "chance-constrained",
            "optimization",
            "optimal bidding",
            "optimal offering",
            "decision-dependent",
            "risk-constrained",
            "cvar",
            "model predictive",
            "linear decision rule",
            "scheduling",
            "随机",
            "鲁棒",
            "机会约束",
            "优化",
            "风险约束",
            "调度",
        ],
        "D3": [
            "market mechanism",
            "truthful bidding",
            "market clearing",
            "game-theoretic",
            "game theoretic",
            "auction",
            "price-maker",
            "withholding",
            "social welfare",
            "bid bounds",
            "strategic",
            "market model",
            "机制",
            "博弈",
            "出清",
            "策略性",
            "市场势力",
            "福利",
        ],
        "D4": [
            "privacy-preserving",
            "secure multi-party",
            "blockchain",
            "sharing",
            "distributed control",
            "ancillary services",
            "frequency",
            "reserve",
            "fcas",
            "state-of-charge",
            "soc market",
            "voltage",
            "共享",
            "隐私",
            "区块链",
            "分布式控制",
            "辅助服务",
            "调频",
            "备用",
            "荷电状态",
        ],
    }
    scores = {
        direction_id: sum(1 for term in terms if term in text)
        for direction_id, terms in category_terms.items()
    }
    best_id, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_id
    return "D2" if "bidding" in text or "bid" in text or "报价" in text else "D4"


def _fallback_directions(
    papers: list[dict[str, Any]],
    *,
    force_assign_all: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[str]] = {"D1": [], "D2": [], "D3": [], "D4": []}
    relevant_papers: list[dict[str, Any]] = []
    for paper in papers:
        direction_id = _heuristic_direction_id(paper, force_assign_all=force_assign_all)
        if not direction_id:
            continue
        grouped[direction_id].append(str(paper["candidate_id"]))
        relevant_papers.append(paper)
    non_empty_count = sum(1 for ids in grouped.values() if ids)
    if non_empty_count < 2 and len(relevant_papers) > 1:
        ordered = [str(paper["candidate_id"]) for paper in relevant_papers]
        midpoint = max(1, len(ordered) // 2)
        grouped = {"D1": ordered[:midpoint], "D2": ordered[midpoint:], "D3": [], "D4": []}
    direction_templates = [
        (
            "D1",
            "数据驱动与强化学习报价",
            "Data-driven and reinforcement-learning bidding",
            "聚焦用强化学习、深度学习、预测后决策或智能体方法学习储能报价策略。",
            ["强化学习", "深度学习", "预测后报价", "智能体", "数据驱动"],
        ),
        (
            "D2",
            "优化建模与风险约束报价",
            "Optimization and risk-constrained bidding",
            "聚焦随机优化、鲁棒优化、机会约束、风险约束和调度协同下的储能报价模型。",
            ["随机优化", "鲁棒优化", "机会约束", "风险约束", "调度协同"],
        ),
        (
            "D3",
            "市场机制与策略性竞价",
            "Market mechanism and strategic bidding",
            "聚焦市场机制、真实报价、出清规则、博弈行为、市场势力和社会福利影响。",
            ["市场机制", "真实报价", "市场出清", "博弈", "市场势力"],
        ),
        (
            "D4",
            "辅助服务与分布式协同",
            "Ancillary services and distributed coordination",
            "聚焦辅助服务、备用/调频、SOC 市场模型、共享储能、隐私保护和分布式控制。",
            ["辅助服务", "调频", "备用", "SOC", "共享储能", "隐私保护"],
        ),
    ]
    directions = [
        {
            "direction_id": direction_id,
            "direction_name_cn": name_cn,
            "direction_name_en": name_en,
            "direction_summary_cn": summary_cn,
            "display_keywords": keywords,
            "paper_ids": grouped[direction_id],
        }
        for direction_id, name_cn, name_en, summary_cn, keywords in direction_templates
        if grouped[direction_id]
    ]
    paper_direction = {
        candidate_id: direction_id
        for direction_id, ids in grouped.items()
        for candidate_id in ids
    }
    assignments = [
        {
            "candidate_id": str(paper["candidate_id"]),
            "direction_id": paper_direction.get(str(paper["candidate_id"]), "D2"),
            "direction_role": "main" if _topic_relevance_guard(paper)[0] else "boundary",
            "assignment_confidence": 0.72 if _topic_relevance_guard(paper)[0] else 0.45,
            "method_or_object_summary_cn": "",
            "method_summary_cn": "",
            "assignment_reason_cn": f"按题名、摘要、关键词和正文摘录识别方法族、市场机制或运行场景后归类；{_topic_relevance_guard(paper)[1]}。",
        }
        for paper in relevant_papers
    ]
    scores = [
        {
            "candidate_id": str(paper["candidate_id"]),
            "relevance_score": round(6.0 + _topic_relevance_guard(paper)[2] * 2.0, 3),
            "decision": "include" if _topic_relevance_guard(paper)[0] else "borderline",
            "reason_cn": f"AI 分类失败后的规则兜底；{_topic_relevance_guard(paper)[1]}。",
        }
        for paper in relevant_papers
    ]
    fast_check = {
        "all_papers_assigned_once": True,
        "empty_directions": [],
        "duplicated_paper_ids": [],
        "unassigned_paper_ids": [],
        "representative_notes_cn": "heuristic_bidding_method_taxonomy_v2; force_assign_all=" + str(force_assign_all).lower(),
    }
    return directions, assignments, scores, fast_check


def _ensure_minimum_direction_groups(
    result: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]],
    papers: list[dict[str, Any]],
    *,
    force_assign_all: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    directions, assignments, scores, fast_check = result
    non_empty = [direction for direction in directions if direction.get("paper_ids")]
    if len(non_empty) >= 2:
        return result
    fallback = _fallback_directions(papers, force_assign_all=force_assign_all)
    fallback[3]["representative_notes_cn"] = "validated_result_collapsed_to_one_direction; heuristic fallback applied"
    return fallback


def infer_candidate_directions(
    topic: str,
    papers: list[dict[str, Any]],
    input_mode: str = "search_metadata",
    force_assign_all: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Use the flash model to group candidate papers into directions."""
    if not papers:
        return [], [], [], {}

    def build_prompt(
        batch_papers: list[dict[str, Any]],
        abstract_limit: int,
        retry: bool = False,
        existing_directions: list[dict[str, Any]] | None = None,
    ) -> str:
        retry_hint = "上一次输出不是完整合法 JSON。请只返回一个完整 JSON 对象，不要 Markdown，不要解释。\n\n" if retry else ""
        prompt = render_prompt(
            "download_prescreen",
            topic=topic,
            candidate_papers_json={
                "input_mode": input_mode,
                "existing_directions": existing_directions or [],
                "papers": _paper_payload(batch_papers, abstract_limit=abstract_limit),
            },
        )
        return retry_hint + prompt

    def call_model_for_batch(
        batch_papers: list[dict[str, Any]],
        existing_directions: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        last_error: Exception | None = None
        for attempt, abstract_limit in enumerate([500, 250, 120], start=1):
            try:
                resp = llm_request(
                    messages=[
                        {"role": "system", "content": load_prompt("system_strict_legal_json_cn")},
                        {
                            "role": "user",
                            "content": build_prompt(
                                batch_papers,
                                abstract_limit,
                                retry=attempt > 1,
                                existing_directions=existing_directions,
                            ),
                        },
                    ],
                    model=get_flash_model(),
                    max_tokens=8192,
                    temperature=0.0,
                )
                payload = _extract_json(resp.choices[0].message.content)
                directions = payload.get("directions", []) if isinstance(payload, dict) else []
                assignments = payload.get("assignments", []) if isinstance(payload, dict) else []
                relevance_scores = payload.get("relevance_scores", []) if isinstance(payload, dict) else []
                fast_check = payload.get("fast_check", {}) if isinstance(payload, dict) else {}
                validated = validate_directions(batch_papers, directions, assignments, relevance_scores, fast_check)
                if validated:
                    return validated
                last_error = ValueError("LLM JSON passed parsing but failed direction validation.")
            except Exception as exc:
                last_error = exc
        raise RuntimeError(str(last_error or "LLM direction classification failed"))

    if len(papers) <= 80:
        try:
            return _ensure_minimum_direction_groups(
                call_model_for_batch(papers),
                papers,
                force_assign_all=force_assign_all,
            )
        except Exception as exc:
            print(f"[预筛警告] AI 方向归纳失败，使用规则兜底：{exc}")
            return _fallback_directions(papers, force_assign_all=force_assign_all)

    if len(papers) > 80:
        print("[预筛] 候选池超过 80 篇，下载前使用规则兜底排序；成功下载后会对展示论文重新进行大模型分类。")
        return _fallback_directions(papers, force_assign_all=force_assign_all)

    print(f"[预筛] 候选文献 {len(papers)} 篇，采用大模型分块方向归纳。")
    chunks = [papers[index : index + 20] for index in range(0, len(papers), 20)]
    merged_directions: dict[str, dict[str, Any]] = {}
    merged_assignments: list[dict[str, Any]] = []
    merged_scores: list[dict[str, Any]] = []
    representative_notes: list[str] = []

    try:
        for index, chunk in enumerate(chunks, start=1):
            existing = [
                {
                    "direction_id": item.get("direction_id"),
                    "direction_name_cn": item.get("direction_name_cn"),
                    "direction_summary_cn": item.get("direction_summary_cn"),
                    "display_keywords": item.get("display_keywords", []),
                    "inclusion_rule_cn": item.get("inclusion_rule_cn", ""),
                    "exclusion_rule_cn": item.get("exclusion_rule_cn", ""),
                }
                for item in merged_directions.values()
            ]
            directions, assignments, scores, fast_check = call_model_for_batch(
                chunk,
                existing_directions=existing if index > 1 else None,
            )
            representative_notes.append(str(fast_check.get("representative_notes_cn") or ""))
            for direction in directions:
                direction_id = str(direction.get("direction_id") or "").strip()
                if not direction_id:
                    continue
                stored = merged_directions.setdefault(direction_id, {**direction, "paper_ids": []})
                stored.setdefault("direction_name_cn", direction.get("direction_name_cn", direction_id))
                stored.setdefault("direction_name_en", direction.get("direction_name_en", ""))
                stored.setdefault("direction_summary_cn", direction.get("direction_summary_cn", ""))
                stored.setdefault("display_keywords", direction.get("display_keywords", []))
                for paper_id in direction.get("paper_ids", []):
                    if paper_id not in stored["paper_ids"]:
                        stored["paper_ids"].append(paper_id)
            merged_assignments.extend(assignments)
            merged_scores.extend(scores)
        fast_check = {
            "all_papers_assigned_once": True,
            "empty_directions": [],
            "duplicated_paper_ids": [],
            "unassigned_paper_ids": [],
            "representative_notes_cn": "chunked_llm_direction_classification; " + " | ".join(note for note in representative_notes if note),
        }
        validated = validate_directions(papers, list(merged_directions.values()), merged_assignments, merged_scores, fast_check)
        if validated:
            return _ensure_minimum_direction_groups(validated, papers, force_assign_all=force_assign_all)
        raise RuntimeError("chunked LLM result failed global validation")
    except Exception as exc:
        print(f"[预筛警告] 分块 AI 方向归纳失败，使用规则兜底：{exc}")
        return _fallback_directions(papers, force_assign_all=force_assign_all)


def validate_directions(
    papers: list[dict[str, Any]],
    directions: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    relevance_scores: list[dict[str, Any]] | None = None,
    fast_check: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None:
    paper_ids = {str(paper.get("candidate_id")) for paper in papers if paper.get("candidate_id")}
    if not paper_ids:
        return None

    clean_dirs: list[dict[str, Any]] = []
    direction_ids: set[str] = set()
    for index, direction in enumerate(directions):
        direction_id = str(direction.get("direction_id") or f"D{index + 1}").strip()
        if not direction_id:
            continue
        paper_list = [str(pid) for pid in direction.get("paper_ids", []) if str(pid) in paper_ids]
        clean_dirs.append(
            {
                "direction_id": direction_id,
                "direction_name_cn": str(direction.get("direction_name_cn") or direction.get("direction_name") or direction_id),
                "direction_name_en": str(direction.get("direction_name_en") or ""),
                "direction_summary_cn": str(direction.get("direction_summary_cn") or ""),
                "display_keywords": direction.get("display_keywords", []),
                "inclusion_rule_cn": str(direction.get("inclusion_rule_cn") or ""),
                "exclusion_rule_cn": str(direction.get("exclusion_rule_cn") or ""),
                "paper_ids": paper_list,
            }
        )
        direction_ids.add(direction_id)

    clean_assignments: list[dict[str, Any]] = []
    assigned: set[str] = set()
    for assignment in assignments:
        candidate_id = str(assignment.get("candidate_id") or "").strip()
        direction_id = str(assignment.get("direction_id") or "").strip()
        if candidate_id not in paper_ids or direction_id not in direction_ids or candidate_id in assigned:
            continue
        clean_assignments.append(
            {
                "candidate_id": candidate_id,
                "direction_id": direction_id,
                "direction_role": str(assignment.get("direction_role") or "main"),
                "assignment_confidence": safe_float(assignment.get("assignment_confidence"), 0.5),
                "method_or_object_summary_cn": str(
                    assignment.get("method_or_object_summary_cn") or assignment.get("method_summary_cn") or ""
                ),
                "method_summary_cn": str(
                    assignment.get("method_or_object_summary_cn") or assignment.get("method_summary_cn") or ""
                ),
                "assignment_reason_cn": str(assignment.get("assignment_reason_cn") or ""),
            }
        )
        assigned.add(candidate_id)

    if assigned != paper_ids:
        return None

    membership: dict[str, list[str]] = {direction["direction_id"]: [] for direction in clean_dirs}
    for assignment in clean_assignments:
        membership.setdefault(assignment["direction_id"], []).append(assignment["candidate_id"])

    non_empty_dirs = []
    for direction in clean_dirs:
        ids = membership.get(direction["direction_id"], [])
        if ids:
            direction["paper_ids"] = ids
            non_empty_dirs.append(direction)
    if not non_empty_dirs:
        return None

    score_by_id: dict[str, dict[str, Any]] = {}
    for row in relevance_scores or []:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id in paper_ids:
            score_by_id[candidate_id] = {
                "candidate_id": candidate_id,
                "relevance_score": clamp_score(row.get("relevance_score"), default=5.0),
                "decision": str(row.get("decision") or "borderline"),
                "reason_cn": str(row.get("reason_cn") or ""),
            }
    clean_scores = [
        score_by_id.get(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "relevance_score": 5.0,
                "decision": "borderline",
                "reason_cn": "10 未返回该论文分数，使用默认分。",
            },
        )
        for candidate_id in sorted(paper_ids)
    ]
    computed_fast_check = {
        "all_papers_assigned_once": assigned == paper_ids,
        "empty_directions": [],
        "duplicated_paper_ids": [],
        "unassigned_paper_ids": sorted(paper_ids - assigned),
        "representative_notes_cn": str((fast_check or {}).get("representative_notes_cn") or ""),
    }
    return non_empty_dirs, clean_assignments, clean_scores, computed_fast_check


def build_screening_state(
    topic: str,
    papers: list[dict[str, Any]],
    journal_levels_path: str | Path | None = None,
    *,
    input_mode: str = "search_metadata",
    force_assign_all: bool = False,
) -> dict[str, Any]:
    papers_with_ids = with_candidate_ids(papers)
    titles = [str(paper.get("title") or "") for paper in papers_with_ids]
    if len(titles) > 50:
        print(f"[预筛] 候选标题 {len(titles)} 篇，跳过批量翻译以避免阻塞；使用原始标题。")
        title_cn = titles
    else:
        print(f"[预筛] 正在翻译候选标题：{len(titles)} 篇")
        title_cn = translate_titles(titles)
    for index, paper in enumerate(papers_with_ids):
        paper["title_cn"] = title_cn[index] if index < len(title_cn) else paper.get("title", "")

    directions, assignments, relevance_scores, fast_check = infer_candidate_directions(
        topic,
        papers_with_ids,
        input_mode=input_mode,
        force_assign_all=force_assign_all,
    )
    by_id = {str(paper["candidate_id"]): paper for paper in papers_with_ids}
    assignment_by_id = {str(item["candidate_id"]): item for item in assignments}
    score_by_id = {str(item["candidate_id"]): item for item in relevance_scores}

    direction_cards: list[dict[str, Any]] = []
    for direction in directions:
        card_papers = []
        for candidate_id in direction.get("paper_ids", []):
            paper = by_id.get(str(candidate_id))
            if not paper:
                continue
            assignment = assignment_by_id.get(str(candidate_id), {})
            card_papers.append(
                {
                    "candidate_id": paper["candidate_id"],
                    "title": paper.get("title", ""),
                    "title_cn": paper.get("title_cn", ""),
                    "venue": paper.get("venue", ""),
                    "year": paper.get("year"),
                    "source": paper.get("source", ""),
                    "method_or_object_summary_cn": assignment.get("method_or_object_summary_cn", ""),
                    "method_summary_cn": assignment.get("method_summary_cn", ""),
                    "assignment_reason_cn": assignment.get("assignment_reason_cn", ""),
                    "direction_role": assignment.get("direction_role", ""),
                    "assignment_confidence": assignment.get("assignment_confidence", ""),
                    "relevance_score": score_by_id.get(str(candidate_id), {}).get("relevance_score", ""),
                    "decision": score_by_id.get(str(candidate_id), {}).get("decision", ""),
                }
            )
        direction_cards.append({**direction, "papers": card_papers})

    return {
        "topic": topic,
        "generated_at": now_text(),
        "input_mode": input_mode,
        "force_assign_all": bool(force_assign_all),
        "journal_levels_path": str(journal_levels_path or ""),
        "papers": papers_with_ids,
        "directions": direction_cards,
        "assignments": assignments,
        "relevance_scores": relevance_scores,
        "fast_check": fast_check,
    }


def save_screening_state(state: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screening_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "candidate_directions.json").write_text(
        json.dumps(state.get("directions", []), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_screening_state(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def all_direction_ids(state: dict[str, Any]) -> list[str]:
    return [str(direction.get("direction_id")) for direction in state.get("directions", []) if direction.get("direction_id")]


def filter_papers_by_directions(
    state: dict[str, Any],
    selected_directions: list[str] | None,
) -> list[dict[str, Any]]:
    selected = set(selected_directions or all_direction_ids(state))
    assignments = {
        str(item.get("candidate_id")): str(item.get("direction_id"))
        for item in state.get("assignments", [])
    }
    result = []
    for paper in state.get("papers", []):
        direction_id = assignments.get(str(paper.get("candidate_id")), "")
        if direction_id in selected:
            item = dict(paper)
            item["direction_id"] = direction_id
            result.append(item)
    return result


def score_relevance_batch(topic: str, papers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not papers:
        return {}
    prompt = render_prompt(
        "download_relevance_score",
        topic=topic,
        candidate_papers_json=json.dumps(_paper_payload(papers), ensure_ascii=False, indent=2),
    )
    try:
        resp = llm_request(
            messages=[
                {"role": "system", "content": load_prompt("system_strict_legal_json_cn")},
                {"role": "user", "content": prompt},
            ],
            model=get_flash_model(),
            temperature=0.0,
        )
        rows = _extract_json(resp.choices[0].message.content)
        if isinstance(rows, dict):
            rows = rows.get("scores", [])
        if isinstance(rows, list):
            result = {}
            for row in rows:
                candidate_id = str(row.get("candidate_id") or "").strip()
                if not candidate_id:
                    continue
                result[candidate_id] = {
                    "relevance_score": clamp_score(row.get("relevance_score"), default=5.0),
                    "decision": str(row.get("decision") or "borderline"),
                    "reason_cn": str(row.get("reason_cn") or ""),
                }
            return result
    except Exception as exc:
        print(f"[预筛警告] AI 相关度评分失败，使用默认分：{exc}")
    return {}


def score_and_rank_candidates(
    topic: str,
    state: dict[str, Any],
    selected_directions: list[str] | None = None,
    journal_levels_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    papers = filter_papers_by_directions(state, selected_directions)
    if not papers:
        raise RuntimeError("方向筛选后没有候选论文。请至少保留一个方向。")

    state_scores = {
        str(item.get("candidate_id")): {
            "relevance_score": clamp_score(item.get("relevance_score"), default=5.0),
            "decision": str(item.get("decision") or "borderline"),
            "reason_cn": str(item.get("reason_cn") or ""),
        }
        for item in state.get("relevance_scores", [])
        if item.get("candidate_id")
    }
    missing_papers = [
        paper for paper in papers
        if str(paper.get("candidate_id")) not in state_scores
    ]
    score_rows = dict(state_scores)
    if missing_papers:
        score_rows.update(score_relevance_batch(topic, missing_papers))
    lookup_path = journal_levels_path or state.get("journal_levels_path") or None
    journal_lookup = load_journal_levels(lookup_path)

    ranked: list[dict[str, Any]] = []
    for paper in papers:
        item = dict(paper)
        score_meta = score_rows.get(str(item.get("candidate_id")), {})
        relevance = clamp_score(score_meta.get("relevance_score"), default=5.0)
        guard_ok, guard_reason, guard_score = _topic_relevance_guard(item)
        decision = str(score_meta.get("decision") or "borderline")
        reason_cn = str(score_meta.get("reason_cn") or "")
        if not guard_ok:
            relevance = min(relevance, 2.0)
            decision = "exclude"
            reason_cn = f"{reason_cn}；规则筛选排除：{guard_reason}".strip("；")
        else:
            relevance = max(relevance, 6.0 + guard_score * 2.0)
            reason_cn = f"{reason_cn}；规则筛选通过：{guard_reason}".strip("；")
        journal_meta = journal_score_for_paper(item, journal_lookup)
        item.update(score_meta)
        item.update(journal_meta)
        item["relevance_score"] = relevance
        item["decision"] = decision
        item["reason_cn"] = reason_cn
        item["topic_guard_passed"] = guard_ok
        item["topic_guard_reason_cn"] = guard_reason
        item["final_score"] = final_score(relevance, journal_meta.get("journal_level_score"))
        item["ranking_formula"] = (
            "0.7*relevance_score + 0.3*journal_level_score"
            if journal_meta.get("journal_score_available")
            else "relevance_score"
        )
        ranked.append(item)

    ranked.sort(key=lambda paper: paper.get("final_score", 0), reverse=True)
    for index, paper in enumerate(ranked, start=1):
        paper["rank"] = index
    return ranked


def selected_for_download(ranked: list[dict[str, Any]], max_papers: int | None) -> list[dict[str, Any]]:
    if max_papers is None:
        return [paper for paper in ranked if str(paper.get("decision") or "").lower() != "exclude"]

    def download_priority(paper: dict[str, Any]) -> int:
        if paper.get("_pdf_path"):
            return 4
        if paper.get("arxiv_id"):
            return 3
        if paper.get("pdf_url"):
            return 2
        if paper.get("pdf_urls"):
            return 2
        if paper.get("oa_url"):
            return 1
        return 0

    eligible = [
        paper for paper in ranked
        if str(paper.get("decision") or "").lower() != "exclude"
        and str(paper.get("direction_id") or "").lower() != "d_excluded"
        and bool(paper.get("topic_guard_passed", True))
        and float(paper.get("relevance_score") or 0) >= 5.5
    ]
    if len(eligible) < (max_papers or 0):
        eligible = [
            paper for paper in ranked
            if str(paper.get("decision") or "").lower() != "exclude"
            and str(paper.get("direction_id") or "").lower() != "d_excluded"
            and bool(paper.get("topic_guard_passed", True))
        ]

    downloadable_first = sorted(
        eligible,
        key=lambda paper: (
            download_priority(paper),
            float(paper.get("final_score") or paper.get("relevance_score") or 0),
            int(paper.get("cited_by_count") or 0),
        ),
        reverse=True,
    )
    return downloadable_first[:max_papers]

