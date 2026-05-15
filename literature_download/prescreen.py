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

from analysis_pipeline.prompt_loader import render_prompt
from backend.llm_client import llm_request
from literature_download.paper_table import translate_titles


def get_flash_model() -> str:
    return os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash")


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


def _paper_payload(papers: list[dict[str, Any]], abstract_limit: int = 1200) -> list[dict[str, Any]]:
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
                "cited_by_count": paper.get("cited_by_count", 0),
            }
        )
    return payload


def _fallback_directions(
    papers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ids = [str(paper["candidate_id"]) for paper in papers]
    directions = [
        {
            "direction_id": "D1",
            "direction_name_cn": "候选相关文献",
            "direction_name_en": "Candidate relevant papers",
            "direction_summary_cn": "模型方向归纳失败时的默认分组，包含所有检索候选文献。",
            "paper_ids": ids,
        }
    ]
    assignments = [
        {
            "candidate_id": str(paper["candidate_id"]),
            "direction_id": "D1",
            "direction_role": "main",
            "assignment_confidence": 0.5,
            "method_or_object_summary_cn": "",
            "method_summary_cn": "",
            "assignment_reason_cn": "默认保留候选文献。",
        }
        for paper in papers
    ]
    scores = [
        {
            "candidate_id": str(paper["candidate_id"]),
            "relevance_score": 5.0,
            "decision": "borderline",
            "reason_cn": "AI 预筛失败后的默认分。",
        }
        for paper in papers
    ]
    fast_check = {
        "all_papers_assigned_once": True,
        "empty_directions": [],
        "duplicated_paper_ids": [],
        "unassigned_paper_ids": [],
        "representative_notes_cn": "fallback",
    }
    return directions, assignments, scores, fast_check


def infer_candidate_directions(
    topic: str,
    papers: list[dict[str, Any]],
    input_mode: str = "search_metadata",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Use the flash model to group metadata-only candidate papers into directions."""
    if not papers:
        return [], [], [], {}

    def build_prompt(abstract_limit: int, retry: bool = False) -> str:
        retry_hint = "上一次输出不是完整合法 JSON。请只返回一个完整 JSON 对象，不要 Markdown，不要解释。\n\n" if retry else ""
        prompt = render_prompt(
            "download_prescreen",
            topic=topic,
            candidate_papers_json={
                "input_mode": input_mode,
                "papers": _paper_payload(papers, abstract_limit=abstract_limit),
            },
        )
        return retry_hint + prompt

    last_error: Exception | None = None
    for attempt, abstract_limit in enumerate([900, 450], start=1):
        try:
            resp = llm_request(
                messages=[
                    {"role": "system", "content": "你只返回严格合法的 JSON。"},
                    {"role": "user", "content": build_prompt(abstract_limit, retry=attempt > 1)},
                ],
                model=get_flash_model(),
                temperature=0.0,
            )
            payload = _extract_json(resp.choices[0].message.content)
            directions = payload.get("directions", []) if isinstance(payload, dict) else []
            assignments = payload.get("assignments", []) if isinstance(payload, dict) else []
            relevance_scores = payload.get("relevance_scores", []) if isinstance(payload, dict) else []
            fast_check = payload.get("fast_check", {}) if isinstance(payload, dict) else {}
            validated = validate_directions(papers, directions, assignments, relevance_scores, fast_check)
            if validated:
                return validated
            last_error = ValueError("LLM JSON passed parsing but failed direction validation.")
        except Exception as exc:
            last_error = exc

    print(f"[预筛警告] AI 方向归纳失败，使用默认方向：{last_error}")
    return _fallback_directions(papers)


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
) -> dict[str, Any]:
    papers_with_ids = with_candidate_ids(papers)
    titles = [str(paper.get("title") or "") for paper in papers_with_ids]
    print(f"[预筛] 正在翻译候选标题：{len(titles)} 篇")
    title_cn = translate_titles(titles)
    for index, paper in enumerate(papers_with_ids):
        paper["title_cn"] = title_cn[index] if index < len(title_cn) else paper.get("title", "")

    directions, assignments, relevance_scores, fast_check = infer_candidate_directions(topic, papers_with_ids)
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
    prompt = f"""你是下载前文献相关度评分助手。

研究主题：
{topic}

请只依据标题、中文标题、摘要、方向、期刊/来源、年份和关键词，为每篇候选论文打相关度分。
相关度是一个总分，包含研究对象、方法、方向、主题匹配程度；不要再拆成方法匹配分。

返回严格 JSON 数组：
[
  {{
    "candidate_id": "候选论文 ID",
    "relevance_score": 0-10,
    "decision": "include/borderline/exclude",
    "reason_cn": "一句话中文理由"
  }}
]

候选论文：
{json.dumps(_paper_payload(papers), ensure_ascii=False, indent=2)}
"""
    try:
        resp = llm_request(
            messages=[
                {"role": "system", "content": "你只返回严格合法的 JSON。"},
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
        journal_meta = journal_score_for_paper(item, journal_lookup)
        item.update(score_meta)
        item.update(journal_meta)
        item["relevance_score"] = relevance
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
        return ranked
    return ranked[:max_papers]
