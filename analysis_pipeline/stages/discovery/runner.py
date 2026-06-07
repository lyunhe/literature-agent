from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any

import requests

from analysis_pipeline.core.common import extract_text_from_pdf, load_json, safe_output_stem
from analysis_pipeline.core.logging import add_skipped_step, run_tracked_block
from analysis_pipeline.core.llm import llm_request
from analysis_pipeline.core.prompts import load_prompt, render_prompt
from analysis_pipeline.stages.discovery.direction_workspace import (
    build_direction_workspace,
    build_local_pdf_candidates,
    build_virtual_single_direction_state,
    load_direction_dirs,
)
from analysis_pipeline.stages.discovery import search_arxiv, search_ieee, search_openalex
from analysis_pipeline.stages.discovery.paper_table import build_paper_table, get_flash_model, save_paper_table
from analysis_pipeline.stages.discovery.prescreen import (
    build_screening_state,
    load_screening_state,
    save_screening_state,
    score_and_rank_candidates,
    selected_for_download,
    with_candidate_ids,
)
from analysis_pipeline.stages.discovery.topic_filtering import TopicFilter


PDF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_prescreen_state(state: dict[str, Any], output_dir: Path) -> None:
    save_json(output_dir / "prescreen_state.json", state)
    save_json(output_dir / "prescreen_candidate_directions.json", state.get("directions", []))


def canonical_candidate(item: dict[str, Any]) -> dict[str, Any]:
    """Keep a stable metadata contract for local and online discovery outputs."""
    row = dict(item)
    row.setdefault("candidate_id", "")
    row.setdefault("title", "")
    row.setdefault("authors", "")
    row.setdefault("year", "")
    row.setdefault("doi", "")
    row.setdefault("source", "")
    row.setdefault("pdf_path", row.get("_pdf_path", ""))
    row.setdefault("txt_path", "")
    row.setdefault("abstract", "")
    row.setdefault("keywords", row.get("concepts", []))
    row.setdefault("download_status", "")
    row.setdefault("is_pdf_available", bool(row.get("_pdf_path") or row.get("pdf_path")))
    if row.get("_pdf_path") and not row.get("pdf_path"):
        row["pdf_path"] = row["_pdf_path"]
    return row


def _text_excerpt(path: Path, max_chars: int = 5000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def enrich_selected_candidates_after_text(
    selected_candidates: list[dict[str, Any]],
    selected_pdfs: list[Path],
    txt_paths: list[Path],
) -> list[dict[str, Any]]:
    txt_by_pdf_name = {pdf.name: txt.resolve() for pdf, txt in zip(selected_pdfs, txt_paths)}
    pdf_by_name = {pdf.name: pdf.resolve() for pdf in selected_pdfs}
    enriched: list[dict[str, Any]] = []
    for raw in selected_candidates:
        if not isinstance(raw, dict):
            continue
        item = canonical_candidate(raw)
        raw_pdf = item.get("_pdf_path") or item.get("pdf_path")
        pdf_name = Path(str(raw_pdf)).name if raw_pdf else ""
        pdf_path = pdf_by_name.get(pdf_name)
        if pdf_path is None and raw_pdf and Path(str(raw_pdf)).exists():
            pdf_path = Path(str(raw_pdf)).resolve()
        if pdf_path is not None:
            item["_pdf_path"] = str(pdf_path)
            item["pdf_path"] = str(pdf_path)
            item["download_status"] = "success"
            item["is_pdf_available"] = True
            pdf_name = pdf_path.name
        txt_path = txt_by_pdf_name.get(pdf_name)
        if txt_path is not None:
            item["txt_path"] = str(txt_path)
            item["pdf_text_excerpt"] = _text_excerpt(txt_path)
        enriched.append(item)
    return enriched


def copy_pdfs_to_run(pdf_paths: list[Path], pdf_dir: Path) -> list[Path]:
    ensure_dir(pdf_dir)
    copied: list[Path] = []
    for source in pdf_paths:
        source = source.resolve()
        target = pdf_dir / source.name
        if source != target.resolve():
            shutil.copy2(source, target)
        copied.append(target.resolve())
        print(f"[PDF归档] {source.name} -> {target}")
    return copied


def convert_pdfs_to_txt(pdf_paths: list[Path], txt_dir: Path, overwrite: bool, max_workers: int = 6) -> list[Path]:
    ensure_dir(txt_dir)
    txt_paths = [txt_dir / f"{safe_output_stem(pdf_path.stem)}.txt" for pdf_path in pdf_paths]

    def convert_one(pdf_path: Path, txt_path: Path) -> Path:
        txt_path = txt_dir / f"{safe_output_stem(pdf_path.stem)}.txt"
        if txt_path.exists() and not overwrite:
            print(f"[TXT] 复用已有文本：{txt_path.name}", flush=True)
            return txt_path
        print(f"[TXT] 正在提取：{pdf_path.name}", flush=True)
        text = extract_text_from_pdf(pdf_path, add_page_mark=True)
        txt_path.write_text(text + "\n", encoding="utf-8")
        print(f"[TXT] 已生成：{txt_path}", flush=True)
        return txt_path

    workers = max(1, min(max_workers, len(pdf_paths)))
    results: dict[Path, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(convert_one, pdf_path, txt_path): pdf_path
            for pdf_path, txt_path in zip(pdf_paths, txt_paths)
        }
        for future in as_completed(futures):
            pdf_path = futures[future]
            results[pdf_path] = future.result()
    return [results.get(pdf_path, txt_dir / f"{safe_output_stem(pdf_path.stem)}.txt") for pdf_path in pdf_paths]


def load_pdf_metadata_candidates(pdf_files: list[Path], metadata_path: Path | None) -> list[dict[str, Any]]:
    if metadata_path is None:
        rows = build_local_pdf_candidates(pdf_files)
        candidates: list[dict[str, Any]] = []
        for row in rows:
            item = canonical_candidate(row)
            pdf_path = Path(str(item.get("_pdf_path") or item.get("pdf_path"))).resolve()
            item["_pdf_path"] = str(pdf_path)
            item["pdf_path"] = str(pdf_path)
            item["source"] = "local"
            item["download_status"] = "local_pdf"
            item["is_pdf_available"] = True
            candidates.append(item)
        return candidates
    payload = load_json(metadata_path)
    rows = payload.get("papers", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("--pdf-metadata-path 必须是数组或包含 papers 数组的 JSON")
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item.setdefault("candidate_id", f"P{index + 1:03d}")
        raw_pdf = item.get("_pdf_path") or item.get("pdf_path") or item.get("filename")
        pdf_path = Path(str(raw_pdf)) if raw_pdf else (pdf_files[index] if index < len(pdf_files) else None)
        if pdf_path is None or not pdf_path.exists():
            raise FileNotFoundError(f"PDF 元数据缺少可匹配文件：{item.get('title') or item.get('candidate_id')}")
        item["_pdf_path"] = str(pdf_path.resolve())
        item.setdefault("source", "local")
        item.setdefault("concepts", [])
        item.setdefault("cited_by_count", 0)
        item["pdf_path"] = item["_pdf_path"]
        item.setdefault("download_status", "local_pdf")
        item["is_pdf_available"] = True
        candidates.append(canonical_candidate(item))
    return candidates


def build_topic_filter(args: Any) -> TopicFilter | None:
    if args.filter_config is not None:
        return TopicFilter.from_config(args.filter_config)
    has_cli = args.filter_and_groups or args.filter_or_groups or args.filter_not_groups
    if not has_cli:
        return None
    return TopicFilter.from_cli_args(
        and_groups=[g.split(",") for g in (args.filter_and_groups or [])],
        or_groups=[g.split(",") for g in (args.filter_or_groups or [])],
        not_groups=[g.split(",") for g in (args.filter_not_groups or [])],
    )


def _clean_llm_keyword(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = text.strip(" \"'`")
    if not text or len(text) > 80:
        return ""
    if any(marker in text for marker in ["{", "}", "[", "]", "\n", "\r"]):
        return ""
    if re.search(r"\b(AND|OR|NOT)\b", text, flags=re.IGNORECASE):
        return ""
    return text


def _extract_keyword_expansion_payload(payload: Any, group_ids: list[str], limit: int) -> dict[str, list[str]]:
    raw_groups: Any
    if isinstance(payload, dict):
        raw_groups = payload.get("groups") or payload.get("filter_groups") or payload.get("expansions") or []
    else:
        raw_groups = payload
    if not isinstance(raw_groups, list):
        return {}

    allowed_ids = set(group_ids)
    expansions: dict[str, list[str]] = {}
    for index, item in enumerate(raw_groups):
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("group_id") or item.get("id") or f"G{index + 1}").strip()
        if group_id not in allowed_ids:
            continue
        raw_keywords = (
            item.get("expanded_keywords")
            or item.get("keywords")
            or item.get("terms")
            or item.get("synonyms")
            or []
        )
        if isinstance(raw_keywords, str):
            raw_keywords = [raw_keywords]
        if not isinstance(raw_keywords, list):
            continue
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw_keyword in raw_keywords:
            keyword = _clean_llm_keyword(raw_keyword)
            key = keyword.lower()
            if keyword and key not in seen:
                cleaned.append(keyword)
                seen.add(key)
            if len(cleaned) >= limit:
                break
        expansions[group_id] = cleaned
    return expansions


def expand_topic_filter_keywords(
    topic: str,
    topic_filter: TopicFilter | None,
    max_terms_per_group: int = 12,
) -> dict[str, Any] | None:
    """Use the flash model to add topic-aware keyword expansions to filter groups.

    The built-in bilingual dictionary remains the deterministic fallback. LLM failures are
    recorded but do not fail discovery.
    """
    if topic_filter is None or not topic_filter.groups:
        return None

    prompt_groups = [
        {
            "group_id": f"G{index + 1}",
            "logic": group.logic,
            "input_keywords": group.input_keywords,
            "current_keywords": group.keywords,
        }
        for index, group in enumerate(topic_filter.groups)
    ]
    report: dict[str, Any] = {
        "topic": topic,
        "model": get_flash_model(),
        "max_terms_per_group": max_terms_per_group,
        "groups": [
            {
                "group_id": item["group_id"],
                "logic": item["logic"],
                "input_keywords": item["input_keywords"],
                "manual_and_saved_expanded_keywords": [
                    kw for kw in item["current_keywords"] if kw not in item["input_keywords"]
                ],
                "llm_added_keywords": [],
            }
            for item in prompt_groups
        ],
    }

    try:
        prompt = render_prompt(
            "filter_keyword_expansion",
            topic=topic,
            filter_groups_json=prompt_groups,
            max_terms_per_group=max_terms_per_group,
        )
        resp = llm_request(
            messages=[
                {"role": "system", "content": load_prompt("system_strict_json_only")},
                {"role": "user", "content": prompt},
            ],
            model=get_flash_model(),
            temperature=0.1,
            max_tokens=1600,
        )
        payload = _extract_json_from_text(resp.choices[0].message.content)
        expansions = _extract_keyword_expansion_payload(
            payload,
            [item["group_id"] for item in prompt_groups],
            max_terms_per_group,
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        print(f"[filter warning] AI keyword expansion failed; using built-in expansions only: {exc}")
        return report

    total_added = 0
    by_id = {item["group_id"]: item for item in report["groups"]}
    for index, group in enumerate(topic_filter.groups):
        group_id = f"G{index + 1}"
        suggested = expansions.get(group_id, [])
        added = group.add_expanded_keywords(suggested)
        total_added += len(added)
        by_id[group_id]["llm_suggested_keywords"] = suggested
        by_id[group_id]["llm_added_keywords"] = added
        by_id[group_id]["keywords_after"] = group.keywords

    report["status"] = "completed" if total_added else "completed_no_new_terms"
    report["total_added"] = total_added
    if total_added:
        print(f"[filter] AI keyword expansion added {total_added} terms.")
    else:
        print("[filter] AI keyword expansion returned no new terms.")
    return report


def selected_direction_ids(raw: str) -> list[str] | None:
    values = [part.strip() for part in (raw or "").split(",") if part.strip()]
    return values or None


def safe_filename(text: str, fallback: str = "paper") -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", text.strip(), flags=re.UNICODE).strip("_")
    return (cleaned or fallback)[:90]


def paper_key(paper: dict[str, Any]) -> str:
    if paper.get("doi"):
        return f"doi:{str(paper['doi']).lower()}"
    if paper.get("arxiv_id"):
        return f"arxiv:{paper['arxiv_id']}"
    if paper.get("openalex_id"):
        return f"openalex:{paper['openalex_id']}"
    return "title:" + re.sub(r"\s+", " ", str(paper.get("title", "")).lower()).strip()


def _extract_json_from_text(text: str) -> Any:
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


def _normalize_query_list(value: Any, limit: int) -> list[str]:
    if isinstance(value, dict):
        value = value.get("queries") or value.get("search_queries") or value.get("query_variations")
    if not isinstance(value, list):
        return []

    unique: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, dict):
            query = str(item.get("query") or item.get("search_query") or "").strip()
        else:
            query = str(item or "").strip()
        key = re.sub(r"\s+", " ", query).lower()
        if key and key not in seen:
            unique.append(query)
            seen.add(key)
        if len(unique) >= limit:
            break
    return unique


def required_topic_concept_groups(topic: str) -> list[list[str]]:
    """Return concept groups that must co-occur for compound topics."""
    text = (topic or "").lower()
    has_storage = any(term in text for term in ["储能", "蓄能", "energy storage", "battery storage", "bess", "pumped hydro", "storage"])
    has_market = any(term in text for term in ["电力市场", "电能市场", "市场", "electricity market", "power market", "energy market", "market"])
    if has_storage and has_market:
        return [
            ["energy storage", "battery storage", "bess", "storage", "pumped hydro", "储能", "蓄能"],
            [
                "electricity market",
                "power market",
                "energy market",
                "wholesale market",
                "ancillary services market",
                "market participation",
                "bidding",
                "market clearing",
                "电力市场",
                "电能市场",
                "市场",
                "报价",
                "竞价",
                "辅助服务",
            ],
        ]
    return []


def _text_has_group(text: str, group: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in group)


def paper_has_required_concepts(topic: str, paper: dict[str, Any]) -> bool:
    groups = required_topic_concept_groups(topic)
    if not groups:
        return True
    text = " ".join(
        str(part or "")
        for part in [
            paper.get("title"),
            paper.get("title_cn"),
            paper.get("abstract"),
            paper.get("venue"),
            " ".join(str(item) for item in (paper.get("concepts") or [])),
            " ".join(str(item) for item in (paper.get("keywords") or [])),
        ]
    )
    return all(_text_has_group(text, group) for group in groups)


def filter_required_topic_concepts(topic: str, papers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = required_topic_concept_groups(topic)
    if not groups:
        return papers, []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for paper in papers:
        item = dict(paper)
        if paper_has_required_concepts(topic, item):
            accepted.append(item)
        else:
            item["download_status"] = "skipped"
            item["is_pdf_available"] = False
            item["skip_reason"] = "missing required compound-topic concept group"
            rejected.append(item)
    return accepted, rejected


def enforce_query_required_concepts(topic: str, queries: list[str], limit: int) -> list[str]:
    groups = required_topic_concept_groups(topic)
    if not groups:
        return queries[:limit]
    seeds = [
        "energy storage electricity market",
        "battery storage electricity market bidding",
        "energy storage market participation",
        "energy storage ancillary services market",
        "pumped hydro electricity market",
        "battery energy storage wholesale market",
        "energy storage market clearing",
        "energy storage power market optimization",
    ]
    normalized: list[str] = []
    for query in [*seeds, *queries]:
        text = str(query or "").strip()
        if not text:
            continue
        for group, canonical in zip(groups, ["energy storage", "electricity market"]):
            if not _text_has_group(text, group):
                text = f"{canonical} {text}"
        text = re.sub(r"\s+", " ", text).strip()
        key = text.lower()
        if key not in {item.lower() for item in normalized}:
            normalized.append(text)
        if len(normalized) >= limit:
            break
    return normalized


def expand_queries(topic: str, max_queries: int = 8) -> list[str]:
    """Use the flash model to turn a research topic into English academic search queries."""
    topic = topic.strip()
    if not topic:
        return []

    prompt = render_prompt("query_expansion", topic=topic, max_queries=max_queries)
    try:
        resp = llm_request(
            messages=[
                {"role": "system", "content": load_prompt("system_strict_json_only")},
                {"role": "user", "content": prompt},
            ],
            model=get_flash_model(),
            temperature=0.2,
            max_tokens=1024,
        )
        payload = _extract_json_from_text(resp.choices[0].message.content)
        queries = _normalize_query_list(payload, limit=max_queries)
        if queries:
            return enforce_query_required_concepts(topic, queries, max_queries)
        raise ValueError("query expansion returned an empty list")
    except Exception as exc:
        print(f"[search warning] AI query expansion failed, using original topic only: {exc}")
        return enforce_query_required_concepts(topic, [topic], max_queries) or [topic]


def _year_in_range(paper: dict[str, Any], year_from: int | None, year_to: int | None) -> bool:
    raw_year = paper.get("year")
    try:
        year = int(raw_year)
    except (TypeError, ValueError):
        return True
    if year_from is not None and year < year_from:
        return False
    if year_to is not None and year > year_to:
        return False
    return True


def _search_limit_attempts(source: str, max_results: int) -> list[int]:
    try:
        requested = max(1, int(max_results or 1))
    except (TypeError, ValueError):
        requested = 5
    first = min(requested, 200) if source == "openalex" else requested
    attempts = [first]
    for fallback in (200, 100, 50, 25, 10):
        if fallback < first and fallback not in attempts:
            attempts.append(fallback)
    return attempts


def _source_search(
    source: str,
    query: str,
    max_results: int,
    year_from: int | None,
    year_to: int | None,
) -> list[dict[str, Any]]:
    if source == "openalex":
        return search_openalex.search(query, max_results=max_results, min_year=year_from, max_year=year_to)
    if source == "arxiv":
        return search_arxiv.search(query, max_results=max_results, min_year=year_from, max_year=year_to)
    if source == "ieee":
        return search_ieee.search(query, max_results=max_results, min_year=year_from, max_year=year_to)
    return []


def search_literature(
    topic: str,
    sources: list[str],
    max_results: int,
    year_from: int | None = None,
    year_to: int | None = None,
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """Search configured literature sources and return de-duplicated metadata rows."""
    all_results: dict[str, dict[str, Any]] = {}
    queries = expand_queries(topic)

    print(f"[检索] 研究主题：{topic}")
    print(f"[检索] 英文/扩展检索词：{', '.join(queries)}")
    if year_from or year_to:
        print(f"[检索] 年份范围：{year_from or '-'} 至 {year_to or '-'}")

    tasks: list[tuple[str, str]] = [
        (source, query)
        for query in queries
        for source in sources
        if source in {"openalex", "arxiv", "ieee"}
    ]

    def run_one(source: str, query: str) -> tuple[str, str, list[dict[str, Any]], list[str], int]:
        errors: list[str] = []
        attempts = _search_limit_attempts(source, max_results)
        for limit in attempts:
            try:
                rows = _source_search(source, query, limit, year_from, year_to)
            except Exception as exc:
                errors.append(f"max_results={limit}: {type(exc).__name__}: {exc}")
                continue
            row_errors = [str(paper.get("error")) for paper in rows if isinstance(paper, dict) and paper.get("error")]
            if row_errors:
                errors.extend(f"max_results={limit}: {error}" for error in row_errors[:3])
                continue
            valid = [
                paper
                for paper in rows
                if isinstance(paper, dict) and "error" not in paper and _year_in_range(paper, year_from, year_to)
            ]
            return source, query, valid, errors, limit
        return source, query, [], errors, attempts[-1] if attempts else max_results

    workers = max(1, min(max_workers, len(tasks)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, source, query) for source, query in tasks]
        for future in as_completed(futures):
            source, query, valid, errors, used_limit = future.result()
            for error in errors[:3]:
                print(f"[检索 warning] {source}: {query} -> {error}", flush=True)
            if used_limit != max_results:
                print(
                    f"[检索 retry] {source}: {query} 使用 max_results={used_limit}（原请求 {max_results}）",
                    flush=True,
                )
            print(f"[检索] {source}: {query} -> {len(valid)} 条 (parallel={workers})", flush=True)
            for paper in valid:
                all_results.setdefault(paper_key(paper), paper)

    print(f"[检索完成] 去重后候选文献：{len(all_results)} 篇")
    return list(all_results.values())


def arxiv_id_from_url(url: str) -> str:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#/]+)", url or "", flags=re.I)
    if not match:
        return ""
    return match.group(1).removesuffix(".pdf")


def arxiv_pdf_url(arxiv_id: str) -> str:
    safe_id = str(arxiv_id).strip().removeprefix("arXiv:").removesuffix(".pdf")
    return f"https://arxiv.org/pdf/{safe_id}.pdf" if safe_id else ""


def _append_unique_url(urls: list[str], url: Any) -> None:
    value = str(url or "").strip()
    if value and value not in urls:
        urls.append(value)


def _openalex_nested_urls(paper: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    open_access = paper.get("open_access") or {}
    if isinstance(open_access, dict):
        _append_unique_url(urls, open_access.get("oa_url"))
    for location in [
        paper.get("primary_location") or {},
        paper.get("best_oa_location") or {},
        *(paper.get("locations") or []),
    ]:
        if not isinstance(location, dict):
            continue
        _append_unique_url(urls, location.get("pdf_url"))
        _append_unique_url(urls, location.get("landing_page_url"))
    return urls


def pdf_candidate_urls(paper: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    arxiv_id = str(paper.get("arxiv_id") or "").strip()
    if not arxiv_id:
        arxiv_id = arxiv_id_from_url(str(paper.get("url") or paper.get("oa_url") or ""))
    if arxiv_id:
        _append_unique_url(urls, arxiv_pdf_url(arxiv_id))
    for key in ("pdf_url", "oa_url"):
        _append_unique_url(urls, paper.get(key))
    for value in paper.get("pdf_urls") or []:
        _append_unique_url(urls, value)
    for value in paper.get("landing_page_urls") or []:
        _append_unique_url(urls, value)
    for value in _openalex_nested_urls(paper):
        _append_unique_url(urls, value)
    return urls


def _status_reason(resp: requests.Response) -> str:
    status = resp.status_code
    if status in {401, 403}:
        return f"access denied ({status})"
    if status == 404:
        return "not found (404)"
    if status >= 400:
        return f"http error ({status})"
    return f"status {status}"


def _looks_like_pdf(content_type: str, first_bytes: bytes) -> bool:
    return first_bytes.startswith(b"%PDF") or "pdf" in (content_type or "").lower()


def check_pdf_downloadable(urls: list[str], timeout: int = 4, max_bytes: int = 4096) -> dict[str, Any]:
    """Verify direct PDF availability before a candidate enters the downloadable pool."""
    attempts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_url in urls:
        url = str(raw_url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        attempt: dict[str, Any] = {"url": url, "head": {}, "get": {}, "ok": False, "reason": ""}

        try:
            head = requests.head(url, headers=PDF_HEADERS, timeout=timeout, allow_redirects=True)
            attempt["head"] = {
                "status_code": head.status_code,
                "content_type": head.headers.get("content-type", ""),
                "final_url": head.url,
            }
            if 200 <= head.status_code < 400 and "pdf" in (head.headers.get("content-type", "") or "").lower():
                attempt["ok"] = True
                attempt["reason"] = "HEAD content-type is PDF"
                attempts.append(attempt)
                return {
                    "ok": True,
                    "url": head.url or url,
                    "reason": attempt["reason"],
                    "attempts": attempts,
                }
            if head.status_code in {401, 403, 404} or head.status_code >= 500:
                attempt["reason"] = _status_reason(head)
                attempts.append(attempt)
                continue
        except requests.RequestException as exc:
            attempt["head"] = {"error": str(exc)}

        try:
            with requests.get(url, headers=PDF_HEADERS, timeout=timeout, allow_redirects=True, stream=True) as resp:
                first = resp.raw.read(max_bytes, decode_content=True)
                content_type = resp.headers.get("content-type", "")
                attempt["get"] = {
                    "status_code": resp.status_code,
                    "content_type": content_type,
                    "final_url": resp.url,
                    "first_bytes": first[:16].hex(),
                }
                if resp.status_code >= 400:
                    attempt["reason"] = _status_reason(resp)
                elif _looks_like_pdf(content_type, first):
                    attempt["ok"] = True
                    attempt["reason"] = "GET first bytes/content-type verify PDF"
                    attempts.append(attempt)
                    return {
                        "ok": True,
                        "url": resp.url or url,
                        "reason": attempt["reason"],
                        "attempts": attempts,
                    }
                else:
                    attempt["reason"] = f"not a PDF ({content_type or 'unknown content-type'})"
        except requests.RequestException as exc:
            attempt["get"] = {"error": str(exc)}
            attempt["reason"] = f"request failed: {exc}"
        attempts.append(attempt)

    return {
        "ok": False,
        "url": "",
        "reason": attempts[-1]["reason"] if attempts else "no candidate URLs",
        "attempts": attempts,
    }


def download_pdf_url(url: str, title: str, output_dir: Path) -> str | None:
    if not url:
        return None
    ensure_dir(output_dir)
    filename = safe_filename(title) + ".pdf"
    output_path = output_dir / filename
    tmp_path = output_dir / f"{filename}.part"
    try:
        with requests.get(url, headers=PDF_HEADERS, timeout=60, allow_redirects=True, stream=True) as response:
            response.raise_for_status()
            iterator = response.iter_content(chunk_size=1024 * 64)
            first = next(iterator, b"")
            content_type = response.headers.get("content-type", "")
            if not _looks_like_pdf(content_type, first):
                print(f"[download warning] response is not PDF: {url} ({content_type})")
                return None
            with tmp_path.open("wb") as fh:
                fh.write(first)
                for chunk in iterator:
                    if chunk:
                        fh.write(chunk)
        tmp_path.replace(output_path)
        return str(output_path.resolve())
    except requests.RequestException as exc:
        print(f"[download warning] request failed: {url} ({exc})")
    except Exception as exc:
        print(f"[download warning] save failed: {url} ({exc})")
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return None


def download_papers(
    papers: list[dict[str, Any]],
    max_papers: int | None,
    output_dir: Path,
    fallback_to_existing: bool = True,
) -> list[Path]:
    """Download PDFs into the current run's discovery PDF directory."""
    ensure_dir(output_dir)
    selected: list[Path] = []

    for paper in papers:
        if max_papers is not None and len(selected) >= max_papers:
            break

        pdf_path: str | None = None
        title = str(paper.get("title") or "Untitled")
        print(f"[下载] 尝试下载：{title[:100]}")
        if paper.get("arxiv_id"):
            result = search_arxiv.download_pdf(str(paper["arxiv_id"]), str(output_dir))
            if result and not result.startswith("Error"):
                pdf_path = result
            else:
                print(f"[下载警告] arXiv 下载失败：{result}")
        if not pdf_path and paper.get("oa_url"):
            pdf_path = download_pdf_url(str(paper["oa_url"]), title, output_dir)

        if pdf_path:
            paper["_pdf_path"] = pdf_path

        if pdf_path:
            resolved = Path(pdf_path).resolve()
            selected.append(resolved)
            print(f"[下载完成] {title[:80]} -> {resolved}")
        else:
            print(f"[下载跳过] 未找到可直接下载的 PDF：{title[:100]}")

    if selected:
        print(f"[下载完成] 本次选中文献 PDF：{len(selected)} 篇")
        return selected

    if not fallback_to_existing:
        print("[下载提示] 未下载到所选候选的 PDF，且当前流程不回退到本地 PDF。")
        return []

    print("[下载提示] 未下载到新 PDF，检查本次运行 PDF 目录中已有 PDF。")
    local_pdfs = sorted(output_dir.glob("*.pdf"))
    if max_papers is not None:
        local_pdfs = local_pdfs[:max_papers]
    return [path.resolve() for path in local_pdfs]


def download_papers(
    papers: list[dict[str, Any]],
    max_papers: int | None,
    output_dir: Path,
    fallback_to_existing: bool = True,
) -> list[Path]:
    """Download PDFs into the current run's discovery PDF directory.

    This definition intentionally appears after the legacy implementation so it
    can add more download strategies without disturbing older call sites.
    """
    ensure_dir(output_dir)
    selected: list[Path] = []

    for paper in papers:
        if max_papers is not None and len(selected) >= max_papers:
            break

        pdf_path: str | None = None
        title = str(paper.get("title") or "Untitled")
        print(f"[download] trying paper: {title[:100]}")

        verified_url = str(paper.get("verified_pdf_url") or "").strip()
        if verified_url:
            print(f"[download] using verified URL: {verified_url}")
            pdf_path = download_pdf_url(verified_url, title, output_dir)

        if not pdf_path:
            arxiv_id = str(paper.get("arxiv_id") or "").strip()
            if not arxiv_id:
                arxiv_id = arxiv_id_from_url(str(paper.get("url") or paper.get("oa_url") or ""))
            if arxiv_id:
                pdf_url = arxiv_pdf_url(arxiv_id)
                availability = check_pdf_downloadable([pdf_url])
                paper["download_attempts"] = availability.get("attempts", [])
                if availability.get("ok"):
                    pdf_path = download_pdf_url(str(availability.get("url") or pdf_url), title, output_dir)
                else:
                    print(f"[download warning] arXiv PDF failed verification: {availability.get('reason')}")

        if not pdf_path:
            for url in pdf_candidate_urls(paper):
                if url == verified_url:
                    continue
                print(f"[download] trying URL: {url}")
                availability = check_pdf_downloadable([url])
                paper["download_attempts"] = availability.get("attempts", [])
                if not availability.get("ok"):
                    print(f"[download warning] URL failed verification: {availability.get('reason')}")
                    continue
                pdf_path = download_pdf_url(str(availability.get("url") or url), title, output_dir)
                if pdf_path:
                    break

        if pdf_path:
            paper["_pdf_path"] = pdf_path
            paper["pdf_path"] = pdf_path
            paper["download_status"] = "success"
            paper["is_pdf_available"] = True
            resolved = Path(pdf_path).resolve()
            selected.append(resolved)
            print(f"[download ok] {title[:80]} -> {resolved}")
        else:
            paper["download_status"] = "failed"
            paper["is_pdf_available"] = False
            print(f"[download skip] no direct PDF found: {title[:100]}")

    if selected:
        print(f"[download ok] selected PDFs: {len(selected)}")
        return selected

    if not fallback_to_existing:
        print("[download note] no PDF was downloaded for selected candidates; local fallback is disabled.")
        return []

    print("[download note] no new PDF was downloaded; checking existing PDFs in this run directory.")
    local_pdfs = sorted(output_dir.glob("*.pdf"))
    if max_papers is not None:
        local_pdfs = local_pdfs[:max_papers]
    return [path.resolve() for path in local_pdfs]


def _topic_terms(topic: str) -> list[str]:
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", topic.lower(), flags=re.UNICODE)
    return [term for term in text.split() if len(term) >= 2]


def candidate_prefilter_score(topic: str, paper: dict[str, Any]) -> float:
    terms = _topic_terms(topic)
    title = str(paper.get("title") or "").lower()
    abstract = str(paper.get("abstract") or "").lower()
    concepts = " ".join(str(x) for x in paper.get("concepts") or paper.get("keywords") or []).lower()
    combined = " ".join([title, abstract, concepts])
    score = 0.0
    for term in terms:
        if term in title:
            score += 3.0
        if term in abstract:
            score += 1.2
        if term in concepts:
            score += 1.0
    for group in required_topic_concept_groups(topic):
        if _text_has_group(combined, group):
            score += 3.0
    if paper.get("pdf_url") or paper.get("pdf_urls") or paper.get("arxiv_id"):
        score += 2.0
    if paper.get("is_open_access"):
        score += 1.0
    score += min(float(paper.get("cited_by_count") or 0), 200.0) / 200.0
    return round(score, 4)


def build_downloadable_pool(
    topic: str,
    candidates: list[dict[str, Any]],
    max_candidates: int | None,
    max_workers: int = 12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = with_candidate_ids([canonical_candidate(paper) for paper in candidates])
    for paper in ranked:
        paper["prefilter_score"] = candidate_prefilter_score(topic, paper)
    ranked.sort(key=lambda item: item.get("prefilter_score", 0), reverse=True)
    for rank, paper in enumerate(ranked, start=1):
        paper["prefilter_rank"] = rank

    downloadable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    limit = max_candidates if max_candidates is not None else len(ranked)
    checked_count = 0

    def verify(paper: dict[str, Any]) -> dict[str, Any]:
        urls = pdf_candidate_urls(paper)
        item = dict(paper)
        item["pdf_url_candidates"] = urls
        item["_availability"] = check_pdf_downloadable(urls)
        return item

    next_index = 0
    futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    worker_count = max(1, min(max_workers, len(ranked)))
    executor = ThreadPoolExecutor(max_workers=worker_count)
    try:
        while next_index < len(ranked) and len(futures) < worker_count:
            paper = ranked[next_index]
            futures[executor.submit(verify, paper)] = paper
            next_index += 1

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                original = futures.pop(future)
                try:
                    paper = future.result()
                    availability = paper.pop("_availability", {})
                except Exception as exc:
                    paper = dict(original)
                    availability = {"ok": False, "reason": f"verification error: {exc}", "attempts": []}

                checked_count += 1
                paper["download_attempts"] = availability.get("attempts", [])
                paper["is_pdf_available"] = bool(availability.get("ok"))
                if availability.get("ok"):
                    paper["verified_pdf_url"] = str(availability.get("url") or "")
                    paper["download_status"] = "verified"
                    downloadable.append(paper)
                    print(
                        f"[downloadable] verified {len(downloadable)}/{limit or len(ranked)} "
                        f"after checking {checked_count}/{len(ranked)} candidates "
                        f"(parallel={worker_count})",
                        flush=True,
                    )
                else:
                    paper["download_status"] = "skipped"
                    paper["skip_reason"] = str(availability.get("reason") or "not downloadable")
                    skipped.append(paper)

                if checked_count % 25 == 0:
                    print(
                        f"[downloadable] checked {checked_count}/{len(ranked)} candidates; "
                        f"verified {len(downloadable)} (parallel={worker_count})",
                        flush=True,
                    )

                if limit is not None and len(downloadable) >= limit:
                    for pending in futures:
                        pending.cancel()
                    for item in futures.values():
                        row = dict(item)
                        row["download_status"] = "not_checked"
                        row["is_pdf_available"] = False
                        row["skip_reason"] = "max_candidates reached"
                        skipped.append(row)
                    for item in ranked[next_index:]:
                        item["download_status"] = "not_checked"
                        item["is_pdf_available"] = False
                        item["skip_reason"] = "max_candidates reached"
                        skipped.append(item)
                    futures.clear()
                    break

                if next_index < len(ranked):
                    paper = ranked[next_index]
                    futures[executor.submit(verify, paper)] = paper
                    next_index += 1
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    downloadable.sort(key=lambda item: int(item.get("prefilter_rank") or 0))
    skipped.sort(key=lambda item: int(item.get("prefilter_rank") or 0))
    return downloadable, skipped


def _source_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[str(row.get("source") or "unknown")] += 1
    return counter


def write_source_comparison(
    output_dir: Path,
    raw_candidates: list[dict[str, Any]],
    downloadable_candidates: list[dict[str, Any]],
    skipped_candidates: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
) -> None:
    sources = sorted(
        set(_source_counts(raw_candidates))
        | set(_source_counts(downloadable_candidates))
        | set(_source_counts(skipped_candidates))
        | set(_source_counts(selected_candidates))
    )
    rows: list[dict[str, Any]] = []
    raw_counts = _source_counts(raw_candidates)
    dl_counts = _source_counts(downloadable_candidates)
    skip_counts = _source_counts(skipped_candidates)
    selected_counts = _source_counts(selected_candidates)
    for source in sources:
        raw_count = raw_counts.get(source, 0)
        downloadable_count = dl_counts.get(source, 0)
        selected_count = selected_counts.get(source, 0)
        skipped_count = skip_counts.get(source, 0)
        rows.append(
            {
                "source": source,
                "raw_candidates": raw_count,
                "downloadable_candidates": downloadable_count,
                "skipped_candidates": skipped_count,
                "selected_downloads": selected_count,
                "downloadable_rate": round(downloadable_count / raw_count, 4) if raw_count else 0.0,
                "selected_rate": round(selected_count / raw_count, 4) if raw_count else 0.0,
            }
        )

    save_json(output_dir / "source_comparison.json", {"rows": rows})
    csv_path = output_dir / "source_comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        fieldnames = [
            "source",
            "raw_candidates",
            "downloadable_candidates",
            "skipped_candidates",
            "selected_downloads",
            "downloadable_rate",
            "selected_rate",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["downloadable_candidates"], default=None)
    lines = ["# Source comparison", ""]
    lines.append("| Source | Raw | Downloadable | Skipped | Selected | Downloadable rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['source']} | {row['raw_candidates']} | {row['downloadable_candidates']} | "
            f"{row['skipped_candidates']} | {row['selected_downloads']} | {row['downloadable_rate']} |"
        )
    lines.append("")
    if best:
        lines.append(
            f"Summary: {best['source']} supplied the largest verified downloadable pool "
            f"({best['downloadable_candidates']} papers). Selected papers are counted only after an actual PDF file is saved."
        )
    (output_dir / "source_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_discovery_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Discovery summary",
        "",
        f"- Topic: {payload.get('topic', '')}",
        f"- Input mode: {payload.get('input_mode', '')}",
        f"- Raw candidates: {payload.get('raw_count', 0)}",
        f"- Downloadable candidates: {payload.get('downloadable_count', 0)}",
        f"- Selected PDFs: {payload.get('selected_count', 0)}",
        f"- Requested max papers: {payload.get('max_papers', '')}",
        f"- Max downloadable candidates: {payload.get('max_candidates', '')}",
    ]
    if payload.get("note"):
        lines.append(f"- Note: {payload['note']}")
    if payload.get("directions"):
        lines.append("")
        lines.append("## Directions")
        for direction in payload["directions"]:
            lines.append(
                f"- {direction.get('direction_id', '')}: {direction.get('direction_name_cn') or direction.get('direction_name_en') or ''} "
                f"({len(direction.get('paper_ids') or [])} papers)"
            )
    (output_dir / "discovery_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_figures_for_pdfs_parallel(ctx: Any, pdf_paths: list[Path], max_workers: int = 4) -> list[dict[str, Any]]:
    ensure_dir(ctx.figures_output_dir)
    script_path = Path(__file__).resolve().parent / "figures_tables.py"
    workers = max(1, min(max_workers, len(pdf_paths)))

    def run_one(pdf_path: Path) -> dict[str, Any]:
        command = [
            sys.executable,
            str(script_path),
            "--pdf",
            str(pdf_path),
            "--output-dir",
            str(ctx.figures_output_dir),
        ]
        started = time.time()
        process = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[3],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "pdf": str(pdf_path),
            "returncode": process.returncode,
            "elapsed_seconds": round(time.time() - started, 3),
            "output": process.stdout,
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, pdf_path): pdf_path for pdf_path in pdf_paths}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "完成" if result["returncode"] == 0 else "失败"
            print(f"[图表提取] {status}: {Path(result['pdf']).name} ({result['elapsed_seconds']}s)", flush=True)
            if result["output"]:
                print(result["output"][-4000:], flush=True)
    failed = [item for item in results if item["returncode"] != 0]
    save_json(ctx.discovery_dir / "figures_tables_manifest.json", results)
    if failed:
        raise RuntimeError(f"图表提取失败：{len(failed)} 篇")
    return results


def state_has_assigned_papers(state: dict[str, Any]) -> bool:
    for direction in state.get("directions", []) if isinstance(state.get("directions"), list) else []:
        if direction.get("paper_ids"):
            return True
    return False


def search_and_download(
    topic: str,
    sources: list[str],
    max_results: int,
    max_papers: int | None,
    output_dir: Path,
    topic_filter: TopicFilter | None = None,
    *,
    ai_prescreen: bool = True,
    screen_only: bool = False,
    screening_state_path: Path | None = None,
    selected_directions: list[str] | None = None,
    journal_levels_path: Path | None = None,
    max_candidates: int | None = None,
    require_pdf: bool = True,
    compare_sources: bool = False,
    year_from: int | None = None,
    year_to: int | None = None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Run search, filter, download, and write outputs into output_dir."""
    ensure_dir(output_dir)
    keyword_expansion_report = expand_topic_filter_keywords(topic, topic_filter)
    if keyword_expansion_report is not None:
        save_json(output_dir / "filter_keyword_expansion.json", keyword_expansion_report)

    if screening_state_path is not None:
        state = load_screening_state(screening_state_path)
        search_results = list(state.get("papers", []))
        save_prescreen_state(state, output_dir)
        ranked = score_and_rank_candidates(
            topic=topic,
            state=state,
            selected_directions=selected_directions,
            journal_levels_path=journal_levels_path,
        )
        save_json(output_dir / "scored_candidates.json", ranked)
        download_candidates = selected_for_download(ranked, len(ranked) if max_papers is not None else None)
        selected_pdfs = download_papers(download_candidates, max_papers=max_papers, output_dir=output_dir / "pdfs", fallback_to_existing=False)
        selected_candidates = [paper for paper in download_candidates if paper.get("_pdf_path")]
        if max_papers is not None:
            selected_candidates = selected_candidates[:max_papers]
        save_json(output_dir / "selected_candidates.json", selected_candidates)
        save_json(output_dir / "selected_pdfs.json", [str(path) for path in selected_pdfs])

        downloaded_names: set[str] = set()
        for paper in selected_candidates:
            if paper.get("_pdf_path"):
                downloaded_names.add(Path(str(paper["_pdf_path"])).name)
        rows = build_paper_table(ranked, downloaded_names, topic_filter)
        save_paper_table(rows, output_dir)
        return search_results, selected_pdfs

    search_results = with_candidate_ids([
        canonical_candidate(item)
        for item in search_literature(
            topic,
            sources,
            max_results,
            year_from=year_from,
            year_to=year_to,
        )
    ])
    save_json(output_dir / "search_results.json", search_results)
    save_json(output_dir / "raw_candidates.json", search_results)

    # Topic filter: between search and download
    accepted = search_results
    if topic_filter is not None:
        accepted, rejected = topic_filter.filter_papers(search_results)
        print(f"[过滤] 主题过滤：{len(accepted)} 篇通过，{len(rejected)} 篇被排除")
        save_json(output_dir / "filter_config.json", topic_filter.to_dict())
        save_json(
            output_dir / "filtered_results.json",
            {
                "accepted": accepted,
                "rejected": rejected,
                "summary": topic_filter.filter_report(search_results),
            },
        )

    required_accepted, required_rejected = filter_required_topic_concepts(topic, accepted)
    if required_rejected:
        print(
            "[filter] compound-topic AND gate: "
            f"{len(required_accepted)} passed, {len(required_rejected)} rejected."
        )
        save_json(
            output_dir / "required_concept_filter.json",
            {
                "topic": topic,
                "required_groups": required_topic_concept_groups(topic),
                "accepted_count": len(required_accepted),
                "rejected_count": len(required_rejected),
                "rejected": required_rejected,
            },
        )
    accepted = required_accepted

    if require_pdf:
        downloadable_candidates, skipped_candidates = build_downloadable_pool(
            topic=topic,
            candidates=accepted,
            max_candidates=max_candidates,
        )
        skipped_candidates = [*required_rejected, *skipped_candidates]
        save_json(output_dir / "downloadable_candidates.json", downloadable_candidates)
        save_json(output_dir / "skipped_candidates.json", skipped_candidates)

        if max_candidates is not None and len(downloadable_candidates) < max_candidates:
            print(
                "[downloadable warning] fewer verified PDFs than requested: "
                f"{len(downloadable_candidates)}/{max_candidates}. Continuing with available papers."
            )

        if ai_prescreen:
            state_candidates = downloadable_candidates
            if screen_only:
                state = build_screening_state(topic, state_candidates, journal_levels_path) if state_candidates else {
                    "topic": topic,
                    "input_mode": "online",
                    "papers": [],
                    "directions": [],
                    "assignments": [],
                    "relevance_scores": [],
                }
                save_screening_state(state, output_dir)
                save_json(output_dir / "selected_pdfs.json", [])
                return search_results, []

            prescreen_pool = downloadable_candidates
            ranked_pool: list[dict[str, Any]] = []
            state: dict[str, Any] = {
                "topic": topic,
                "input_mode": "online_prescreen",
                "papers": [],
                "directions": [],
                "assignments": [],
                "relevance_scores": [],
            }
            if prescreen_pool:
                print(f"[prescreen] scoring verified downloadable candidates before download: {len(prescreen_pool)}")
                state = build_screening_state(topic, prescreen_pool, journal_levels_path, input_mode="online_prescreen")
                if not state_has_assigned_papers(state):
                    print("[prescreen warning] AI/rule screening produced no assigned papers; using single-direction fallback for downloadable PDFs.")
                    state = build_virtual_single_direction_state(topic, prescreen_pool)
                save_prescreen_state(state, output_dir)
                try:
                    ranked_pool = score_and_rank_candidates(
                        topic=topic,
                        state=state,
                        selected_directions=selected_directions,
                        journal_levels_path=journal_levels_path,
                    )
                except RuntimeError:
                    state = build_virtual_single_direction_state(topic, prescreen_pool)
                    save_prescreen_state(state, output_dir)
                    ranked_pool = score_and_rank_candidates(
                        topic=topic,
                        state=state,
                        selected_directions=None,
                        journal_levels_path=journal_levels_path,
                    )
            ranked_downloadable = []
            for paper in ranked_pool:
                source = next(
                    (item for item in downloadable_candidates if str(item.get("candidate_id")) == str(paper.get("candidate_id"))),
                    None,
                )
                if source:
                    merged = dict(source)
                    merged.update(paper)
                    ranked_downloadable.append(merged)
            if ranked_downloadable:
                download_candidates = selected_for_download(ranked_downloadable, len(ranked_downloadable))
                ranked_ids = {str(item.get("candidate_id")) for item in download_candidates}
                download_candidates.extend(
                    item for item in downloadable_candidates
                    if str(item.get("candidate_id")) not in ranked_ids
                )
            else:
                download_candidates = list(downloadable_candidates)
            selected_pdfs = download_papers(
                download_candidates,
                max_papers=max_papers,
                output_dir=output_dir / "pdfs",
                fallback_to_existing=False,
            )
            selected_candidates = [canonical_candidate(paper) for paper in download_candidates if paper.get("_pdf_path")]
            if max_papers is not None:
                selected_candidates = selected_candidates[:max_papers]

            ranked = ranked_pool
            save_json(output_dir / "prescreen_scored_candidates.json", ranked_pool)
            save_json(output_dir / "scored_candidates.json", ranked)
            save_json(output_dir / "selected_candidates.json", selected_candidates)
            save_json(output_dir / "selected_pdfs.json", [str(path) for path in selected_pdfs])

            downloaded_names = {Path(str(item["_pdf_path"])).name for item in selected_candidates if item.get("_pdf_path")}
            rows = build_paper_table(ranked or selected_candidates, downloaded_names, topic_filter) if selected_candidates else []
            save_paper_table(rows, output_dir)
            if compare_sources:
                write_source_comparison(output_dir, search_results, downloadable_candidates, skipped_candidates, selected_candidates)
            write_discovery_summary(
                output_dir,
                {
                    "topic": topic,
                    "input_mode": "online",
                    "raw_count": len(search_results),
                    "downloadable_count": len(downloadable_candidates),
                    "selected_count": len(selected_candidates),
                    "max_papers": max_papers,
                    "max_candidates": max_candidates,
                    "note": (
                        f"Only {len(downloadable_candidates)} verified downloadable PDFs were found."
                        if max_candidates is not None and len(downloadable_candidates) < max_candidates
                        else ""
                    ),
                    "directions": state.get("directions", []),
                },
            )
            return search_results, selected_pdfs

    if ai_prescreen:
        if max_papers is not None:
            prescreen_limit = min(len(accepted), max(80, max_papers * 20))
            prescreen_limit = min(prescreen_limit, 120)
            prescreen_candidates = accepted[:prescreen_limit]
            print(f"[预筛] 下载目标 {max_papers} 篇，优先对前 {len(prescreen_candidates)} 篇高相关候选进行大模型方向分类。")
        else:
            prescreen_candidates = accepted
        state = build_screening_state(topic, prescreen_candidates, journal_levels_path, input_mode="online_prescreen")
        save_prescreen_state(state, output_dir)
        print(f"[预筛] 已生成候选方向：{output_dir / 'prescreen_candidate_directions.json'}")
        if screen_only:
            save_screening_state(state, output_dir)
            save_json(output_dir / "selected_pdfs.json", [])
            return search_results, []

        ranked = score_and_rank_candidates(
            topic=topic,
            state=state,
            selected_directions=selected_directions,
            journal_levels_path=journal_levels_path,
        )
        save_json(output_dir / "scored_candidates.json", ranked)
        download_candidates = selected_for_download(ranked, len(ranked) if max_papers is not None else None)
        selected_pdfs = download_papers(download_candidates, max_papers=max_papers, output_dir=output_dir / "pdfs", fallback_to_existing=False)
        selected_candidates = [paper for paper in download_candidates if paper.get("_pdf_path")]
        if max_papers is not None:
            selected_candidates = selected_candidates[:max_papers]
        save_json(output_dir / "selected_candidates.json", selected_candidates)
        save_json(output_dir / "selected_pdfs.json", [str(path) for path in selected_pdfs])

        downloaded_names: set[str] = set()
        for paper in selected_candidates:
            if paper.get("_pdf_path"):
                downloaded_names.add(Path(str(paper["_pdf_path"])).name)
        if ranked:
            rows = build_paper_table(ranked, downloaded_names, topic_filter)
            save_paper_table(rows, output_dir)
        return search_results, selected_pdfs

    selected_pdfs = download_papers(accepted, max_papers, output_dir=output_dir / "pdfs")
    save_json(output_dir / "selected_pdfs.json", [str(path) for path in selected_pdfs])

    # Build downloaded PDF names for table matching
    downloaded_names: set[str] = set()
    for paper in accepted:
        if paper.get("_pdf_path"):
            downloaded_names.add(Path(str(paper["_pdf_path"])).name)

    # Generate paper summary table
    if accepted:
        rows = build_paper_table(accepted, downloaded_names, topic_filter)
        save_paper_table(rows, output_dir)

    return search_results, selected_pdfs


def run_discovery(ctx: Any) -> None:
    args = ctx.args
    max_papers = None if args.all_papers else args.max_papers
    sources = [source.strip().lower() for source in args.sources.split(",") if source.strip()]

    def prepare_papers() -> tuple[list[dict[str, Any]], list[Path]]:
        if args.from_pdf_only:
            pdf_paths = sorted(args.pdf_dir.glob("*.pdf"))
            if max_papers is not None:
                pdf_paths = pdf_paths[:max_papers]
            if not pdf_paths:
                raise RuntimeError(f"PDF-only 模式未找到 PDF：{args.pdf_dir}")
            candidates = load_pdf_metadata_candidates([path.resolve() for path in pdf_paths], args.pdf_metadata_path)
            save_json(ctx.discovery_dir / "local_pdfs_manifest.json", candidates)
            if args.single_direction_only:
                state = build_virtual_single_direction_state(args.topic, candidates)
                ctx.report["direction_source"] = "user_single_direction"
            else:
                state = build_screening_state(args.topic, candidates, args.journal_levels, input_mode="pdf_metadata_prescreen")
                ctx.report["direction_source"] = "pdf_metadata_10"
            save_prescreen_state(state, ctx.discovery_dir)
            save_json(ctx.discovery_dir / "pdf_metadata_direction_mapping.json", state)
            if args.screen_only:
                save_screening_state(state, ctx.discovery_dir)
                save_json(ctx.discovery_dir / "selected_candidates.json", [])
                save_json(ctx.discovery_dir / "selected_pdfs.json", [])
                return candidates, []
            ranked = score_and_rank_candidates(
                topic=args.topic,
                state=state,
                selected_directions=selected_direction_ids(args.selected_directions),
                journal_levels_path=args.journal_levels,
            )
            save_json(ctx.discovery_dir / "scored_candidates.json", ranked)
            selected_candidates = selected_for_download(ranked, max_papers)
            if not selected_candidates:
                selected_candidates = ranked if max_papers is None else ranked[:max_papers]
            selected_candidates = [canonical_candidate(item) for item in selected_candidates]
            save_json(ctx.discovery_dir / "selected_candidates.json", selected_candidates)
            pdfs = [Path(str(item.get("_pdf_path"))).resolve() for item in selected_candidates if item.get("_pdf_path")]
            save_json(ctx.discovery_dir / "selected_pdfs.json", [str(path) for path in pdfs])
            downloaded_names = {path.name for path in pdfs}
            rows = build_paper_table(ranked, downloaded_names, build_topic_filter(args))
            save_paper_table(rows, ctx.discovery_dir)
            write_discovery_summary(
                ctx.discovery_dir,
                {
                    "topic": args.topic,
                    "input_mode": "local",
                    "raw_count": len(candidates),
                    "downloadable_count": len(candidates),
                    "selected_count": len(selected_candidates),
                    "max_papers": max_papers,
                    "max_candidates": args.max_candidates,
                    "directions": state.get("directions", []),
                },
            )
            return candidates, pdfs

        if args.skip_ai_prescreen:
            raise RuntimeError("新版流程需要 10 作为唯一方向来源，不能使用 --skip-ai-prescreen。")
        ctx.report["direction_source"] = "download_prescreen_10"
        return search_and_download(
            topic=args.topic,
            sources=sources,
            max_results=args.max_results,
            max_papers=max_papers,
            output_dir=ctx.discovery_dir,
            topic_filter=build_topic_filter(args),
            ai_prescreen=True,
            screen_only=args.screen_only,
            screening_state_path=args.screening_state,
            selected_directions=selected_direction_ids(args.selected_directions),
            journal_levels_path=args.journal_levels,
            max_candidates=args.max_candidates,
            require_pdf=bool(args.require_pdf),
            compare_sources=bool(args.compare_sources),
            year_from=args.year_from,
            year_to=args.year_to,
        )

    _, selected_pdfs = run_tracked_block(ctx, "0. 发现阶段：检索/方向预筛/PDF 准备", prepare_papers)
    if args.screen_only:
        ctx.report["status"] = "screening_completed"
        ctx.report["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        ctx.report["screening_state"] = str((ctx.discovery_dir / "screening_state.json").resolve())
        ctx.save_report()
        print(f"\n下载前方向筛选完成：{ctx.discovery_dir / 'screening_state.json'}")
        return
    if not selected_pdfs:
        raise RuntimeError("没有可处理的 PDF。请检查检索结果、网络连接，或先放入 PDF 到 input_pdfs。")

    ctx.report["source_papers"] = [str(path) for path in selected_pdfs]
    save_json(ctx.discovery_dir / "selected_source_pdfs.json", ctx.report["source_papers"])
    selected_pdfs = run_tracked_block(ctx, "0.1 发现阶段：PDF 归档", lambda: copy_pdfs_to_run(selected_pdfs, ctx.run_pdf_dir))
    ctx.report["papers"] = [str(path) for path in selected_pdfs]
    save_json(ctx.discovery_dir / "selected_pdfs.json", ctx.report["papers"])
    ctx.save_report()

    txt_dir = ensure_dir(ctx.discovery_dir / "txt_output")
    txt_paths = run_tracked_block(ctx, "0.2 发现阶段：PDF 正文提取", lambda: convert_pdfs_to_txt(selected_pdfs, txt_dir, args.overwrite))
    ctx.report["txt_output"] = [str(path) for path in txt_paths]
    selected_candidates_path = ctx.discovery_dir / "selected_candidates.json"
    selected_candidates: list[dict[str, Any]] = []
    if selected_candidates_path.exists():
        selected_candidates_for_txt = load_json(selected_candidates_path)
        if isinstance(selected_candidates_for_txt, list):
            selected_candidates = enrich_selected_candidates_after_text(selected_candidates_for_txt, selected_pdfs, txt_paths)
            save_json(selected_candidates_path, selected_candidates)
    if not selected_candidates:
        selected_candidates = enrich_selected_candidates_after_text(
            load_pdf_metadata_candidates(selected_pdfs, None),
            selected_pdfs,
            txt_paths,
        )
        save_json(selected_candidates_path, selected_candidates)
    ctx.save_report()

    def build_final_screening_state() -> dict[str, Any]:
        if args.single_direction_only:
            state = build_virtual_single_direction_state(args.topic, selected_candidates)
            state["input_mode"] = "pdf_text_final"
            state["force_assign_all"] = True
            return state
        return build_screening_state(
            args.topic,
            selected_candidates,
            args.journal_levels,
            input_mode="pdf_text_final",
            force_assign_all=True,
        )

    screening_state = run_tracked_block(
        ctx,
        "0.3 发现阶段：基于 PDF 正文最终方向分类",
        build_final_screening_state,
    )
    save_screening_state(screening_state, ctx.discovery_dir)
    ctx.report["screening_state"] = str((ctx.discovery_dir / "screening_state.json").resolve())
    ctx.report["direction_source"] = "user_single_direction" if args.single_direction_only else "pdf_text_final_10"
    ranked = score_and_rank_candidates(
        topic=args.topic,
        state=screening_state,
        selected_directions=None,
        journal_levels_path=args.journal_levels,
    )
    save_json(ctx.discovery_dir / "scored_candidates.json", ranked)
    downloaded_names = {
        Path(str(item.get("_pdf_path") or item.get("pdf_path"))).name
        for item in selected_candidates
        if item.get("_pdf_path") or item.get("pdf_path")
    }
    save_paper_table(build_paper_table(ranked or selected_candidates, downloaded_names, build_topic_filter(args)), ctx.discovery_dir)
    ctx.save_report()

    if args.extract_figures_tables:
        run_tracked_block(
            ctx,
            "0.4 发现阶段：图表并行提取",
            lambda: extract_figures_for_pdfs_parallel(
                ctx,
                selected_pdfs,
                max_workers=max(1, min(args.parallel_papers, 4)),
            ),
        )
    else:
        add_skipped_step(ctx, "0.4 发现阶段：图表提取", "默认不提取图表；如需提取请添加 --extract-figures-tables")

    screening_state = load_json(ctx.discovery_dir / "screening_state.json")
    selected_candidates = load_json(ctx.discovery_dir / "selected_candidates.json")
    ctx.direction_dirs = run_tracked_block(
        ctx,
        "0.5 发现阶段：构建方向工作区",
        lambda: build_direction_workspace(
            output_dir=ctx.output_dir,
            screening_state=screening_state,
            selected_candidates=selected_candidates,
            pdf_dir=ctx.run_pdf_dir,
            txt_dir=txt_dir,
            figures_dir=ctx.figures_output_dir if args.extract_figures_tables else None,
        ),
    )
    save_json(
        ctx.discovery_dir / "discovery_manifest.json",
        {
            "stage": "discovery",
            "topic": args.topic,
            "input_mode": args.input_mode,
            "paper_table_csv": str((ctx.discovery_dir / "paper_table.csv").resolve()),
            "paper_table_json": str((ctx.discovery_dir / "paper_table.json").resolve()),
            "pdf_dir": str(ctx.run_pdf_dir.resolve()),
            "txt_dir": str((ctx.discovery_dir / "txt_output").resolve()),
            "figures_tables_dir": str(ctx.figures_output_dir.resolve()),
            "directions": [str(path.resolve()) for path in ctx.direction_dirs],
        },
    )


def load_discovery_direction_dirs(ctx: Any) -> list[Path]:
    if ctx.args.discovery_dir is not None:
        root = ctx.args.discovery_dir / "directions"
        return sorted(path for path in root.iterdir() if (path / "assigned_papers.json").exists()) if root.exists() else []
    if (ctx.reviews_dir / "directions").exists():
        return sorted(path for path in (ctx.reviews_dir / "directions").iterdir() if (path / "assigned_papers.json").exists())
    return load_direction_dirs(ctx.output_dir)
