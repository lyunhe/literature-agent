from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests

from analysis_pipeline.prompt_loader import render_prompt
from backend import db
from backend.llm_client import llm_request
from backend.paths import LIBRARY_PDF_DIR, normalize_library_path
from literature_download import arxiv_search, ieee_search, openalex_search
from literature_download.doi_enrichment import enrich_papers_by_doi
from literature_download.frontend_report import write_frontend_reports
from literature_download.pdf_resolver import download_best_pdf, save_download_reports
from literature_download.query_planner import build_query_plan, flatten_query_plan
from literature_download.seed_ignition import prepare_seed_ignition
from literature_download.topic_filter import TopicFilter
from literature_download.paper_table import build_paper_table, get_flash_model, save_paper_table
from literature_download.prescreen import (
    build_screening_state,
    load_screening_state,
    save_screening_state,
    score_and_rank_candidates,
    selected_for_download,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def expand_queries(topic: str, max_queries: int = 8) -> list[str]:
    """Use the flash model to turn a research topic into English academic search queries."""
    topic = topic.strip()
    if not topic:
        return []

    prompt = render_prompt("query_expansion", topic=topic, max_queries=max_queries)
    try:
        resp = llm_request(
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            model=get_flash_model(),
            temperature=0.2,
            max_tokens=1024,
        )
        payload = _extract_json_from_text(resp.choices[0].message.content)
        queries = _normalize_query_list(payload, limit=max_queries)
        if queries:
            return queries
        raise ValueError("query expansion returned an empty list")
    except Exception as exc:
        print(f"[search warning] AI query expansion failed, using original topic only: {exc}")
        return [topic]


def _source_enabled_for_query(enabled_sources: list[str], source: str, source_hint: str) -> bool:
    if source not in enabled_sources:
        return False
    hint = (source_hint or "all").lower().strip()
    if hint in {"", "all"}:
        return True
    parts = {part.strip() for part in re.split(r"[,/|;]+", hint) if part.strip()}
    return source in parts


def search_literature(
    topic: str,
    sources: list[str],
    max_results: int,
    *,
    max_queries: int = 8,
    trace: dict[str, Any] | None = None,
    seed_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search configured literature sources and return de-duplicated metadata rows."""
    all_results: dict[str, dict[str, Any]] = {}
    query_plan = build_query_plan(topic, max_queries=max_queries, seed_context=seed_context)
    queries = flatten_query_plan(query_plan)
    source_queries: list[dict[str, Any]] = []
    query_rows = query_plan.get("executable_queries", [])

    print(f"[检索] 研究主题：{topic}")
    print(f"[检索] 英文/扩展检索词：{', '.join(queries)}")
    for row in query_rows:
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        source_hint = str(row.get("source_hint") or "all")
        query_meta = {
            "query_id": row.get("query_id", ""),
            "level": row.get("level", ""),
            "rationale_cn": row.get("rationale_cn", ""),
            "source_hint": source_hint,
        }
        if _source_enabled_for_query(sources, "openalex", source_hint):
            rows = openalex_search.search(query, max_results=max_results)
            valid = [paper for paper in rows if "error" not in paper]
            print(f"[检索] OpenAlex: {query} -> {len(valid)} 条")
            source_queries.append({"source": "openalex", "query": query, "result_count": len(valid), **query_meta})
            for paper in valid:
                all_results.setdefault(paper_key(paper), paper)
        if _source_enabled_for_query(sources, "arxiv", source_hint):
            rows = arxiv_search.search(query, max_results=max_results)
            valid = [paper for paper in rows if "error" not in paper]
            print(f"[检索] arXiv: {query} -> {len(valid)} 条")
            source_queries.append({"source": "arxiv", "query": query, "result_count": len(valid), **query_meta})
            for paper in valid:
                all_results.setdefault(paper_key(paper), paper)
        if _source_enabled_for_query(sources, "ieee", source_hint):
            rows = ieee_search.search(query, max_results=max_results)
            valid = [paper for paper in rows if "error" not in paper]
            print(f"[检索] IEEE: {query} -> {len(valid)} 条")
            source_queries.append({"source": "ieee", "query": query, "result_count": len(valid), **query_meta})
            for paper in valid:
                all_results.setdefault(paper_key(paper), paper)

    print(f"[检索完成] 去重后候选文献：{len(all_results)} 篇")
    if trace is not None:
        trace["topic"] = topic
        trace["expanded_queries"] = queries
        trace["query_plan"] = query_plan
        trace["source_queries"] = source_queries
    return list(all_results.values())


def merge_seed_candidates(
    candidates: list[dict[str, Any]],
    seed_papers: list[dict[str, Any]],
    *,
    include_seed_candidates: bool,
) -> list[dict[str, Any]]:
    if not include_seed_candidates or not seed_papers:
        return candidates
    merged: dict[str, dict[str, Any]] = {paper_key(paper): dict(paper) for paper in candidates}
    for seed in seed_papers:
        item = dict(seed)
        item.setdefault("source", "seed")
        item["seed_injected"] = True
        merged.setdefault(paper_key(item), item)
    return list(merged.values())


def _seed_candidate_rows(seed_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(seed_payload.get("seed_papers") or [])
    rows.extend(seed_payload.get("seed_reference_candidates") or [])
    return rows


def download_pdf_url(url: str, title: str, output_dir: Path) -> str | None:
    if not url:
        return None
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException:
        return None

    content_type = response.headers.get("content-type", "")
    if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
        return None

    filename = safe_filename(title) + ".pdf"
    output_path = output_dir / filename
    output_path.write_bytes(response.content)
    return str(output_path.resolve())


def download_papers(
    papers: list[dict[str, Any]],
    max_papers: int | None,
    fallback_to_library: bool = True,
    *,
    report_dir: Path | None = None,
) -> list[Path]:
    """Download PDFs into library/pdfs and save metadata into the SQLite library."""
    ensure_dir(LIBRARY_PDF_DIR)
    selected: list[Path] = []
    download_results: list[dict[str, Any]] = []

    for paper in papers:
        if max_papers is not None and len(selected) >= max_papers:
            break

        title = str(paper.get("title") or "Untitled")
        print(f"[下载] 尝试下载：{title[:100]}")
        result = download_best_pdf(paper, LIBRARY_PDF_DIR)
        result["title"] = title
        result["candidate_id"] = paper.get("candidate_id", "")
        download_results.append(result)

        paper_for_db = dict(paper)
        pdf_path = result.get("pdf_path") or ""
        if pdf_path:
            paper_for_db["pdf_path"] = result.get("normalized_pdf_path") or normalize_library_path(pdf_path)
            paper["_pdf_path"] = pdf_path
        db.add_paper(paper_for_db)

        if pdf_path:
            resolved = Path(pdf_path).resolve()
            selected.append(resolved)
            print(f"[下载完成] {title[:80]} -> {resolved}")
        elif result.get("manual_required"):
            print(f"[下载待手动] 未找到开放 PDF，已加入浏览器辅助下载队列：{title[:100]}")
        else:
            print(f"[下载跳过] 未找到可直接下载的 PDF：{title[:100]}")

    if report_dir is not None:
        save_download_reports(report_dir, download_results)

    if selected:
        print(f"[下载完成] 本次选中文献 PDF：{len(selected)} 篇")
        return selected

    if not fallback_to_library:
        print("[下载提示] 未下载到所选候选的 PDF，且当前流程不回退到本地 PDF。")
        return []

    print("[下载提示] 未下载到新 PDF，改用 library/pdfs 中已有 PDF。")
    local_pdfs = sorted(LIBRARY_PDF_DIR.glob("*.pdf"))
    if max_papers is not None:
        local_pdfs = local_pdfs[:max_papers]
    return [path.resolve() for path in local_pdfs]


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
    max_queries: int = 8,
    seed_papers_path: Path | None = None,
    include_seed_candidates: bool = True,
    doi_enrich: bool = True,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Run search, filter, download, and write outputs into output_dir."""
    ensure_dir(output_dir)
    search_trace: dict[str, Any] = {}

    if screening_state_path is not None:
        state = load_screening_state(screening_state_path)
        search_results = list(state.get("papers", []))
        search_trace = {
            "topic": topic,
            "expanded_queries": [],
            "source_queries": [{"source": "screening_state", "query": str(screening_state_path), "result_count": len(search_results)}],
        }
        save_screening_state(state, output_dir)
        ranked = score_and_rank_candidates(
            topic=topic,
            state=state,
            selected_directions=selected_directions,
            journal_levels_path=journal_levels_path,
        )
        save_json(output_dir / "scored_candidates.json", ranked)
        selected_candidates = selected_for_download(ranked, max_papers)
        selected_pdfs = download_papers(selected_candidates, max_papers=None, fallback_to_library=False, report_dir=output_dir)
        save_json(output_dir / "selected_candidates.json", selected_candidates)
        save_json(output_dir / "selected_pdfs.json", [str(path) for path in selected_pdfs])

        downloaded_names: set[str] = set()
        for paper in selected_candidates:
            if paper.get("_pdf_path"):
                downloaded_names.add(Path(str(paper["_pdf_path"])).name)
        rows = build_paper_table(ranked, downloaded_names, topic_filter)
        save_paper_table(rows, output_dir)
        download_results = json.loads((output_dir / "download_attempts.json").read_text(encoding="utf-8")) if (output_dir / "download_attempts.json").exists() else []
        write_frontend_reports(
            output_dir,
            topic=topic,
            queries=search_trace.get("expanded_queries", []),
            source_queries=search_trace.get("source_queries", []),
            search_results=search_results,
            accepted_results=search_results,
            ranked_candidates=ranked,
            selected_candidates=selected_candidates,
            download_results=download_results,
            query_plan=search_trace.get("query_plan", {}),
            doi_enrichment=[],
        )
        return search_results, selected_pdfs

    seed_payload = prepare_seed_ignition(seed_papers_path, output_dir) if seed_papers_path else {
        "seed_papers": [],
        "seed_context": {"seed_papers": [], "seed_terms": [], "seed_venues": []},
        "doi_enrichment": [],
    }
    search_results = search_literature(
        topic,
        sources,
        max_results,
        max_queries=max_queries,
        trace=search_trace,
        seed_context=seed_payload.get("seed_context") or None,
    )
    seed_rows = _seed_candidate_rows(seed_payload)
    before_seed_merge = len(search_results)
    search_results = merge_seed_candidates(search_results, seed_rows, include_seed_candidates=include_seed_candidates)
    search_trace["seed_ignition"] = {
        "seed_papers": len(seed_payload.get("seed_papers") or []),
        "seed_reference_candidates": len(seed_payload.get("seed_reference_candidates") or []),
        "search_results_before_seed_merge": before_seed_merge,
        "search_results_after_seed_merge": len(search_results),
        "include_seed_candidates": include_seed_candidates,
    }
    doi_reports: list[dict[str, Any]] = []
    if doi_enrich:
        search_results, doi_reports = enrich_papers_by_doi(search_results, output_dir, fetch_landing_page=True)
    save_json(output_dir / "search_results.json", search_results)
    save_json(output_dir / "search_trace.json", search_trace)
    save_json(output_dir / "query_plan.json", search_trace.get("query_plan", {}))

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

    if ai_prescreen:
        state = build_screening_state(topic, accepted, journal_levels_path)
        save_screening_state(state, output_dir)
        print(f"[预筛] 已生成候选方向：{output_dir / 'candidate_directions.json'}")
        if screen_only:
            save_json(output_dir / "selected_pdfs.json", [])
            return search_results, []

        ranked = score_and_rank_candidates(
            topic=topic,
            state=state,
            selected_directions=selected_directions,
            journal_levels_path=journal_levels_path,
        )
        save_json(output_dir / "scored_candidates.json", ranked)
        selected_candidates = selected_for_download(ranked, max_papers)
        selected_pdfs = download_papers(selected_candidates, max_papers=None, fallback_to_library=False, report_dir=output_dir)
        save_json(output_dir / "selected_candidates.json", selected_candidates)
        save_json(output_dir / "selected_pdfs.json", [str(path) for path in selected_pdfs])

        downloaded_names: set[str] = set()
        for paper in selected_candidates:
            if paper.get("_pdf_path"):
                downloaded_names.add(Path(str(paper["_pdf_path"])).name)
        if ranked:
            rows = build_paper_table(ranked, downloaded_names, topic_filter)
            save_paper_table(rows, output_dir)
        download_results = json.loads((output_dir / "download_attempts.json").read_text(encoding="utf-8")) if (output_dir / "download_attempts.json").exists() else []
        write_frontend_reports(
            output_dir,
            topic=topic,
            queries=search_trace.get("expanded_queries", []),
            source_queries=search_trace.get("source_queries", []),
            search_results=search_results,
            accepted_results=accepted,
            ranked_candidates=ranked,
            selected_candidates=selected_candidates,
            download_results=download_results,
            query_plan=search_trace.get("query_plan", {}),
            doi_enrichment=doi_reports,
        )
        return search_results, selected_pdfs

    selected_pdfs = download_papers(accepted, max_papers, report_dir=output_dir)
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
    download_results = json.loads((output_dir / "download_attempts.json").read_text(encoding="utf-8")) if (output_dir / "download_attempts.json").exists() else []
    write_frontend_reports(
        output_dir,
        topic=topic,
        queries=search_trace.get("expanded_queries", []),
        source_queries=search_trace.get("source_queries", []),
        search_results=search_results,
        accepted_results=accepted,
        ranked_candidates=accepted,
        selected_candidates=accepted[:max_papers] if max_papers is not None else accepted,
        download_results=download_results,
        query_plan=search_trace.get("query_plan", {}),
        doi_enrichment=doi_reports,
    )

    return search_results, selected_pdfs
