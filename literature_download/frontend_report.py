from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _venue_counter(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter = Counter(str(p.get("venue") or "Unknown") for p in papers)
    return [{"venue": venue, "count": count} for venue, count in counter.most_common()]


def _source_counter(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter = Counter(str(p.get("source") or "unknown") for p in papers)
    return [{"source": source, "count": count} for source, count in counter.most_common()]


def write_frontend_reports(
    output_dir: Path,
    *,
    topic: str,
    queries: list[str],
    source_queries: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    accepted_results: list[dict[str, Any]],
    ranked_candidates: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
    download_results: list[dict[str, Any]],
    query_plan: dict[str, Any] | None = None,
    doi_enrichment: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = [item for item in download_results if item.get("status") in {"downloaded", "local_exists"}]
    manual = [item for item in download_results if item.get("manual_required")]
    summary = {
        "topic": topic,
        "expanded_queries": queries,
        "query_plan": query_plan or {},
        "searched_sources": sorted({item.get("source") for item in source_queries if item.get("source")}),
        "source_queries": source_queries,
        "counts": {
            "search_results": len(search_results),
            "accepted_after_rule_filter": len(accepted_results),
            "ranked_candidates": len(ranked_candidates),
            "selected_candidates": len(selected_candidates),
            "downloaded_or_local": len(downloaded),
            "manual_required": len(manual),
        },
        "journals": _venue_counter(search_results),
        "sources": _source_counter(search_results),
        "selected_candidates": selected_candidates,
        "download_results": download_results,
        "doi_enrichment": doi_enrichment or [],
    }
    json_path = output_dir / "frontend_retrieval_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# 检索下载阶段报告",
        "",
        f"研究主题：{topic}",
        "",
        "## LLM 扩展检索词",
        "",
        *[f"- {query}" for query in queries],
        "",
        "## 多层级检索计划",
        "",
        *[
            f"- {name}: {', '.join(query_plan.get(name, [])[:12])}"
            for name in ["domain_terms", "problem_terms", "method_terms", "venue_anchor_terms", "seed_terms", "negative_terms"]
            if query_plan and query_plan.get(name)
        ],
        "",
        "## 检索信息来源",
        "",
        *[f"- {row['source']}: {row['query']} -> {row.get('result_count', 0)} 条" for row in source_queries],
        "",
        "## 候选文献期刊/会议分布",
        "",
        *[f"- {row['venue']}: {row['count']}" for row in summary["journals"][:30]],
        "",
        "## DOI/出版社来源追溯",
        "",
        *[
            f"- {item.get('input_doi') or item.get('input_title')}: "
            f"{(item.get('merged') or {}).get('venue') or 'Unknown'} / "
            f"{(item.get('merged') or {}).get('publisher') or 'Unknown'}"
            for item in (doi_enrichment or [])[:30]
        ],
        "",
        "## 下载结果",
        "",
        *[
            f"- {item.get('status')}: {item.get('title') or item.get('paper_key')} -> {item.get('pdf_path') or 'manual/failed'}"
            for item in download_results
        ],
    ]
    md_path = output_dir / "retrieval_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"json_path": str(json_path.resolve()), "markdown_path": str(md_path.resolve()), "summary": summary}
