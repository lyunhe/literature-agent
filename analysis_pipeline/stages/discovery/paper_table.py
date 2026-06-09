"""Build a paper summary table with Chinese title translation.

Translates all titles in a single batch API call using the flash model
(LLM_FLASH_MODEL, default deepseek-v4-flash) and outputs both JSON
and CSV files.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from analysis_pipeline.core.llm import llm_request
from analysis_pipeline.core.prompts import load_prompt, render_prompt
from analysis_pipeline.stages.discovery.candidate_links import get_doi_link, pdf_candidate_urls

PAPER_TABLE_COLUMNS = [
    "downloaded",
    "download_status",
    "rank",
    "direction_id",
    "final_score",
    "relevance_score",
    "journal_level_score",
    "journal_level",
    "title",
    "title_cn",
    "abstract_summary_cn",
    "venue",
    "year",
    "source",
    "doi",
    "doi_link",
    "pdf_url",
    "pdf_url_candidates",
    "oa_url",
    "arxiv_id",
    "keywords",
]


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


def _truncate_abstract(text: str, limit: int = 280) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def translate_titles(titles: list[str]) -> list[str]:
    """Batch-translate all titles to Chinese in one API call.

    Returns a list of Chinese translations in the same order as input.
    Falls back to original title on any failure.
    """
    if not titles:
        return []

    non_empty = [(i, t) for i, t in enumerate(titles) if t and t.strip()]
    if not non_empty:
        return list(titles)

    numbered = "\n".join(
        f"{idx}. {text}" for idx, (_, text) in enumerate(non_empty)
    )

    prompt = render_prompt("batch_title_translation", title_list=numbered)

    try:
        resp = llm_request(
            messages=[
                {
                    "role": "system",
                    "content": load_prompt("system_academic_translation_json_array"),
                },
                {"role": "user", "content": prompt},
            ],
            model=get_flash_model(),
            temperature=0.0,
            max_tokens=4096,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        translated = json.loads(raw)

        if isinstance(translated, list) and len(translated) == len(non_empty):
            result = list(titles)
            for (idx, _), cn in zip(non_empty, translated):
                result[idx] = str(cn)
            return result
    except Exception:
        pass

    results: list[str] = []
    for title in titles:
        if not title or not title.strip():
            results.append(title)
            continue
        try:
            resp = llm_request(
                messages=[
                    {"role": "system", "content": load_prompt("system_strict_legal_json_cn")},
                    {"role": "user", "content": render_prompt("single_title_translation", title=str(title))},
                ],
                model=get_flash_model(),
                temperature=0.0,
                max_tokens=256,
            )
            results.append(resp.choices[0].message.content.strip())
        except Exception:
            results.append(title)
    return results


def summarize_abstracts_cn(abstracts: list[str]) -> list[str]:
    """Batch-summarize abstracts in Chinese. Falls back to truncated abstract on failure."""
    if not abstracts:
        return []

    non_empty = [(i, text) for i, text in enumerate(abstracts) if text and str(text).strip()]
    if not non_empty:
        return [_truncate_abstract(text) for text in abstracts]

    numbered = "\n".join(
        f"{idx}. {text}" for idx, (_, text) in enumerate(non_empty)
    )
    prompt = render_prompt("batch_abstract_summary_cn", abstract_list=numbered)

    try:
        resp = llm_request(
            messages=[
                {
                    "role": "system",
                    "content": load_prompt("system_academic_translation_json_array"),
                },
                {"role": "user", "content": prompt},
            ],
            model=get_flash_model(),
            temperature=0.0,
            max_tokens=8192,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        summarized = json.loads(raw)
        if isinstance(summarized, list) and len(summarized) == len(non_empty):
            result = [_truncate_abstract(text) for text in abstracts]
            for (idx, _), summary in zip(non_empty, summarized):
                result[idx] = str(summary).strip() or result[idx]
            return result
    except Exception:
        pass

    return [_truncate_abstract(text) for text in abstracts]


def _paper_download_status(paper: dict[str, Any]) -> str:
    status = str(paper.get("download_status") or "").strip()
    if status:
        return status
    if paper.get("_pdf_path") or paper.get("pdf_path"):
        return "success"
    return "not_attempted"


def build_paper_table(
    papers: list[dict[str, Any]],
    topic_filter: Any | None = None,
) -> list[dict[str, Any]]:
    """Build paper summary rows with translation, links, and keyword matches."""
    titles = [str(p.get("title") or "") for p in papers]
    existing_cn = [str(p.get("title_cn") or "") for p in papers]
    if papers and all(existing_cn):
        title_cn_list = existing_cn
    elif len(titles) > 50:
        print(f"[表格] 候选标题 {len(titles)} 个，跳过批量翻译以避免阻塞；使用原始标题。")
        title_cn_list = titles
    else:
        print(f"[表格] 正在批量翻译 {len(titles)} 个标题...")
        title_cn_list = translate_titles(titles)

    abstracts = [str(p.get("abstract") or "") for p in papers]
    if len(abstracts) > 50:
        print(f"[表格] 候选摘要 {len(abstracts)} 个，跳过批量概括；使用截断摘要。")
        abstract_summary_cn_list = [_truncate_abstract(text) for text in abstracts]
    else:
        print(f"[表格] 正在批量概括 {len(abstracts)} 个摘要...")
        abstract_summary_cn_list = summarize_abstracts_cn(abstracts)

    rows: list[dict[str, Any]] = []
    for i, paper in enumerate(papers):
        title = str(paper.get("title") or "")
        pdf_path = str(paper.get("_pdf_path") or paper.get("pdf_path") or "")
        downloaded = bool(pdf_path)
        download_status = _paper_download_status(paper)
        candidate_urls = pdf_candidate_urls(paper)

        matched_kw: list[str] = []
        if topic_filter is not None:
            _, matched_kw = topic_filter.evaluate_with_matches(paper)

        doi = str(paper.get("doi") or "").strip()
        rows.append(
            {
                "downloaded": downloaded,
                "download_status": download_status,
                "rank": paper.get("rank", ""),
                "direction_id": paper.get("direction_id", ""),
                "final_score": paper.get("final_score", ""),
                "relevance_score": paper.get("relevance_score", ""),
                "journal_level_score": paper.get("journal_level_score", ""),
                "journal_level": paper.get("journal_level", ""),
                "title": title,
                "title_cn": title_cn_list[i] if i < len(title_cn_list) else title,
                "abstract_summary_cn": abstract_summary_cn_list[i] if i < len(abstract_summary_cn_list) else "",
                "venue": paper.get("venue", ""),
                "year": paper.get("year", ""),
                "source": paper.get("source", ""),
                "doi": doi,
                "doi_link": get_doi_link(paper),
                "pdf_url": candidate_urls[0] if candidate_urls else "",
                "pdf_url_candidates": "; ".join(candidate_urls),
                "oa_url": paper.get("oa_url", ""),
                "arxiv_id": paper.get("arxiv_id", ""),
                "keywords": "; ".join(matched_kw) if matched_kw else "",
            }
        )
    return rows


def save_paper_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Write paper_table.json and paper_table.csv to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "paper_table.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[表格] 已保存 JSON: {json_path}")

    csv_path = output_dir / "paper_table.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PAPER_TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[表格] 已保存 CSV: {csv_path}")


def update_paper_table_download_status(
    output_dir: Path,
    papers: list[dict[str, Any]],
) -> None:
    """Update only downloaded/download_status in the existing paper_table."""
    json_path = output_dir / "paper_table.json"
    if not json_path.exists():
        save_paper_table(build_paper_table(papers, None), output_dir)
        return

    rows = json.loads(json_path.read_text(encoding="utf-8"))
    by_rank = {
        paper.get("rank"): paper
        for paper in papers
        if paper.get("rank") is not None
    }
    by_title = {
        str(paper.get("title") or "").strip().lower(): paper
        for paper in papers
    }
    for row in rows:
        paper = by_rank.get(row.get("rank"))
        if paper is None:
            paper = by_title.get(str(row.get("title") or "").strip().lower())
        if paper is None:
            continue
        row["downloaded"] = bool(paper.get("_pdf_path") or paper.get("pdf_path"))
        row["download_status"] = _paper_download_status(paper)
    save_paper_table(rows, output_dir)
    print(f"[表格] 已更新下载状态：{output_dir / 'paper_table.csv'}")
