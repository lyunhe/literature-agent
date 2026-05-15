"""Build a paper summary table with Chinese title translation.

Translates all titles in a single batch API call using the flash model
(DEEPSEEK_FLASH_MODEL, default deepseek-v4-flash) and outputs both JSON
and CSV files.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from analysis_pipeline.prompt_loader import render_prompt
from backend.llm_client import llm_request


def get_flash_model() -> str:
    return os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash")


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
                    "content": "你是一个专业的学术翻译。返回严格合法的 JSON 字符串数组。",
                },
                {"role": "user", "content": prompt},
            ],
            model=get_flash_model(),
            temperature=0.0,
            max_tokens=4096,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
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

    # Fallback: translate one by one
    results: list[str] = []
    for title in titles:
        if not title or not title.strip():
            results.append(title)
            continue
        try:
            resp = llm_request(
                messages=[
                    {"role": "system", "content": "将以下英文标题翻译为中文，只返回中文翻译。"},
                    {"role": "user", "content": str(title)},
                ],
                model=get_flash_model(),
                temperature=0.0,
                max_tokens=256,
            )
            results.append(resp.choices[0].message.content.strip())
        except Exception:
            results.append(title)
    return results


def _get_doi_link(paper: dict[str, Any]) -> str:
    doi = paper.get("doi")
    if doi and str(doi).strip():
        return f"https://doi.org/{str(doi).strip()}"
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id and str(arxiv_id).strip():
        return f"https://arxiv.org/abs/{str(arxiv_id).strip()}"
    url = paper.get("url")
    if url and str(url).strip():
        return str(url).strip()
    return ""


def build_paper_table(
    papers: list[dict[str, Any]],
    downloaded_pdfs: set[str],
    topic_filter: Any | None = None,
) -> list[dict[str, Any]]:
    """Build paper summary rows with translation and keyword matches.

    Args:
        papers: List of paper metadata dicts (must pass filter already).
        downloaded_pdfs: Set of PDF filenames that were successfully downloaded.
        topic_filter: Optional TopicFilter for recording matched keywords.

    Returns:
        List of row dicts with keys:
        downloaded, rank, direction_id, final_score, relevance_score,
        journal_level_score, journal_level, title, title_cn, keywords, doi_link
    """
    titles = [str(p.get("title") or "") for p in papers]
    existing_cn = [str(p.get("title_cn") or "") for p in papers]
    if papers and all(existing_cn):
        title_cn_list = existing_cn
    else:
        print(f"[表格] 正在批量翻译 {len(titles)} 个标题...")
        title_cn_list = translate_titles(titles)

    rows: list[dict[str, Any]] = []
    for i, paper in enumerate(papers):
        title = str(paper.get("title") or "")
        # Check if downloaded via _pdf_path marker set by download_papers()
        pdf_path = str(paper.get("_pdf_path", ""))
        downloaded = bool(pdf_path)

        # Get matched keywords
        matched_kw: list[str] = []
        if topic_filter is not None:
            _, matched_kw = topic_filter.evaluate_with_matches(paper)

        rows.append(
            {
                "downloaded": downloaded,
                "rank": paper.get("rank", ""),
                "direction_id": paper.get("direction_id", ""),
                "final_score": paper.get("final_score", ""),
                "relevance_score": paper.get("relevance_score", ""),
                "journal_level_score": paper.get("journal_level_score", ""),
                "journal_level": paper.get("journal_level", ""),
                "title": title,
                "title_cn": title_cn_list[i] if i < len(title_cn_list) else title,
                "keywords": "; ".join(matched_kw) if matched_kw else "",
                "doi_link": _get_doi_link(paper),
            }
        )
    return rows


def save_paper_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Write paper_table.json and paper_table.csv to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "paper_table.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[表格] 已保存 JSON: {json_path}")

    # CSV (UTF-8 BOM for Excel)
    csv_path = output_dir / "paper_table.csv"
    fieldnames = [
        "downloaded",
        "rank",
        "direction_id",
        "final_score",
        "relevance_score",
        "journal_level_score",
        "journal_level",
        "title",
        "title_cn",
        "keywords",
        "doi_link",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[表格] 已保存 CSV: {csv_path}")
