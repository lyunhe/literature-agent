from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests

from backend import db
from backend.paths import LIBRARY_PDF_DIR, normalize_library_path
from literature_download import arxiv_search, ieee_search, openalex_search


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


def expand_queries(topic: str) -> list[str]:
    """Expand a short Chinese or English research topic into search-oriented queries."""
    queries = [topic]
    text = topic.lower()
    if "储能" in topic or "energy storage" in text:
        queries.extend(
            [
                "energy storage electricity market",
                "battery energy storage power market bidding",
                "energy storage participation electricity markets",
            ]
        )
    if "报价" in topic or "竞价" in topic or "bidding" in text or "bid" in text:
        queries.extend(
            [
                "energy storage bidding strategies in electricity markets",
                "battery energy storage bidding strategy electricity market",
                "energy storage price bidding electricity markets",
            ]
        )
    if (
        "收益" in topic
        or "分配" in topic
        or "利润" in topic
        or "revenue" in text
        or "profit" in text
        or "benefit sharing" in text
        or "allocation" in text
    ):
        queries.extend(
            [
                "energy storage revenue allocation electricity market",
                "battery energy storage profit sharing electricity market",
                "energy storage benefit allocation power market",
                "shared energy storage revenue sharing electricity market",
            ]
        )
    if "电力市场" in topic or "electricity market" in text or "power market" in text:
        queries.extend(
            [
                "electricity market energy storage optimization",
                "power market battery storage scheduling",
            ]
        )

    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.strip().lower()
        if key and key not in seen:
            unique.append(query.strip())
            seen.add(key)
    return unique


def search_literature(topic: str, sources: list[str], max_results: int) -> list[dict[str, Any]]:
    """Search configured literature sources and return de-duplicated metadata rows."""
    all_results: dict[str, dict[str, Any]] = {}
    queries = expand_queries(topic)

    print(f"[检索] 研究主题：{topic}")
    print(f"[检索] 英文/扩展检索词：{', '.join(queries)}")
    for query in queries:
        if "openalex" in sources:
            rows = openalex_search.search(query, max_results=max_results)
            valid = [paper for paper in rows if "error" not in paper]
            print(f"[检索] OpenAlex: {query} -> {len(valid)} 条")
            for paper in valid:
                all_results.setdefault(paper_key(paper), paper)
        if "arxiv" in sources:
            rows = arxiv_search.search(query, max_results=max_results)
            valid = [paper for paper in rows if "error" not in paper]
            print(f"[检索] arXiv: {query} -> {len(valid)} 条")
            for paper in valid:
                all_results.setdefault(paper_key(paper), paper)
        if "ieee" in sources:
            rows = ieee_search.search(query, max_results=max_results)
            valid = [paper for paper in rows if "error" not in paper]
            print(f"[检索] IEEE: {query} -> {len(valid)} 条")
            for paper in valid:
                all_results.setdefault(paper_key(paper), paper)

    print(f"[检索完成] 去重后候选文献：{len(all_results)} 篇")
    return list(all_results.values())


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


def download_papers(papers: list[dict[str, Any]], max_papers: int | None) -> list[Path]:
    """Download PDFs into library/pdfs and save metadata into the SQLite library."""
    ensure_dir(LIBRARY_PDF_DIR)
    selected: list[Path] = []

    for paper in papers:
        if max_papers is not None and len(selected) >= max_papers:
            break

        pdf_path: str | None = None
        title = str(paper.get("title") or "Untitled")
        print(f"[下载] 尝试下载：{title[:100]}")
        if paper.get("arxiv_id"):
            result = arxiv_search.download_pdf(str(paper["arxiv_id"]), str(LIBRARY_PDF_DIR))
            if result and not result.startswith("Error"):
                pdf_path = result
            else:
                print(f"[下载警告] arXiv 下载失败：{result}")
        if not pdf_path and paper.get("oa_url"):
            pdf_path = download_pdf_url(str(paper["oa_url"]), title, LIBRARY_PDF_DIR)

        paper_for_db = dict(paper)
        if pdf_path:
            paper_for_db["pdf_path"] = normalize_library_path(pdf_path)
        db.add_paper(paper_for_db)

        if pdf_path:
            resolved = Path(pdf_path).resolve()
            selected.append(resolved)
            print(f"[下载完成] {title[:80]} -> {resolved}")
        else:
            print(f"[下载跳过] 未找到可直接下载的 PDF：{title[:100]}")

    if selected:
        print(f"[下载完成] 本次选中文献 PDF：{len(selected)} 篇")
        return selected

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
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Run search and download, writing search metadata into output_dir."""
    search_results = search_literature(topic, sources, max_results)
    save_json(output_dir / "search_results.json", search_results)
    selected_pdfs = download_papers(search_results, max_papers)
    save_json(output_dir / "selected_pdfs.json", [str(path) for path in selected_pdfs])
    return search_results, selected_pdfs
