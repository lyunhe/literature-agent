"""Retry PDF downloads for failed discovery candidates (publisher URLs)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis_pipeline.stages.discovery.candidate_links import pdf_candidate_urls
from analysis_pipeline.stages.discovery.paper_table import update_paper_table_download_status
from analysis_pipeline.stages.discovery.runner import (
    canonical_candidate,
    download_pdf_url,
    ensure_dir,
    save_json,
)


def extra_urls(paper: dict) -> list[str]:
    urls: list[str] = []
    doi = str(paper.get("doi") or "").strip().removeprefix("https://doi.org/")
    if doi.startswith("10.1007/"):
        urls.append(f"https://link.springer.com/content/pdf/{doi}.pdf")
    if doi.startswith("10.1140/"):
        urls.append(f"https://link.springer.com/content/pdf/{doi}.pdf")
    if doi.startswith("10.2514/"):
        urls.extend(
            [
                f"https://arc.aiaa.org/doi/pdf/{doi}",
                f"https://arc.aiaa.org/doi/pdfdirect/{doi}?download=true",
                f"https://doi.org/{doi}",
            ]
        )
    if doi.startswith("10.1209/"):
        urls.append(f"https://iopscience.iop.org/article/{doi}/pdf")
        urls.append(f"https://doi.org/{doi}")
    if doi.startswith("10.23919/"):
        urls.append(f"https://doi.org/{doi}")
    for key in ("pdf_url", "oa_url"):
        value = str(paper.get(key) or "").strip()
        if value:
            urls.append(value)
    for value in pdf_candidate_urls(paper):
        if value not in urls:
            urls.append(value)
    for value in str(paper.get("pdf_url_candidates") or "").split(";"):
        value = value.strip()
        if value and value not in urls:
            urls.append(value)
    return urls


def has_pdf(paper: dict) -> bool:
    path = paper.get("_pdf_path") or paper.get("pdf_path")
    return bool(path and Path(str(path)).exists())


def try_download(paper: dict, pdf_dir: Path) -> bool:
    if has_pdf(paper):
        return True
    title = str(paper.get("title") or "paper")
    for url in extra_urls(paper):
        print(f"  try {url[:120]}")
        path = download_pdf_url(url, title, pdf_dir)
        if path:
            paper["_pdf_path"] = path
            paper["pdf_path"] = path
            paper["download_status"] = "success"
            paper["is_pdf_available"] = True
            print(f"  OK -> {Path(path).name}")
            return True
    paper["download_status"] = "failed"
    return False


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python tools/retry_failed_downloads.py <01_discovery_dir>")
        raise SystemExit(2)

    disc = Path(sys.argv[1]).resolve()
    papers = json.loads((disc / "scored_candidates.json").read_text(encoding="utf-8"))
    pdf_dir = ensure_dir(disc / "pdfs")

    failed = [p for p in papers if not has_pdf(p)]
    print(f"Retry {len(failed)} papers via publisher URLs...")
    new_ok = 0
    for paper in failed:
        rank = paper.get("rank")
        title = str(paper.get("title") or "")[:70]
        print(f"[{rank}] {title}")
        if try_download(paper, pdf_dir):
            new_ok += 1

    selected = [canonical_candidate(p) for p in papers if has_pdf(p)]
    save_json(disc / "selected_candidates.json", selected)
    unique: list[str] = []
    seen: set[str] = set()
    for item in selected:
        resolved = str(Path(str(item["_pdf_path"])).resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    save_json(disc / "selected_pdfs.json", unique)

    update_paper_table_download_status(disc, papers)

    print(f"SUMMARY new_ok={new_ok} total_pdfs={len(unique)}/{len(papers)}")


if __name__ == "__main__":
    main()
