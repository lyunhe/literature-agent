from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from literature_download.doi_enrichment import clean_doi, enrich_paper_by_doi, probe_publisher_page


def _load_papers(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("papers", payload.get("results", payload)) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("input JSON must be a list, or an object with papers/results.")
    return [dict(row) for row in rows]


def _playwright_probe(doi: str, url: str) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    target = url or (f"https://doi.org/{doi}" if doi else "")
    if not target:
        return {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(target, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(1200)
        data = page.evaluate(
            """() => {
                const meta = {};
                for (const item of document.querySelectorAll('meta')) {
                    const key = item.getAttribute('name') || item.getAttribute('property') || item.getAttribute('itemprop');
                    const value = item.getAttribute('content');
                    if (key && value) meta[key] = value;
                }
                const links = Array.from(document.querySelectorAll('link')).map(link => ({
                    rel: link.getAttribute('rel') || '',
                    type: link.getAttribute('type') || '',
                    href: link.href || link.getAttribute('href') || ''
                }));
                const jsonld = Array.from(document.querySelectorAll('script[type="application/ld+json"]'))
                    .map(script => script.textContent || '').slice(0, 5);
                return {
                    resolved_url: location.href,
                    html_title: document.title,
                    meta,
                    links,
                    jsonld
                };
            }"""
        )
        browser.close()
    meta = data.get("meta") or {}
    pdf_links = []
    if meta.get("citation_pdf_url"):
        pdf_links.append(meta["citation_pdf_url"])
    for link in data.get("links") or []:
        href = link.get("href") or ""
        rel = (link.get("rel") or "").lower()
        link_type = (link.get("type") or "").lower()
        if href and ("pdf" in rel or "pdf" in link_type or href.lower().endswith(".pdf")):
            pdf_links.append(href)
    return {
        "resolved_url": data.get("resolved_url", ""),
        "html_title": data.get("html_title", ""),
        "citation_title": meta.get("citation_title") or meta.get("dc.title") or "",
        "citation_journal_title": meta.get("citation_journal_title") or meta.get("prism.publicationName") or "",
        "citation_conference_title": meta.get("citation_conference_title") or "",
        "citation_publisher": meta.get("citation_publisher") or meta.get("dc.publisher") or "",
        "citation_publication_date": meta.get("citation_publication_date") or meta.get("citation_online_date") or "",
        "citation_doi": meta.get("citation_doi") or doi,
        "citation_pdf_url": pdf_links[0] if pdf_links else "",
        "pdf_candidates": list(dict.fromkeys(pdf_links))[:8],
        "jsonld": data.get("jsonld", []),
        "source": "playwright_browser",
    }


def probe_paper(paper: dict[str, Any], use_playwright: bool = True) -> dict[str, Any]:
    doi = clean_doi(paper.get("doi"))
    url = str(paper.get("landing_page_url") or paper.get("url") or "")
    if use_playwright:
        try:
            page = _playwright_probe(doi, url)
            if page:
                return {
                    "candidate_id": paper.get("candidate_id", ""),
                    "title": paper.get("title", ""),
                    "doi": doi,
                    "browser_metadata": page,
                    "diagnostics": [],
                }
        except Exception as exc:
            fallback = enrich_paper_by_doi(paper, fetch_landing_page=True)
            fallback.setdefault("diagnostics", []).append({"source": "playwright_browser", "error": str(exc)})
            return fallback
    try:
        page = probe_publisher_page(doi, url)
        return {
            "candidate_id": paper.get("candidate_id", ""),
            "title": paper.get("title", ""),
            "doi": doi,
            "browser_metadata": page,
            "diagnostics": [],
        }
    except Exception as exc:
        return {
            "candidate_id": paper.get("candidate_id", ""),
            "title": paper.get("title", ""),
            "doi": doi,
            "browser_metadata": {},
            "diagnostics": [{"source": "publisher_page", "error": str(exc)}],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe DOI/publisher pages for venue metadata with Playwright when available.")
    parser.add_argument("--input", type=Path, required=True, help="JSON list, search_results.json, or selected_candidates.json")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--no-playwright", action="store_true", help="Use requests fallback instead of Playwright")
    parser.add_argument("--limit", type=int, default=0, help="Maximum papers to probe; 0 means all")
    args = parser.parse_args()

    papers = _load_papers(args.input)
    if args.limit:
        papers = papers[: args.limit]
    reports = [probe_paper(paper, use_playwright=not args.no_playwright) for paper in papers]
    output = args.output or args.input.with_name("browser_metadata_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
