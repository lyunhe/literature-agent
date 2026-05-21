from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"


def clean_doi(doi: str | None) -> str:
    return (doi or "").strip().removeprefix("https://doi.org/").removeprefix("doi:")


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}
        self.links: list[dict[str, str]] = []
        self.title = ""
        self._in_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() == "meta":
            name = attr.get("name") or attr.get("property") or attr.get("itemprop")
            content = attr.get("content")
            if name and content:
                self.meta[name.strip()] = html.unescape(content.strip())
        if tag.lower() == "link":
            self.links.append(attr)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
            self.title = html.unescape("".join(self._title_parts).strip())

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)


def _first_list(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _published_year(message: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "published", "issued"):
        parts = (message.get(key) or {}).get("date-parts") or []
        if parts and parts[0]:
            try:
                return int(parts[0][0])
            except Exception:
                return None
    return None


def _crossref_enrichment(doi: str) -> dict[str, Any]:
    if not doi:
        return {}
    resp = requests.get(f"{CROSSREF_WORKS_URL}/{doi}", headers={"Accept": "application/json"}, timeout=20)
    resp.raise_for_status()
    msg = resp.json().get("message") or {}
    return {
        "doi": msg.get("DOI") or doi,
        "title": _first_list(msg.get("title")),
        "venue": _first_list(msg.get("container-title")),
        "publisher": msg.get("publisher") or "",
        "issn": msg.get("ISSN") or [],
        "isbn": msg.get("ISBN") or [],
        "type": msg.get("type") or "",
        "url": msg.get("URL") or "",
        "year": _published_year(msg),
        "source": "crossref",
    }


def _openalex_enrichment(doi: str, title: str = "") -> dict[str, Any]:
    params: dict[str, Any] = {"per-page": 1}
    if doi:
        params["filter"] = f"doi:{doi}"
    elif title:
        params["search"] = title
    else:
        return {}
    resp = requests.get(OPENALEX_WORKS_URL, params=params, timeout=20)
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        return {}
    work = results[0]
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    return {
        "openalex_id": work.get("id") or "",
        "doi": (work.get("doi") or "").removeprefix("https://doi.org/"),
        "title": work.get("title") or "",
        "venue": source.get("display_name") or "",
        "venue_type": source.get("type") or "",
        "publisher": source.get("host_organization_name") or "",
        "issn": source.get("issn") or [],
        "url": work.get("doi") or work.get("id") or "",
        "landing_page_url": primary.get("landing_page_url") or "",
        "pdf_url": primary.get("pdf_url") or "",
        "is_open_access": (work.get("open_access") or {}).get("is_oa", False),
        "oa_url": (work.get("open_access") or {}).get("oa_url") or "",
        "cited_by_count": work.get("cited_by_count", 0),
        "concepts": [
            item.get("display_name")
            for item in work.get("concepts") or []
            if item.get("display_name")
        ][:12],
        "year": work.get("publication_year"),
        "source": "openalex",
    }


def _publisher_page_metadata(doi: str, url: str = "") -> dict[str, Any]:
    target = url or (f"https://doi.org/{doi}" if doi else "")
    if not target:
        return {}
    resp = requests.get(
        target,
        timeout=25,
        allow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 literature-agent metadata resolver"},
    )
    resp.raise_for_status()
    parser = _MetaParser()
    parser.feed(resp.text[:800000])
    meta = parser.meta

    pdf_candidates = []
    for key in ("citation_pdf_url", "dc.identifier", "DC.Identifier"):
        value = meta.get(key)
        if value and ".pdf" in value.lower():
            pdf_candidates.append(value)
    for link in parser.links:
        href = link.get("href") or ""
        rel = (link.get("rel") or "").lower()
        link_type = (link.get("type") or "").lower()
        if href and ("pdf" in link_type or "pdf" in rel or href.lower().endswith(".pdf")):
            pdf_candidates.append(href)

    return {
        "resolved_url": resp.url,
        "site_domain": urlparse(resp.url).netloc,
        "html_title": parser.title,
        "citation_title": meta.get("citation_title") or meta.get("dc.title") or meta.get("DC.Title") or "",
        "citation_journal_title": meta.get("citation_journal_title") or meta.get("prism.publicationName") or "",
        "citation_conference_title": meta.get("citation_conference_title") or "",
        "citation_publisher": meta.get("citation_publisher") or meta.get("dc.publisher") or "",
        "citation_publication_date": meta.get("citation_publication_date") or meta.get("citation_online_date") or "",
        "citation_doi": meta.get("citation_doi") or doi,
        "citation_pdf_url": pdf_candidates[0] if pdf_candidates else "",
        "pdf_candidates": list(dict.fromkeys(pdf_candidates))[:8],
        "source": "publisher_page",
    }


def probe_publisher_page(doi: str = "", url: str = "") -> dict[str, Any]:
    """Public wrapper for DOI/publisher landing-page metadata extraction."""
    return _publisher_page_metadata(clean_doi(doi), url)


def enrich_paper_by_doi(paper: dict[str, Any], fetch_landing_page: bool = True) -> dict[str, Any]:
    doi = clean_doi(paper.get("doi"))
    title = str(paper.get("title") or "")
    result: dict[str, Any] = {
        "candidate_id": paper.get("candidate_id", ""),
        "input_title": title,
        "input_doi": doi,
        "sources": {},
        "diagnostics": [],
    }
    for name, callback in [
        ("crossref", lambda: _crossref_enrichment(doi)),
        ("openalex", lambda: _openalex_enrichment(doi, title)),
    ]:
        try:
            data = callback()
            if data:
                result["sources"][name] = data
        except requests.RequestException as exc:
            result["diagnostics"].append({"source": name, "error": str(exc)})

    page_url = (
        (result["sources"].get("openalex") or {}).get("landing_page_url")
        or (result["sources"].get("crossref") or {}).get("url")
        or str(paper.get("url") or "")
    )
    if fetch_landing_page and (doi or page_url):
        try:
            page = _publisher_page_metadata(doi, page_url)
            if page:
                result["sources"]["publisher_page"] = page
        except requests.RequestException as exc:
            result["diagnostics"].append({"source": "publisher_page", "error": str(exc)})

    merged: dict[str, Any] = {}
    for source_name in ("crossref", "openalex", "publisher_page"):
        source = result["sources"].get(source_name) or {}
        for key in (
            "title",
            "venue",
            "publisher",
            "issn",
            "year",
            "url",
            "openalex_id",
            "venue_type",
            "landing_page_url",
            "pdf_url",
            "oa_url",
            "is_open_access",
            "cited_by_count",
            "concepts",
        ):
            if source.get(key) and not merged.get(key):
                merged[key] = source[key]
        if source_name == "publisher_page":
            if source.get("citation_journal_title") and not merged.get("venue"):
                merged["venue"] = source["citation_journal_title"]
            if source.get("citation_publisher") and not merged.get("publisher"):
                merged["publisher"] = source["citation_publisher"]
            if source.get("citation_pdf_url") and not merged.get("pdf_url"):
                merged["pdf_url"] = source["citation_pdf_url"]
            if source.get("resolved_url") and not merged.get("landing_page_url"):
                merged["landing_page_url"] = source["resolved_url"]

    result["merged"] = merged
    return result


def enrich_papers_by_doi(
    papers: list[dict[str, Any]],
    output_dir: str | Path | None = None,
    fetch_landing_page: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    enriched: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for paper in papers:
        item = dict(paper)
        if clean_doi(item.get("doi")) or item.get("title"):
            report = enrich_paper_by_doi(item, fetch_landing_page=fetch_landing_page)
            reports.append(report)
            merged = report.get("merged") or {}
            for key, value in merged.items():
                if value and (not item.get(key) or key in {"cited_by_count", "concepts"}):
                    item[key] = value
            item["doi_enriched"] = bool(merged)
        enriched.append(item)
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "doi_enrichment_report.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return enriched, reports
