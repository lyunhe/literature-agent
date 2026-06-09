"""PDF and DOI link helpers shared by discovery runner and paper table."""

from __future__ import annotations

import re
from typing import Any


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


def get_doi_link(paper: dict[str, Any]) -> str:
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
