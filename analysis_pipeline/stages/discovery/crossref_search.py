"""
CrossRef API — no auth required.
Used to resolve DOI -> paper metadata, and optionally find arXiv ID.
"""
from __future__ import annotations

import requests


def _format_result(msg: dict) -> dict:
    authors = msg.get("author", [])
    return {
        "title":    msg.get("title", [""])[0],
        "authors":  "; ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in authors
        ),
        "abstract": msg.get("abstract", ""),
        "doi":      msg.get("DOI", ""),
        "url":      msg.get("URL", ""),
        "year":     int(msg.get("published-print", msg.get("published-online", {})).get("date-parts", [[None]])[0][0] or 0),
        "source":   "crossref",
        # arXiv ID may be embedded in the resource
        "arxiv_id": _extract_arxiv(msg),
    }


def _extract_arxiv(msg: dict) -> str:
    """Try to find an arXiv ID in the CrossRef record's relations or URLs."""
    try:
        rels = msg.get("relation", {})
        if "arxiv" in rels:
            arxiv_url = rels["arxiv"].get("url", "")
            if "arxiv.org/abs" in arxiv_url:
                return arxiv_url.split("/")[-1]
    except Exception:
        pass
    # Fallback: check URL field
    for url in msg.get("URL", "").split():
        if "arxiv.org/abs" in url:
            return url.split("/")[-1]
    return ""


def resolve_doi(doi: str) -> dict | str:
    """
    Resolve a DOI to paper metadata via CrossRef.
    Returns a dict or an error string.
    """
    if not doi:
        return "Error: empty DOI"
    doi = doi.strip()
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    if doi.startswith("doi:"):
        doi = doi[4:]

    url = f"https://api.crossref.org/works/{doi}"
    try:
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        resp.raise_for_status()
        return _format_result(resp.json()["message"])
    except requests.RequestException as e:
        return f"Error resolving DOI: {e}"
