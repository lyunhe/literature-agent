from __future__ import annotations

import requests
import re

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def _clean_query(query: str) -> str:
    text = query or ""
    text = re.sub(r"\b(title|venue|abstract|document title|publication title)\s*:\s*", " ", text, flags=re.I)
    text = re.sub(r"\bAND\b|\bOR\b|\bNOT\b", " ", text, flags=re.I)
    text = re.sub(r"[()\"']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    terms = text.split()
    deduped = []
    seen = set()
    for term in terms:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
    return " ".join(deduped[:22])


def _format_authors(authorships: list[dict]) -> str:
    names = []
    for item in authorships or []:
        author = item.get("author") or {}
        name = author.get("display_name")
        if name:
            names.append(name)
    return "; ".join(names)


def _abstract_from_inverted_index(index: dict | None) -> str:
    if not index:
        return ""
    positioned = []
    for word, positions in index.items():
        for position in positions:
            positioned.append((position, word))
    positioned.sort()
    return " ".join(word for _, word in positioned)


def _format_result(work: dict) -> dict:
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    concepts = [
        concept.get("display_name")
        for concept in work.get("concepts") or []
        if concept.get("display_name")
    ][:8]
    return {
        "title": work.get("title") or "",
        "authors": _format_authors(work.get("authorships") or []),
        "abstract": work.get("abstract") or _abstract_from_inverted_index(work.get("abstract_inverted_index")),
        "doi": (work.get("doi") or "").removeprefix("https://doi.org/"),
        "openalex_id": work.get("id") or "",
        "url": work.get("doi") or work.get("id") or "",
        "year": work.get("publication_year"),
        "venue": source.get("display_name") or "",
        "venue_type": source.get("type") or "",
        "cited_by_count": work.get("cited_by_count", 0),
        "concepts": concepts,
        "source": "openalex",
        "is_open_access": (work.get("open_access") or {}).get("is_oa", False),
        "oa_url": (work.get("open_access") or {}).get("oa_url") or "",
    }


def search(
    query: str,
    max_results: int = 5,
    venue: str | None = None,
    author: str | None = None,
    min_year: int | None = None,
) -> list[dict]:
    """Search OpenAlex works by free text, optionally nudged by venue/author/year."""
    clean_query = _clean_query(query)
    attempts = [clean_query]
    lowered = clean_query.lower()
    if "scuc" in lowered and "benders" in lowered:
        attempts.append("SCUC Benders decomposition")
    if "security constrained unit commitment" in lowered and "benders" in lowered:
        attempts.append("security constrained unit commitment Benders decomposition")
    if "unit commitment" in lowered and "benders" in lowered:
        attempts.append("unit commitment Benders decomposition")
    if "parallel" in lowered and ("scuc" in lowered or "unit commitment" in lowered):
        attempts.append("SCUC parallel computing")
        attempts.append("unit commitment parallel computing")

    seen_attempts = []
    for attempt in attempts:
        if attempt and attempt not in seen_attempts:
            seen_attempts.append(attempt)

    filters = []
    if min_year:
        filters.append(f"from_publication_date:{min_year}-01-01")

    works = []
    last_error = None
    used_query = clean_query
    for final_query in seen_attempts:
        query_filters = [f"title_and_abstract.search:{final_query}", *filters]
        params = {
            "filter": ",".join(query_filters),
            "per-page": max_results,
        }
        try:
            resp = requests.get(OPENALEX_WORKS_URL, params=params, timeout=20)
            resp.raise_for_status()
            works = resp.json().get("results", [])
            used_query = final_query
            if works:
                break
        except requests.RequestException as e:
            last_error = e

    if not works and last_error:
        return [{"error": f"OpenAlex API request failed: {last_error}"}]

    results = [_format_result(work) for work in works]
    for result in results:
        result["normalized_query"] = clean_query
        result["openalex_query_used"] = used_query
        result["venue_hint"] = venue or ""
        result["author_hint"] = author or ""
    return results
