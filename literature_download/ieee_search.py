"""
IEEE Xplore search via REST API.
Requires IEEE API Key (institution subscription).
PDF download is done via the arnumber-based stamp URL.
"""
from __future__ import annotations

import requests
from backend.config import IEEE_API_KEY

IEEE_BASE = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
IEEE_DOCUMENT = "https://ieeexploreapi.ieee.org/api/v1/search/document"
IEEE_STAMP = "https://ieeexplore.ieee.org/stamp/stamp.jsp"


def _format_result(record: dict) -> dict:
    authors_value = record.get("authors", {})
    if isinstance(authors_value, dict):
        authors_list = authors_value.get("authors", [])
        authors = "; ".join(a.get("full_name") or a.get("name", "") for a in authors_list)
    elif isinstance(authors_value, list):
        authors = "; ".join(str(a.get("full_name") or a.get("name") or a) for a in authors_value)
    else:
        authors = str(authors_value or "")
    article_number = record.get("article_number") or record.get("articleNumber") or ""
    return {
        "title":    record.get("title", ""),
        "authors":  authors,
        "abstract": record.get("abstract", ""),
        "ieee_id":  str(article_number),
        "doi":      record.get("doi", ""),
        "url":      record.get("html_url") or f"https://ieeexplore.ieee.org/document/{article_number}",
        "year":     record.get("publication_year") or record.get("year"),
        "venue":    record.get("publication_title", ""),
        "source":   "ieee",
        "access_type": record.get("access_type") or record.get("accessType") or "",
        "pdf_url":  record.get("pdf_url") or f"{IEEE_STAMP}?tp=&arnumber={article_number}",
    }


def search(query: str, max_results: int = 5, api_key: str = None) -> list[dict]:
    """
    Search IEEE Xplore.
    api_key: from env.yaml or IEEE_API_KEY in config.
    """
    key = api_key or IEEE_API_KEY
    if not key:
        return [{"error": "IEEE API key not found. Set api_keys.ieee_xplore in env.yaml."}]

    params = {
        "apikey": key,
        "format": "json",
        "querytext": query,
        "max_records": max_results,
    }
    try:
        resp = requests.get(IEEE_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return [{"error": f"IEEE API request failed: {e}"}]

    records = data.get("articles", data.get("records", []))
    return [_format_result(r) for r in records]


def download_pdf(ieee_id: str, output_path: str, api_key: str = None) -> str:
    """
    Download IEEE paper PDF to output_path.
    ieee_id: the article_number from the search result.
    """
    key = api_key or IEEE_API_KEY
    if not key:
        return "Error: IEEE API key not found."

    # Step 1: IEEE Open Access / Full-Text endpoint. Availability depends on
    # the API product attached to the key and the article's access type.
    fulltext_url = f"{IEEE_DOCUMENT}/{ieee_id}/fulltext"
    try:
        meta_resp = requests.get(
            fulltext_url,
            params={"apikey": key, "format": "json"},
            timeout=30,
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        pdf_url = meta.get("pdf_url") or meta.get("downloadLink", "")
    except requests.RequestException as e:
        return f"Error fetching IEEE full-text metadata: {e}"

    if not pdf_url:
        # Fallback: construct stamp URL (may require session cookie for full PDF)
        pdf_url = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={ieee_id}"

    # Step 2: download the PDF
    try:
        pdf_resp = requests.get(pdf_url, timeout=60, stream=True)
        pdf_resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in pdf_resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path
    except requests.RequestException as e:
        return f"Error downloading IEEE PDF: {e}"
    except IOError as e:
        return f"Error writing PDF file: {e}"


def get_info(ieee_id: str, api_key: str = None) -> dict | str:
    """Get detailed metadata for a single IEEE article."""
    key = api_key or IEEE_API_KEY
    if not key:
        return "Error: IEEE API key not found."
    params = {
        "apikey": key,
        "format": "json",
        "article_number": ieee_id,
        "max_records": 1,
    }
    try:
        resp = requests.get(IEEE_BASE, params=params, timeout=30)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        if not articles:
            return f"Error: IEEE article '{ieee_id}' not found."
        return _format_result(articles[0])
    except requests.RequestException as e:
        return f"Error fetching IEEE document: {e}"
