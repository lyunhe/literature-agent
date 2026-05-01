"""
IEEE Xplore search via REST API.
Requires IEEE API Key (institution subscription).
PDF download is done via the arnumber-based stamp URL.
"""
from __future__ import annotations

import requests
from backend.config import IEEE_API_KEY

IEEE_BASE = "https://ieeexplore.ieee.org/rest/search"
IEEE_STAMP = "https://ieeexplore.ieee.org/stamp/stamp.jsp"


def _format_result(record: dict) -> dict:
    authors_list = record.get("authors", {}).get("authors", [])
    return {
        "title":    record.get("title", ""),
        "authors":  "; ".join(a.get("name", "") for a in authors_list),
        "abstract": record.get("abstract", ""),
        "ieee_id":  str(record.get("article_number", "")),
        "doi":      record.get("doi", ""),
        "url":      f"https://ieeexplore.ieee.org/document/{record.get('article_number', '')}",
        "year":     record.get("year"),
        "venue":    record.get("publication_title", ""),
        "source":   "ieee",
        "pdf_url":  f"{IEEE_STAMP}?tp=&arnumber={record.get('article_number', '')}",
    }


def search(query: str, max_results: int = 5, api_key: str = None) -> list[dict]:
    """
    Search IEEE Xplore.
    api_key: from env.yaml or IEEE_API_KEY in config.
    """
    key = api_key or IEEE_API_KEY
    if not key:
        return [{"error": "IEEE API key not found. Set api_keys.ieee_xplore in env.yaml."}]

    headers = {
        "Content-Type": "application/json",
        "Accept":       "application/json",
        "X-API-Key":    key,
    }
    payload = {
        "QueryText":      query,
        "maxRecords":     max_results,
        "highlight":      True,
        "fields":         "title,authors,abstract,doi,article_number,year,publication_title",
    }
    try:
        resp = requests.post(IEEE_BASE, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return [{"error": f"IEEE API request failed: {e}"}]

    records = data.get("records", [])
    return [_format_result(r) for r in records]


def download_pdf(ieee_id: str, output_path: str, api_key: str = None) -> str:
    """
    Download IEEE paper PDF to output_path.
    ieee_id: the article_number from the search result.
    """
    key = api_key or IEEE_API_KEY
    if not key:
        return "Error: IEEE API key not found."

    # Step 1: get the actual PDF download URL via metadata endpoint
    meta_url = f"https://ieeexplore.ieee.org/rest/document/{ieee_id}"
    headers = {"Accept": "application/json", "X-API-Key": key}
    try:
        meta_resp = requests.get(meta_url, headers=headers, timeout=30)
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        pdf_url = meta.get("downloadLink", "")
    except requests.RequestException as e:
        return f"Error fetching IEEE document metadata: {e}"

    if not pdf_url:
        # Fallback: construct stamp URL (may require session cookie for full PDF)
        pdf_url = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={ieee_id}"

    # Step 2: download the PDF
    try:
        pdf_resp = requests.get(pdf_url, headers=headers, timeout=60, stream=True)
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
    url = f"https://ieeexplore.ieee.org/rest/document/{ieee_id}"
    headers = {"Accept": "application/json", "X-API-Key": key}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        r = resp.json()
        # Extract needed fields
        authors_list = r.get("authors", {}).get("authors", [])
        return {
            "title":    r.get("title", ""),
            "authors":  "; ".join(a.get("name", "") for a in authors_list),
            "abstract": r.get("abstract", ""),
            "ieee_id":  str(r.get("articleNumber", ieee_id)),
            "doi":      r.get("doi", ""),
            "url":      f"https://ieeexplore.ieee.org/document/{ieee_id}",
            "year":     r.get("year"),
            "venue":    r.get("publicationTitle", ""),
            "source":   "ieee",
        }
    except requests.RequestException as e:
        return f"Error fetching IEEE document: {e}"
