"""
arXiv search and PDF download.
Uses the `arxiv` package — no API key required.
"""
from __future__ import annotations

import os, json, arxiv, requests
from analysis_pipeline.core.paths import INPUT_PDFS_DIR


def _format_result(paper) -> dict:
    return {
        "title":    paper.title,
        "authors":  ", ".join(a.name for a in paper.authors),
        "abstract": paper.summary,
        "arxiv_id": paper.entry_id.split("/")[-1],
        "url":      paper.entry_id,
        "year":     paper.published.year,
        "doi":      paper.doi or "",
        "source":   "arxiv",
    }


def search(query: str, max_results: int = 5, min_year: int | None = None, max_year: int | None = None) -> list[dict]:
    """
    Search arXiv by query string.
    Returns a list of paper metadata dicts.
    """
    try:
        result_limit = max(1, int(max_results or 1))
    except (TypeError, ValueError):
        result_limit = 5
    client = arxiv.Client()
    search_obj = arxiv.Search(
        query=query,
        max_results=result_limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    try:
        results = list(client.results(search_obj))
    except Exception as e:
        return [{"error": str(e)}]
    rows = [_format_result(r) for r in results]
    if min_year is not None:
        rows = [row for row in rows if int(row.get("year") or 0) >= min_year]
    if max_year is not None:
        rows = [row for row in rows if int(row.get("year") or 9999) <= max_year]
    return rows


def download_pdf(arxiv_id: str, output_dir: str = None) -> str:
    """
    Download the PDF for a given arXiv ID.
    Returns the absolute path of the saved PDF.
    """
    if output_dir is None:
        output_dir = str(INPUT_PDFS_DIR)
    os.makedirs(output_dir, exist_ok=True)

    direct = _download_pdf_direct(arxiv_id, output_dir)
    if not direct.startswith("Error"):
        return direct

    client = arxiv.Client()
    search_obj = arxiv.Search(id_list=[arxiv_id])
    try:
        paper = next(client.results(search_obj))
    except StopIteration:
        return f"Error: arXiv ID '{arxiv_id}' not found."
    except Exception as e:
        return f"Error fetching arXiv paper: {e}; direct PDF fallback also failed: {direct}"

    try:
        filename = f"{arxiv_id}.pdf"
        saved_path = paper.download_pdf(dirpath=output_dir, filename=filename)
        return os.path.abspath(saved_path)
    except Exception as e:
        return f"Error downloading PDF: {e}; direct PDF fallback also failed: {direct}"


def _download_pdf_direct(arxiv_id: str, output_dir: str) -> str:
    """Download directly from arxiv.org/pdf when the API metadata path is flaky."""
    safe_id = str(arxiv_id).strip().removeprefix("arXiv:").split("/")[-1]
    url = f"https://arxiv.org/pdf/{safe_id}.pdf"
    filename = f"{safe_id}.pdf"
    output_path = os.path.abspath(os.path.join(output_dir, filename))
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not resp.content.startswith(b"%PDF"):
            return f"Error: direct arXiv response was not a PDF ({content_type})"
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return output_path
    except Exception as e:
        return f"Error direct-downloading arXiv PDF: {e}"


def get_info(arxiv_id: str) -> dict | str:
    """Get detailed metadata for a single arXiv paper."""
    client = arxiv.Client()
    search_obj = arxiv.Search(id_list=[arxiv_id])
    try:
        paper = next(client.results(search_obj))
    except StopIteration:
        return f"Error: arXiv ID '{arxiv_id}' not found."
    except Exception as e:
        return f"Error: {e}"
    return _format_result(paper)
