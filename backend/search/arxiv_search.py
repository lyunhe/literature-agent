"""
arXiv search and PDF download.
Uses the `arxiv` package — no API key required.
"""
import os, json, arxiv
from backend.config import LIBRARY_PDF_DIR


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


def search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search arXiv by query string.
    Returns a list of paper metadata dicts.
    """
    client = arxiv.Client()
    search_obj = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    try:
        results = list(client.results(search_obj))
    except Exception as e:
        return [{"error": str(e)}]
    return [_format_result(r) for r in results]


def download_pdf(arxiv_id: str, output_dir: str = None) -> str:
    """
    Download the PDF for a given arXiv ID.
    Returns the absolute path of the saved PDF.
    """
    if output_dir is None:
        output_dir = LIBRARY_PDF_DIR
    os.makedirs(output_dir, exist_ok=True)

    client = arxiv.Client()
    search_obj = arxiv.Search(id_list=[arxiv_id])
    try:
        paper = next(client.results(search_obj))
    except StopIteration:
        return f"Error: arXiv ID '{arxiv_id}' not found."
    except Exception as e:
        return f"Error fetching arXiv paper: {e}"

    try:
        filename = f"{arxiv_id}.pdf"
        saved_path = paper.download_pdf(dirpath=output_dir, filename=filename)
        return os.path.abspath(saved_path)
    except Exception as e:
        return f"Error downloading PDF: {e}"


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
