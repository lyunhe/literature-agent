"""
Literature Agent tools — specifications + implementations.
The `tools` list is passed to the LLM so it can decide when to call each function.
Each function below is a real implementation that gets called when the LLM requests it.
"""
import json, os

# ============ Tool Specifications (passed to LLM) ============

tools = [
    {
        "type": "function",
        "function": {
            "name": "advanced_search",
            "description": (
                "Advanced multi-round literature search with LLM planning and iterative refinement. "
                "The LLM first expands the topic, identifies subtopics, likely venues and authors, "
                "then runs a first search round, scores the results, proposes follow-up queries, "
                "runs a second search round, and returns the best papers with relevance assessment. "
                "Use this for serious research tasks where you want exploration plus ranking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Research query (can be Chinese or English)"},
                    "final_results": {"type": "integer", "description": "Number of top papers to return after scoring", "default": 10},
                    "preferred_sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional source list such as ['openalex', 'arxiv', 'ieee']",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_openalex",
            "description": (
                "Search OpenAlex for scholarly works by free text, with optional venue/author hints. "
                "Useful for journal-oriented discovery, citation counts, DOI metadata, and high-quality venue screening."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum number of papers to return", "default": 5},
                    "venue": {"type": "string", "description": "Optional journal/conference name hint"},
                    "author": {"type": "string", "description": "Optional author name hint"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": (
                "Basic arXiv search - fast but less precise. "
                "Use for quick lookups or when you need many results. "
                "For important queries, prefer advanced_search instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":      {"type": "string",  "description": "Search query, e.g. 'transformer protein folding'"},
                    "max_results": {"type": "integer", "description": "Maximum number of papers to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_ieee",
            "description": (
                "Search IEEE Xplore digital library for papers by free-text query. "
                "Best for electrical engineering, computer science, and electronics. "
                "Requires IEEE API key in env.yaml."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum number of papers to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_doi",
            "description": (
                "Resolve a DOI to paper metadata via CrossRef. "
                "Use this when the user provides a DOI to look up paper details, "
                "or to find an arXiv ID linked to a DOI."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doi": {"type": "string", "description": "The DOI string (with or without https://doi.org/ prefix)"},
                },
                "required": ["doi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_pdf",
            "description": "Download the PDF of a paper by its arXiv ID or DOI into the local library folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier":  {"type": "string", "description": "arXiv ID (e.g. '2301.00001') or DOI"},
                    "source":      {"type": "string", "description": "'arxiv' or 'doi'", "default": "arxiv"},
                    "output_dir":  {"type": "string", "description": "Directory to save PDF", "default": "./library/pdfs"},
                },
                "required": ["identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_info",
            "description": "Get detailed metadata for a specific paper by arXiv ID or DOI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "arXiv ID or DOI"},
                    "source":     {"type": "string", "description": "'arxiv' or 'doi'", "default": "arxiv"},
                },
                "required": ["identifier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_library",
            "description": "List all papers saved in the local SQLite literature library.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum number of papers to return", "default": 50},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_library",
            "description": (
                "Save a paper's metadata to the local SQLite literature library. "
                "Call this after downloading a PDF or after reviewing search results "
                "to bookmark papers the user wants to keep."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title":    {"type": "string"},
                    "authors":  {"type": "string"},
                    "abstract": {"type": "string"},
                    "arxiv_id": {"type": "string"},
                    "doi":      {"type": "string"},
                    "ieee_id":  {"type": "string"},
                    "pdf_path": {"type": "string"},
                    "source":   {"type": "string", "description": "'arxiv' / 'ieee' / 'crossref'"},
                    "year":     {"type": "integer"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_library",
            "description": "Search the local literature library by keyword (title, abstract, or authors).",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                },
                "required": ["keyword"],
            },
        },
    },
]


# ============ Tool Implementations ============

def search_arxiv(query: str, max_results: int = 5) -> str:
    from .search import arxiv_search
    results = arxiv_search.search(query, max_results)
    return json.dumps(results, ensure_ascii=False, indent=2)


def search_ieee(query: str, max_results: int = 5) -> str:
    from .search import ieee_search
    results = ieee_search.search(query, max_results)
    return json.dumps(results, ensure_ascii=False, indent=2)


def search_openalex(query: str, max_results: int = 5, venue: str = "", author: str = "") -> str:
    from .search import openalex_search
    results = openalex_search.search(
        query,
        max_results=max_results,
        venue=venue or None,
        author=author or None,
    )
    return json.dumps(results, ensure_ascii=False, indent=2)


def resolve_doi(doi: str) -> str:
    from .search import crossref_search
    result = crossref_search.resolve_doi(doi)
    if isinstance(result, str):          # error string
        return result
    return json.dumps(result, ensure_ascii=False, indent=2)


def download_pdf(identifier: str, source: str = "arxiv", output_dir: str = "./library/pdfs") -> str:
    from .search import arxiv_search, crossref_search, ieee_search
    from .config import LIBRARY_PDF_DIR
    from .paths import normalize_library_path

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if source == "arxiv":
        result = arxiv_search.download_pdf(identifier, output_dir)
        return normalize_library_path(result) or result

    if source == "doi":
        meta = crossref_search.resolve_doi(identifier)
        if isinstance(meta, str):         # error
            return meta
        # Try arXiv if ID is available
        if meta.get("arxiv_id"):
            result = arxiv_search.download_pdf(meta["arxiv_id"], output_dir)
            return normalize_library_path(result) or result
        return json.dumps(meta, ensure_ascii=False, indent=2) + \
            "\n(No arXiv ID found — PDF not downloadable via DOI)"

    if source == "ieee":
        ieee_id = identifier
        filename = f"ieee_{ieee_id}.pdf"
        output_path = os.path.join(output_dir, filename)
        result = ieee_search.download_pdf(ieee_id, output_path)
        return normalize_library_path(result) or result

    return f"Error: unknown source '{source}'. Use 'arxiv', 'doi', or 'ieee'."


def get_paper_info(identifier: str, source: str = "arxiv") -> str:
    from .search import arxiv_search, crossref_search, ieee_search

    if source == "arxiv":
        result = arxiv_search.get_info(identifier)
    elif source == "doi":
        result = crossref_search.resolve_doi(identifier)
    elif source == "ieee":
        result = ieee_search.get_info(identifier)
    else:
        return f"Error: unknown source '{source}'"

    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, indent=2)


def list_library(limit: int = 50) -> str:
    from . import db
    papers = db.list_papers(limit=limit)
    if not papers:
        return "Library is empty. Use search_arxiv / search_ieee then save_to_library."
    return json.dumps(papers, ensure_ascii=False, indent=2)


def save_to_library(
    title: str, authors: str = "", abstract: str = "",
    arxiv_id: str = None, doi: str = None, ieee_id: str = None,
    pdf_path: str = None, source: str = "unknown", year: int = None
) -> str:
    from . import db
    paper = {
        "title":    title,
        "authors":  authors,
        "abstract": abstract,
        "arxiv_id": arxiv_id,
        "doi":      doi,
        "ieee_id":  ieee_id,
        "pdf_path": pdf_path,
        "source":   source,
        "year":     year,
    }
    return db.add_paper(paper)


def search_library(keyword: str) -> str:
    from . import db
    papers = db.search_papers(keyword)
    if not papers:
        return f"No papers found in library matching '{keyword}'."
    return json.dumps(papers, ensure_ascii=False, indent=2)


def clean_library_paths() -> str:
    from . import db
    return json.dumps(db.normalize_pdf_paths(), ensure_ascii=False, indent=2)


def advanced_search(query: str, final_results: int = 10, preferred_sources: list[str] = None) -> str:
    """
    Multi-round search with LLM planning, scoring, and iterative refinement.
    Returns a structured report with search plans, round diagnostics, and top papers.
    """
    from .advanced_search import multi_round_search

    results = multi_round_search(
        user_query=query,
        final_results=final_results,
        preferred_sources=preferred_sources,
    )

    return json.dumps(results, ensure_ascii=False, indent=2)
