"""
Command-line entry point with argparse sub-commands.
Mirrors the demo's main.py pattern.
"""
import argparse, sys
import io
import json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backend import research, db, lit_tools
from backend.llm_client import analyze_pdf


def cmd_search(args):
    """One-shot search + optional download via LLM agent."""
    prompt = args.query
    if args.source:
        prompt += f" (focus on {args.source} if available)"
    if args.max:
        prompt += f" (return up to {args.max} results)"
    if args.download:
        prompt += " and download PDFs for the most relevant ones."
    response = research(prompt)
    print(response)


def cmd_list(args):
    """List papers in local library."""
    papers = db.list_papers(limit=args.limit or 50)
    if not papers:
        print("Library is empty.")
        return
    for p in papers:
        yr = p.get("year") or ""
        src = p.get("source") or ""
        pdf_value = p.get("pdf_abs_path") or p.get("pdf_path")
        pdf = f"  PDF: {pdf_value}" if pdf_value else ""
        print(f"[{src}] {yr} — {p['title'][:70]}")
        print(f"  Authors: {p.get('authors','')[:80]}{pdf}\n")


def cmd_info(args):
    """Get detailed paper info."""
    identifier = args.identifier
    source = "doi" if identifier.startswith("10.") else args.source or "arxiv"
    result = lit_tools.get_paper_info(identifier, source)
    print(result)


def cmd_download(args):
    """Download paper PDF."""
    identifier = args.identifier
    source = args.source or ("doi" if identifier.startswith("10.") else "arxiv")
    result = lit_tools.download_pdf(identifier, source=source)
    print(result)


def cmd_clean_paths(args):
    """Normalize stored library file paths for the current project checkout."""
    from backend import db

    summary = db.normalize_pdf_paths()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_run(args):
    """Full multi-turn agent session (non-interactive)."""
    response = research(args.prompt)
    print(response)


def cmd_analyze_pdf(args):
    """Analyze a local PDF with the configured LLM API."""
    print(analyze_pdf(args.pdf_path, args.prompt))


def cmd_graph(args):
    """Build a simple literature relationship graph."""
    from backend.lit_graph import export_similarity_graph

    summary = export_similarity_graph(
        output_path=args.output,
        limit=args.limit,
        threshold=args.threshold,
        top_k=args.top_k,
        infer_llm=args.llm,
        max_llm_relations=args.max_llm_relations,
        include_citations=args.citations,
        use_pdf=args.pdf,
        max_pdf_relations=args.max_pdf_relations,
    )
    print(f"Graph HTML: {summary['html_path']}")
    print(f"Graph JSON: {summary['json_path']}")
    print(f"Nodes: {summary['nodes']}")
    print(f"Edges: {summary['edges']}")
    if summary["relations"]:
        print("Top relations:")
        for rel in summary["relations"][:10]:
            print(
                f"  {rel['weight']:.3f} | {rel['type']} | {rel['source_title'][:48]} "
                f"<-> {rel['target_title'][:48]}"
            )


def cmd_guided_search(args):
    """Run LLM-planned source/venue/author-aware literature discovery."""
    from backend.advanced_search import multi_round_search

    sources = [s.strip() for s in (args.sources or "").split(",") if s.strip()] or None
    result = multi_round_search(
        user_query=args.query,
        per_query_limit=args.per_query_limit,
        first_round_queries=args.first_round_queries,
        second_round_queries=args.second_round_queries,
        final_results=args.final_results,
        preferred_sources=sources,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Literature Agent — search, download, and manage academic papers."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # search
    p = subparsers.add_parser("search", help="Run a research query via the LLM agent")
    p.add_argument("query", help="Research query")
    p.add_argument("--source", choices=["arxiv", "ieee"], help="Preferred source")
    p.add_argument("--max", type=int, help="Max results to consider")
    p.add_argument("--download", action="store_true", help="Also download PDFs")

    # list
    p = subparsers.add_parser("list", help="List papers in the local library")
    p.add_argument("--limit", type=int, help="Max papers to show (default: 50)")

    # clean-paths
    subparsers.add_parser("clean-paths", help="Normalize stored PDF paths in the local library")

    # info
    p = subparsers.add_parser("info", help="Get paper metadata")
    p.add_argument("identifier", help="arXiv ID, DOI, or IEEE article number")
    p.add_argument("--source", choices=["arxiv", "doi", "ieee"], help="Source type")

    # download
    p = subparsers.add_parser("download", help="Download paper PDF")
    p.add_argument("identifier", help="arXiv ID, DOI, or IEEE article number")
    p.add_argument("--source", choices=["arxiv", "doi", "ieee"], help="Source type")

    # run
    p = subparsers.add_parser("run", help="Full multi-turn agent (preserves context)")
    p.add_argument("prompt", help="Research prompt")

    # analyze-pdf
    p = subparsers.add_parser("analyze-pdf", help="Analyze a local PDF via the LLM API")
    p.add_argument("pdf_path", help="Path to local PDF")
    p.add_argument("prompt", help="Question or analysis instruction")

    # graph
    p = subparsers.add_parser("graph", help="Generate a simple literature relationship graph")
    p.add_argument("--output", help="Output HTML path")
    p.add_argument("--limit", type=int, default=50, help="Max papers from local library")
    p.add_argument("--threshold", type=float, default=0.18, help="Minimum similarity edge weight")
    p.add_argument("--top-k", type=int, default=3, help="Max edges kept per paper")
    p.add_argument("--llm", action="store_true", help="Use the configured LLM to classify candidate relations")
    p.add_argument("--max-llm-relations", type=int, default=12, help="Max candidate edges sent to the LLM")
    p.add_argument("--citations", action="store_true", help="Use OpenAlex to enrich graph with real citation data")
    p.add_argument("--pdf", action="store_true", help="Let the LLM read local PDFs for high-value candidate relations")
    p.add_argument("--max-pdf-relations", type=int, default=3, help="Max candidate edges analyzed with PDF input")

    # guided-search
    p = subparsers.add_parser("guided-search", help="Plan search domains, venues, authors, and run source-aware retrieval")
    p.add_argument("query", help="Research query or download/search intent")
    p.add_argument("--sources", default="openalex,arxiv,ieee", help="Comma-separated sources: openalex,arxiv,ieee")
    p.add_argument("--per-query-limit", type=int, default=5, help="Max results per generated query")
    p.add_argument("--first-round-queries", type=int, default=5, help="Number of first-round planned queries")
    p.add_argument("--second-round-queries", type=int, default=3, help="Number of refined queries")
    p.add_argument("--final-results", type=int, default=8, help="Number of final scored papers")

    args = parser.parse_args()

    if args.command == "search":
        cmd_search(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "clean-paths":
        cmd_clean_paths(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "analyze-pdf":
        cmd_analyze_pdf(args)
    elif args.command == "graph":
        cmd_graph(args)
    elif args.command == "guided-search":
        cmd_guided_search(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
