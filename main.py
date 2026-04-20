"""
Command-line entry point with argparse sub-commands.
Mirrors the demo's main.py pattern.
"""
import argparse, sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backend import research, db, lit_tools


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
        pdf = f"  PDF: {p['pdf_path']}" if p.get("pdf_path") else ""
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


def cmd_run(args):
    """Full multi-turn agent session (non-interactive)."""
    response = research(args.prompt)
    print(response)


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

    args = parser.parse_args()

    if args.command == "search":
        cmd_search(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
