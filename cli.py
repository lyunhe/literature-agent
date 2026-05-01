"""
Interactive CLI for the literature agent.
Mirrors the demo's cli.py pattern with cmd.Cmd.
"""
import cmd, io, sys
import json
import shlex
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from backend import research


class LiteratureCLI(cmd.Cmd):
    intro = (
        "Literature Agent CLI — powered by LLM + arXiv / IEEE Xplore.\n"
        "Type 'help' for available commands.\n"
    )
    prompt = "(lit) "

    messages = None   # shared conversation history for multi-turn session

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def do_search(self, arg):
        """search <query>  — run a research query through the LLM agent"""
        if not arg.strip():
            print("Usage: search <query>")
            return
        print(f"\n[Searching for: {arg}]\n")
        response = research(arg, self.messages)
        print(response)
        # research() mutates self.messages in-place, no need to reassign

    def do_list(self, arg):
        """list [limit]  — list papers saved in the local library"""
        from backend import db
        limit = int(arg.strip()) if arg.strip().isdigit() else 50
        papers = db.list_papers(limit=limit)
        if not papers:
            print("Library is empty.")
            return
        for p in papers:
            yr = p.get("year") or ""
            src = p.get("source") or ""
            print(f"  [{src}] {yr} — {p['title'][:70]}")
            print(f"    Authors: {p.get('authors','')[:80]}")
            pdf_value = p.get("pdf_abs_path") or p.get("pdf_path")
            if pdf_value:
                print(f"    PDF: {pdf_value}")
            print()

    def do_clean_paths(self, arg):
        """clean_paths  — normalize stored PDF paths for this checkout"""
        from backend import db
        print(json.dumps(db.normalize_pdf_paths(), ensure_ascii=False, indent=2))

    def do_info(self, arg):
        """info <arxiv_id|doi>  — get detailed info for a specific paper"""
        if not arg.strip():
            print("Usage: info <arxiv_id|doi>")
            return
        identifier = arg.strip()
        # auto-detect source
        source = "doi" if identifier.startswith("10.") else "arxiv"
        from backend import lit_tools
        result = lit_tools.get_paper_info(identifier, source)
        print(result)

    def do_download(self, arg):
        """download <arxiv_id|doi> [--source arxiv|doi|ieee]  — download a paper PDF"""
        parts = arg.strip().split()
        if not parts:
            print("Usage: download <identifier> [--source arxiv|doi|ieee]")
            return
        identifier = parts[0]
        source = "arxiv"
        if "--source" in parts:
            idx = parts.index("--source")
            source = parts[idx + 1] if idx + 1 < len(parts) else "arxiv"
        from backend import lit_tools
        result = lit_tools.download_pdf(identifier, source=source)
        print(result)

    def do_guided_search(self, arg):
        """guided_search <query> [--sources openalex,arxiv]  — plan domains/venues/authors and search"""
        parts = shlex.split(arg)
        if not parts:
            print("Usage: guided_search <query> [--sources openalex,arxiv]")
            return
        sources = "openalex,arxiv"
        if "--sources" in parts:
            idx = parts.index("--sources")
            if idx + 1 < len(parts):
                sources = parts[idx + 1]
                del parts[idx:idx + 2]
        query = " ".join(parts).strip()
        if not query:
            print("Usage: guided_search <query> [--sources openalex,arxiv]")
            return
        from backend.advanced_search import multi_round_search
        result = multi_round_search(
            user_query=query,
            per_query_limit=3,
            first_round_queries=3,
            second_round_queries=1,
            final_results=5,
            preferred_sources=[s.strip() for s in sources.split(",") if s.strip()],
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    def do_graph(self, arg):
        """graph [--llm] [--citations] [--pdf]  — generate the literature relationship graph"""
        parts = shlex.split(arg)
        from backend.lit_graph import export_similarity_graph
        summary = export_similarity_graph(
            threshold=0.05,
            top_k=4,
            infer_llm="--llm" in parts,
            include_citations="--citations" in parts,
            use_pdf="--pdf" in parts,
            max_pdf_relations=2,
            max_llm_relations=8,
        )
        print(f"Graph HTML: {summary['html_path']}")
        print(f"Graph JSON: {summary['json_path']}")
        print(f"Nodes: {summary['nodes']}")
        print(f"Edges: {summary['edges']}")
        print(f"Relation types: {', '.join(summary['relation_types'])}")

    def do_analyze_pdf(self, arg):
        """analyze_pdf <pdf_path> <prompt>  — analyze a local PDF with the configured LLM"""
        parts = shlex.split(arg)
        if len(parts) < 2:
            print("Usage: analyze_pdf <pdf_path> <prompt>")
            return
        from backend.llm_client import analyze_pdf
        print(analyze_pdf(parts[0], " ".join(parts[1:])))

    def do_run(self, arg):
        """run <prompt>  — full multi-turn agent interaction (preserves context)"""
        if not arg.strip():
            print("Usage: run <prompt>")
            return
        print(f"\n[Running agent...]\n")
        response = research(arg, self.messages)
        print(response)

    def do_clear(self, arg):
        """clear  — reset conversation history"""
        self.messages = None
        print("Conversation history cleared.")

    def do_exit(self, arg):
        """exit  — quit the CLI"""
        print("Goodbye!")
        return True

    def do_EOF(self, arg):
        """Exit on Ctrl+D"""
        print()
        return True

    # ------------------------------------------------------------------
    # Completions
    # ------------------------------------------------------------------
    def do_help(self, arg):
        CMDS = [
            ("search <query>",    "Run a research query through the LLM agent"),
            ("guided_search <query>", "Plan domains/venues/authors and run source-aware retrieval"),
            ("list [limit]",      "List saved papers in the local library"),
            ("clean_paths",       "Normalize stored PDF paths"),
            ("info <id|doi>",     "Get detailed info for a specific paper"),
            ("download <id>",     "Download a paper PDF (auto-detects source)"),
            ("analyze_pdf <pdf> <prompt>", "Analyze a local PDF via the LLM API"),
            ("graph [--llm --citations --pdf]", "Generate a literature relationship graph"),
            ("run <prompt>",      "Full multi-turn agent session"),
            ("clear",             "Reset conversation history"),
            ("exit / Ctrl+D",     "Quit"),
        ]
        if not arg.strip():
            print("\nAvailable commands:")
            for cmd, desc in CMDS:
                print(f"  {cmd:<22} — {desc}")
            print()
        else:
            super().do_help(arg)


if __name__ == "__main__":
    LiteratureCLI().cmdloop()
