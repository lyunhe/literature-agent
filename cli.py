"""
Interactive CLI for the literature agent.
Mirrors the demo's cli.py pattern with cmd.Cmd.
"""
import cmd, io, sys
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
            if p.get("pdf_path"):
                print(f"    PDF: {p['pdf_path']}")
            print()

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
            ("list [limit]",      "List saved papers in the local library"),
            ("info <id|doi>",     "Get detailed info for a specific paper"),
            ("download <id>",     "Download a paper PDF (auto-detects source)"),
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
