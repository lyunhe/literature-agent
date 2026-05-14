"""
Literature Agent — Agent Loop.
Mirrors the demo's backend/api_interface.py pattern:
  user prompt → LLM (with tools) → tool call loop → final answer.
"""
import json
import time
from .llm_client import llm_request
from . import lit_tools, db

# Tool list passed to the LLM so it knows what it can call
lit_tools_list = lit_tools.tools

# Initialise SQLite library on import
db.init_db()


def research(prompt: str, messages: list = None) -> str:
    """
    Multi-turn Agent Loop:
      1. Append user message to history
      2. Ask LLM with full tool list
      3. If LLM calls a tool → execute it, feed result back, ask again
      4. If LLM returns text → done
    """
    if messages is None:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research librarian powered by a large language model. "
                    "Your job is to help the user find, download, and organise academic papers. "
                    "Available actions (use the tools below):\n"
                    "- advanced_search: preferred for substantive literature review tasks because it performs multi-round planning, scoring, and refinement\n"
                    "- search_arxiv: basic arXiv search for quick lookups\n"
                    "- search_ieee: basic IEEE Xplore search\n"
                    "- resolve_doi: look up paper metadata from a DOI\n"
                    "- download_pdf: download a paper PDF to the local library\n"
                    "- get_paper_info: get detailed metadata for a specific paper\n"
                    "- list_library: show all papers saved locally\n"
                    "- save_to_library: bookmark a paper's metadata in the SQLite library\n"
                    "- search_library: search saved papers by keyword\n"
                    "For broad or important topics, start with advanced_search before deciding on further tool calls. "
                    "Use the returned search plan, venue hints, author hints, and relevance scores when summarising the results. "
                    "Always offer to download relevant PDFs and save them to the library."
                ),
            }
        ]

    messages.append({"role": "user", "content": prompt})

    while True:
        for attempt in range(5):
            try:
                resp = llm_request(messages, tools=lit_tools_list)
                break
            except Exception as e:
                if attempt < 4:
                    time.sleep(5)
                    continue
                raise

        msg = resp.choices[0]

        if msg.finish_reason in ("tool_calls", "tool_use"):
            for tc in msg.message.tool_calls:
                func_name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"\n[Agent] Calling tool: {func_name}({args})")
                try:
                    impl = getattr(lit_tools, func_name)
                    result = impl(**args)
                except Exception as e:
                    result = f"Tool execution error: {e}"
                print(f"[Agent] Result: {str(result)[:300]}{'...' if len(str(result)) > 300 else ''}")
                messages.append(msg.message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })
            continue

        # LLM returned a text answer
        messages.append(msg.message)
        return msg.message.content.strip()
