from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = [
    "test system",
    "modified ieee",
    "118-bus",
    "generator",
    "wind",
    "load",
    "uncertainty",
    "box uncertainty",
    "master problem",
    "subproblem",
    "column-and-constraint",
    "benders",
    "cutting-plane",
    "load-shift-factor",
    "solver",
    "cplex",
    "ampl",
    "computational",
    "table",
    "appendix",
]


def load_text_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_evidence(
    text_json: dict[str, Any],
    patterns: list[str] | None = None,
    context_chars: int = 650,
    max_snippets: int = 60,
    max_per_page: int = 8,
) -> list[dict[str, Any]]:
    patterns = patterns or DEFAULT_PATTERNS
    regex = re.compile("|".join(re.escape(p) for p in patterns), re.IGNORECASE)
    snippets: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    page_counts: dict[int, int] = {}
    for page in text_json.get("pages", []):
        text = page.get("text", "")
        for match in regex.finditer(text):
            page_number = page["page"]
            if page_counts.get(page_number, 0) >= max_per_page:
                continue
            start = max(0, match.start() - context_chars)
            end = min(len(text), match.end() + context_chars)
            key = (page_number, start // 200)
            if key in seen:
                continue
            seen.add(key)
            page_counts[page_number] = page_counts.get(page_number, 0) + 1
            snippets.append(
                {
                    "page": page_number,
                    "keyword": match.group(0),
                    "text": text[start:end].strip(),
                }
            )
            if len(snippets) >= max_snippets:
                return snippets
    return snippets


def evidence_as_prompt(snippets: list[dict[str, Any]], max_chars: int = 18000) -> str:
    chunks = []
    total = 0
    for item in snippets:
        block = f"[page {item['page']}; keyword: {item['keyword']}]\n{item['text']}"
        if total + len(block) > max_chars:
            break
        chunks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(chunks)
