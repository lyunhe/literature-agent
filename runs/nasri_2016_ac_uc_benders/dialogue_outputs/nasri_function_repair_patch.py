"""Nasri 2016 dialogue-generated validation helper draft.

This patch is intentionally placed in dialogue_outputs. It demonstrates how
the multi-turn LLM workspace can turn a user request into a reviewable script
without overwriting the formal reproduction implementation.
"""
from __future__ import annotations

import csv
from pathlib import Path


OPTIONAL_TABLES = {
    "reserves.csv": "The implemented baseline can run without explicit reserves, but reserve assumptions should be reviewed for paper-style reporting.",
    "uncertainty_bounds.csv": "Wind/load uncertainty bounds can be regenerated from paper rules before final AC-Benders experiments.",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_table(data_dir: str | Path, filename: str) -> dict[str, object]:
    path = Path(data_dir) / filename
    rows = read_rows(path)
    return {
        "file": filename,
        "rows": len(rows),
        "status": "optional_candidate_needed" if not rows and filename in OPTIONAL_TABLES else ("ok" if rows else "missing_or_empty"),
        "note": OPTIONAL_TABLES.get(filename, ""),
    }


def validate_optional_assumptions(data_dir: str | Path) -> list[dict[str, object]]:
    """Return review items that the UI can show before finalizing assumptions."""
    return [summarize_table(data_dir, filename) for filename in OPTIONAL_TABLES]
