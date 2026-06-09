"""Sync paper_table.csv from paper_table.json (single table only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis_pipeline.stages.discovery.paper_table import save_paper_table


def main() -> None:
    disc = Path(sys.argv[1]).resolve()
    rows = json.loads((disc / "paper_table.json").read_text(encoding="utf-8"))
    save_paper_table(rows, disc)
    success = sum(1 for row in rows if row.get("download_status") == "success")
    failed = sum(1 for row in rows if row.get("download_status") == "failed")
    print(f"Updated {disc / 'paper_table.csv'}: success={success}, failed={failed}")


if __name__ == "__main__":
    main()
