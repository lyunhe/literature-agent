from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis_pipeline.core.common import load_json


def load_assigned_papers(direction_dir: Path) -> dict[str, Any]:
    return load_json(direction_dir / "assigned_papers.json")
