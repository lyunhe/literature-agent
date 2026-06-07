from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis_pipeline.core.common import save_json


def write_reviews_manifest(path: Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)
