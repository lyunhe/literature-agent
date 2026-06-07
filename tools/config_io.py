from __future__ import annotations

from pathlib import Path
from typing import Any


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load the small YAML subset used by target configs.

    This intentionally avoids a PyYAML dependency. It supports top-level scalars
    and top-level lists introduced by `key:` followed by `  - value` lines.
    """
    result: dict[str, Any] = {}
    current_key: str | None = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - "):
            if current_key is None:
                raise ValueError(f"List item without key in {path}: {line}")
            result.setdefault(current_key, []).append(_parse_scalar(line[4:]))
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line in {path}: {line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_key = key
        if value:
            result[key] = _parse_scalar(value)
        else:
            result[key] = []
    return result


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

