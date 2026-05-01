from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_DIR = PROJECT_ROOT / "library"
LIBRARY_PDF_DIR = LIBRARY_DIR / "pdfs"
DB_PATH = LIBRARY_DIR / "library.db"


def ensure_library_dirs() -> None:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_PDF_DIR.mkdir(parents=True, exist_ok=True)


def _portable_parts(path: str) -> list[str]:
    return [part for part in path.replace("\\", "/").split("/") if part]


def display_path(path: str | os.PathLike | None) -> str:
    """Return an absolute path that is useful for CLI display on this machine."""
    resolved = resolve_library_path(path)
    if resolved:
        return str(resolved)
    return str(path or "")


def normalize_library_path(path: str | os.PathLike | None) -> str | None:
    """
    Store paths relative to library/ when possible.

    This keeps the SQLite database portable across macOS, Linux, and Windows
    clones while still accepting old absolute paths copied from another system.
    """
    if not path:
        return None

    raw = str(path).strip()
    if not raw:
        return None

    resolved = resolve_library_path(raw)
    if resolved:
        try:
            return resolved.relative_to(LIBRARY_DIR).as_posix()
        except ValueError:
            return str(resolved)

    parts = _portable_parts(raw)
    if "library" in parts:
        idx = len(parts) - 1 - parts[::-1].index("library")
        rel = "/".join(parts[idx + 1 :])
        return rel or None
    if "pdfs" in parts:
        idx = len(parts) - 1 - parts[::-1].index("pdfs")
        return "/".join(parts[idx:])
    if len(parts) == 1 and parts[0].lower().endswith(".pdf"):
        return f"pdfs/{parts[0]}"

    return raw


def resolve_library_path(path: str | os.PathLike | None) -> Path | None:
    """Resolve portable and legacy library paths to an existing local file."""
    if not path:
        return None

    raw = str(path).strip()
    if not raw:
        return None

    candidates: list[Path] = []
    as_path = Path(raw).expanduser()
    candidates.append(as_path)

    if not as_path.is_absolute():
        candidates.append(LIBRARY_DIR / raw)
        candidates.append(PROJECT_ROOT / raw)

    parts = _portable_parts(raw)
    if "library" in parts:
        idx = len(parts) - 1 - parts[::-1].index("library")
        suffix = Path(*parts[idx + 1 :]) if parts[idx + 1 :] else None
        if suffix:
            candidates.append(LIBRARY_DIR / suffix)
    if "pdfs" in parts:
        idx = len(parts) - 1 - parts[::-1].index("pdfs")
        suffix = Path(*parts[idx:])
        candidates.append(LIBRARY_DIR / suffix)
    if parts:
        filename = parts[-1]
        if filename.lower().endswith(".pdf"):
            candidates.append(LIBRARY_PDF_DIR / filename)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    return None


def pdf_candidates(arxiv_id: str | None = None, pdf_path: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    resolved = resolve_library_path(pdf_path)
    if resolved:
        candidates.append(resolved)
    if pdf_path:
        parts = _portable_parts(pdf_path)
        if parts:
            candidates.append(LIBRARY_PDF_DIR / parts[-1])
    if arxiv_id:
        arxiv = str(arxiv_id)
        candidates.append(LIBRARY_PDF_DIR / f"{arxiv}.pdf")
        candidates.append(LIBRARY_PDF_DIR / f"{arxiv.split('v')[0]}.pdf")

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique
