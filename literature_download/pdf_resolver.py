from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests

from backend.paths import LIBRARY_PDF_DIR, normalize_library_path
from literature_download import arxiv_search, ieee_search


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
UNPAYWALL_URL = "https://api.unpaywall.org/v2"


def _clean_doi(doi: str | None) -> str:
    return (doi or "").strip().removeprefix("https://doi.org/").removeprefix("doi:")


def _candidate(
    url: str,
    source: str,
    access: str,
    *,
    note: str = "",
    priority: int = 100,
) -> dict[str, Any]:
    return {
        "url": url,
        "source": source,
        "access": access,
        "note": note,
        "priority": priority,
    }


def _safe_pdf_name(paper: dict[str, Any]) -> str:
    raw = str(
        paper.get("arxiv_id")
        or paper.get("doi")
        or paper.get("ieee_id")
        or paper.get("openalex_id")
        or paper.get("title")
        or "paper"
    )
    raw = raw.removeprefix("https://doi.org/")
    cleaned = re.sub(r"[^\w.-]+", "_", raw, flags=re.UNICODE).strip("_")
    return (cleaned or "paper")[:120] + ".pdf"


def _existing_local_candidates(paper: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    stems = []
    for key in ("arxiv_id", "doi", "ieee_id"):
        value = str(paper.get(key) or "").strip()
        if value:
            stems.append(re.sub(r"[^\w.-]+", "_", value.removeprefix("https://doi.org/"), flags=re.UNICODE).strip("_"))
    title = str(paper.get("title") or "").strip()
    if title:
        stems.append(re.sub(r"[^\w.-]+", "_", title, flags=re.UNICODE).strip("_")[:90])

    candidates = []
    for directory in [output_dir, Path(LIBRARY_PDF_DIR)]:
        if not directory.exists():
            continue
        for pdf in directory.glob("*.pdf"):
            name = pdf.name.lower()
            if any(stem and stem.lower() in name for stem in stems):
                candidates.append(_candidate(str(pdf.resolve()), "local", "local", note="Existing local PDF", priority=0))
    return candidates


def _openalex_candidates(paper: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    if paper.get("oa_url"):
        candidates.append(_candidate(str(paper["oa_url"]), "openalex", "open", note="Search result oa_url", priority=20))
    if paper.get("pdf_url"):
        candidates.append(_candidate(str(paper["pdf_url"]), "openalex", "open", note="Search result pdf_url", priority=20))

    clean_doi = _clean_doi(paper.get("doi"))
    params: dict[str, Any] = {"per-page": 1}
    if clean_doi:
        params["filter"] = f"doi:{clean_doi}"
    elif paper.get("title"):
        params["search"] = str(paper["title"])
    else:
        return candidates

    resp = requests.get(OPENALEX_WORKS_URL, params=params, timeout=20)
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        return candidates
    work = results[0]
    oa = work.get("open_access") or {}
    if oa.get("oa_url"):
        candidates.append(_candidate(oa["oa_url"], "openalex", "open", note="open_access.oa_url", priority=25))
    locations = []
    if work.get("primary_location"):
        locations.append(work["primary_location"])
    locations.extend(work.get("locations") or [])
    for location in locations:
        if location.get("pdf_url"):
            candidates.append(_candidate(location["pdf_url"], "openalex", "open", note="location.pdf_url", priority=30))
        if location.get("landing_page_url"):
            candidates.append(_candidate(location["landing_page_url"], "openalex", "landing-page", note="Publisher landing page", priority=80))
    return candidates


def _unpaywall_candidates(paper: dict[str, Any]) -> list[dict[str, Any]]:
    doi = _clean_doi(paper.get("doi"))
    email = os.getenv("UNPAYWALL_EMAIL")
    if not doi or not email:
        return []
    resp = requests.get(f"{UNPAYWALL_URL}/{doi}", params={"email": email}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    candidates = []
    best = data.get("best_oa_location") or {}
    if best.get("url_for_pdf"):
        candidates.append(_candidate(best["url_for_pdf"], "unpaywall", "open", note="best_oa_location.url_for_pdf", priority=35))
    if best.get("url"):
        candidates.append(_candidate(best["url"], "unpaywall", "landing-page", note="best_oa_location.url", priority=82))
    for location in data.get("oa_locations") or []:
        if location.get("url_for_pdf"):
            candidates.append(_candidate(location["url_for_pdf"], "unpaywall", "open", note="oa_locations.url_for_pdf", priority=40))
    return candidates


def _crossref_candidates(paper: dict[str, Any]) -> list[dict[str, Any]]:
    doi = _clean_doi(paper.get("doi"))
    if not doi:
        return []
    resp = requests.get(f"{CROSSREF_WORKS_URL}/{doi}", headers={"Accept": "application/json"}, timeout=20)
    resp.raise_for_status()
    message = resp.json().get("message") or {}
    candidates = []
    for link in message.get("link") or []:
        url = link.get("URL") or link.get("url")
        content_type = str(link.get("content-type") or "").lower()
        if not url:
            continue
        if "pdf" in content_type:
            candidates.append(_candidate(url, "crossref", "open-or-unknown", note=f"Crossref link {content_type}", priority=45))
        else:
            candidates.append(_candidate(url, "crossref", "landing-page", note=f"Crossref link {content_type}", priority=90))
    if message.get("URL"):
        candidates.append(_candidate(message["URL"], "crossref", "landing-page", note="Crossref work URL", priority=95))
    return candidates


def _ieee_candidates(paper: dict[str, Any]) -> list[dict[str, Any]]:
    ieee_id = str(paper.get("ieee_id") or "").strip()
    if not ieee_id:
        return []
    candidates = []
    if paper.get("pdf_url"):
        candidates.append(_candidate(str(paper["pdf_url"]), "ieee", "restricted-or-open", note="IEEE search pdf_url", priority=60))
    candidates.append(_candidate(
        f"https://ieeexplore.ieee.org/document/{ieee_id}",
        "ieee",
        "landing-page",
        note="IEEE Xplore document page for browser-assisted download",
        priority=85,
    ))
    if ieee_id:
        candidates.append(_candidate(
            f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={ieee_id}",
            "ieee",
            "restricted-or-open",
            note="IEEE stamp URL; requires authorized access for many articles",
            priority=65,
        ))
    return candidates


def resolve_pdf_candidates(paper: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    output_path = Path(output_dir or LIBRARY_PDF_DIR)
    candidates: list[dict[str, Any]] = []
    diagnostics: list[dict[str, str]] = []

    candidates.extend(_existing_local_candidates(paper, output_path))
    if paper.get("arxiv_id"):
        safe_id = str(paper["arxiv_id"]).strip().removeprefix("arXiv:").split("/")[-1]
        candidates.append(_candidate(f"https://arxiv.org/pdf/{safe_id}.pdf", "arxiv", "open", note="arXiv direct PDF", priority=10))

    for source, resolver in [
        ("openalex", _openalex_candidates),
        ("unpaywall", _unpaywall_candidates),
        ("crossref", _crossref_candidates),
        ("ieee", _ieee_candidates),
    ]:
        try:
            candidates.extend(resolver(paper))
        except requests.RequestException as exc:
            diagnostics.append({"source": source, "error": str(exc)})

    deduped = []
    seen = set()
    for item in sorted(candidates, key=lambda c: c.get("priority", 100)):
        url = str(item.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(item)
    return {
        "paper_key": paper.get("candidate_id") or paper.get("doi") or paper.get("arxiv_id") or paper.get("title"),
        "title": paper.get("title") or "",
        "doi": paper.get("doi") or "",
        "ieee_id": paper.get("ieee_id") or "",
        "candidates": deduped,
        "diagnostics": diagnostics,
    }


def _download_url(url: str, output_path: Path) -> tuple[bool, str]:
    try:
        resp = requests.get(url, timeout=80, allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return False, f"request failed: {exc}"
    content_type = resp.headers.get("content-type", "")
    if "pdf" not in content_type.lower() and not resp.content.startswith(b"%PDF"):
        return False, f"not a PDF response ({content_type})"
    output_path.write_bytes(resp.content)
    return True, "downloaded"


def download_best_pdf(
    paper: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    allow_restricted_direct: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir or LIBRARY_PDF_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    resolution = resolve_pdf_candidates(paper, output_path)
    attempts = []

    for candidate in resolution["candidates"]:
        access = candidate.get("access")
        url = str(candidate.get("url") or "")
        if access == "local" and Path(url).exists():
            return {
                "status": "local_exists",
                "pdf_path": url,
                "normalized_pdf_path": normalize_library_path(url),
                "resolution": resolution,
                "attempts": attempts,
                "manual_required": False,
            }
        if access not in {"open", "open-or-unknown"} and not allow_restricted_direct:
            continue
        target = output_path / _safe_pdf_name(paper)
        ok, message = _download_url(url, target)
        attempts.append({**candidate, "status": "success" if ok else "failed", "message": message})
        if ok:
            return {
                "status": "downloaded",
                "pdf_path": str(target.resolve()),
                "normalized_pdf_path": normalize_library_path(target),
                "resolution": resolution,
                "attempts": attempts,
                "manual_required": False,
            }

    manual_candidates = [
        c for c in resolution["candidates"]
        if c.get("access") in {"landing-page", "restricted-or-open"}
    ]
    return {
        "status": "manual_required" if manual_candidates else "failed",
        "pdf_path": "",
        "normalized_pdf_path": "",
        "resolution": resolution,
        "attempts": attempts,
        "manual_required": bool(manual_candidates),
        "manual_candidates": manual_candidates,
    }


def save_download_reports(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "download_attempts.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manual = [item for item in results if item.get("manual_required")]
    (output_dir / "manual_download_queue.json").write_text(
        json.dumps(manual, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
