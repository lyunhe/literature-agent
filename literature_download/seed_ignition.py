from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from literature_download.doi_enrichment import enrich_papers_by_doi


def load_seed_papers(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    seed_path = Path(path)
    if not seed_path.exists():
        raise FileNotFoundError(f"种子文献文件不存在：{seed_path}")
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    rows = payload.get("papers", payload.get("seed_papers", payload)) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("种子文献 JSON 必须是数组，或包含 papers/seed_papers 数组。")
    seeds: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item.setdefault("candidate_id", f"S{index + 1:03d}")
        item.setdefault("source", "seed")
        item["seed_role"] = item.get("seed_role") or "seed"
        seeds.append(item)
    return seeds


def extract_seed_reference_candidates(seed_papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten references embedded in seed papers into candidate-paper rows."""
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    ref_index = 1
    for seed in seed_papers:
        seed_title = str(seed.get("title") or seed.get("seed_title") or "").strip()
        rows = seed.get("references") or seed.get("cited_references") or seed.get("reference_papers") or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, str):
                item = {"title": row}
            elif isinstance(row, dict):
                item = dict(row)
            else:
                continue
            key = (
                str(item.get("doi") or "").lower().strip()
                or str(item.get("arxiv_id") or "").lower().strip()
                or re.sub(r"\s+", " ", str(item.get("title") or "").lower()).strip()
            )
            if not key or key in seen:
                continue
            seen.add(key)
            item.setdefault("candidate_id", f"SR{ref_index:03d}")
            item.setdefault("source", "seed_reference")
            item["seed_reference"] = True
            item["seed_source_title"] = seed_title
            item.setdefault("seed_role", "reference")
            references.append(item)
            ref_index += 1
    return references


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text or "")
    stop = {
        "the", "and", "for", "with", "from", "into", "using", "based", "review",
        "survey", "paper", "study", "system", "systems", "method", "methods",
    }
    return [word for word in words if word.lower() not in stop]


def build_seed_context(seed_papers: list[dict[str, Any]], max_terms: int = 30) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    venues: Counter[str] = Counter()
    compact_papers: list[dict[str, Any]] = []
    for paper in seed_papers:
        title = str(paper.get("title") or "")
        abstract = str(paper.get("abstract") or "")
        venue = str(paper.get("venue") or "")
        concepts = paper.get("concepts") or []
        if venue:
            venues[venue] += 1
        for token in _tokenize(" ".join([title, abstract, " ".join(map(str, concepts))])):
            counter[token] += 1
        compact_papers.append(
            {
                "candidate_id": paper.get("candidate_id", ""),
                "title": title,
                "venue": venue,
                "year": paper.get("year"),
                "doi": paper.get("doi", ""),
                "concepts": concepts[:10] if isinstance(concepts, list) else [],
                "seed_role": paper.get("seed_role", "seed"),
                "abstract_snippet": abstract[:700],
                "reference_count": len(paper.get("references") or paper.get("cited_references") or []),
            }
        )
    return {
        "seed_papers": compact_papers,
        "seed_terms": [term for term, _ in counter.most_common(max_terms)],
        "seed_venues": [venue for venue, _ in venues.most_common(15)],
    }


def prepare_seed_ignition(
    seed_path: str | Path | None,
    output_dir: str | Path | None = None,
    *,
    enrich_doi: bool = True,
) -> dict[str, Any]:
    seeds = load_seed_papers(seed_path)
    seed_references = extract_seed_reference_candidates(seeds)
    reports: list[dict[str, Any]] = []
    if seeds and enrich_doi:
        seeds, reports = enrich_papers_by_doi(seeds, output_dir=None, fetch_landing_page=False)
    context = build_seed_context([*seeds, *seed_references])
    payload = {
        "seed_papers": seeds,
        "seed_reference_candidates": seed_references,
        "seed_context": context,
        "doi_enrichment": reports,
    }
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "seed_ignition.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload
