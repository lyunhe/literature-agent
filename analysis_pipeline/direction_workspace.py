from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from analysis_pipeline.pipeline_common import (
    ensure_dir,
    extract_pdf_metadata,
    load_json,
    safe_output_stem,
    safe_plain_stem,
    save_json,
)


def direction_dir_name(direction: dict[str, Any]) -> str:
    direction_id = str(direction.get("direction_id") or "DX").strip()
    name = str(direction.get("direction_name_cn") or direction.get("direction_name") or direction_id)
    return safe_output_stem(f"{direction_id}_{name}", max_base_len=48)


def find_manifest_for_pdf(pdf_path: Path, figures_dir: Path | None) -> Path | None:
    if figures_dir is None or not figures_dir.exists():
        return None
    direct = figures_dir / safe_plain_stem(pdf_path.stem) / "manifest.json"
    if direct.exists():
        return direct
    raw_stem = pdf_path.stem
    for subdir in figures_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name.startswith(raw_stem) or raw_stem in subdir.name:
            manifest_path = subdir / "manifest.json"
            if manifest_path.exists():
                return manifest_path
    return None


def txt_path_for_pdf(pdf_path: Path, txt_dir: Path) -> Path:
    return txt_dir / f"{safe_output_stem(pdf_path.stem)}.txt"


def _candidate_pdf_path(candidate: dict[str, Any], pdf_dir: Path) -> Path | None:
    raw = candidate.get("_pdf_path") or candidate.get("pdf_path")
    if raw:
        name = Path(str(raw)).name
        copied = pdf_dir / name
        if copied.exists():
            return copied.resolve()
        original = Path(str(raw))
        if original.exists():
            return original.resolve()
    title = str(candidate.get("title") or "").strip()
    if title:
        title_key = re.sub(r"\W+", "", title.lower())
        for path in pdf_dir.glob("*.pdf"):
            if title_key and title_key[:40] in re.sub(r"\W+", "", path.stem.lower()):
                return path.resolve()
    return None


def build_local_pdf_candidates(pdf_files: list[Path]) -> list[dict[str, Any]]:
    return [extract_pdf_metadata(path.resolve(), f"P{idx + 1:03d}") for idx, path in enumerate(pdf_files)]


def build_virtual_single_direction_state(topic: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(item["candidate_id"]) for item in candidates]
    return {
        "topic": topic,
        "input_mode": "user_single_direction",
        "papers": candidates,
        "directions": [
            {
                "direction_id": "D1",
                "direction_name_cn": topic,
                "direction_name_en": "",
                "direction_summary_cn": "用户指定所有 PDF 属于同一研究方向。",
                "display_keywords": [],
                "paper_ids": ids,
                "papers": [
                    {
                        "candidate_id": item["candidate_id"],
                        "title": item.get("title", ""),
                        "title_cn": item.get("title_cn", ""),
                        "venue": item.get("venue", ""),
                        "year": item.get("year"),
                        "source": item.get("source", "local_pdf"),
                        "method_or_object_summary_cn": "",
                        "assignment_reason_cn": "用户指定单方向分析。",
                    }
                    for item in candidates
                ],
            }
        ],
        "assignments": [
            {
                "candidate_id": item["candidate_id"],
                "direction_id": "D1",
                "direction_role": "main",
                "assignment_confidence": 1.0,
                "method_or_object_summary_cn": "",
                "method_summary_cn": "",
                "assignment_reason_cn": "用户指定单方向分析。",
            }
            for item in candidates
        ],
        "relevance_scores": [
            {
                "candidate_id": item["candidate_id"],
                "relevance_score": 10.0,
                "decision": "include",
                "reason_cn": "用户指定纳入。",
            }
            for item in candidates
        ],
    }


def build_direction_workspace(
    output_dir: Path,
    screening_state: dict[str, Any],
    selected_candidates: list[dict[str, Any]],
    pdf_dir: Path,
    txt_dir: Path,
    figures_dir: Path | None = None,
) -> list[Path]:
    analysis_dir = ensure_dir(output_dir / "analysis")
    directions_root = ensure_dir(analysis_dir / "directions")
    selected_with_pdf: dict[str, dict[str, Any]] = {}
    for item in selected_candidates:
        candidate_id = str(item.get("candidate_id") or "")
        if not candidate_id:
            continue
        if _candidate_pdf_path(item, pdf_dir) is not None:
            selected_with_pdf[candidate_id] = item
    selected_ids = set(selected_with_pdf)
    selected_by_id = selected_with_pdf
    all_papers = {str(item.get("candidate_id")): item for item in screening_state.get("papers", [])}
    assignments = {str(item.get("candidate_id")): item for item in screening_state.get("assignments", [])}
    scores = {str(item.get("candidate_id")): item for item in screening_state.get("relevance_scores", [])}

    created: list[Path] = []
    for direction in screening_state.get("directions", []):
        direction_id = str(direction.get("direction_id") or "").strip()
        paper_ids = [
            str(pid)
            for pid in direction.get("paper_ids", [])
            if str(pid) in selected_ids
        ]
        if not paper_ids:
            continue
        direction_dir = ensure_dir(directions_root / direction_dir_name(direction))
        ensure_dir(direction_dir / "enriched_single_papers")
        papers: list[dict[str, Any]] = []
        for candidate_id in paper_ids:
            candidate = dict(all_papers.get(candidate_id, {}))
            candidate.update(selected_by_id.get(candidate_id, {}))
            pdf_path = _candidate_pdf_path(candidate, pdf_dir)
            if pdf_path is None:
                raise FileNotFoundError(f"无法为候选论文找到 PDF：{candidate_id}")
            txt_path = txt_path_for_pdf(pdf_path, txt_dir)
            manifest_path = find_manifest_for_pdf(pdf_path, figures_dir)
            assignment = assignments.get(candidate_id, {})
            score = scores.get(candidate_id, {})
            papers.append(
                {
                    "paper_id": candidate.get("paper_id") or candidate_id,
                    "candidate_id": candidate_id,
                    "title": candidate.get("title", ""),
                    "title_cn": candidate.get("title_cn", ""),
                    "abstract": candidate.get("abstract", ""),
                    "authors": candidate.get("authors", []),
                    "year": candidate.get("year", ""),
                    "venue": candidate.get("venue", ""),
                    "doi": candidate.get("doi", ""),
                    "source": candidate.get("source", ""),
                    "pdf_path": str(pdf_path),
                    "txt_path": str(txt_path.resolve()),
                    "figures_tables_manifest_path": str(manifest_path.resolve()) if manifest_path else "",
                    "prescreen": {
                        "direction_id": direction_id,
                        "direction_role": assignment.get("direction_role", "main"),
                        "assignment_confidence": assignment.get("assignment_confidence", ""),
                        "method_or_object_summary_cn": assignment.get("method_or_object_summary_cn")
                        or assignment.get("method_summary_cn", ""),
                        "assignment_reason_cn": assignment.get("assignment_reason_cn", ""),
                        "relevance_score": score.get("relevance_score", candidate.get("relevance_score", "")),
                        "decision": score.get("decision", candidate.get("decision", "")),
                        "score_reason_cn": score.get("reason_cn", candidate.get("reason_cn", "")),
                    },
                }
            )
        assigned = {
            "topic": screening_state.get("topic", ""),
            "direction_id": direction_id,
            "direction_name_cn": direction.get("direction_name_cn", direction_id),
            "direction_name_en": direction.get("direction_name_en", ""),
            "direction_summary_cn": direction.get("direction_summary_cn", ""),
            "display_keywords": direction.get("display_keywords", []),
            "inclusion_rule_cn": direction.get("inclusion_rule_cn", ""),
            "exclusion_rule_cn": direction.get("exclusion_rule_cn", ""),
            "papers": papers,
        }
        save_json(direction_dir / "assigned_papers.json", assigned)
        created.append(direction_dir)
    save_json(analysis_dir / "direction_workspace_manifest.json", [str(path.resolve()) for path in created])
    return created


def load_direction_dirs(output_dir: Path) -> list[Path]:
    root = output_dir / "analysis" / "directions"
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if (path / "assigned_papers.json").exists())
