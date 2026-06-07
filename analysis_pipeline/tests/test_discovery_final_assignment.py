from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from analysis_pipeline.stages.discovery.direction_workspace import build_direction_workspace
from analysis_pipeline.stages.discovery.prescreen import _fallback_directions, infer_candidate_directions


def _llm_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _paper(candidate_id: str, title: str, abstract: str = "", excerpt: str = "") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "title": title,
        "abstract": abstract,
        "concepts": [],
        "pdf_text_excerpt": excerpt,
    }


def _ids_from_directions(directions: list[dict[str, object]]) -> list[str]:
    return [
        str(candidate_id)
        for direction in directions
        for candidate_id in direction.get("paper_ids", [])
    ]


def test_final_direction_validation_fallback_covers_paper_missing_from_model_output() -> None:
    papers = [
        _paper(
            "P001",
            "Reinforcement learning bidding for battery energy storage in electricity markets",
            "Battery energy storage bidding in day-ahead electricity markets.",
            "The PDF discusses reinforcement learning agents for battery bidding.",
        ),
        _paper(
            "P002",
            "Robust optimization bidding for battery storage in day-ahead markets",
            "A robust optimization model for battery energy storage market bidding.",
            "The PDF formulates risk-constrained bids for storage.",
        ),
        _paper(
            "P003",
            "Dark energy constraints from a synoptic survey telescope",
            "Astronomy and cosmology survey data.",
            "The PDF is weakly related, but final classification must still assign it.",
        ),
    ]

    incomplete_model_output = json.dumps(
        {
            "directions": [
                {"direction_id": "D1", "direction_name_cn": "Learning", "paper_ids": ["P001"]},
                {"direction_id": "D2", "direction_name_cn": "Optimization", "paper_ids": ["P002"]},
            ],
            "assignments": [
                {"candidate_id": "P001", "direction_id": "D1"},
                {"candidate_id": "P002", "direction_id": "D2"},
            ],
            "relevance_scores": [
                {"candidate_id": "P001", "relevance_score": 8, "decision": "include"},
                {"candidate_id": "P002", "relevance_score": 8, "decision": "include"},
            ],
            "fast_check": {"all_papers_assigned_once": False},
        }
    )

    with patch(
        "analysis_pipeline.stages.discovery.prescreen.llm_request",
        return_value=_llm_response(incomplete_model_output),
    ):
        directions, assignments, scores, fast_check = infer_candidate_directions(
            "battery storage bidding in electricity markets",
            papers,
            input_mode="pdf_text_final",
            force_assign_all=True,
        )

    expected_ids = {str(paper["candidate_id"]) for paper in papers}
    assigned_ids = [str(row["candidate_id"]) for row in assignments]
    direction_ids = {str(row["direction_id"]) for row in assignments}
    direction_member_ids = _ids_from_directions(directions)

    assert set(assigned_ids) == expected_ids
    assert Counter(assigned_ids) == Counter(expected_ids)
    assert Counter(direction_member_ids) == Counter(expected_ids)
    assert {str(row["candidate_id"]) for row in scores} == expected_ids
    assert direction_ids <= {"D1", "D2", "D3", "D4"}
    assert "D_excluded" not in direction_ids
    assert fast_check["all_papers_assigned_once"] is True


def test_rule_fallback_splits_collapsed_direction_without_duplicate_membership() -> None:
    papers = [
        _paper(
            f"P{index:03d}",
            "Robust stochastic optimization bidding for battery energy storage",
            "Battery energy storage bidding in electricity markets with robust optimization.",
            "Risk-constrained optimization and market bidding for storage.",
        )
        for index in range(1, 5)
    ]

    directions, assignments, scores, _ = _fallback_directions(papers, force_assign_all=True)
    expected_ids = {str(paper["candidate_id"]) for paper in papers}

    assert Counter(_ids_from_directions(directions)) == Counter(expected_ids)
    assert Counter(str(row["candidate_id"]) for row in assignments) == Counter(expected_ids)
    assert {str(row["candidate_id"]) for row in scores} == expected_ids
    assert len([direction for direction in directions if direction.get("paper_ids")]) == 2


def test_direction_workspace_assigned_paper_count_matches_selected_candidates(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    pdf_dir = output_dir / "01_discovery" / "pdfs"
    txt_dir = output_dir / "02_text"
    pdf_dir.mkdir(parents=True)
    txt_dir.mkdir(parents=True)
    pdf_a = pdf_dir / "paper_a.pdf"
    pdf_b = pdf_dir / "paper_b.pdf"
    pdf_a.write_bytes(b"%PDF-1.4\n")
    pdf_b.write_bytes(b"%PDF-1.4\n")
    (txt_dir / "paper_a.txt").write_text("paper a text", encoding="utf-8")
    (txt_dir / "paper_b.txt").write_text("paper b text", encoding="utf-8")

    selected_candidates = [
        {"candidate_id": "P001", "title": "Paper A", "_pdf_path": str(pdf_a)},
        {"candidate_id": "P002", "title": "Paper B", "_pdf_path": str(pdf_b)},
    ]
    screening_state = {
        "topic": "storage market bidding",
        "papers": [
            {"candidate_id": "P001", "title": "Paper A"},
            {"candidate_id": "P002", "title": "Paper B"},
        ],
        "directions": [
            {"direction_id": "D1", "direction_name_cn": "A", "paper_ids": ["P001"]},
            {"direction_id": "D2", "direction_name_cn": "B", "paper_ids": ["P002"]},
        ],
        "assignments": [
            {"candidate_id": "P001", "direction_id": "D1"},
            {"candidate_id": "P002", "direction_id": "D2"},
        ],
        "relevance_scores": [
            {"candidate_id": "P001", "relevance_score": 8, "decision": "include"},
            {"candidate_id": "P002", "relevance_score": 8, "decision": "include"},
        ],
    }

    created = build_direction_workspace(output_dir, screening_state, selected_candidates, pdf_dir, txt_dir)
    assigned_total = sum(
        len(json.loads((path / "assigned_papers.json").read_text(encoding="utf-8"))["papers"])
        for path in created
    )

    assert assigned_total == len(selected_candidates)


def test_direction_workspace_rejects_duplicate_selected_pdf_assignments(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    pdf_dir = output_dir / "01_discovery" / "pdfs"
    txt_dir = output_dir / "02_text"
    pdf_dir.mkdir(parents=True)
    txt_dir.mkdir(parents=True)
    pdf_path = pdf_dir / "paper_a.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    selected_candidates = [{"candidate_id": "P001", "title": "Paper A", "_pdf_path": str(pdf_path)}]
    screening_state = {
        "topic": "storage market bidding",
        "papers": [{"candidate_id": "P001", "title": "Paper A"}],
        "directions": [
            {"direction_id": "D1", "direction_name_cn": "A", "paper_ids": ["P001"]},
            {"direction_id": "D2", "direction_name_cn": "B", "paper_ids": ["P001"]},
        ],
        "assignments": [{"candidate_id": "P001", "direction_id": "D1"}],
        "relevance_scores": [{"candidate_id": "P001", "relevance_score": 8, "decision": "include"}],
    }

    with pytest.raises(RuntimeError, match="more than once"):
        build_direction_workspace(output_dir, screening_state, selected_candidates, pdf_dir, txt_dir)
