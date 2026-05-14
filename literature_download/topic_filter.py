"""AND/OR/NOT topic filter for literature papers.

Keyword-based filtering with zero external dependencies.
Filters operate on paper title, abstract, and OpenAlex concepts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class FilterGroup:
    logic: Literal["AND", "OR", "NOT"]
    keywords: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.keywords = [kw.strip() for kw in self.keywords if kw.strip()]

    def matches(self, text: str) -> bool:
        """Return True if any keyword is found in text (case-insensitive)."""
        text_lower = text.lower()
        for kw in self.keywords:
            if kw.lower() in text_lower:
                return True
        return False

    def matched_keywords(self, text: str) -> list[str]:
        """Return which keywords matched (case-insensitive)."""
        text_lower = text.lower()
        return [kw for kw in self.keywords if kw.lower() in text_lower]

    def to_dict(self) -> dict[str, Any]:
        return {"logic": self.logic, "keywords": self.keywords}


class TopicFilter:
    """Filter papers by AND/OR/NOT logic across keyword groups.

    Logic:
    - All AND groups must have at least one keyword match.
    - At least one OR group must have a keyword match (when OR groups exist).
    - No NOT group may have any keyword match.
    """

    def __init__(self, groups: list[FilterGroup] | None = None) -> None:
        self.groups: list[FilterGroup] = groups or []
        self._and_groups = [g for g in self.groups if g.logic == "AND"]
        self._or_groups = [g for g in self.groups if g.logic == "OR"]
        self._not_groups = [g for g in self.groups if g.logic == "NOT"]

    # ── factory methods ────────────────────────────────────────────

    @classmethod
    def from_config(cls, path: str | Path) -> "TopicFilter":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        groups = [FilterGroup(**g) for g in raw["groups"]]
        return cls(groups)

    @classmethod
    def from_cli_args(
        cls,
        and_groups: list[list[str]] | None = None,
        or_groups: list[list[str]] | None = None,
        not_groups: list[list[str]] | None = None,
    ) -> "TopicFilter":
        groups: list[FilterGroup] = []
        for kw_list in and_groups or []:
            groups.append(FilterGroup(logic="AND", keywords=kw_list))
        for kw_list in or_groups or []:
            groups.append(FilterGroup(logic="OR", keywords=kw_list))
        for kw_list in not_groups or []:
            groups.append(FilterGroup(logic="NOT", keywords=kw_list))
        return cls(groups)

    # ── serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {"groups": [g.to_dict() for g in self.groups]}

    # ── text extraction ────────────────────────────────────────────

    @staticmethod
    def _paper_text(paper: dict[str, Any]) -> str:
        parts: list[str] = []
        for field in ("title", "abstract"):
            val = paper.get(field)
            if val:
                parts.append(str(val))
        concepts = paper.get("concepts")
        if concepts and isinstance(concepts, list):
            parts.append(" ".join(str(c) for c in concepts if c))
        return " ".join(parts)

    # ── evaluation ─────────────────────────────────────────────────

    def evaluate(self, paper: dict[str, Any]) -> bool:
        """Return True if the paper passes all filter rules."""
        text = self._paper_text(paper)

        for group in self._not_groups:
            if group.matches(text):
                return False

        for group in self._and_groups:
            if not group.matches(text):
                return False

        if self._or_groups:
            if not any(g.matches(text) for g in self._or_groups):
                return False

        return True

    def evaluate_with_matches(
        self, paper: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Return (pass, matched_keywords) for the paper."""
        text = self._paper_text(paper)

        for group in self._not_groups:
            if group.matches(text):
                return False, []

        matched: list[str] = []
        for group in self._and_groups:
            hits = group.matched_keywords(text)
            if not hits:
                return False, []
            matched.extend(hits)

        if self._or_groups:
            or_matched = False
            for group in self._or_groups:
                hits = group.matched_keywords(text)
                if hits:
                    matched.extend(hits)
                    or_matched = True
            if not or_matched:
                return False, []
            return True, matched

        return True, matched

    # ── batch operations ───────────────────────────────────────────

    def filter_papers(
        self, papers: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for paper in papers:
            if self.evaluate(paper):
                accepted.append(paper)
            else:
                rejected.append(paper)
        return accepted, rejected

    def filter_report(self, papers: list[dict[str, Any]]) -> dict[str, Any]:
        accepted, rejected = self.filter_papers(papers)
        return {
            "total": len(papers),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "groups": [g.to_dict() for g in self.groups],
        }


# ── inline sanity check ────────────────────────────────────────────────

if __name__ == "__main__":
    sample_papers = [
        {
            "title": "Battery storage bidding in electricity markets",
            "abstract": "We study optimal bidding strategies.",
            "concepts": ["Energy storage", "Electricity market"],
        },
        {
            "title": "A review of energy storage technologies",
            "abstract": "Survey paper on storage tech.",
            "concepts": ["Energy storage"],
        },
        {
            "title": "Power system SCUC optimization",
            "abstract": "Unit commitment formulation.",
            "concepts": ["Power systems"],
        },
    ]

    # Test 1: AND filter
    tf = TopicFilter.from_cli_args(
        and_groups=[["energy storage", "battery"], ["electricity market", "bidding"]],
    )
    accepted, _ = tf.filter_papers(sample_papers)
    assert len(accepted) == 1, f"Test 1: expected 1, got {len(accepted)}"
    assert "Battery storage" in accepted[0]["title"]

    # Test 2: AND + NOT filter
    tf2 = TopicFilter.from_cli_args(
        and_groups=[["energy storage"]],
        not_groups=[["review", "survey"]],
    )
    accepted2, _ = tf2.filter_papers(sample_papers)
    assert len(accepted2) == 1, f"Test 2: expected 1, got {len(accepted2)}"
    assert "Battery" in accepted2[0]["title"]

    # Test 3: evaluate_with_matches
    passed, matched = tf.evaluate_with_matches(sample_papers[0])
    assert passed and "energy storage" in matched, f"Test 3: bad matches {matched}"

    # Test 4: empty groups (backward compatible)
    tf3 = TopicFilter()
    accepted3, _ = tf3.filter_papers(sample_papers)
    assert len(accepted3) == 3, f"Test 4: expected 3, got {len(accepted3)}"

    # Test 5: from_config round-trip
    import tempfile, os
    config = {"groups": [{"logic": "AND", "keywords": ["energy"]}]}
    tmp = Path(tempfile.mktemp(suffix=".json"))
    try:
        tmp.write_text(json.dumps(config), encoding="utf-8")
        tf5 = TopicFilter.from_config(tmp)
        assert len(tf5.groups) == 1 and tf5.groups[0].logic == "AND"
    finally:
        os.unlink(tmp)

    print("All 5 tests passed.")
