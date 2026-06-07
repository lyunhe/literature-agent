from __future__ import annotations

from unittest.mock import patch

from analysis_pipeline.stages.discovery.runner import search_literature


def test_openalex_search_caps_requested_limit_before_call() -> None:
    calls: list[int] = []

    def fake_search(*_args, **kwargs):
        calls.append(int(kwargs["max_results"]))
        return [{"title": "Battery bidding", "year": 2024, "source": "openalex"}]

    with (
        patch("analysis_pipeline.stages.discovery.runner.expand_queries", return_value=["battery bidding"]),
        patch("analysis_pipeline.stages.discovery.runner.search_openalex.search", side_effect=fake_search),
    ):
        rows = search_literature("storage bidding", ["openalex"], max_results=400, max_workers=1)

    assert len(rows) == 1
    assert calls == [200]


def test_search_retries_with_lower_limit_after_api_error() -> None:
    calls: list[int] = []

    def fake_search(*_args, **kwargs):
        limit = int(kwargs["max_results"])
        calls.append(limit)
        if limit == 400:
            return [{"error": "API limit exceeded"}]
        return [{"title": "Storage market", "year": 2024, "source": "arxiv"}]

    with (
        patch("analysis_pipeline.stages.discovery.runner.expand_queries", return_value=["storage market"]),
        patch("analysis_pipeline.stages.discovery.runner.search_arxiv.search", side_effect=fake_search),
    ):
        rows = search_literature("storage market", ["arxiv"], max_results=400, max_workers=1)

    assert len(rows) == 1
    assert calls == [400, 200]
