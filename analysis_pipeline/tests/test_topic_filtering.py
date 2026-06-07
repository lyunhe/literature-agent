from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from analysis_pipeline.stages.discovery.runner import expand_topic_filter_keywords
from analysis_pipeline.stages.discovery.topic_filtering import TopicFilter


def _llm_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_chinese_dispatch_filter_matches_english_dispatch_terms() -> None:
    topic_filter = TopicFilter.from_cli_args(and_groups=[["调度"]])
    paper = {
        "title": "Integrated unit commitment and economic dispatch of power systems",
        "abstract": "The model improves optimal scheduling under renewable uncertainty.",
        "concepts": ["Power systems", "Economic dispatch"],
    }

    passed, matched = topic_filter.evaluate_with_matches(paper)

    assert passed is True
    assert "economic dispatch" in matched
    assert "unit commitment" in matched
    assert "optimal scheduling" in matched


def test_filter_config_rehydrates_dispatch_expansions(tmp_path: Path) -> None:
    config_path = tmp_path / "filter_config.json"
    config_path.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "logic": "AND",
                        "input_keywords": ["调度"],
                        "expanded_keywords": ["security constrained economic dispatch"],
                        "keywords": ["调度", "security constrained economic dispatch"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    topic_filter = TopicFilter.from_config(config_path)

    assert "dispatch" in topic_filter.groups[0].keywords
    assert "security constrained economic dispatch" in topic_filter.groups[0].keywords
    assert topic_filter.evaluate(
        {
            "title": "Frequency-constrained unit commitment for power systems",
            "abstract": "A scheduling process is proposed for secure operation.",
            "concepts": ["Power systems"],
        }
    )


def test_llm_keyword_expansion_updates_topic_filter() -> None:
    topic_filter = TopicFilter.from_cli_args(and_groups=[["调度"]])
    payload = json.dumps(
        {
            "groups": [
                {
                    "group_id": "G1",
                    "expanded_keywords": ["SCED", "security constrained dispatch"],
                }
            ]
        },
        ensure_ascii=False,
    )

    with patch(
        "analysis_pipeline.stages.discovery.runner.llm_request",
        return_value=_llm_response(payload),
    ):
        report = expand_topic_filter_keywords("电力系统 调度", topic_filter)

    assert report is not None
    assert report["status"] == "completed"
    assert "SCED" in topic_filter.groups[0].expanded_keywords
    assert topic_filter.evaluate(
        {
            "title": "SCED for power systems with renewable uncertainty",
            "abstract": "",
            "concepts": ["Power systems"],
        }
    )
