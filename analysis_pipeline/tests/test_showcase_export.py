from __future__ import annotations

from analysis_pipeline.stages.showcase_export import _method_distribution, build_quality_report


def test_method_distribution_uses_display_facts_modeling_type_without_visualization_stage() -> None:
    methods = _method_distribution(
        {},
        [],
        [
            {
                "paper_id": "P001",
                "display_facts": {
                    "method_family": [],
                    "modeling_or_experiment_type": ["robust optimization"],
                },
                "method": {
                    "summary_cn": "Battery bidding model",
                    "workflow": "",
                    "object_cn": "",
                },
            }
        ],
    )

    assert methods
    assert methods[0]["paper_ids"] == ["P001"]


def test_irrelevant_papers_check_is_advisory_pass() -> None:
    report = build_quality_report(
        {
            "topic": "储能 电力市场 报价",
            "target_paper_total": 1,
            "corpus": {"target_paper_total": 1},
            "directions": [
                {
                    "id": "D1",
                    "papers": [
                        {
                            "id": "P001",
                            "title": "Enhancing Transactive Energy Trading Framework for Residential End Users",
                            "formulas": [{"id": "f1", "formula": "\\\\[x=1\\\\]"}],
                        }
                    ],
                },
                {"id": "D2", "papers": []},
            ],
        }
    )

    irrelevant_check = next(item for item in report["checks"] if item["id"] == "irrelevant_papers")
    assert irrelevant_check["actual"] == 1
    assert irrelevant_check["status"] == "pass"
    assert report["status"] == "pass"
