from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "docs" / "prompts"

PROMPT_FILES = {
    "download_prescreen": "10A-download-prescreen-improved.txt",
    "enriched_single_by_direction": "11-enriched-single-paper-by-direction.txt",
    "direction_records": "12-direction-records.txt",
    "single_direction_review": "13-single-direction-review-md.txt",
    "single_direction_plot": "14-single-direction-plot.txt",
    "cross_direction_review": "15-cross-direction-review-md.txt",
    "cross_direction_plot": "16-cross-direction-plot.txt",
    "json_local_repair": "17-json-local-repair.txt",
    "plot_text_repair": "18-plot-text-repair.txt",
}


def load_prompt(name: str) -> str:
    try:
        filename = PROMPT_FILES[name]
    except KeyError as exc:
        raise KeyError(f"未知 prompt 名称：{name}") from exc
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt 文件不存在：{path}")
    return path.read_text(encoding="utf-8")


def prompt_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_prompt(name: str, **values: Any) -> str:
    text = load_prompt(name)
    for key, value in values.items():
        if isinstance(value, str):
            rendered = value
        else:
            rendered = prompt_json(value)
        text = text.replace("{" + key + "}", rendered)
    return text


def assert_prompt_placeholders_resolved(prompt: str) -> None:
    unresolved = [
        token
        for token in [
            "{topic}",
            "{candidate_papers_json}",
            "{direction_info_json}",
            "{paper_metadata_and_prescreen_json}",
            "{formula_candidates_json}",
            "{figures_tables_json}",
            "{paper_text}",
            "{assigned_papers_json}",
            "{enriched_single_papers_json}",
            "{direction_records_json}",
            "{enriched_supporting_info_json}",
            "{literature_review_md}",
            "{key_formulas_figures_json}",
            "{direction_mapping_json}",
            "{all_direction_records_json}",
            "{all_direction_reviews_json}",
            "{corpus_literature_review_md}",
            "{all_single_direction_plot_ready_json}",
            "{target_schema_json}",
            "{validation_errors_json}",
            "{invalid_output_text}",
            "{plot_ready_json}",
            "{violations_json}",
        ]
        if token in prompt
    ]
    if unresolved:
        raise ValueError("Prompt 存在未替换占位符：" + ", ".join(unresolved))
