from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"

PROMPT_REGISTRY = {
    "discovery": {
        "01": "01_discovery/01_research_librarian.txt",
        "02": "01_discovery/02_query_variants.txt",
        "03": "01_discovery/03_search_strategy.txt",
        "04": "01_discovery/04_single_relevance_score.txt",
        "05": "01_discovery/05_batch_relevance_score.txt",
        "06": "01_discovery/06_query_refinement.txt",
        "07": "01_discovery/07_search_plan_refinement.txt",
        "08": "01_discovery/08_flash_query_expansion.txt",
        "09": "01_discovery/09_title_translation.txt",
        "10": "01_discovery/10_direction_prescreen.txt",
        "11": "01_discovery/11_download_relevance_score.txt",
        "12": "01_discovery/12_single_title_translation.txt",
        "13": "01_discovery/13_filter_keyword_expansion.txt",
        "14": "01_discovery/14_batch_abstract_summary_cn.txt",
    },
    "reviews": {
        "01": "02_reviews/01_single_paper_lit_card.txt",
        "02": "02_reviews/02_direction_literature_review.txt",
        "03": "02_reviews/03_corpus_literature_review.txt",
    },
    "repair": {
        "01": "repair/01_json_local_repair.txt",
    },
    "system": {
        "default_system_prompt": "system/default_system_prompt.txt",
        "strict_json_only": "system/strict_json_only.txt",
        "strict_legal_json_cn": "system/strict_legal_json_cn.txt",
        "academic_translation_json_array": "system/academic_translation_json_array.txt",
    },
}

PROMPT_ALIASES = {
    "00_agent_system": ("discovery", "01"),
    "query_variations": ("discovery", "02"),
    "plan_search_strategy": ("discovery", "03"),
    "score_relevance": ("discovery", "04"),
    "batch_score_papers": ("discovery", "05"),
    "refine_query": ("discovery", "06"),
    "refine_search_plan": ("discovery", "07"),
    "query_expansion": ("discovery", "08"),
    "batch_title_translation": ("discovery", "09"),
    "download_prescreen": ("discovery", "10"),
    "download_relevance_score": ("discovery", "11"),
    "single_title_translation": ("discovery", "12"),
    "filter_keyword_expansion": ("discovery", "13"),
    "batch_abstract_summary_cn": ("discovery", "14"),
    "single_paper_lit_card": ("reviews", "01"),
    "enriched_single_by_direction": ("reviews", "01"),
    "direction_literature_review": ("reviews", "02"),
    "single_direction_review": ("reviews", "02"),
    "corpus_literature_review": ("reviews", "03"),
    "cross_direction_review": ("reviews", "03"),
    "json_local_repair": ("repair", "01"),
    "system_default": ("system", "default_system_prompt"),
    "system_strict_json_only": ("system", "strict_json_only"),
    "system_strict_legal_json_cn": ("system", "strict_legal_json_cn"),
    "system_academic_translation_json_array": ("system", "academic_translation_json_array"),
}


def prompt_path(stage: str, prompt_id: str) -> Path:
    try:
        relative = PROMPT_REGISTRY[stage][prompt_id]
    except KeyError as exc:
        raise KeyError(f"未知 prompt：stage={stage}, prompt_id={prompt_id}") from exc
    path = PROMPTS_DIR / relative
    if not path.exists():
        raise FileNotFoundError(f"Prompt 文件不存在：{path}")
    return path


def load_prompt(name: str | None = None, *, stage: str | None = None, prompt_id: str | None = None) -> str:
    if stage is not None and prompt_id is not None:
        return prompt_path(stage, prompt_id).read_text(encoding="utf-8")
    if not name:
        raise ValueError("必须提供 name，或同时提供 stage 与 prompt_id")
    try:
        resolved_stage, resolved_id = PROMPT_ALIASES[name]
    except KeyError as exc:
        raise KeyError(f"未知 prompt 名称：{name}") from exc
    return prompt_path(resolved_stage, resolved_id).read_text(encoding="utf-8")


def prompt_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_prompt(name: str | None = None, *, stage: str | None = None, prompt_id: str | None = None, **values: Any) -> str:
    text = load_prompt(name, stage=stage, prompt_id=prompt_id)
    for key, value in values.items():
        rendered = value if isinstance(value, str) else prompt_json(value)
        text = text.replace("{" + key + "}", rendered)
    return text


def assert_prompt_placeholders_resolved(prompt: str) -> None:
    unresolved = [
        token
        for token in [
            "{topic}",
            "{max_queries}",
            "{title_list}",
            "{candidate_papers_json}",
            "{title}",
            "{direction_info_json}",
            "{paper_metadata_and_prescreen_json}",
            "{formula_candidates_json}",
            "{figures_tables_json}",
            "{paper_text}",
            "{assigned_papers_json}",
            "{paper_cards_json}",
            "{direction_review_md}",
            "{direction_review_summary_json}",
            "{all_direction_reviews_json}",
            "{all_direction_review_summaries_json}",
            "{corpus_literature_review_md}",
            "{corpus_review_summary_json}",
            "{target_schema_json}",
            "{validation_errors_json}",
            "{invalid_output_text}",
            "{violations_json}",
            "{user_query}",
            "{glossary_text}",
            "{num_variations}",
            "{preferred_sources}",
            "{topic_summary}",
            "{title}",
            "{abstract_truncated_700_chars}",
            "{expanded_topics_json}",
            "{target_venues_json}",
            "{target_authors_json}",
            "{search_domains_json}",
            "{papers_json}",
            "{original_query}",
            "{titles}",
            "{initial_plan_json}",
            "{top_results_json}",
            "{filter_groups_json}",
            "{max_terms_per_group}",
            "{abstract_list}",
        ]
        if token in prompt
    ]
    if unresolved:
        raise ValueError("Prompt 存在未替换占位符：" + ", ".join(unresolved))
