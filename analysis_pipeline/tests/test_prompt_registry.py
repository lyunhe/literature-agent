from __future__ import annotations

from pathlib import Path

from analysis_pipeline.core.prompts import PROMPT_ALIASES, PROMPT_REGISTRY, load_prompt, prompt_path


def test_prompt_aliases_load_current_renumbered_files() -> None:
    aliases = [
        "download_prescreen",
        "filter_keyword_expansion",
        "single_paper_lit_card",
        "enriched_single_by_direction",
        "direction_literature_review",
        "single_direction_review",
        "corpus_literature_review",
        "cross_direction_review",
        "json_local_repair",
    ]

    for alias in aliases:
        assert alias in PROMPT_ALIASES
        assert load_prompt(alias).strip()

    assert prompt_path("reviews", "01").name == "01_single_paper_lit_card.txt"
    assert prompt_path("reviews", "02").name == "02_direction_literature_review.txt"
    assert prompt_path("reviews", "03").name == "03_corpus_literature_review.txt"
    assert prompt_path("repair", "01").name == "01_json_local_repair.txt"


def test_old_prompt_numbered_files_are_not_referenced_or_present() -> None:
    project_root = Path(__file__).resolve().parents[2]
    old_paths = [
        project_root / "prompts" / "02_reviews" / "11_single_paper_lit_card.txt",
        project_root / "prompts" / "02_reviews" / "12_direction_literature_review.txt",
        project_root / "prompts" / "02_reviews" / "14_corpus_literature_review.txt",
        project_root / "prompts" / "repair" / "16_json_local_repair.txt",
    ]
    registry_paths = {
        relative_path
        for stage in PROMPT_REGISTRY.values()
        for relative_path in stage.values()
    }

    assert all(not path.exists() for path in old_paths)
    assert all(str(path.relative_to(project_root / "prompts")).replace("\\", "/") not in registry_paths for path in old_paths)


def test_download_prescreen_prompt_supports_pdf_text_final_classification() -> None:
    prompt = load_prompt("download_prescreen")

    assert "pdf_text_excerpt" in prompt
    assert "pdf_text_final" in prompt
    assert "不能遗漏" in prompt
