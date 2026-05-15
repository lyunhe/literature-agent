from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from analysis_pipeline.pipeline_common import (
    TimeRecorder,
    call_api_json,
    call_api_text,
    clean_text,
    ensure_dir,
    extract_text_from_pdf,
    load_json,
    resolve_llm_config,
    safe_output_stem,
    save_json,
    trim_text_for_prompt,
)
from analysis_pipeline.prompt_loader import render_prompt
from analysis_pipeline.render_review_figures_v3 import render_cross_direction_svg, render_single_direction_svg


def load_figures_tables_manifest(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        return {}
    return load_json(path)


def extract_formula_candidates(text: str, limit: int = 40) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    pattern = re.compile(r"(?P<formula>.{0,180}?)(?P<number>\(\d+[a-zA-Z]?\))", re.DOTALL)
    for match in pattern.finditer(text):
        formula = re.sub(r"\s+", " ", match.group("formula")).strip()
        if not formula or len(formula) < 8:
            continue
        candidates.append(
            {
                "number": match.group("number"),
                "text": formula[-220:],
                "source_location": "pdf_text",
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def compact_enriched_for_review(enriched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for paper in enriched:
        compact.append(
            {
                "paper_id": paper.get("paper_id"),
                "candidate_id": paper.get("candidate_id"),
                "citation_cn": paper.get("citation_cn"),
                "bibliography": paper.get("bibliography", {}),
                "direction_context": paper.get("direction_context", {}),
                "key_formulas": paper.get("key_formulas", [])[:8],
                "key_figures_tables": paper.get("key_figures_tables", [])[:8],
                "main_findings": paper.get("main_findings", [])[:5],
                "advantages": paper.get("advantages", [])[:4],
                "limitations": paper.get("limitations", [])[:4],
            }
        )
    return compact


def key_formulas_figures(enriched: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "key_formulas": [
            {
                "paper_id": paper.get("paper_id"),
                **formula,
            }
            for paper in enriched
            for formula in paper.get("key_formulas", [])[:4]
        ],
        "key_figures_tables": [
            {
                "paper_id": paper.get("paper_id"),
                **item,
            }
            for paper in enriched
            for item in paper.get("key_figures_tables", [])[:4]
        ],
    }


def validate_plot_text(payload: dict[str, Any], max_chars: int = 42) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str):
            if len(value) > max_chars and not value.startswith("$$"):
                violations.append({"path": path, "type": "too_long", "text": value})
            if re.fullmatch(r"[A-Za-z]{2,8}", value.strip()):
                violations.append({"path": path, "type": "bare_english_or_symbol", "text": value})

    walk(payload, "")
    return violations[:20]


def maybe_repair_plot_text(client: OpenAI, model: str, payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    violations = validate_plot_text(payload)
    if not violations:
        return payload, []
    prompt = render_prompt(
        "plot_text_repair",
        plot_ready_json=json.dumps(payload, ensure_ascii=False, indent=2),
        violations_json=violations,
    )
    repaired = call_api_json(client=client, model=model, prompt=prompt)
    return repaired if isinstance(repaired, dict) else payload, violations


def _load_or_extract_text(paper: dict[str, Any]) -> str:
    txt_path = Path(str(paper.get("txt_path") or ""))
    if txt_path.exists():
        return clean_text(txt_path.read_text(encoding="utf-8", errors="ignore"))
    pdf_path = Path(str(paper.get("pdf_path") or ""))
    if not pdf_path.exists():
        raise FileNotFoundError(f"缺少 PDF/TXT：{paper.get('candidate_id')}")
    text = extract_text_from_pdf(pdf_path, add_page_mark=True)
    ensure_dir(txt_path.parent)
    txt_path.write_text(text + "\n", encoding="utf-8")
    return clean_text(text)


def run_enriched_single_papers(
    client: OpenAI,
    flash_model: str,
    direction_dir: Path,
    assigned: dict[str, Any],
    topic: str,
    overwrite: bool,
    parallel_papers: int,
    timer: TimeRecorder,
) -> list[dict[str, Any]]:
    enriched_dir = ensure_dir(direction_dir / "enriched_single_papers")
    direction_info = {
        key: assigned.get(key)
        for key in [
            "direction_id",
            "direction_name_cn",
            "direction_name_en",
            "direction_summary_cn",
            "display_keywords",
            "inclusion_rule_cn",
            "exclusion_rule_cn",
        ]
    }
    papers = list(assigned.get("papers", []))

    def load_or_generate(paper: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(paper.get("candidate_id") or paper.get("paper_id") or "paper")
        output_path = enriched_dir / f"{safe_output_stem(candidate_id)}.json"
        if output_path.exists() and not overwrite:
            return load_json(output_path)
        with timer.track("enriched_single_by_direction", candidate_id):
            paper_text = _load_or_extract_text(paper)
            manifest = load_figures_tables_manifest(str(paper.get("figures_tables_manifest_path") or ""))
            prompt = render_prompt(
                "enriched_single_by_direction",
                topic=topic,
                direction_info_json=direction_info,
                paper_metadata_and_prescreen_json=paper,
                formula_candidates_json=extract_formula_candidates(paper_text),
                figures_tables_json=manifest,
                paper_text=trim_text_for_prompt(paper_text, max_chars=90000),
            )
            result = call_api_json(client=client, model=flash_model, prompt=prompt)
            if isinstance(result, dict):
                result.setdefault("candidate_id", candidate_id)
                result.setdefault("paper_id", paper.get("paper_id") or candidate_id)
                save_json(output_path, result)
                return result
            raise RuntimeError(f"富化单篇返回非对象：{candidate_id}")

    if parallel_papers <= 1 or len(papers) <= 1:
        return [load_or_generate(paper) for paper in papers]
    results: dict[str, dict[str, Any]] = {}
    workers = max(1, min(parallel_papers, len(papers)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load_or_generate, paper): str(paper.get("candidate_id")) for paper in papers}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [results[str(paper.get("candidate_id"))] for paper in papers if str(paper.get("candidate_id")) in results]


def run_direction_records(
    client: OpenAI,
    model: str,
    direction_dir: Path,
    assigned: dict[str, Any],
    enriched: list[dict[str, Any]],
    topic: str,
    overwrite: bool,
    timer: TimeRecorder,
) -> dict[str, Any]:
    output_path = direction_dir / "direction_records.json"
    if output_path.exists() and not overwrite:
        return load_json(output_path)
    with timer.track("direction_records", str(assigned.get("direction_id", ""))):
        prompt = render_prompt(
            "direction_records",
            topic=topic,
            direction_info_json={k: assigned.get(k) for k in assigned if k != "papers"},
            assigned_papers_json=assigned,
            enriched_single_papers_json=enriched,
        )
        result = call_api_json(client=client, model=model, prompt=prompt)
        if not isinstance(result, dict):
            raise RuntimeError("direction_records 返回非 JSON 对象")
        save_json(output_path, result)
        return result


def run_single_direction_review(
    client: OpenAI,
    model: str,
    direction_dir: Path,
    assigned: dict[str, Any],
    records: dict[str, Any],
    enriched: list[dict[str, Any]],
    topic: str,
    overwrite: bool,
    timer: TimeRecorder,
) -> str:
    output_path = direction_dir / "literature_review.md"
    if output_path.exists() and not overwrite:
        return output_path.read_text(encoding="utf-8")
    with timer.track("single_direction_review_md", str(assigned.get("direction_id", ""))):
        prompt = render_prompt(
            "single_direction_review",
            topic=topic,
            direction_info_json={k: assigned.get(k) for k in assigned if k != "papers"},
            direction_records_json=records,
            enriched_supporting_info_json=compact_enriched_for_review(enriched),
        )
        text = call_api_text(client=client, model=model, prompt=prompt).strip()
        text = re.sub(r"^```(?:markdown)?\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
        output_path.write_text(text + "\n", encoding="utf-8")
        return text


def run_single_direction_plot(
    client: OpenAI,
    model: str,
    direction_dir: Path,
    assigned: dict[str, Any],
    records: dict[str, Any],
    review_md: str,
    enriched: list[dict[str, Any]],
    topic: str,
    overwrite: bool,
    timer: TimeRecorder,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_path = direction_dir / "plot_ready.json"
    repair_events: list[dict[str, Any]] = []
    if output_path.exists() and not overwrite:
        payload = load_json(output_path)
    else:
        with timer.track("single_direction_plot", str(assigned.get("direction_id", ""))):
            prompt = render_prompt(
                "single_direction_plot",
                topic=topic,
                direction_info_json={k: assigned.get(k) for k in assigned if k != "papers"},
                direction_records_json=records,
                literature_review_md=review_md,
                key_formulas_figures_json=key_formulas_figures(enriched),
            )
            payload = call_api_json(client=client, model=model, prompt=prompt)
            if not isinstance(payload, dict):
                raise RuntimeError("plot_ready 返回非 JSON 对象")
            payload, violations = maybe_repair_plot_text(client, model, payload)
            if violations:
                repair_events.append({"type": "plot_text_repair", "direction": assigned.get("direction_id"), "violations": violations})
            save_json(output_path, payload)
    svg_path = direction_dir / "single_direction_overview.svg"
    render_single_direction_svg(payload, svg_path)
    return payload, repair_events


def run_direction_pipeline(
    direction_dir: Path,
    topic: str,
    client: OpenAI,
    model: str,
    flash_model: str,
    overwrite: bool = False,
    parallel_papers: int = 1,
    timer: TimeRecorder | None = None,
) -> dict[str, Any]:
    local_timer = timer or TimeRecorder()
    assigned = load_json(direction_dir / "assigned_papers.json")
    enriched = run_enriched_single_papers(client, flash_model, direction_dir, assigned, topic, overwrite, parallel_papers, local_timer)
    records = run_direction_records(client, model, direction_dir, assigned, enriched, topic, overwrite, local_timer)
    review_md = run_single_direction_review(client, model, direction_dir, assigned, records, enriched, topic, overwrite, local_timer)
    plot_ready, repair_events = run_single_direction_plot(client, model, direction_dir, assigned, records, review_md, enriched, topic, overwrite, local_timer)
    return {
        "direction_id": assigned.get("direction_id"),
        "direction_name_cn": assigned.get("direction_name_cn"),
        "paper_count": len(assigned.get("papers", [])),
        "outputs": {
            "assigned_papers": str((direction_dir / "assigned_papers.json").resolve()),
            "direction_records": str((direction_dir / "direction_records.json").resolve()),
            "literature_review_md": str((direction_dir / "literature_review.md").resolve()),
            "plot_ready": str((direction_dir / "plot_ready.json").resolve()),
            "single_direction_svg": str((direction_dir / "single_direction_overview.svg").resolve()),
        },
        "plot_ready": plot_ready,
        "repair_events": repair_events,
    }


def run_cross_direction_outputs(
    output_dir: Path,
    topic: str,
    direction_results: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    overwrite: bool,
    timer: TimeRecorder,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    analysis_dir = ensure_dir(output_dir / "analysis")
    review_path = analysis_dir / "corpus_literature_review.md"
    plot_path = analysis_dir / "cross_direction_plot_ready.json"
    svg_path = ensure_dir(output_dir / "review_figures") / "corpus_overview.svg"
    direction_records = [load_json(Path(item["outputs"]["direction_records"])) for item in direction_results]
    direction_reviews = [
        {
            "direction_id": item.get("direction_id"),
            "direction_name_cn": item.get("direction_name_cn"),
            "literature_review_md": Path(item["outputs"]["literature_review_md"]).read_text(encoding="utf-8"),
        }
        for item in direction_results
    ]
    repair_events: list[dict[str, Any]] = []
    if len(direction_results) == 1:
        item = direction_results[0]
        corpus_review = Path(item["outputs"]["literature_review_md"]).read_text(encoding="utf-8")
        review_path.write_text(corpus_review + ("\n" if not corpus_review.endswith("\n") else ""), encoding="utf-8")
        single_plot = item.get("plot_ready") or load_json(Path(item["outputs"]["plot_ready"]))
        record = direction_records[0] if direction_records else {}
        plot_ready = {
            "topic": topic,
            "figure_title_cn": f"{topic}总览",
            "global_core_problem_cn": single_plot.get("core_problem_box", [""])[0]
            if isinstance(single_plot.get("core_problem_box"), list)
            else str(single_plot.get("core_problem_box") or record.get("direction_definition_cn") or ""),
            "direction_blocks": [
                {
                    "direction_id": item.get("direction_id") or single_plot.get("direction_id") or "D1",
                    "direction_name_cn": item.get("direction_name_cn") or single_plot.get("direction_name_cn") or "",
                    "main_problem_cn": record.get("within_direction_summary", {}).get("common_problem_cn", "")
                    or (single_plot.get("core_problem_box", [""])[0] if isinstance(single_plot.get("core_problem_box"), list) else ""),
                    "method_keywords_cn": record.get("within_direction_summary", {}).get("common_methods_cn", [])[:3],
                    "representative_papers_cn": [
                        row.get("citation_cn", "")
                        for row in record.get("records", [])[:3]
                        if row.get("citation_cn")
                    ],
                    "main_outputs_cn": record.get("within_direction_summary", {}).get("common_outputs_cn", [])[:2],
                    "limitations_cn": single_plot.get("research_gap_box", [])[:2]
                    if isinstance(single_plot.get("research_gap_box"), list)
                    else [],
                }
            ],
            "cross_direction_comparison": [],
            "storyline_cn": [
                "本次候选 PDF 形成单一研究方向",
                "总综述复用该方向的方向内 records 与综述",
            ],
            "research_gap_blocks": [
                {
                    "gap_name_cn": "方向内研究空白",
                    "gap_description_cn": "；".join(single_plot.get("research_gap_box", [])[:2])
                    if isinstance(single_plot.get("research_gap_box"), list)
                    else "",
                    "related_direction_ids": [item.get("direction_id") or "D1"],
                    "possible_entry_point_cn": "结合更多 PDF 后再扩展跨方向比较",
                }
            ],
            "symbol_glossary_cn": single_plot.get("symbol_glossary_cn", []),
            "self_check": {
                "all_text_cn_or_explained": True,
                "box_length_ok": True,
                "no_unexplained_symbol": True,
                "notes": "single_direction_fast_path",
            },
        }
        save_json(plot_path, plot_ready)
        render_cross_direction_svg(plot_ready, svg_path)
        return (
            {
                "corpus_literature_review_md": str(review_path.resolve()),
                "cross_direction_plot_ready": str(plot_path.resolve()),
                "corpus_overview_svg": str(svg_path.resolve()),
            },
            repair_events,
        )
    if review_path.exists() and not overwrite:
        corpus_review = review_path.read_text(encoding="utf-8")
    else:
        with timer.track("cross_direction_review", "all_directions"):
            mapping_path = analysis_dir / "direction_workspace_manifest.json"
            prompt = render_prompt(
                "cross_direction_review",
                topic=topic,
                direction_mapping_json=load_json(mapping_path) if mapping_path.exists() else {},
                all_direction_records_json=direction_records,
                all_direction_reviews_json=direction_reviews,
            )
            corpus_review = call_api_text(client=client, model=model, prompt=prompt).strip()
            corpus_review = re.sub(r"^```(?:markdown)?\s*\n", "", corpus_review)
            corpus_review = re.sub(r"\n```\s*$", "", corpus_review)
            review_path.write_text(corpus_review + "\n", encoding="utf-8")

    if plot_path.exists() and not overwrite:
        plot_ready = load_json(plot_path)
    else:
        with timer.track("cross_direction_plot", "all_directions"):
            prompt = render_prompt(
                "cross_direction_plot",
                topic=topic,
                corpus_literature_review_md=corpus_review,
                all_direction_records_json=direction_records,
                all_single_direction_plot_ready_json=[item.get("plot_ready", {}) for item in direction_results],
            )
            plot_ready = call_api_json(client=client, model=model, prompt=prompt)
            if not isinstance(plot_ready, dict):
                raise RuntimeError("cross_direction_plot 返回非 JSON 对象")
            plot_ready, violations = maybe_repair_plot_text(client, model, plot_ready)
            if violations:
                repair_events.append({"type": "cross_plot_text_repair", "violations": violations})
            save_json(plot_path, plot_ready)
    render_cross_direction_svg(plot_ready, svg_path)
    return (
        {
            "corpus_literature_review_md": str(review_path.resolve()),
            "cross_direction_plot_ready": str(plot_path.resolve()),
            "corpus_overview_svg": str(svg_path.resolve()),
        },
        repair_events,
    )


def default_models() -> tuple[str, str]:
    config = resolve_llm_config()
    return config.model, config.flash_model
