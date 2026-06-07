from __future__ import annotations

import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from analysis_pipeline.core.common import (
    TimeRecorder,
    build_client,
    call_api_json,
    clean_text,
    ensure_dir,
    extract_text_from_pdf,
    load_json,
    resolve_llm_config,
    safe_output_stem,
    save_json,
    trim_text_for_prompt,
)
from analysis_pipeline.core.logging import run_tracked_block
from analysis_pipeline.core.prompts import render_prompt
from analysis_pipeline.stages.discovery.runner import load_discovery_direction_dirs


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


def _direction_info(assigned: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in assigned.items() if key != "papers"}


def _load_or_extract_text(paper: dict[str, Any]) -> str:
    txt_path = Path(str(paper.get("txt_path") or ""))
    if txt_path.exists():
        return clean_text(txt_path.read_text(encoding="utf-8", errors="ignore"))
    pdf_path = Path(str(paper.get("pdf_path") or ""))
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF/TXT for paper: {paper.get('candidate_id')}")
    text = extract_text_from_pdf(pdf_path, add_page_mark=True)
    ensure_dir(txt_path.parent)
    txt_path.write_text(text + "\n", encoding="utf-8")
    return clean_text(text)


def _review_direction_dir(discovery_direction_dir: Path, reviews_dir: Path) -> Path:
    return ensure_dir(reviews_dir / "directions" / discovery_direction_dir.name)


def run_paper_cards(
    client: OpenAI,
    flash_model: str,
    discovery_direction_dir: Path,
    review_direction_dir: Path,
    assigned: dict[str, Any],
    topic: str,
    overwrite: bool,
    parallel_papers: int,
    timer: TimeRecorder,
) -> list[dict[str, Any]]:
    cards_dir = ensure_dir(review_direction_dir / "paper_cards")
    papers = list(assigned.get("papers", []))

    def load_or_generate(paper: dict[str, Any]) -> dict[str, Any]:
        candidate_id = str(paper.get("candidate_id") or paper.get("paper_id") or "paper")
        output_path = cards_dir / f"{safe_output_stem(candidate_id)}.json"
        if output_path.exists() and not overwrite:
            return load_json(output_path)
        with timer.track("single_paper_lit_card", candidate_id):
            paper_text = _load_or_extract_text(paper)
            manifest = load_figures_tables_manifest(str(paper.get("figures_tables_manifest_path") or ""))
            prompt = render_prompt(
                "single_paper_lit_card",
                topic=topic,
                direction_info_json=_direction_info(assigned),
                paper_metadata_and_prescreen_json=paper,
                formula_candidates_json=extract_formula_candidates(paper_text),
                figures_tables_json=manifest,
                paper_text=trim_text_for_prompt(paper_text, max_chars=90000),
            )
            result = call_api_json(client=client, model=flash_model, prompt=prompt)
            if not isinstance(result, dict):
                raise RuntimeError(f"single_paper_lit_card returned non-object JSON for {candidate_id}")
            result.setdefault("candidate_id", candidate_id)
            result.setdefault("paper_id", paper.get("paper_id") or candidate_id)
            save_json(output_path, result)
            return result

    if len(papers) <= 1:
        return [load_or_generate(paper) for paper in papers]

    results: dict[str, dict[str, Any]] = {}
    workers = max(1, min(parallel_papers, len(papers)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load_or_generate, paper): str(paper.get("candidate_id")) for paper in papers}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [results[str(paper.get("candidate_id"))] for paper in papers if str(paper.get("candidate_id")) in results]


def run_direction_review(
    client: OpenAI,
    model: str,
    review_direction_dir: Path,
    assigned: dict[str, Any],
    paper_cards: list[dict[str, Any]],
    topic: str,
    overwrite: bool,
    timer: TimeRecorder,
) -> tuple[str, dict[str, Any]]:
    review_path = review_direction_dir / "direction_review.md"
    summary_path = review_direction_dir / "direction_review_summary.json"
    if review_path.exists() and summary_path.exists() and not overwrite:
        return review_path.read_text(encoding="utf-8"), load_json(summary_path)
    with timer.track("direction_literature_review", str(assigned.get("direction_id", ""))):
        prompt = render_prompt(
            "direction_literature_review",
            topic=topic,
            direction_info_json=_direction_info(assigned),
            assigned_papers_json=assigned,
            paper_cards_json=paper_cards,
        )
        payload = call_api_json(client=client, model=model, prompt=prompt)
        if not isinstance(payload, dict):
            raise RuntimeError("direction_literature_review returned non-object JSON")
        review_md = str(payload.get("direction_review_md") or "").strip()
        summary = payload.get("direction_review_summary") or {}
        if not review_md or not isinstance(summary, dict):
            raise RuntimeError("direction review output missing direction_review_md or direction_review_summary")
        review_md = re.sub(r"^```(?:markdown)?\s*\n", "", review_md)
        review_md = re.sub(r"\n```\s*$", "", review_md)
        review_path.write_text(review_md + "\n", encoding="utf-8")
        save_json(summary_path, summary)
        return review_md, summary


def run_direction_pipeline(
    direction_dir: Path,
    topic: str,
    client: OpenAI,
    model: str,
    flash_model: str,
    reviews_dir: Path | None = None,
    overwrite: bool = False,
    parallel_papers: int = 1,
    timer: TimeRecorder | None = None,
) -> dict[str, Any]:
    local_timer = timer or TimeRecorder()
    assigned = load_json(direction_dir / "assigned_papers.json")
    reviews_root = ensure_dir(reviews_dir or direction_dir.parents[2] / "02_reviews")
    review_direction_dir = _review_direction_dir(direction_dir, reviews_root)
    source_assigned = (direction_dir / "assigned_papers.json").resolve()
    target_assigned = (review_direction_dir / "assigned_papers.json").resolve()
    if source_assigned != target_assigned:
        shutil.copy2(source_assigned, target_assigned)
    paper_cards = run_paper_cards(
        client,
        flash_model,
        direction_dir,
        review_direction_dir,
        assigned,
        topic,
        overwrite,
        parallel_papers,
        local_timer,
    )
    run_direction_review(
        client,
        model,
        review_direction_dir,
        assigned,
        paper_cards,
        topic,
        overwrite,
        local_timer,
    )
    return {
        "direction_id": assigned.get("direction_id"),
        "direction_name_cn": assigned.get("direction_name_cn"),
        "paper_count": len(assigned.get("papers", [])),
        "outputs": {
            "assigned_papers": str((review_direction_dir / "assigned_papers.json").resolve()),
            "paper_cards_dir": str((review_direction_dir / "paper_cards").resolve()),
            "direction_review_md": str((review_direction_dir / "direction_review.md").resolve()),
            "direction_review_summary": str((review_direction_dir / "direction_review_summary.json").resolve()),
        },
    }


def run_cross_direction_outputs(
    reviews_dir: Path,
    topic: str,
    direction_results: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    overwrite: bool,
    timer: TimeRecorder,
    discovery_dir: Path | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    ensure_dir(reviews_dir)
    review_path = reviews_dir / "corpus_literature_review.md"
    summary_path = reviews_dir / "corpus_review_summary.json"
    direction_reviews = [
        {
            "direction_id": item.get("direction_id"),
            "direction_name_cn": item.get("direction_name_cn"),
            "direction_review_md": Path(item["outputs"]["direction_review_md"]).read_text(encoding="utf-8"),
        }
        for item in direction_results
    ]
    direction_summaries = [load_json(Path(item["outputs"]["direction_review_summary"])) for item in direction_results]
    mapping_path = (discovery_dir / "direction_workspace_manifest.json") if discovery_dir else None
    direction_mapping = load_json(mapping_path) if mapping_path and mapping_path.exists() else {}

    if review_path.exists() and summary_path.exists() and not overwrite:
        corpus_summary = load_json(summary_path)
    else:
        with timer.track("corpus_literature_review", "all_directions"):
            prompt = render_prompt(
                "corpus_literature_review",
                topic=topic,
                direction_mapping_json=direction_mapping,
                all_direction_reviews_json=direction_reviews,
                all_direction_review_summaries_json=direction_summaries,
            )
            payload = call_api_json(client=client, model=model, prompt=prompt)
            if not isinstance(payload, dict):
                raise RuntimeError("corpus_literature_review returned non-object JSON")
            corpus_review = str(payload.get("corpus_literature_review_md") or "").strip()
            corpus_summary = payload.get("corpus_review_summary") or {}
            if not corpus_review or not isinstance(corpus_summary, dict):
                raise RuntimeError("corpus review output missing corpus_literature_review_md or corpus_review_summary")
            corpus_review = re.sub(r"^```(?:markdown)?\s*\n", "", corpus_review)
            corpus_review = re.sub(r"\n```\s*$", "", corpus_review)
            review_path.write_text(corpus_review + "\n", encoding="utf-8")
            save_json(summary_path, corpus_summary)

    return {
        "corpus_literature_review_md": str(review_path.resolve()),
        "corpus_review_summary": str(summary_path.resolve()),
    }, []


def default_models() -> tuple[str, str]:
    config = resolve_llm_config()
    return config.model, config.flash_model


def run_reviews(ctx: Any) -> None:
    args = ctx.args
    if not ctx.direction_dirs:
        ctx.direction_dirs = load_discovery_direction_dirs(ctx)
    if not ctx.direction_dirs:
        raise RuntimeError("Direction workspace is empty. Run discovery first or pass --discovery-dir.")

    config = resolve_llm_config()
    client = build_client(config)
    timer = TimeRecorder()

    def run_all_directions() -> list[dict[str, Any]]:
        if len(ctx.direction_dirs) <= 1:
            direction_dir = ctx.direction_dirs[0]
            print(f"[Direction] {direction_dir.name}")
            return [
                run_direction_pipeline(
                    direction_dir=direction_dir,
                    topic=ctx.topic_for_model,
                    client=client,
                    model=config.model,
                    flash_model=config.flash_model,
                    reviews_dir=ctx.reviews_dir,
                    overwrite=args.overwrite,
                    parallel_papers=args.parallel_papers,
                    timer=timer,
                )
            ]

        results_map: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(ctx.direction_dirs)) as executor:
            futures = {}
            for direction_dir in ctx.direction_dirs:
                print(f"[Direction] {direction_dir.name}")
                future = executor.submit(
                    run_direction_pipeline,
                    direction_dir=direction_dir,
                    topic=ctx.topic_for_model,
                    client=client,
                    model=config.model,
                    flash_model=config.flash_model,
                    reviews_dir=ctx.reviews_dir,
                    overwrite=args.overwrite,
                    parallel_papers=args.parallel_papers,
                    timer=timer,
                )
                futures[future] = direction_dir.name
            for future in as_completed(futures):
                results_map[futures[future]] = future.result()
        return [results_map[d.name] for d in ctx.direction_dirs if d.name in results_map]

    ctx.direction_results = run_tracked_block(
        ctx,
        "1. Reviews: paper cards and direction reviews",
        run_all_directions,
    )
    ctx.report["directions"] = ctx.direction_results
    ctx.save_report()

    ctx.corpus_outputs, cross_repairs = run_tracked_block(
        ctx,
        "2. Reviews: corpus literature review",
        lambda: run_cross_direction_outputs(
            reviews_dir=ctx.reviews_dir,
            topic=ctx.topic_for_model,
            direction_results=ctx.direction_results,
            client=client,
            model=config.model,
            overwrite=args.overwrite,
            timer=timer,
            discovery_dir=ctx.discovery_dir,
        ),
    )
    ctx.report["corpus_outputs"] = ctx.corpus_outputs
    ctx.report.setdefault("repair_events", []).extend(cross_repairs)
    save_json(
        ctx.reviews_dir / "reviews_manifest.json",
        {
            "stage": "reviews",
            "topic": args.topic,
            "source_discovery_dir": str(ctx.discovery_dir.resolve()),
            "direction_reviews": [
                item.get("outputs", {}).get("direction_review_md", "")
                for item in ctx.direction_results
            ],
            "direction_review_summaries": [
                item.get("outputs", {}).get("direction_review_summary", "")
                for item in ctx.direction_results
            ],
            "corpus_review": ctx.corpus_outputs.get("corpus_literature_review_md", ""),
            "corpus_review_summary": ctx.corpus_outputs.get("corpus_review_summary", ""),
        },
    )
    timer.save(ctx.output_dir / "time_records")
    ctx.save_report()
