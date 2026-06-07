from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis_pipeline._bootstrap import PROJECT_ROOT
from analysis_pipeline.core.common import ensure_dir, save_json


def safe_output_name(topic: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", topic.strip())
    cleaned = re.sub(r"\s+", "_", cleaned, flags=re.UNICODE).strip("._ ")
    return (cleaned or "关键研究领域")[:80]


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class RunContext:
    args: Any
    run_parts: list[str]
    output_dir: Path
    discovery_dir: Path
    reviews_dir: Path
    logs_dir: Path
    run_pdf_dir: Path
    figures_output_dir: Path
    topic_for_model: str
    report: dict[str, Any]
    direction_dirs: list[Path] = field(default_factory=list)
    direction_results: list[dict[str, Any]] = field(default_factory=list)
    corpus_outputs: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_args(cls, args: Any) -> "RunContext":
        if args.input_mode == "local":
            args.from_pdf_only = True
        elif args.input_mode == "online":
            args.from_pdf_only = False
        else:
            args.from_pdf_only = bool(args.from_pdf_only or args.skip_search)
            args.input_mode = "local" if args.from_pdf_only else "online"
        args.skip_search = bool(args.skip_search or args.from_pdf_only)
        args.single_direction_only = bool(args.single_direction_only or args.single_only)
        if args.max_candidates is None and args.max_papers is not None:
            args.max_candidates = max(1, int(args.max_papers) * max(1, int(args.candidate_multiplier)))
        run_parts = [part.strip().lower() for part in args.run_parts.split(",") if part.strip()]
        valid_parts = {"discovery", "reviews"}
        invalid_parts = [part for part in run_parts if part not in valid_parts]
        if invalid_parts:
            raise ValueError("--run-parts 只支持 discovery,reviews；未知阶段：" + ",".join(invalid_parts))

        stamp = time.strftime("%Y%m%d_%H%M")
        output_dir = ensure_dir(args.output_dir or PROJECT_ROOT / "output" / f"{stamp}_{safe_output_name(args.topic)}")
        discovery_dir = ensure_dir(args.discovery_dir or output_dir / "01_discovery")
        reviews_dir = ensure_dir(args.reviews_dir or output_dir / "02_reviews")
        logs_dir = ensure_dir(output_dir / "logs")
        run_pdf_dir = ensure_dir(discovery_dir / "pdfs")
        figures_output_dir = discovery_dir / "figures_tables"
        topic_for_model = f"{args.topic}。请主要使用中文输出，保留必要英文术语。"
        entry_mode = "pdf_only" if args.from_pdf_only else "search_to_pdf"
        if args.single_direction_only:
            entry_mode = "single_direction_only"
        report = {
            "topic": args.topic,
            "topic_for_model": topic_for_model,
            "sources": [source.strip().lower() for source in args.sources.split(",") if source.strip()],
            "max_results": args.max_results,
            "max_papers": args.max_papers,
            "year_from": args.year_from,
            "year_to": args.year_to,
            "candidate_multiplier": args.candidate_multiplier,
            "max_candidates": args.max_candidates,
            "require_pdf": bool(args.require_pdf),
            "compare_sources": bool(args.compare_sources),
            "extract_figures_tables": bool(args.extract_figures_tables),
            "pipeline_version": "three_stage_reviews_v3",
            "entry_mode": entry_mode,
            "input_mode": args.input_mode,
            "run_parts": run_parts,
            "direction_source": "",
            "prompt_layout": "three_stage_continuous_numbering",
            "direction_records_removed": True,
            "repair_events": [],
            "status": "running",
            "started_at": now_text(),
            "output_dir": str(output_dir.resolve()),
            "output_layout": {
                "01_discovery": str(discovery_dir.resolve()),
                "02_reviews": str(reviews_dir.resolve()),
                "pdfs": str(run_pdf_dir.resolve()),
                "logs": str(logs_dir.resolve()),
                "figures_tables": str(figures_output_dir.resolve()),
            },
            "steps": [],
            "papers": [],
        }
        ctx = cls(
            args=args,
            run_parts=run_parts,
            output_dir=output_dir,
            discovery_dir=discovery_dir,
            reviews_dir=reviews_dir,
            logs_dir=logs_dir,
            run_pdf_dir=run_pdf_dir,
            figures_output_dir=figures_output_dir,
            topic_for_model=topic_for_model,
            report=report,
        )
        ctx.save_report()
        save_json(
            discovery_dir / "input_mode.json",
            {
                "entry_mode": entry_mode,
                "input_mode": args.input_mode,
                "run_parts": run_parts,
                "topic": args.topic,
                "max_results": args.max_results,
                "max_papers": args.max_papers,
                "candidate_multiplier": args.candidate_multiplier,
                "max_candidates": args.max_candidates,
                "require_pdf": bool(args.require_pdf),
                "compare_sources": bool(args.compare_sources),
                "extract_figures_tables": bool(args.extract_figures_tables),
                "pdf_dir": str(args.pdf_dir.resolve()) if args.pdf_dir else "",
            },
        )
        return ctx

    def should_run(self, stage: str) -> bool:
        return stage in self.run_parts

    def save_report(self) -> None:
        save_json(self.output_dir / "unified_run_report.json", self.report)

    def add_repair_events(self, events: list[dict[str, Any]]) -> None:
        self.report.setdefault("repair_events", []).extend(events)
        self.save_report()

    def finish(self) -> None:
        self.report["status"] = "completed"
        self.report["completed_at"] = now_text()
        self.save_report()


def write_step_records(logs_dir: Path, steps: list[dict[str, Any]]) -> None:
    jsonl_path = logs_dir / "step_records.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(step, ensure_ascii=False) + "\n" for step in steps),
        encoding="utf-8",
    )
