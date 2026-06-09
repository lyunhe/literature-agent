from __future__ import annotations

import argparse
from pathlib import Path

try:
    from analysis_pipeline._bootstrap import PROJECT_ROOT
except ModuleNotFoundError:
    from _bootstrap import PROJECT_ROOT

from analysis_pipeline.core.run_context import RunContext, now_text
from analysis_pipeline.core.logging import run_tracked_block
from analysis_pipeline.core.timing_summary import write_timing_summary
from analysis_pipeline.stages.discovery.runner import load_discovery_direction_dirs, run_discovery
from analysis_pipeline.stages.reviews.runner import run_reviews
from analysis_pipeline.stages.showcase_export import write_three_stage_review


DEFAULT_TOPIC = "储能参与电力市场"


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Literature discovery and review pipeline.")
    parser.add_argument("topic", nargs="?", default=DEFAULT_TOPIC, help="Research topic.")
    parser.add_argument("--topic", dest="topic_option", default=None, help="Research topic. Overrides the positional topic.")
    parser.add_argument("--input-mode", choices=["local", "online"], default=None, help="Input mode: local reads PDFs from --pdf-dir; online searches and downloads PDFs.")
    parser.add_argument("--sources", default="openalex,arxiv", help="Comma-separated search sources: openalex, arxiv, ieee.")
    parser.add_argument("--max-results", type=int, default=5, help="Max results per query per source.")
    parser.add_argument("--max-papers", type=int, default=1, help="Max PDFs to process.")
    parser.add_argument("--year-from", type=int, default=None, help="Earliest publication year to include in online search.")
    parser.add_argument("--year-to", type=int, default=None, help="Latest publication year to include in online search.")
    parser.add_argument("--candidate-multiplier", type=int, default=2, help="Online mode: downloadable candidate pool target = max_papers * candidate_multiplier.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Online mode: max verified downloadable candidates before final PDF selection.")
    parser.add_argument("--require-pdf", type=str_to_bool, default=True, help="Online mode: only admit verified direct PDFs to downloadable_candidates.")
    parser.add_argument("--compare-sources", action="store_true", help="Write source_comparison.json/.csv/.md for OpenAlex/arXiv/other sources.")
    parser.add_argument("--all-papers", action="store_true", help="Process every available/downloaded PDF.")
    parser.add_argument("--pdf-dir", type=Path, default=PROJECT_ROOT / "input_pdfs", help="PDF directory for PDF-only mode.")
    parser.add_argument("--from-pdf-only", action="store_true", help="Skip online search and start from --pdf-dir.")
    parser.add_argument("--pdf-metadata-path", type=Path, default=None, help="Optional metadata JSON for PDF-only mode.")
    parser.add_argument("--skip-search", action="store_true", help="Compatibility alias for --from-pdf-only.")
    parser.add_argument("--single-direction-only", action="store_true", help="Treat all PDFs as one direction.")
    parser.add_argument("--single-only", action="store_true", help="Compatibility alias for --single-direction-only.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing intermediate outputs.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Run output directory.")
    parser.add_argument("--run-parts", default="discovery,reviews", help="Stages to run: discovery,reviews.")
    parser.add_argument("--discovery-dir", type=Path, default=None, help="Existing 01_discovery directory for later stages.")
    parser.add_argument("--reviews-dir", type=Path, default=None, help="Existing 02_reviews directory.")
    parser.add_argument("--extract-figures-tables", action="store_true", help="Extract PDF figures/tables during discovery.")
    parser.add_argument("--screen-only", action="store_true", help="Stop after direction prescreening.")
    parser.add_argument("--table-only", action="store_true", help="Stop after search, filter, rank, and paper_table export; skip PDF verification and download.")
    parser.add_argument("--screening-state", type=Path, default=None, help="Reuse an existing screening_state.json.")
    parser.add_argument("--selected-directions", default="", help="Comma-separated direction IDs to keep, e.g. D1,D3.")
    parser.add_argument("--journal-levels", type=Path, default=PROJECT_ROOT / "journal_levels.csv", help="Journal level CSV.")
    parser.add_argument("--skip-ai-prescreen", action="store_true", help="Disabled: 10_direction_prescreen is required.")
    parser.add_argument("--parallel-papers", type=int, default=7, help="Parallel single-paper card workers per direction (default: number of papers, max 7).")
    parser.add_argument("--filter-and", action="append", dest="filter_and_groups", default=None, help="AND keyword group, comma-separated.")
    parser.add_argument("--filter-or", action="append", dest="filter_or_groups", default=None, help="OR keyword group, comma-separated.")
    parser.add_argument("--filter-not", action="append", dest="filter_not_groups", default=None, help="NOT keyword group, comma-separated.")
    parser.add_argument("--filter-config", type=Path, default=None, help="JSON topic filter config.")
    args = parser.parse_args()
    if args.topic_option:
        args.topic = args.topic_option
    delattr(args, "topic_option")
    return args


def main() -> None:
    args = parse_args()
    ctx = RunContext.from_args(args)

    try:
        if ctx.should_run("discovery"):
            run_discovery(ctx)
            if args.screen_only or args.table_only:
                return
        else:
            ctx.direction_dirs = load_discovery_direction_dirs(ctx)

        if not ctx.direction_dirs and ctx.should_run("reviews"):
            raise RuntimeError("Direction workspace is empty. Run discovery first or pass --discovery-dir.")

        if ctx.should_run("reviews"):
            run_reviews(ctx)

        showcase_path = run_tracked_block(
            ctx,
            "3. Showcase export: three_stage_review and quality_report",
            lambda: write_three_stage_review(ctx.output_dir),
        )
        ctx.report["three_stage_review_json"] = str(showcase_path.resolve())
        ctx.save_report()

        ctx.finish()
        timing_summary = write_timing_summary(ctx.output_dir)
        ctx.report["timing_summary_csv"] = str(timing_summary.resolve())
        ctx.save_report()
        print(f"\nPipeline completed. Report: {ctx.output_dir / 'unified_run_report.json'}")
    except Exception as exc:
        if ctx.report.get("status") != "failed":
            ctx.report.pop("current_step", None)
            ctx.report["status"] = "failed"
            ctx.report["failed_at"] = now_text()
            ctx.report["failure"] = str(exc)
            ctx.save_report()
        raise


if __name__ == "__main__":
    main()
