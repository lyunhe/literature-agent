from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from analysis_pipeline._bootstrap import PROJECT_ROOT
except ModuleNotFoundError:
    from _bootstrap import PROJECT_ROOT

from backend import db
from backend.paths import LIBRARY_PDF_DIR
from analysis_pipeline.direction_pipeline import run_cross_direction_outputs, run_direction_pipeline
from analysis_pipeline.direction_workspace import (
    build_direction_workspace,
    build_local_pdf_candidates,
    build_virtual_single_direction_state,
)
from analysis_pipeline.pipeline_common import (
    TimeRecorder,
    build_client,
    extract_text_from_pdf,
    load_json,
    resolve_llm_config,
    safe_output_stem,
)
from literature_download.prescreen import (
    build_screening_state,
    save_screening_state,
    score_and_rank_candidates,
    selected_for_download,
)
from literature_download.topic_filter import TopicFilter
from literature_download.workflow import search_and_download


PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_TOPIC = "储能参与电力市场"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_output_name(topic: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", topic.strip())
    cleaned = re.sub(r"\s+", "_", cleaned, flags=re.UNICODE).strip("._ ")
    return (cleaned or "关键研究领域")[:80]


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def make_step_log_path(logs_dir: Path, index: int, name: str) -> Path:
    return logs_dir / f"{index:02d}_{safe_output_name(name)}.log"


class TeeWriter:
    def __init__(self, *writers: Any) -> None:
        self.writers = writers

    def write(self, data: str) -> int:
        for writer in self.writers:
            writer.write(data)
            writer.flush()
        return len(data)

    def flush(self) -> None:
        for writer in self.writers:
            writer.flush()


def write_step_records(logs_dir: Path, steps: list[dict[str, Any]]) -> None:
    ensure_dir(logs_dir)
    jsonl_path = logs_dir / "step_records.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(step, ensure_ascii=False) + "\n" for step in steps),
        encoding="utf-8",
    )
    csv_path = logs_dir / "step_records.csv"
    fieldnames = [
        "index",
        "name",
        "status",
        "returncode",
        "elapsed_seconds",
        "start_time",
        "end_time",
        "log_file",
        "reason",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for step in steps:
            writer.writerow({key: step.get(key, "") for key in fieldnames})


def start_step(
    report: dict[str, Any],
    output_dir: Path,
    logs_dir: Path,
    name: str,
) -> tuple[dict[str, Any], Path]:
    index = len(report["steps"]) + 1
    log_path = make_step_log_path(logs_dir, index, name)
    step = {
        "index": index,
        "name": name,
        "status": "running",
        "start_time": now_text(),
        "log_file": str(log_path.resolve()),
    }
    report["current_step"] = step
    save_json(output_dir / "unified_run_report.json", report)
    save_json(logs_dir / "current_step.json", step)
    return step, log_path


def finish_step(
    report: dict[str, Any],
    output_dir: Path,
    logs_dir: Path,
    step: dict[str, Any],
    *,
    status: str,
    started: float,
    reason: str | None = None,
) -> None:
    step["status"] = status
    step["end_time"] = now_text()
    step["elapsed_seconds"] = round(time.time() - started, 3)
    step["returncode"] = 0 if status == "completed" else ""
    if reason:
        step["reason"] = reason
    report.pop("current_step", None)
    report["steps"].append(step)
    report["updated_at"] = now_text()
    save_json(output_dir / "unified_run_report.json", report)
    save_json(logs_dir / "latest_step.json", step)
    write_step_records(logs_dir, report["steps"])


def add_skipped_step(report: dict[str, Any], output_dir: Path, logs_dir: Path, name: str, reason: str) -> None:
    started = time.time()
    step, log_path = start_step(report, output_dir, logs_dir, name)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"=== {name} ===\n")
        log.write(f"开始时间：{step['start_time']}\n")
        log.write(f"跳过原因：{reason}\n")
    print(f"[跳过] {name}: {reason}")
    finish_step(report, output_dir, logs_dir, step, status="skipped", started=started, reason=reason)


def run_tracked_block(
    name: str,
    report: dict[str, Any],
    output_dir: Path,
    logs_dir: Path,
    callback: Any,
) -> Any:
    started = time.time()
    step, log_path = start_step(report, output_dir, logs_dir, name)
    print(f"\n=== {name} ===")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"=== {name} ===\n")
        log.write(f"开始时间：{step['start_time']}\n\n")
        try:
            with contextlib.redirect_stdout(TeeWriter(sys.stdout, log)):
                with contextlib.redirect_stderr(TeeWriter(sys.stderr, log)):
                    result = callback()
        except Exception as exc:
            log.write(f"\n[失败] {exc}\n")
            finish_step(report, output_dir, logs_dir, step, status="failed", started=started, reason=str(exc))
            report["status"] = "failed"
            report["failed_at"] = now_text()
            report["failure"] = str(exc)
            save_json(output_dir / "unified_run_report.json", report)
            raise
        log.write(f"\n结束时间：{now_text()}\n")
    finish_step(report, output_dir, logs_dir, step, status="completed", started=started)
    return result


def run_existing_script(
    name: str,
    script_name: str,
    args: list[str],
    report: dict[str, Any],
    output_dir: Path,
    logs_dir: Path,
    required: bool = True,
) -> bool:
    script_path = PIPELINE_DIR / script_name
    started = time.time()
    step, log_path = start_step(report, output_dir, logs_dir, name)
    command = [sys.executable, str(script_path), *args]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print(f"\n=== {name} ===")
    with log_path.open("a", encoding="utf-8") as log:
        log.write("命令:\n" + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        returncode = process.wait()
    finish_step(
        report,
        output_dir,
        logs_dir,
        step,
        status="completed" if returncode == 0 else "failed",
        started=started,
        reason="" if returncode == 0 else f"退出码 {returncode}",
    )
    if returncode != 0 and required:
        raise RuntimeError(f"{name} 失败，退出码：{returncode}")
    return returncode == 0


def copy_pdfs_to_run(pdf_paths: list[Path], pdf_dir: Path) -> list[Path]:
    ensure_dir(pdf_dir)
    copied: list[Path] = []
    for source in pdf_paths:
        source = source.resolve()
        target = pdf_dir / source.name
        if source != target.resolve():
            shutil.copy2(source, target)
        copied.append(target.resolve())
        print(f"[PDF归档] {source.name} -> {target}")
    return copied


def convert_pdfs_to_txt(pdf_paths: list[Path], txt_dir: Path, overwrite: bool) -> list[Path]:
    ensure_dir(txt_dir)
    txt_paths: list[Path] = []
    for pdf_path in pdf_paths:
        txt_path = txt_dir / f"{safe_output_stem(pdf_path.stem)}.txt"
        txt_paths.append(txt_path)
        if txt_path.exists() and not overwrite:
            print(f"[TXT] 复用已有文本：{txt_path.name}")
            continue
        print(f"[TXT] 正在提取：{pdf_path.name}")
        text = extract_text_from_pdf(pdf_path, add_page_mark=True)
        txt_path.write_text(text + "\n", encoding="utf-8")
        print(f"[TXT] 已生成：{txt_path}")
    return txt_paths


def load_pdf_metadata_candidates(pdf_files: list[Path], metadata_path: Path | None) -> list[dict[str, Any]]:
    if metadata_path is None:
        return build_local_pdf_candidates(pdf_files)
    payload = load_json(metadata_path)
    rows = payload.get("papers", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("--pdf-metadata-path 必须是数组或包含 papers 数组的 JSON")
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item.setdefault("candidate_id", f"P{index + 1:03d}")
        raw_pdf = item.get("_pdf_path") or item.get("pdf_path") or item.get("filename")
        pdf_path = Path(str(raw_pdf)) if raw_pdf else (pdf_files[index] if index < len(pdf_files) else None)
        if pdf_path is None or not pdf_path.exists():
            raise FileNotFoundError(f"PDF 元数据缺少可匹配文件：{item.get('title') or item.get('candidate_id')}")
        item["_pdf_path"] = str(pdf_path.resolve())
        item.setdefault("source", "local_pdf")
        item.setdefault("concepts", [])
        item.setdefault("cited_by_count", 0)
        candidates.append(item)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一文献检索、PDF 后处理与综述可视化流程 v3")
    parser.add_argument("topic", nargs="?", default=DEFAULT_TOPIC, help="简单研究主题提示词")
    parser.add_argument("--sources", default="openalex,arxiv", help="检索源，逗号分隔：openalex,arxiv,ieee")
    parser.add_argument("--max-results", type=int, default=5, help="每个查询词在每个来源最多返回多少条结果")
    parser.add_argument("--max-papers", type=int, default=1, help="最多处理多少篇 PDF")
    parser.add_argument("--all-papers", action="store_true", help="处理能下载或本地已有的全部 PDF")
    parser.add_argument("--pdf-dir", type=Path, default=LIBRARY_PDF_DIR, help="PDF-only 模式使用的 PDF 目录")
    parser.add_argument("--from-pdf-only", action="store_true", help="跳过在线检索，直接从 --pdf-dir 中的 PDF 开始")
    parser.add_argument("--pdf-metadata-path", type=Path, default=None, help="PDF-only 模式可选元数据 JSON")
    parser.add_argument("--skip-search", action="store_true", help="兼容旧参数：等同于 --from-pdf-only")
    parser.add_argument("--single-direction-only", action="store_true", help="明确所有 PDF 属于同一方向，跳过 10 分方向")
    parser.add_argument("--single-only", action="store_true", help="兼容旧参数：等同于 --single-direction-only")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有中间结果")
    parser.add_argument("--output-dir", type=Path, default=None, help="统一流程输出目录")
    parser.add_argument("--extract-figures-tables", action="store_true", help="额外提取 PDF 图表截图和表格")
    parser.add_argument("--screen-only", action="store_true", help="只生成下载前方向筛选结果，不下载/分析 PDF")
    parser.add_argument("--screening-state", type=Path, default=None, help="复用已有下载前方向筛选状态继续运行")
    parser.add_argument("--selected-directions", default="", help="只保留指定方向 ID，逗号分隔，如 D1,D3")
    parser.add_argument("--journal-levels", type=Path, default=PROJECT_ROOT / "journal_levels.csv", help="期刊分区评分 CSV")
    parser.add_argument("--skip-ai-prescreen", action="store_true", help="旧参数已禁用：新版流程需要 10")
    parser.add_argument("--parallel-papers", type=int, default=1, help="并发处理方向内单篇富化数量")
    parser.add_argument("--filter-and", action="append", dest="filter_and_groups", default=None, help="AND 主题组：逗号分隔关键词。")
    parser.add_argument("--filter-or", action="append", dest="filter_or_groups", default=None, help="OR 主题组：逗号分隔关键词。")
    parser.add_argument("--filter-not", action="append", dest="filter_not_groups", default=None, help="NOT 主题组：逗号分隔关键词。")
    parser.add_argument("--filter-config", type=Path, default=None, help="JSON 过滤配置文件。")
    return parser.parse_args()


def _build_topic_filter(args: argparse.Namespace) -> TopicFilter | None:
    if args.filter_config is not None:
        return TopicFilter.from_config(args.filter_config)
    has_cli = args.filter_and_groups or args.filter_or_groups or args.filter_not_groups
    if not has_cli:
        return None
    return TopicFilter.from_cli_args(
        and_groups=[g.split(",") for g in (args.filter_and_groups or [])],
        or_groups=[g.split(",") for g in (args.filter_or_groups or [])],
        not_groups=[g.split(",") for g in (args.filter_not_groups or [])],
    )


def _selected_direction_ids(raw: str) -> list[str] | None:
    values = [part.strip() for part in (raw or "").split(",") if part.strip()]
    return values or None


def main() -> None:
    args = parse_args()
    db.init_db()
    args.from_pdf_only = bool(args.from_pdf_only or args.skip_search)
    args.single_direction_only = bool(args.single_direction_only or args.single_only)

    stamp = time.strftime("%Y%m%d_%H%M")
    output_dir = ensure_dir(args.output_dir or PROJECT_ROOT / "output" / f"{stamp}_{safe_output_name(args.topic)}")
    download_dir = ensure_dir(output_dir / "download")
    logs_dir = ensure_dir(output_dir / "logs")
    run_pdf_dir = ensure_dir(output_dir / "pdfs")
    analysis_output_dir = ensure_dir(output_dir / "analysis")
    figures_output_dir = output_dir / "figures_tables"
    review_output_dir = output_dir / "review_figures"

    max_papers = None if args.all_papers else args.max_papers
    sources = [source.strip().lower() for source in args.sources.split(",") if source.strip()]
    topic_for_model = f"{args.topic}。请主要使用中文输出，保留必要英文术语。"
    entry_mode = "pdf_only" if args.from_pdf_only else "search_to_pdf"
    if args.single_direction_only:
        entry_mode = "single_direction_only"

    report: dict[str, Any] = {
        "topic": args.topic,
        "topic_for_model": topic_for_model,
        "sources": sources,
        "pipeline_version": "pdf_postprocess_v3",
        "entry_mode": entry_mode,
        "direction_source": "",
        "deleted_legacy_flow": True,
        "legacy_steps_skipped": [
            "PDF 后二次方向划分",
            "方向 schema 层",
            "全集综合结构化",
            "旧全量修复",
            "旧综述图渲染",
            "默认文献关系图",
        ],
        "repair_events": [],
        "status": "running",
        "started_at": now_text(),
        "output_dir": str(output_dir.resolve()),
        "output_layout": {
            "download": str(download_dir.resolve()),
            "pdfs": str(run_pdf_dir.resolve()),
            "logs": str(logs_dir.resolve()),
            "analysis": str(analysis_output_dir.resolve()),
            "figures_tables": str(figures_output_dir.resolve()),
            "review_figures": str(review_output_dir.resolve()),
        },
        "steps": [],
        "papers": [],
    }
    save_json(output_dir / "unified_run_report.json", report)

    def prepare_papers() -> tuple[list[dict[str, Any]], list[Path]]:
        if args.from_pdf_only:
            pdf_paths = sorted(args.pdf_dir.glob("*.pdf"))
            if max_papers is not None:
                pdf_paths = pdf_paths[:max_papers]
            if not pdf_paths:
                raise RuntimeError(f"PDF-only 模式未找到 PDF：{args.pdf_dir}")
            candidates = load_pdf_metadata_candidates([path.resolve() for path in pdf_paths], args.pdf_metadata_path)
            if args.single_direction_only:
                state = build_virtual_single_direction_state(args.topic, candidates)
                report["direction_source"] = "user_single_direction"
            else:
                state = build_screening_state(args.topic, candidates, args.journal_levels)
                report["direction_source"] = "pdf_metadata_10"
            save_screening_state(state, download_dir)
            save_json(analysis_output_dir / "pdf_metadata_direction_mapping.json", state)
            if args.screen_only:
                save_json(download_dir / "selected_candidates.json", [])
                save_json(download_dir / "selected_pdfs.json", [])
                return candidates, []
            ranked = score_and_rank_candidates(
                topic=args.topic,
                state=state,
                selected_directions=_selected_direction_ids(args.selected_directions),
                journal_levels_path=args.journal_levels,
            )
            save_json(download_dir / "scored_candidates.json", ranked)
            selected_candidates = selected_for_download(ranked, max_papers)
            save_json(download_dir / "selected_candidates.json", selected_candidates)
            pdfs = [Path(str(item.get("_pdf_path"))).resolve() for item in selected_candidates if item.get("_pdf_path")]
            save_json(download_dir / "selected_pdfs.json", [str(path) for path in pdfs])
            return candidates, pdfs

        if args.skip_ai_prescreen:
            raise RuntimeError("新版流程需要 10 作为唯一方向来源，不能使用 --skip-ai-prescreen。")
        report["direction_source"] = "download_prescreen_10"
        return search_and_download(
            topic=args.topic,
            sources=sources,
            max_results=args.max_results,
            max_papers=max_papers,
            output_dir=download_dir,
            topic_filter=_build_topic_filter(args),
            ai_prescreen=True,
            screen_only=args.screen_only,
            screening_state_path=args.screening_state,
            selected_directions=_selected_direction_ids(args.selected_directions),
            journal_levels_path=args.journal_levels,
        )

    _, selected_pdfs = run_tracked_block("0. 文献检索/方向预筛/PDF 准备", report, output_dir, logs_dir, prepare_papers)
    if args.screen_only:
        report["status"] = "screening_completed"
        report["completed_at"] = now_text()
        report["screening_state"] = str((download_dir / "screening_state.json").resolve())
        save_json(output_dir / "unified_run_report.json", report)
        print(f"\n下载前方向筛选完成：{download_dir / 'screening_state.json'}")
        return
    if not selected_pdfs:
        raise RuntimeError("没有可处理的 PDF。请检查检索结果、网络连接，或先放入 PDF 到 library/pdfs。")

    report["source_papers"] = [str(path) for path in selected_pdfs]
    save_json(download_dir / "selected_source_pdfs.json", report["source_papers"])
    selected_pdfs = run_tracked_block(
        "0.1 PDF 归档到本次输出目录",
        report,
        output_dir,
        logs_dir,
        lambda: copy_pdfs_to_run(selected_pdfs, run_pdf_dir),
    )
    report["papers"] = [str(path) for path in selected_pdfs]
    save_json(download_dir / "selected_pdfs.json", report["papers"])
    save_json(output_dir / "unified_run_report.json", report)

    txt_dir = ensure_dir(analysis_output_dir / "txt_output")
    txt_paths = run_tracked_block(
        "1. PDF 正文提取",
        report,
        output_dir,
        logs_dir,
        lambda: convert_pdfs_to_txt(selected_pdfs, txt_dir, args.overwrite),
    )
    report["txt_output"] = [str(path) for path in txt_paths]
    save_json(output_dir / "unified_run_report.json", report)

    if args.extract_figures_tables:
        ensure_dir(figures_output_dir)
        for pdf_path in selected_pdfs:
            run_existing_script(
                f"2. 图表提取：{pdf_path.name}",
                "extract_pdf_figures_tables.py",
                ["--pdf", str(pdf_path), "--output-dir", str(figures_output_dir)],
                report,
                output_dir,
                logs_dir,
                required=True,
            )
    else:
        add_skipped_step(report, output_dir, logs_dir, "2. 图表提取", "默认不提取图表；如需提取请添加 --extract-figures-tables")

    screening_state = load_json(download_dir / "screening_state.json")
    selected_candidates = load_json(download_dir / "selected_candidates.json")
    direction_dirs = run_tracked_block(
        "3. 构建方向工作区",
        report,
        output_dir,
        logs_dir,
        lambda: build_direction_workspace(
            output_dir=output_dir,
            screening_state=screening_state,
            selected_candidates=selected_candidates,
            pdf_dir=run_pdf_dir,
            txt_dir=txt_dir,
            figures_dir=figures_output_dir if args.extract_figures_tables else None,
        ),
    )
    if not direction_dirs:
        raise RuntimeError("方向工作区为空，请检查 selected_directions 或 10 结果。")

    config = resolve_llm_config()
    client = build_client(config)
    timer = TimeRecorder()

    def run_all_directions() -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for direction_dir in direction_dirs:
            print(f"[方向处理] {direction_dir.name}")
            results.append(
                run_direction_pipeline(
                    direction_dir=direction_dir,
                    topic=topic_for_model,
                    client=client,
                    model=config.model,
                    flash_model=config.flash_model,
                    overwrite=args.overwrite,
                    parallel_papers=args.parallel_papers,
                    timer=timer,
                )
            )
        return results

    direction_results = run_tracked_block("4. 方向内富化、综述与单方向图", report, output_dir, logs_dir, run_all_directions)
    report["directions"] = [
        {key: value for key, value in item.items() if key not in {"plot_ready", "repair_events"}}
        for item in direction_results
    ]
    for item in direction_results:
        report["repair_events"].extend(item.get("repair_events", []))
    save_json(output_dir / "unified_run_report.json", report)

    corpus_outputs, cross_repairs = run_tracked_block(
        "5. 跨方向总综述与总图",
        report,
        output_dir,
        logs_dir,
        lambda: run_cross_direction_outputs(
            output_dir=output_dir,
            topic=topic_for_model,
            direction_results=direction_results,
            client=client,
            model=config.model,
            overwrite=args.overwrite,
            timer=timer,
        ),
    )
    report["corpus_outputs"] = corpus_outputs
    report["repair_events"].extend(cross_repairs)
    timer.save(output_dir / "time_records")

    report["status"] = "completed"
    report["completed_at"] = now_text()
    save_json(output_dir / "unified_run_report.json", report)
    print(f"\n统一流程完成。运行报告：{output_dir / 'unified_run_report.json'}")


if __name__ == "__main__":
    main()
