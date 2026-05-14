from __future__ import annotations

import argparse
import csv
import contextlib
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
    command: list[str] | None = None,
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
    if command is not None:
        step["command"] = command
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
    returncode: int | None = None,
    reason: str | None = None,
) -> None:
    step["status"] = status
    step["end_time"] = now_text()
    step["elapsed_seconds"] = round(time.time() - started, 3)
    if returncode is not None:
        step["returncode"] = returncode
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
    print(f"[跳过] {name}：{reason}")
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
    finish_step(report, output_dir, logs_dir, step, status="completed", started=started, returncode=0)
    return result


def run_command(
    name: str,
    command: list[str],
    report: dict[str, Any],
    output_dir: Path,
    logs_dir: Path,
) -> int:
    started = time.time()
    step, log_path = start_step(report, output_dir, logs_dir, name, command)
    print(f"\n=== {name} ===")
    print(" ".join(command))
    started = time.time()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"=== {name} ===\n")
        log.write(f"开始时间：{step['start_time']}\n")
        log.write("命令：\n" + " ".join(command) + "\n\n")
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
            log.flush()
        returncode = process.wait()
        log.write(f"\n结束时间：{now_text()}\n")
        log.write(f"退出码：{returncode}\n")

    finish_step(
        report,
        output_dir,
        logs_dir,
        step,
        status="completed" if returncode == 0 else "failed",
        started=started,
        returncode=returncode,
    )
    return returncode


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
    if not script_path.exists():
        print(f"\n=== {name} ===")
        print(f"跳过：未找到 {script_name}")
        add_skipped_step(report, output_dir, logs_dir, name, f"本地未找到脚本：{script_name}")
        return not required

    returncode = run_command(name, [sys.executable, str(script_path), *args], report, output_dir, logs_dir)
    if returncode != 0 and required:
        report["status"] = "failed"
        report["failed_at"] = now_text()
        report["failure"] = f"{name} 失败，退出码：{returncode}"
        save_json(output_dir / "unified_run_report.json", report)
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


def selected_pdf_args(pdf_paths: list[Path]) -> list[str]:
    args: list[str] = []
    for path in pdf_paths:
        args.extend(["--file", path.name])
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一文献检索、PDF结构化与综述可视化流程")
    parser.add_argument("topic", nargs="?", default=DEFAULT_TOPIC, help="简单研究主题提示词")
    parser.add_argument("--sources", default="openalex,arxiv", help="检索源，逗号分隔：openalex,arxiv,ieee")
    parser.add_argument("--max-results", type=int, default=5, help="每个查询词在每个来源最多返回多少条结果")
    parser.add_argument("--max-papers", type=int, default=1, help="最多处理多少篇 PDF；测试建议保留 1")
    parser.add_argument("--all-papers", action="store_true", help="处理能下载或本地已有的全部 PDF")
    parser.add_argument("--skip-search", action="store_true", help="跳过在线检索，直接使用 library/pdfs 中的 PDF")
    parser.add_argument("--single-only", action="store_true", help="只生成单篇正文结构化结果，不进入方向识别和综述图")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有中间结果")
    parser.add_argument("--output-dir", type=Path, default=None, help="统一流程输出目录")
    parser.add_argument("--extract-figures-tables", action="store_true", help="额外提取 PDF 图表截图和表格")
    parser.add_argument("--extract-formulas", action="store_true", help="额外提取公式截图并执行公式 OCR")
    parser.add_argument("--skip-formulas", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--screen-only", action="store_true", help="只生成下载前方向筛选结果，不下载 PDF")
    parser.add_argument("--screening-state", type=Path, default=None, help="复用已有下载前方向筛选状态继续运行")
    parser.add_argument("--selected-directions", default="", help="只保留指定方向 ID，逗号分隔，如 D1,D3")
    parser.add_argument("--journal-levels", type=Path, default=PROJECT_ROOT / "journal_levels.csv", help="期刊分区评分 CSV")
    parser.add_argument("--skip-ai-prescreen", action="store_true", help="跳过下载前 AI 方向筛选与排序，使用旧下载流程")
    parser.add_argument("--parallel-papers", type=int, default=1, help="并发处理单篇全文结构化数量，默认 1")
    parser.add_argument(
        "--filter-and", action="append", dest="filter_and_groups", default=None,
        help="AND 主题组：逗号分隔关键词，论文必须包含组内至少一个词。可重复使用。",
    )
    parser.add_argument(
        "--filter-or", action="append", dest="filter_or_groups", default=None,
        help="OR 主题组：逗号分隔关键词，论文至少命中一组。可重复使用。",
    )
    parser.add_argument(
        "--filter-not", action="append", dest="filter_not_groups", default=None,
        help="NOT 主题组：逗号分隔关键词，命中即排除。可重复使用。",
    )
    parser.add_argument(
        "--filter-config", type=Path, default=None,
        help="JSON 过滤配置文件（优先级高于 --filter-* 参数）",
    )
    return parser.parse_args()


def _build_topic_filter(args: argparse.Namespace) -> TopicFilter | None:
    """Construct TopicFilter from parsed CLI args or config file."""
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

    stamp = time.strftime("%Y%m%d_%H%M")
    output_dir = args.output_dir or PROJECT_ROOT / "output" / f"{stamp}_{safe_output_name(args.topic)}"
    output_dir = ensure_dir(output_dir)
    download_dir = ensure_dir(output_dir / "download")
    logs_dir = ensure_dir(output_dir / "logs")
    run_pdf_dir = ensure_dir(output_dir / "pdfs")
    analysis_output_dir = ensure_dir(output_dir / "analysis")
    figures_output_dir = output_dir / "figures_tables"
    formula_output_dir = output_dir / "formulas" / "regions"
    ocr_output_dir = output_dir / "formulas" / "ocr"
    review_output_dir = output_dir / "review_figures"

    max_papers = None if args.all_papers else args.max_papers
    sources = [source.strip().lower() for source in args.sources.split(",") if source.strip()]
    topic_for_model = f"{args.topic}。请主要使用中文输出，保留必要英文术语。"

    report: dict[str, Any] = {
        "topic": args.topic,
        "topic_for_model": topic_for_model,
        "sources": sources,
        "status": "running",
        "started_at": now_text(),
        "output_dir": str(output_dir.resolve()),
        "output_layout": {
            "download": str(download_dir.resolve()),
            "pdfs": str(run_pdf_dir.resolve()),
            "logs": str(logs_dir.resolve()),
            "analysis": str(analysis_output_dir.resolve()),
            "figures_tables": str(figures_output_dir.resolve()),
            "formulas": str((output_dir / "formulas").resolve()),
            "review_figures": str(review_output_dir.resolve()),
        },
        "steps": [],
        "papers": [],
    }
    save_json(output_dir / "unified_run_report.json", report)

    def prepare_papers() -> tuple[list[dict[str, Any]], list[Path]]:
        if args.skip_search:
            selected_file = download_dir / "selected_pdfs.json"
            if selected_file.exists():
                pdfs = [Path(path).resolve() for path in json.loads(selected_file.read_text(encoding="utf-8"))]
                print(f"[检索跳过] 复用已有选中文献清单：{selected_file}")
            else:
                pdf_paths = sorted(LIBRARY_PDF_DIR.glob("*.pdf"))
                if max_papers is not None:
                    pdf_paths = pdf_paths[:max_papers]
                pdfs = [path.resolve() for path in pdf_paths]
                print(f"[检索跳过] 使用 library/pdfs 中的 PDF：{len(pdfs)} 篇")
            return [], pdfs

        print(f"检索主题：{args.topic}")
        return search_and_download(
            topic=args.topic,
            sources=sources,
            max_results=args.max_results,
            max_papers=max_papers,
            output_dir=download_dir,
            topic_filter=_build_topic_filter(args),
            ai_prescreen=not args.skip_ai_prescreen,
            screen_only=args.screen_only,
            screening_state_path=args.screening_state,
            selected_directions=_selected_direction_ids(args.selected_directions),
            journal_levels_path=args.journal_levels,
        )

    search_results, selected_pdfs = run_tracked_block(
        "0. 文献检索与下载",
        report,
        output_dir,
        logs_dir,
        prepare_papers,
    )

    if args.screen_only:
        report["status"] = "screening_completed"
        report["completed_at"] = now_text()
        report["screening_state"] = str((download_dir / "screening_state.json").resolve())
        save_json(output_dir / "unified_run_report.json", report)
        print(f"\n下载前方向筛选完成：{download_dir / 'screening_state.json'}")
        return

    if not selected_pdfs:
        report["status"] = "failed"
        report["failed_at"] = now_text()
        report["failure"] = "没有可处理的 PDF"
        save_json(output_dir / "unified_run_report.json", report)
        raise RuntimeError("没有可处理的 PDF。请检查检索结果、网络连接，或先放入 PDF 到 library/pdfs。")

    report["source_papers"] = [str(path) for path in selected_pdfs]
    save_json(download_dir / "selected_source_pdfs.json", report["source_papers"])

    selected_pdfs = run_tracked_block(
        "0.1 PDF归档到本次输出目录",
        report,
        output_dir,
        logs_dir,
        lambda: copy_pdfs_to_run(selected_pdfs, run_pdf_dir),
    )
    report["papers"] = [str(path) for path in selected_pdfs]
    save_json(download_dir / "selected_pdfs.json", report["papers"])
    save_json(output_dir / "unified_run_report.json", report)

    print(f"[准备完成] 本次将分析 {len(selected_pdfs)} 篇 PDF。")
    print(f"[输出目录] {output_dir}")

    common_pdf_args = ["--pdf-dir", str(run_pdf_dir), *selected_pdf_args(selected_pdfs)]
    pipeline_args = [
        *common_pdf_args,
        "--output-dir",
        str(analysis_output_dir),
        "--topic",
        topic_for_model,
    ]
    if args.single_only:
        pipeline_args.append("--single-only")
    if args.overwrite:
        pipeline_args.append("--overwrite")
    if args.parallel_papers and args.parallel_papers > 1:
        pipeline_args.extend(["--parallel-papers", str(args.parallel_papers)])

    run_existing_script(
        "1. 正文结构化",
        "multi_paper_structured_pipeline_v2.py",
        pipeline_args,
        report,
        output_dir,
        logs_dir,
        required=True,
    )

    if args.extract_figures_tables:
        ensure_dir(figures_output_dir)
        for pdf_path in selected_pdfs:
            figure_args = [
                "--pdf",
                str(pdf_path),
                "--output-dir",
                str(figures_output_dir),
            ]
            run_existing_script(
                f"2. 图表提取：{pdf_path.name}",
                "extract_pdf_figures_tables.py",
                figure_args,
                report,
                output_dir,
                logs_dir,
                required=True,
            )
    else:
        add_skipped_step(
            report,
            output_dir,
            logs_dir,
            "2. 图表提取",
            "默认一键流程不提取图表；如需提取请添加 --extract-figures-tables",
        )

    if not args.extract_formulas or args.skip_formulas:
        add_skipped_step(
            report,
            output_dir,
            logs_dir,
            "3-4. 公式提取与 OCR",
            "默认一键流程不提取公式；如需提取请添加 --extract-formulas",
        )
    else:
        ensure_dir(formula_output_dir)
        ensure_dir(ocr_output_dir)
        for pdf_path in selected_pdfs:
            formula_args = [
                "--pdf",
                str(pdf_path),
                "--output-dir",
                str(formula_output_dir),
            ]
            if args.overwrite:
                formula_args.append("--overwrite")
            run_existing_script(
                f"3. 公式截图提取：{pdf_path.name}",
                "extract_pdf_formula_regions_v2.py",
                formula_args,
                report,
                output_dir,
                logs_dir,
                required=True,
            )

        ocr_args = ["--input-dir", str(formula_output_dir), "--output-dir", str(ocr_output_dir)]
        if args.overwrite:
            ocr_args.append("--overwrite")
        run_existing_script(
            "4. 公式 OCR",
            "ocr_formula_images_pix2tex.py",
            ocr_args,
            report,
            output_dir,
            logs_dir,
            required=True,
        )

    if args.single_only:
        add_skipped_step(
            report,
            output_dir,
            logs_dir,
            "5. 综述可视化图",
            "single-only 模式不会生成 directions/comparisons",
        )
    else:
        run_existing_script(
            "5. 综述可视化图",
            "generate_review_figures.py",
            ["--input-dir", str(analysis_output_dir), "--output-dir", str(review_output_dir)],
            report,
            output_dir,
            logs_dir,
            required=True,
        )

    report["status"] = "completed"
    report["completed_at"] = now_text()
    save_json(output_dir / "unified_run_report.json", report)
    print(f"\n统一流程完成。运行报告：{output_dir / 'unified_run_report.json'}")


if __name__ == "__main__":
    main()
