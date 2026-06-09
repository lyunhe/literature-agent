from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from markupsafe import Markup


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent if (PROJECT_ROOT.parent / "tools" / "repro_cli.py").exists() else PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from analysis_pipeline.core.run_context import safe_output_name
except Exception:
    def safe_output_name(value: str, max_length: int = 48) -> str:
        import re

        cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", str(value or "").strip())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return (cleaned or "run")[:max_length]

OUTPUT_ROOT = PROJECT_ROOT / "output"
INPUT_PDF_DIR = PROJECT_ROOT / "input_pdfs"
LIBRARY_PDF_DIR = PROJECT_ROOT / "library" / "pdfs"
PIPELINE_SCRIPT = PROJECT_ROOT / "analysis_pipeline" / "unified_literature_pipeline.py"
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
REPRO_PYTHON_EXE = REPO_ROOT / ".venv" / "bin" / "python"
SHOWCASE_DATA_PATH = APP_DIR / "data" / "sample_three_stage_review.json"
SHOWCASE_OUTPUT_NAME = "three_stage_review.json"
QUALITY_OUTPUT_NAME = "quality_report.json"
WEB_JOB_STATUS_NAME = "web_job_status.json"
REPRO_RUN_ROOT = REPO_ROOT / "runs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from tools.llm_client import LLMError, call_openai_text
except Exception:
    LLMError = RuntimeError
    call_openai_text = None

app = Flask(__name__)
jobs: dict[str, dict[str, Any]] = {}
repro_jobs: dict[str, dict[str, Any]] = {}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def persist_job(job: dict[str, Any]) -> None:
    output_dir = Path(str(job.get("output_dir") or ""))
    if not output_dir:
        return
    fields = [
        "id",
        "status",
        "topic",
        "base_topic",
        "output_dir",
        "max_papers",
        "downloadable_limit",
        "run_parts",
        "year_from",
        "year_to",
        "started_at",
        "completed_at",
        "returncode",
        "pid",
        "stderr",
        "last_message",
        "log_tail",
    ]
    save_json(output_dir / WEB_JOB_STATUS_NAME, {key: job.get(key) for key in fields if key in job})


def load_persisted_job(job_id: str) -> dict[str, Any] | None:
    if not OUTPUT_ROOT.exists():
        return None
    for path in sorted(OUTPUT_ROOT.glob(f"*/{WEB_JOB_STATUS_NAME}"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = load_json(path, {})
        if isinstance(payload, dict) and str(payload.get("id") or "") == str(job_id):
            payload.setdefault("output_dir", str(path.parent))
            return payload
    return None


def terminal_report_status(job: dict[str, Any]) -> str:
    output_dir = Path(str(job.get("output_dir") or ""))
    report = load_json(output_dir / "unified_run_report.json", {}) if output_dir else {}
    if not isinstance(report, dict):
        return ""
    status = str(report.get("status") or "")
    if status == "screening_completed":
        return "completed"
    return status if status in {"completed", "failed"} else ""


def list_len_json(path: Path, key: str | None = None) -> int:
    data = load_json(path, None)
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        if key and isinstance(data.get(key), list):
            return len(data[key])
        for candidate in ("papers", "items", "results", "directions"):
            if isinstance(data.get(candidate), list):
                return len(data[candidate])
    return 0


def safe_count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob(pattern) if path.is_file())


def reviewed_paper_count(reviews_dir: Path) -> int:
    if not reviews_dir.exists():
        return 0
    paper_cards = reviews_dir / "paper_cards"
    if paper_cards.exists():
        count = safe_count_files(paper_cards, "*.json")
        if count:
            return count
    return safe_count_files(reviews_dir / "directions", "P*.json")


def reviewed_direction_count(reviews_dir: Path) -> int:
    if not reviews_dir.exists():
        return 0
    count = safe_count_files(reviews_dir / "directions", "direction_review.md")
    if count:
        return count
    return safe_count_files(reviews_dir / "directions", "direction_review_summary.json")


def stage_code(name: str, index: int = 0) -> str:
    import re

    match = re.match(r"\s*(\d+(?:\.\d+)*)", str(name or ""))
    return match.group(1) if match else str(index or "")


def stage_short_label(name: str) -> str:
    text = str(name or "")
    if "corpus literature review" in text:
        return "总综述"
    if "paper cards" in text or "direction reviews" in text:
        return "单篇与方向综述"
    if "构建方向工作区" in text:
        return "方向工作区"
    if "图表提取" in text or ("图表" in text and "提取" in text):
        return "图表截取"
    if "PDF 正文提取" in text:
        return "正文提取"
    if "PDF 归档" in text:
        return "PDF归档"
    if "发现阶段" in text:
        return "检索筛选下载"
    return text[:18] or "阶段"


def step_elapsed_seconds(step: dict[str, Any], *, running: bool = False) -> float | str:
    raw_elapsed = step.get("elapsed_seconds")
    if raw_elapsed not in {"", None}:
        try:
            return round(float(raw_elapsed), 1)
        except (TypeError, ValueError):
            return raw_elapsed
    if not running:
        return ""
    start_time = str(step.get("start_time") or "")
    if not start_time:
        return ""
    try:
        started = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ""
    return round(max(0.0, time.time() - started.timestamp()), 1)


def major_stage_code(step: dict[str, Any]) -> str:
    code = stage_code(str(step.get("name") or ""), int(step.get("index") or 0))
    return code.split(".", 1)[0] if code else ""


def compact_steps(steps: list[dict[str, Any]], current_step: dict[str, Any] | None) -> list[dict[str, Any]]:
    phases: dict[str, dict[str, Any]] = {
        "0": {"code": "1", "label": "文献检索筛选与下载", "status": "", "elapsed_seconds": 0.0},
        "1": {"code": "2", "label": "单篇文献与方向综述", "status": "", "elapsed_seconds": 0.0},
        "2": {"code": "3", "label": "总体综述", "status": "", "elapsed_seconds": 0.0},
    }

    def absorb(step: dict[str, Any]) -> None:
        parent = major_stage_code(step)
        if parent not in phases:
            return
        raw_status = str(step.get("status") or "")
        is_running = raw_status == "running"
        elapsed = step_elapsed_seconds(step, running=is_running)
        if isinstance(elapsed, (int, float)):
            phases[parent]["elapsed_seconds"] = round(float(phases[parent]["elapsed_seconds"]) + float(elapsed), 1)
        if is_running:
            phases[parent]["status"] = "running"
        elif raw_status == "failed":
            phases[parent]["status"] = "failed"
        elif phases[parent]["status"] != "running":
            phases[parent]["status"] = "completed"

    for step in steps:
        absorb(step)
    if current_step:
        absorb({**current_step, "status": "running"})

    ordered = [phases["0"], phases["1"], phases["2"]]
    for index, phase in enumerate(ordered):
        if phase["status"]:
            continue
        previous_done = all(item["status"] == "completed" for item in ordered[:index])
        phase["status"] = "pending" if previous_done or index == 0 else ""
        phase["elapsed_seconds"] = ""
    return ordered


def quality_notes(output_dir: Path | None, target_papers: Any, downloaded_papers: int) -> list[str]:
    notes: list[str] = []
    if not output_dir:
        return notes
    try:
        target_count = int(target_papers)
    except (TypeError, ValueError):
        target_count = 0
    discovery_summary = output_dir / "01_discovery" / "discovery_summary.md"
    raw_candidates = list_len_json(output_dir / "01_discovery" / "raw_candidates.json")
    downloadable = list_len_json(output_dir / "01_discovery" / "downloadable_candidates.json")
    if not raw_candidates and not downloaded_papers:
        notes.append("在线检索没有返回题录候选，请放宽主题条件、检查检索源连接，或降低年份限制。")
    if target_count and downloaded_papers and target_count > downloaded_papers:
        notes.append(f"在线检索只成功下载 {downloaded_papers}/{target_count} 篇 PDF。")
    elif target_count and downloadable and target_count > downloadable:
        notes.append(f"可验证 PDF 候选不足：{downloadable}/{target_count}。")
    if discovery_summary.exists():
        text = discovery_summary.read_text(encoding="utf-8", errors="replace")
        if "Only" in text and "verified downloadable PDFs" in text:
            notes.append("开放可下载 PDF 数量不足，部分候选只有题录或无法直连 PDF。")
    quality = load_json(output_dir / QUALITY_OUTPUT_NAME, {})
    if isinstance(quality, dict) and quality.get("status") in {"fail", "warn"}:
        for check in quality.get("checks") or []:
            if not isinstance(check, dict) or check.get("status") == "pass":
                continue
            if check.get("id") == "paper_total":
                notes.append(f"质量检查：目标 {check.get('expected')} 篇，实际 {check.get('actual')} 篇。")
            elif check.get("id") == "direction_minimum":
                notes.append(f"质量检查：方向编号为 {check.get('actual')}，未满足至少 D1/D2。")
            else:
                notes.append(f"质量检查：{check.get('name')} 为 {check.get('status')}。")
    return notes[:4]


def compact_progress_steps_for_job(
    steps: list[dict[str, Any]],
    current_step: dict[str, Any] | None,
    job_status: str,
    downloaded_papers: int,
) -> list[dict[str, Any]]:
    compacted = compact_steps(steps, current_step)
    if job_status == "failed" and not downloaded_papers:
        for item in compacted:
            if item.get("code") == "1":
                item["status"] = "failed"
                break
        for item in compacted:
            if item.get("code") in {"2", "3"} and not item.get("elapsed_seconds"):
                item["status"] = "pending" if item.get("code") == "2" else ""
    return compacted


def public_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in job.items()
        if key not in {"stdout", "stderr", "log_tail", "last_message", "command"}
    }
    progress = summarize_job_progress(job)
    payload["progress"] = progress
    if progress.get("status"):
        payload["status"] = progress["status"]
    if payload.get("status") == "failed":
        raw_error = str(job.get("stderr") or job.get("last_message") or "")
        if "PDF" in raw_error and "input_pdfs" in raw_error:
            payload["error"] = "没有可处理的 PDF。请增加可下载候选上限、调整主题条件，或先将 PDF 放入 input_pdfs。"
        else:
            payload["error"] = raw_error[:500] or "未知错误"
    return payload


def progress_phase(report: dict[str, Any], job: dict[str, Any]) -> tuple[str, int]:
    status = str(job.get("status") or report.get("status") or "queued")
    current = report.get("current_step") if isinstance(report.get("current_step"), dict) else {}
    current_name = str(current.get("name") or "")
    steps = report.get("steps") if isinstance(report.get("steps"), list) else []
    completed_names = " ".join(str(step.get("name") or "") for step in steps if isinstance(step, dict))

    if status == "queued":
        return "等待启动", 3
    if status == "failed":
        previous = job.get("progress") if isinstance(job.get("progress"), dict) else {}
        return "运行失败", max(5, int(previous.get("percent") or 5))
    if status == "completed" or report.get("status") == "completed":
        return "已完成展示数据生成", 100
    if "corpus literature review" in current_name:
        return "生成总综述与领域综合洞察", 92
    if "paper cards" in current_name or "direction reviews" in current_name:
        return "分析单篇论文与方向综述", 76
    if "构建方向工作区" in current_name:
        return "构建方向分类工作区", 62
    if "图表提取" in current_name:
        return "截取关键图表与表格", 52
    if "PDF 正文提取" in current_name:
        return "提取 PDF 正文", 46
    if "PDF 归档" in current_name:
        return "归档已下载论文", 40
    if "发现阶段" in current_name or not completed_names:
        return "检索、筛选并下载论文", 28
    if "paper cards" in completed_names:
        return "汇总跨方向综述", 88
    if "发现阶段" in completed_names:
        return "准备进入论文分析", 66
    return "后台运行中", 20


def weighted_progress_percent(
    report: dict[str, Any],
    job: dict[str, Any],
    target_papers: Any,
    downloaded_papers: int,
    analyzed_papers: int,
) -> int:
    status = str(job.get("status") or report.get("status") or "queued")
    if status == "queued":
        return 3
    if status == "failed":
        previous = job.get("progress") if isinstance(job.get("progress"), dict) else {}
        return max(5, int(previous.get("percent") or 5))
    if status == "completed" or report.get("status") == "completed":
        return 100

    try:
        target = max(1, int(target_papers or job.get("max_papers") or downloaded_papers or 1))
    except (TypeError, ValueError):
        target = max(1, downloaded_papers or 1)

    steps = report.get("steps") if isinstance(report.get("steps"), list) else []
    current_step = report.get("current_step") if isinstance(report.get("current_step"), dict) else {}
    current_major = major_stage_code(current_step) if current_step else ""
    completed_majors = {
        major_stage_code(step)
        for step in steps
        if isinstance(step, dict) and str(step.get("status") or "") == "completed"
    }

    discovery_part = min(40, round(downloaded_papers / target * 40)) if downloaded_papers else 5
    if "0" in completed_majors or current_major in {"1", "2"}:
        discovery_part = 40

    review_part = min(40, round(analyzed_papers / target * 40)) if analyzed_papers else 0
    if "1" in completed_majors or current_major == "2":
        review_part = 40

    corpus_part = 0
    if current_major == "2":
        corpus_part = 10
    if "2" in completed_majors:
        corpus_part = 20

    return max(3, min(99, int(discovery_part + review_part + corpus_part)))


def summarize_job_progress(job: dict[str, Any]) -> dict[str, Any]:
    output_dir_text = str(job.get("output_dir") or "")
    output_dir = Path(output_dir_text) if output_dir_text else None
    report = load_json(output_dir / "unified_run_report.json", {}) if output_dir else {}
    if not isinstance(report, dict):
        report = {}
    report_status = str(report.get("status") or "")
    effective_job = dict(job)
    if report_status == "screening_completed":
        effective_job["status"] = "completed"
    elif report_status in {"completed", "failed"}:
        effective_job["status"] = report_status
    discovery_dir = output_dir / "01_discovery" if output_dir else Path()
    reviews_dir = output_dir / "02_reviews" if output_dir else Path()
    stage_label, percent = progress_phase(report, effective_job)
    quality = load_json(output_dir / QUALITY_OUTPUT_NAME, {}) if output_dir else {}
    if isinstance(quality, dict) and quality.get("status") == "fail":
        stage_label = "已完成，但质量检查未通过"
    elif isinstance(quality, dict) and quality.get("status") == "warn":
        stage_label = "已完成，有质量提醒"
    target_papers = report.get("max_papers") or job.get("max_papers") or ""
    selected_count = (
        list_len_json(discovery_dir / "selected_pdfs.json")
        or safe_count_files(discovery_dir / "pdfs", "*.pdf")
        or len(report.get("papers") or [])
    )
    analyzed_count = reviewed_paper_count(reviews_dir)
    direction_reviewed_count = reviewed_direction_count(reviews_dir)
    figure_count = safe_count_files(discovery_dir / "figures_tables", "*.png") + safe_count_files(discovery_dir / "figures_tables", "*.jpg")
    steps = report.get("steps") if isinstance(report.get("steps"), list) else []
    current_step = report.get("current_step") if isinstance(report.get("current_step"), dict) else None
    percent = weighted_progress_percent(report, effective_job, target_papers, selected_count, analyzed_count)
    progress_status = str(effective_job.get("status") or report.get("status") or "queued")
    return {
        "status": progress_status,
        "stage_label": stage_label,
        "percent": percent,
        "output_dir": output_dir_text,
        "run_id": output_dir.name if output_dir else "",
        "target_papers": target_papers,
        "searched_papers": list_len_json(discovery_dir / "search_results.json") or list_len_json(discovery_dir / "raw_candidates.json"),
        "filtered_papers": list_len_json(discovery_dir / "filtered_results.json"),
        "downloadable_papers": list_len_json(discovery_dir / "downloadable_candidates.json"),
        "downloaded_papers": selected_count,
        "analyzed_papers": analyzed_count,
        "direction_count": list_len_json(discovery_dir / "direction_workspace_manifest.json") or len(report.get("directions") or []),
        "reviewed_directions": direction_reviewed_count,
        "figure_count": figure_count,
        "completed_steps": len(steps),
        "current_step": current_step,
        "steps": compact_progress_steps_for_job(steps, current_step, progress_status, selected_count),
        "notes": quality_notes(output_dir, target_papers, selected_count),
        "paper_progress": f"{analyzed_count}/{target_papers or selected_count or 0}",
    }


@app.context_processor
def template_helpers() -> dict[str, Any]:
    def render_table(table: dict[str, Any]) -> Markup:
        headers = list(table.get("headers") or [])
        rows = list(table.get("rows") or [])
        if not headers or not rows:
            return Markup('<p class="muted">鏆傛棤鐭╅樀鏁版嵁銆?/p>')
        head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
        body_rows = []
        for row in rows:
            cells = "".join(f"<td>{escape(str(row.get(header, '')))}</td>" for header in headers)
            body_rows.append(f"<tr>{cells}</tr>")
        return Markup(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>')

    return {"render_table": render_table}


@dataclass(frozen=True)
class RunRef:
    run_id: str
    path: Path
    modified: float


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def export_showcase_data(run_path: Path) -> Path | None:
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from analysis_pipeline.stages.showcase_export import write_three_stage_review

        return write_three_stage_review(run_path)
    except Exception as exc:
        app.logger.warning("Unable to export showcase JSON for %s: %s", run_path, exc)
        return None


def showcase_needs_refresh(run_path: Path, target: Path) -> bool:
    if not target.exists():
        return True
    target_mtime = target.stat().st_mtime
    source_roots = [run_path / "01_discovery", run_path / "02_reviews"]
    source_files = [Path(__file__).resolve().parents[1] / "analysis_pipeline" / "stages" / "showcase_export.py"]
    for root in source_roots:
        if root.exists():
            source_files.extend(path for path in root.rglob("*") if path.is_file())
    return any(path.exists() and path.stat().st_mtime > target_mtime for path in source_files)


def load_showcase_data(run_id: str = "") -> dict[str, Any]:
    if run_id:
        run_path = safe_child(OUTPUT_ROOT, run_id)
        target = run_path / SHOWCASE_OUTPUT_NAME
        if showcase_needs_refresh(run_path, target):
            export_showcase_data(run_path)
        data = load_json(target, {})
        if isinstance(data, dict) and data.get("directions"):
            return enrich_showcase_with_reproduction(data, run_path, run_id)
    else:
        selected_id = select_default_run(iter_runs())
        if selected_id:
            data = load_showcase_data(selected_id)
            if data:
                return data
    data = load_json(SHOWCASE_DATA_PATH, {})
    return data if isinstance(data, dict) else {}


def load_quality_report(run_id: str = "") -> dict[str, Any]:
    selected_id = run_id or select_default_run(iter_runs())
    if not selected_id:
        return {}
    run_path = safe_child(OUTPUT_ROOT, selected_id)
    target = run_path / QUALITY_OUTPUT_NAME
    if showcase_needs_refresh(run_path, target):
        export_showcase_data(run_path)
    data = load_json(target, {})
    return data if isinstance(data, dict) else {}


def load_query_meta(run_id: str = "") -> dict[str, Any]:
    if not run_id:
        return {}
    from analysis_pipeline.stages.discovery.topic_filtering import BILINGUAL_KEYWORD_EXPANSIONS

    run_path = safe_child(OUTPUT_ROOT, run_id)
    report = load_json(run_path / "unified_run_report.json", {})
    input_mode = load_json(run_path / "01_discovery" / "input_mode.json", {})
    filter_config = load_json(run_path / "01_discovery" / "filter_config.json", {})
    conditions: list[dict[str, Any]] = []
    condition_input_terms: set[str] = set()
    logic_labels = {"AND": "且", "OR": "或", "NOT": "非"}
    for group in filter_config.get("groups", []) if isinstance(filter_config, dict) else []:
        if not isinstance(group, dict):
            continue
        keywords = [str(item) for item in group.get("keywords", []) if str(item).strip()]
        input_keywords = [str(item) for item in group.get("input_keywords", []) if str(item).strip()]
        expanded_keywords = [str(item) for item in group.get("expanded_keywords", []) if str(item).strip()]
        if not input_keywords and keywords:
            input_keywords = keywords[:1]
        condition_input_terms.update(item.strip().lower() for item in input_keywords if item.strip())
        if not expanded_keywords and keywords:
            input_set = {item.lower() for item in input_keywords}
            expanded_keywords = [item for item in keywords if item.lower() not in input_set]
        if not expanded_keywords:
            expanded_keywords = []
            for item in input_keywords:
                expanded_keywords.extend(BILINGUAL_KEYWORD_EXPANSIONS.get(item, []))
        logic = str(group.get("logic") or "").upper()
        conditions.append(
            {
                "logic": logic_labels.get(logic, logic or "条件"),
                "logic_raw": logic,
                "input_keywords": input_keywords,
                "expanded_keywords": expanded_keywords,
            }
        )
    topic = str(report.get("topic") or input_mode.get("topic") or run_id)
    topic_expanded: list[str] = []
    for key, values in BILINGUAL_KEYWORD_EXPANSIONS.items():
        if key and key in topic and key.strip().lower() not in condition_input_terms:
            topic_expanded.extend(values)
    seen_topic_terms: set[str] = set()
    topic_expanded_keywords = []
    for item in topic_expanded:
        normalized = item.lower()
        if normalized in seen_topic_terms:
            continue
        seen_topic_terms.add(normalized)
        topic_expanded_keywords.append(item)

    return {
        "topic": topic,
        "topic_expanded_keywords": topic_expanded_keywords,
        "input_mode": str(report.get("input_mode") or input_mode.get("input_mode") or ""),
        "year_from": report.get("year_from") or "",
        "year_to": report.get("year_to") or "",
        "conditions": conditions,
    }


def select_default_run(runs: list[RunRef]) -> str:
    showcase_runs = [
        RunRef(run.run_id, run.path, (run.path / SHOWCASE_OUTPUT_NAME).stat().st_mtime)
        for run in runs
        if (run.path / SHOWCASE_OUTPUT_NAME).exists()
    ]
    if showcase_runs:
        return sorted(showcase_runs, key=lambda item: item.modified, reverse=True)[0].run_id
    for run in runs:
        if (run.path / "02_reviews").exists():
            return run.run_id
    return runs[0].run_id if runs else ""


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if limit is None else text[:limit]


def safe_child(root: Path, *parts: str) -> Path:
    root = root.resolve()
    target = root.joinpath(*parts).resolve()
    if root != target and root not in target.parents:
        abort(404)
    return target


def iter_runs() -> list[RunRef]:
    if not OUTPUT_ROOT.exists():
        return []
    runs: list[RunRef] = []
    for path in OUTPUT_ROOT.iterdir():
        if not path.is_dir():
            continue
        if not (path / "unified_run_report.json").exists() and not (path / "01_discovery").exists():
            continue
        runs.append(RunRef(path.name, path, path.stat().st_mtime))
    return sorted(runs, key=lambda item: item.modified, reverse=True)


def run_title(run: RunRef) -> str:
    report = load_json(run.path / "unified_run_report.json", {})
    if not isinstance(report, dict):
        report = {}
    return str(report.get("topic") or run.run_id)


def run_card_meta(run: RunRef) -> dict[str, Any]:
    report = load_json(run.path / "unified_run_report.json", {})
    review = load_json(run.path / SHOWCASE_OUTPUT_NAME, {})
    quality = load_json(run.path / QUALITY_OUTPUT_NAME, {})
    if not isinstance(report, dict):
        report = {}
    if not isinstance(review, dict):
        review = {}
    if not isinstance(quality, dict):
        quality = {}
    directions = review.get("directions") if isinstance(review.get("directions"), list) else []
    corpus = review.get("corpus") if isinstance(review.get("corpus"), dict) else {}
    paper_count = int(corpus.get("paper_total") or 0)
    if not paper_count:
        paper_ids: set[str] = set()
        for direction in directions:
            if not isinstance(direction, dict):
                continue
            for paper in direction.get("papers") or []:
                if not isinstance(paper, dict):
                    continue
                paper_id = paper.get("id") or paper.get("paper_id") or paper.get("candidate_id")
                if paper_id:
                    paper_ids.add(str(paper_id))
        paper_count = len(paper_ids)
    if not paper_count:
        papers = report.get("papers") if isinstance(report.get("papers"), list) else []
        paper_count = len(papers)
    input_mode = load_json(run.path / "01_discovery" / "input_mode.json", {})
    if not isinstance(input_mode, dict):
        input_mode = {}
    return {
        "paper_count": paper_count,
        "direction_count": len(directions),
        "quality_status": str(quality.get("status") or ""),
        "input_mode": str(report.get("input_mode") or input_mode.get("mode") or input_mode.get("input_mode") or ""),
        "modified_label": time.strftime("%m-%d %H:%M", time.localtime(run.modified)),
    }


def has_showcase_data(run: RunRef) -> bool:
    return (run.path / SHOWCASE_OUTPUT_NAME).exists()


def file_url(run_id: str, relative_path: str) -> str:
    cleaned = relative_path.replace("\\", "/").lstrip("/")
    return f"/runs/{run_id}/files/{cleaned}"


def collect_csv(path: Path, max_rows: int = 80) -> dict[str, Any]:
    if not path.exists():
        return {"headers": [], "rows": []}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            rows.append({key: value for key, value in row.items()})
        return {"headers": reader.fieldnames or [], "rows": rows}


def load_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data: dict[str, Any] = {}
    current_key = ""
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith("  - ") and current_key:
            data.setdefault(current_key, []).append(raw_line[4:].strip().strip('"').strip("'"))
            continue
        if raw_line.startswith(" ") or ":" not in raw_line:
            continue
        key, raw_value = raw_line.split(":", 1)
        current_key = key.strip()
        value = raw_value.strip()
        if value:
            data[current_key] = value.strip('"').strip("'")
        else:
            data[current_key] = []
    return data


def repro_file_url(run_id: str, relative_path: str) -> str:
    cleaned = relative_path.replace("\\", "/").lstrip("/")
    return f"/repro-runs/{run_id}/files/{cleaned}"


def repo_file_url(relative_path: str) -> str:
    cleaned = relative_path.replace("\\", "/").lstrip("/")
    return f"/repo-files/{cleaned}"


def file_record(root: Path, run_id: str, relative_path: str) -> dict[str, Any]:
    target = root / relative_path
    return {
        "name": target.name,
        "relative": relative_path,
        "exists": target.exists() and target.is_file(),
        "size": target.stat().st_size if target.exists() and target.is_file() else 0,
        "url": repro_file_url(run_id, relative_path) if target.exists() and target.is_file() else "",
    }


def file_record_with_meta(root: Path, url_builder, relative_path: str, label: str = "") -> dict[str, Any]:
    target = root / relative_path
    record: dict[str, Any] = {
        "name": target.name,
        "label": label or target.stem.replace("_", " ").replace("-", " ").title(),
        "relative": relative_path,
        "exists": target.exists() and target.is_file(),
        "size": target.stat().st_size if target.exists() and target.is_file() else 0,
        "url": url_builder(relative_path) if target.exists() and target.is_file() else "",
        "kind": target.suffix.lower().lstrip(".") or "file",
    }
    if target.exists() and target.is_file() and target.suffix.lower() == ".csv":
        table = collect_csv(target, 3)
        record["rows_preview"] = len(table.get("rows", []))
        record["columns"] = table.get("headers", [])[:10]
    return record


def collect_existing_files(root: Path, url_builder, specs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    records = []
    for relative_path, label in specs:
        record = file_record_with_meta(root, url_builder, relative_path, label)
        if record["exists"]:
            records.append(record)
    return records


def collect_dialogue_output_files(run_id: str, run_path: Path) -> list[dict[str, Any]]:
    output_dir = run_path / "dialogue_outputs"
    if not output_dir.exists():
        return []
    label_map = {
        "nasri_dialogue_rounds.md": "Nasri 多轮对话记录",
        "nasri_candidate_reserves.csv": "候选备用需求表",
        "nasri_candidate_uncertainty_bounds.csv": "候选不确定性边界表",
        "nasri_function_repair_patch.py": "功能修复脚本草稿",
        "nasri_effect_improvement_summary.md": "效果展示说明",
        "nasri_showcase_metrics.json": "展示指标 JSON",
        "demo_data_completion_plan.csv": "示例数据补齐计划 CSV",
        "demo_model_interfaces_patch.py": "示例数据读取代码草稿",
        "README.md": "生成产物说明",
    }
    records: list[dict[str, Any]] = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        relative = str(path.relative_to(run_path))
        label = label_map.get(path.name, path.stem.replace("_", " ").replace("-", " ").title())
        records.append(file_record_with_meta(run_path, lambda rel: repro_file_url(run_id, rel), relative, label))
    return records


def code_preview(path: Path, limit: int = 34) -> str:
    if not path.exists() or not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    kept = [line.rstrip() for line in lines if line.strip()][:limit]
    return "\n".join(kept)


def latest_result_json(run_path: Path) -> dict[str, Any]:
    preferred = run_path / "results" / "benders_auto_loop_coupled_active" / "auto_loop_result.json"
    data = load_json(preferred, None)
    if isinstance(data, dict):
        data["_relative"] = str(preferred.relative_to(run_path))
        return data
    candidates = sorted((run_path / "results").rglob("auto_loop_result.json")) if (run_path / "results").exists() else []
    for path in reversed(candidates):
        data = load_json(path, None)
        if isinstance(data, dict):
            data["_relative"] = str(path.relative_to(run_path))
            return data
    return {}


def first_existing_report(run_path: Path, names: list[str]) -> dict[str, str]:
    for name in names:
        path = run_path / "reports" / name
        if path.exists():
            return {
                "name": name,
                "relative": str(path.relative_to(run_path)),
                "text": read_text(path, 4200),
            }
    return {"name": "", "relative": "", "text": ""}


def collect_repro_code(run_id: str, run_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((run_path / "src").glob("*.py")) if (run_path / "src").exists() else []:
        records.append(
            {
                "name": path.name,
                "role": "大模型引导生成的目标论文复现模块",
                "group": "generated",
                "relative": str(path.relative_to(run_path)),
                "url": repro_file_url(run_id, str(path.relative_to(run_path))),
                "preview": code_preview(path),
            }
        )
    tool_roles = {
        "repro_cli.py": "端到端复现命令入口",
        "llm_client.py": "结构化大模型 JSON 调用",
        "audit.py": "可复现潜力审计",
        "model_spec.py": "模型参数与变量抽取",
        "repro_scaffold.py": "复现工作区脚手架生成",
    }
    for filename, role in tool_roles.items():
        path = REPO_ROOT / "tools" / filename
        if path.exists():
            records.append(
                {
                    "name": filename,
                    "role": role,
                    "group": "toolchain",
                    "relative": f"tools/{filename}",
                    "url": repo_file_url(f"tools/{filename}"),
                    "preview": code_preview(path),
                }
            )
    return records[:18]


def collect_llm_prompt_assets() -> list[dict[str, Any]]:
    specs = [
        ("config/prompts/reproducibility_audit.md", "复现审计提示词模板"),
        ("config/schemas/reproducibility_audit.schema.json", "复现审计输出结构约束"),
        ("config/prompts/model_spec.md", "模型参数抽取提示词模板"),
        ("config/schemas/model_spec.schema.json", "模型参数输出结构约束"),
        ("tools/audit.py", "审计提示词构造脚本"),
        ("tools/model_spec.py", "模型规范提示词构造脚本"),
        ("tools/llm_client.py", "大模型结构化调用封装"),
    ]
    assets = []
    for relative_path, label in specs:
        path = REPO_ROOT / relative_path
        if not path.exists() or not path.is_file():
            continue
        assets.append(
            {
                "name": path.name,
                "label": label,
                "relative": relative_path,
                "url": repo_file_url(relative_path),
                "kind": path.suffix.lower().lstrip(".") or "file",
                "preview": code_preview(path, 18),
            }
        )
    return assets


def collect_repro_materials(run_id: str, run_path: Path) -> dict[str, Any]:
    run_url = lambda relative: repro_file_url(run_id, relative)
    data_specs = [
        ("data/buses.csv", "节点数据"),
        ("data/lines.csv", "线路参数"),
        ("data/generators.csv", "机组参数"),
        ("data/generator_cost_segments.csv", "机组分段成本"),
        ("data/reserves.csv", "备用需求"),
        ("data/load_profile.csv", "负荷曲线"),
        ("data/wind_farms.csv", "风场信息"),
        ("data/wind_profile.csv", "风电曲线"),
        ("data/uncertainty_bounds.csv", "不确定性边界"),
        ("data/scenario_probabilities.csv", "场景概率"),
        ("data/load_factors.csv", "负荷因子"),
        ("data/paper_parameters.csv", "论文参数表"),
        ("data/wind_scenario_statistics.csv", "风电场景统计"),
    ]
    parameter_specs = [
        ("artifacts/model_spec.json", "结构化模型参数"),
        ("artifacts/model_spec.md", "模型参数说明"),
        ("artifacts/equations_manifest.json", "公式清单"),
        ("configs/solver_config.json", "求解器配置"),
        ("configs/experiment_matrix.json", "实验矩阵"),
        ("configs/reproduction_assumptions.json", "复现假设"),
    ]
    processed_specs = [
        ("artifacts/dataset_registry.csv", "数据来源登记表"),
        ("artifacts/source_trace.md", "数据来源追踪"),
        ("artifacts/algorithm_trace.md", "算法追踪"),
        ("artifacts/figures_tables_manifest.json", "图表清单"),
        ("reports/data_validation.md", "数据校验报告"),
        ("reports/data_validation.json", "数据校验 JSON"),
        ("reports/reproduction_checklist.md", "复现检查清单"),
    ]
    return {
        "data_files": collect_existing_files(run_path, run_url, data_specs),
        "parameter_files": collect_existing_files(run_path, run_url, parameter_specs),
        "processed_files": collect_existing_files(run_path, run_url, processed_specs),
        "generated_files": collect_dialogue_output_files(run_id, run_path),
    }


DATA_COMPLETION_HINTS: dict[str, dict[str, str]] = {
    "buses.csv": {
        "source": "MATPOWER/PGLib IEEE 118 base case, then document paper-specific modifications.",
        "action": "Import bus ids, voltage base, area/zone and load allocation notes from a public IEEE 118 source.",
    },
    "lines.csv": {
        "source": "MATPOWER/PGLib branch table and paper load-shift-factor / line-limit description.",
        "action": "Fill from_bus, to_bus, reactance and line rate; mark whether limits are original, scaled, or assumed.",
    },
    "generators.csv": {
        "source": "IEEE 118 UC benchmark, cited unit-commitment data, or documented reconstruction assumptions.",
        "action": "Recover p_min, p_max, startup/shutdown, ramp and minimum up/down parameters for each unit.",
    },
    "generator_cost_segments.csv": {
        "source": "Cited UC benchmark cost curves or piecewise cost tables.",
        "action": "Convert generator production costs into segment rows with marginal_cost and p_max_segment_mw.",
    },
    "reserves.csv": {
        "source": "Paper reserve requirement rules or benchmark operating reserve convention.",
        "action": "Create hourly spinning and operating reserve requirements with source notes.",
    },
    "load_profile.csv": {
        "source": "Paper reported 24-hour profile, figure/table extraction, or benchmark load shape scaled to peak 3733 MWh.",
        "action": "Fill 24 hourly total_load_mw values and record scaling method.",
    },
    "wind_farms.csv": {
        "source": "Paper text, cited wind-storage case, or documented bus-location assumptions.",
        "action": "Identify the three wind stations, bus ids and nominal capacities.",
    },
    "wind_profile.csv": {
        "source": "Paper wind scenario/profile figures, cited source, or reconstructable normalized wind profile.",
        "action": "Fill hourly forecast and +/-20% uncertainty bounds for each wind station.",
    },
    "uncertainty_bounds.csv": {
        "source": "Paper uncertainty set: +/-20% wind and +/-3% load.",
        "action": "Create component-level lower/upper bounds for load and wind by hour.",
    },
    "scenario_probabilities.csv": {
        "source": "Paper scenario-generation description or equal-probability documented assumption.",
        "action": "Record scenario id, probability and source rationale.",
    },
    "load_factors.csv": {
        "source": "Paper load factors or benchmark 24-hour load curve.",
        "action": "Fill normalized hourly load factors used to scale peak load.",
    },
    "paper_parameters.csv": {
        "source": "Paper text, equations, tables and solver setup paragraphs.",
        "action": "Collect scalar parameters such as horizon, uncertainty percentages, solver tolerances and case size.",
    },
    "wind_scenario_statistics.csv": {
        "source": "Derived from wind_profile.csv and scenario probabilities after wind data are filled.",
        "action": "Compute total MWh, average MW and capacity factor for each scenario.",
    },
}


def csv_headers(path: Path) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def collect_data_completion_plan(
    run_id: str,
    run_path: Path,
    target: dict[str, Any],
    validation: dict[str, Any],
    model_spec: dict[str, Any],
    dataset_registry: dict[str, Any],
) -> dict[str, Any]:
    checks = validation.get("checks", []) if isinstance(validation, dict) else []
    tasks = []
    for check in checks:
        filename = str(check.get("file") or "")
        status = str(check.get("status") or "")
        if not filename or status == "ok":
            continue
        path = run_path / "data" / filename
        hint = DATA_COMPLETION_HINTS.get(filename, {})
        tasks.append(
            {
                "file": filename,
                "status": status,
                "rows": check.get("rows", 0),
                "missing_columns": check.get("missing_columns", []),
                "columns": csv_headers(path),
                "url": repro_file_url(run_id, f"data/{filename}") if path.exists() else "",
                "source_hint": hint.get("source", "Use paper text, cited datasets, public benchmark data, or documented assumptions."),
                "action": hint.get("action", "Fill this table and record source/assumption notes."),
            }
        )

    registry_rows = dataset_registry.get("rows", []) if isinstance(dataset_registry, dict) else []
    registry_excerpt = "\n".join(
        f"- {row.get('item', '')}: {row.get('source_hint', '')}; status={row.get('reproduction_status', '')}; notes={row.get('notes', '')}"
        for row in registry_rows[:8]
    )
    task_excerpt = "\n".join(
        f"- {task['file']} ({task['status']}): columns={', '.join(task['columns'])}; source={task['source_hint']}; action={task['action']}"
        for task in tasks[:16]
    )
    prompt = f"""你是论文复现的数据补齐助手。目标不是凭空编造数据，而是为复现工作区补齐可追踪、可验证、可声明假设的数据表。

论文标题：{target.get('title') or target.get('id') or run_id}
复现目标 ID：{run_id}

当前数据来源登记：
{registry_excerpt or '- 暂无 dataset registry 摘要'}

当前空缺/异常数据表：
{task_excerpt or '- 暂无待补齐数据表'}

模型规范摘要：
- Objective: {model_spec.get('objective', '') if isinstance(model_spec, dict) else ''}
- Parameters: {', '.join((model_spec.get('parameters', []) if isinstance(model_spec, dict) else [])[:8])}
- Constraints: {', '.join((model_spec.get('constraints', []) if isinstance(model_spec, dict) else [])[:8])}

请分三步回答：
1. 对每个 CSV 判断应优先从论文、引用文献、公开 IEEE 118/MATPOWER/PGLib/UC benchmark，还是从明确假设补齐。
2. 给出每个 CSV 的补齐方案，必须包含字段来源、单位、是否可直接填表、置信度和需要用户确认的问题。
3. 对可以先行补齐的表，输出 CSV 行片段；对不能补齐的表，输出下一轮检索关键词和需要人工确认的证据。

输出格式请使用 Markdown，按文件分节；不要隐藏假设，不要把无法确认的数据伪装成论文原始数据。"""
    return {
        "status": "ready" if tasks else "complete",
        "tasks": tasks,
        "prompt": prompt,
        "instructions": [
            "先打开 prompt，与大模型或人工检索继续交互。",
            "把确认的数据写回对应 CSV，保留 notes/source 字段。",
            "重新运行 validate-data；所有关键 CSV 有数据行后，再进入阶段二可视化复现。",
        ],
    }


def collect_repro_figures(run_id: str, run_path: Path) -> list[dict[str, str]]:
    labels = {
        "fig3_wind_scenarios.png": "Fig. 3 风电场景重构",
        "fig4_generation_schedule.png": "Fig. 4 发电与承诺调度",
        "fig5_benders_convergence.png": "Fig. 5 Benders 收敛曲线",
        "paper_vs_reproduction_comparison.png": "原文结果与当前实现对比",
        "fig1_price_soc_schedule.svg": "Fig. 1 价格、SOC 与充放电计划",
        "fig2_revenue_breakdown.svg": "Fig. 2 能量、备用与退化成本收益分解",
        "fig3_degradation_sensitivity.svg": "Fig. 3 退化成本敏感性",
        "paper_vs_reproduction_comparison.svg": "原文结果与当前实现对比",
    }
    root = run_path / "results" / "paper_style_figures"
    figures = []
    seen: set[str] = set()
    for filename, label in labels.items():
        path = root / filename
        if path.exists():
            rel = str(path.relative_to(run_path))
            seen.add(rel)
            figures.append({"label": label, "relative": rel, "url": repro_file_url(run_id, rel)})
    if root.exists():
        for path in sorted(root.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg"}:
                continue
            rel = str(path.relative_to(run_path))
            if rel in seen:
                continue
            label = path.stem.replace("_", " ").replace("-", " ").title()
            figures.append({"label": label, "relative": rel, "url": repro_file_url(run_id, rel)})
    return figures


def collect_repro_evidence_snippets(run_path: Path) -> list[dict[str, Any]]:
    rows = load_json(run_path / "extracted_text" / "evidence_snippets.json", [])
    if not isinstance(rows, list):
        return []
    preferred = [
        "data",
        "test system",
        "case studies",
        "load",
        "table",
        "solver",
        "milp",
        "model",
        "parameters",
        "generator",
        "agc",
        "market",
    ]
    seen: set[str] = set()
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if len(text) < 80:
            continue
        signature = re.sub(r"\s+", " ", text[:220]).lower()
        if signature in seen:
            continue
        seen.add(signature)
        keyword = str(row.get("keyword") or "").lower()
        lowered = text.lower()
        score = 0
        for weight, word in enumerate(preferred[::-1], start=1):
            if word in keyword or word in lowered:
                score += weight
        ranked.append(
            (
                score,
                -index,
                {
                    "page": row.get("page") or "",
                    "keyword": row.get("keyword") or "",
                    "text": text,
                },
            )
        )
    ranked.sort(reverse=True)
    return [item for _, __, item in ranked[:10]]


def collect_repro_target(run_path: Path) -> dict[str, Any]:
    run_id = run_path.name
    target = load_simple_yaml(run_path / "target.yaml")
    audit = load_json(run_path / "audits" / "reproducibility_audit.json", {})
    model_spec = load_json(run_path / "artifacts" / "model_spec.json", {})
    validation = load_json(run_path / "reports" / "data_validation.json", {})
    case_data = load_json(run_path / "results" / "case_data_summary.json", {})
    result = latest_result_json(run_path)
    comparison = collect_csv(run_path / "results" / "paper_style_results" / "paper_vs_reproduction_comparison.csv", 20)
    dataset_registry = collect_csv(run_path / "artifacts" / "dataset_registry.csv", 30)
    showcase_index = collect_csv(run_path / "reports" / "stage_15_showcase_materials_index.csv", 80)
    reports = [
        {"title": "大模型多轮对话结构", **first_existing_report(run_path, ["stage_18_llm_dialogue_design_and_file_structure_cn.md"])},
        {"title": "大模型代码生成验证", **first_existing_report(run_path, ["stage_20_llm_code_generation_validation_cn.md"])},
        {"title": "当前复现效果与差距", **first_existing_report(run_path, ["stage_14_paper_style_visual_comparison_report.md"])},
        {"title": "工具链可复用性", **first_existing_report(run_path, ["stage_19_toolchain_reusability_showcase_cn.md"])},
        {"title": "复现计划", **first_existing_report(run_path, ["reproduction_plan.md", "reproduction_checklist.md"])},
    ]
    data_tables = []
    for name, meta in (case_data.get("tables") or {}).items():
        data_tables.append(
            {
                "name": name,
                "rows": meta.get("rows"),
                "columns": meta.get("columns", [])[:8],
            }
        )

    return {
        "id": run_id,
        "path": str(run_path),
        "title": target.get("title") or audit.get("paper_title") or run_id,
        "authors": target.get("authors") or [],
        "year": target.get("year") or "",
        "venue": target.get("venue") or "",
        "doi": target.get("doi") or "",
        "role": target.get("role") or audit.get("recommended_role") or "",
        "scores": audit.get("scores") if isinstance(audit, dict) else {},
        "data_check": audit.get("data_check", []) if isinstance(audit, dict) else [],
        "algorithm_check": audit.get("algorithm_check", []) if isinstance(audit, dict) else [],
        "result_alignment": audit.get("result_alignment", []) if isinstance(audit, dict) else [],
        "blockers": audit.get("blockers", []) if isinstance(audit, dict) else [],
        "next_steps": audit.get("next_steps", []) if isinstance(audit, dict) else [],
        "evidence_snippets": collect_repro_evidence_snippets(run_path),
        "model_spec": {
            "sets": model_spec.get("sets", []) if isinstance(model_spec, dict) else [],
            "parameters": model_spec.get("parameters", []) if isinstance(model_spec, dict) else [],
            "variables": model_spec.get("variables", []) if isinstance(model_spec, dict) else [],
            "objective": model_spec.get("objective", "") if isinstance(model_spec, dict) else "",
            "constraints": model_spec.get("constraints", []) if isinstance(model_spec, dict) else [],
            "uncertainty": model_spec.get("uncertainty", []) if isinstance(model_spec, dict) else [],
            "implementation_notes": model_spec.get("implementation_notes", []) if isinstance(model_spec, dict) else [],
        },
        "data_validation": validation if isinstance(validation, dict) else {},
        "case_data": {
            "base_mva": case_data.get("base_mva") if isinstance(case_data, dict) else "",
            "tables": data_tables,
        },
        "dataset_registry": dataset_registry,
        "comparison": comparison,
        "loop_result": result,
        "figures": collect_repro_figures(run_id, run_path),
        "showcase_index": showcase_index,
        "reports": reports,
        "code_artifacts": collect_repro_code(run_id, run_path),
        "materials": collect_repro_materials(run_id, run_path),
        "llm_prompt_assets": collect_llm_prompt_assets(),
        "data_completion": collect_data_completion_plan(run_id, run_path, target, validation if isinstance(validation, dict) else {}, model_spec if isinstance(model_spec, dict) else {}, dataset_registry),
        "links": {
            "audit_md": repro_file_url(run_id, "audits/reproducibility_audit.md") if (run_path / "audits" / "reproducibility_audit.md").exists() else "",
            "model_spec_md": repro_file_url(run_id, "artifacts/model_spec.md") if (run_path / "artifacts" / "model_spec.md").exists() else "",
            "source_trace_md": repro_file_url(run_id, "artifacts/source_trace.md") if (run_path / "artifacts" / "source_trace.md").exists() else "",
            "algorithm_trace_md": repro_file_url(run_id, "artifacts/algorithm_trace.md") if (run_path / "artifacts" / "algorithm_trace.md").exists() else "",
        },
        "core_files": [
            file_record(run_path, run_id, "target.yaml"),
            file_record(run_path, run_id, "audits/reproducibility_audit.json"),
            file_record(run_path, run_id, "artifacts/model_spec.json"),
            file_record(run_path, run_id, "reports/data_validation.json"),
            file_record(run_path, run_id, "results/paper_style_results/paper_vs_reproduction_comparison.csv"),
        ],
    }


def collect_reproduction_showcase() -> dict[str, Any]:
    targets = []
    if REPRO_RUN_ROOT.exists():
        for path in sorted(REPRO_RUN_ROOT.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            if path.is_dir() and (path / "target.yaml").exists():
                targets.append(collect_repro_target(path))
    return {
        "schema_version": "paper_reproduction_showcase.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "targets": targets,
        "toolchain_commands": [
            "python3 -m tools.repro_cli init-target --target config/targets/<paper>.yaml",
            "python3 -m tools.repro_cli extract-pdf --target config/targets/<paper>.yaml",
            "python3 -m tools.repro_cli audit --target config/targets/<paper>.yaml",
            "python3 -m tools.repro_cli model-spec --target config/targets/<paper>.yaml",
            "python3 -m tools.repro_cli prepare-repro --target config/targets/<paper>.yaml",
            "python3 -m tools.repro_cli validate-data --target config/targets/<paper>.yaml",
        ],
    }


def compact_repro_chat_context(target: dict[str, Any]) -> str:
    scores = target.get("scores") if isinstance(target.get("scores"), dict) else {}
    validation = target.get("data_validation") if isinstance(target.get("data_validation"), dict) else {}
    model_spec = target.get("model_spec") if isinstance(target.get("model_spec"), dict) else {}
    completion = target.get("data_completion") if isinstance(target.get("data_completion"), dict) else {}
    tasks = completion.get("tasks", []) if isinstance(completion, dict) else []
    materials = target.get("materials") if isinstance(target.get("materials"), dict) else {}
    data_files = materials.get("data_files", []) if isinstance(materials, dict) else []
    code_files = target.get("code_artifacts") if isinstance(target.get("code_artifacts"), list) else []
    lines = [
        f"Target id: {target.get('id')}",
        f"Title: {target.get('title')}",
        f"Role: {target.get('role')}",
        f"Scores: overall={scores.get('overall')}, data={scores.get('data')}, algorithm={scores.get('algorithm')}, result_alignment={scores.get('result_alignment')}",
        f"Data validation: complete={validation.get('complete_files')}, empty={validation.get('empty_files')}, missing={validation.get('missing_files')}, bad_header={validation.get('bad_header_files')}",
        f"Objective: {model_spec.get('objective')}",
        "Parameters: " + "; ".join((model_spec.get("parameters") or [])[:10]),
        "Constraints: " + "; ".join((model_spec.get("constraints") or [])[:10]),
        "Open data tasks:",
    ]
    for task in tasks[:14]:
        lines.append(
            f"- {task.get('file')}: status={task.get('status')}; columns={', '.join(task.get('columns') or [])}; source={task.get('source_hint')}; action={task.get('action')}"
        )
    lines.append("Available data files:")
    for file in data_files[:12]:
        lines.append(f"- {file.get('relative')}: {file.get('label')}, columns={', '.join(file.get('columns') or [])}")
    lines.append("Code artifacts:")
    for file in code_files[:10]:
        preview = str(file.get("preview") or "").replace("\n", " ")[:260]
        lines.append(f"- {file.get('relative')}: {file.get('role')}; preview={preview}")
    blockers = target.get("blockers") if isinstance(target.get("blockers"), list) else []
    next_steps = target.get("next_steps") if isinstance(target.get("next_steps"), list) else []
    if blockers:
        lines.append("Blockers: " + "; ".join(str(item) for item in blockers[:6]))
    if next_steps:
        lines.append("Next steps: " + "; ".join(str(item) for item in next_steps[:6]))
    return "\n".join(lines)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_nasri_demo_outputs(target_id: str, target: dict[str, Any]) -> list[dict[str, str]]:
    run_path = repro_run_path(target_id)
    output_dir = run_path / "dialogue_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = target.get("data_validation") if isinstance(target.get("data_validation"), dict) else {}
    figures = target.get("figures") if isinstance(target.get("figures"), list) else []
    comparison = target.get("comparison") if isinstance(target.get("comparison"), dict) else {}
    comparison_rows = comparison.get("rows", []) if isinstance(comparison, dict) else []

    load_rows = _read_csv_rows(run_path / "data" / "load_profile.csv")
    if not load_rows:
        load_rows = [{"hour": str(hour), "total_load_mw": ""} for hour in range(1, 25)]
    wind_rows = _read_csv_rows(run_path / "data" / "wind_farms.csv")
    wind_ids = [row.get("wind_id") or row.get("id") or f"W{index + 1}" for index, row in enumerate(wind_rows)] or ["W1", "W2"]

    reserves_path = output_dir / "nasri_candidate_reserves.csv"
    with reserves_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "hour",
                "load_mw",
                "spinning_reserve_mw",
                "operating_reserve_mw",
                "source_type_cn",
                "construction_rule_cn",
                "needs_user_confirm",
                "source_note",
            ],
        )
        writer.writeheader()
        for row in load_rows:
            load_mw = _float_or_none(row.get("total_load_mw"))
            writer.writerow(
                {
                    "hour": row.get("hour") or "",
                    "load_mw": "" if load_mw is None else round(load_mw, 4),
                    "spinning_reserve_mw": "" if load_mw is None else round(load_mw * 0.03, 4),
                    "operating_reserve_mw": "" if load_mw is None else round(load_mw * 0.05, 4),
                    "source_type_cn": "合理构造/待确认假设",
                    "construction_rule_cn": "演示候选：旋转备用=3%负荷，运行备用=5%负荷；正式复现需按论文或用户确认改写。",
                    "needs_user_confirm": "是",
                    "source_note": "由阶段二多轮对话根据 load_profile.csv 自动生成，不覆盖 data/reserves.csv。",
                }
            )

    uncertainty_path = output_dir / "nasri_candidate_uncertainty_bounds.csv"
    with uncertainty_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "component",
                "component_id",
                "hour",
                "lower_multiplier",
                "upper_multiplier",
                "source_type_cn",
                "paper_basis_cn",
                "needs_user_confirm",
                "source_note",
            ],
        )
        writer.writeheader()
        for row in load_rows:
            writer.writerow(
                {
                    "component": "load",
                    "component_id": "system",
                    "hour": row.get("hour") or "",
                    "lower_multiplier": 0.97,
                    "upper_multiplier": 1.03,
                    "source_type_cn": "论文规则重建/待确认",
                    "paper_basis_cn": "按当前材料中的 +/-3% 负荷边界登记为候选；若最终模型只保留风电不确定性，可删除此行。",
                    "needs_user_confirm": "是",
                    "source_note": "由阶段二对话生成的候选边界，不覆盖正式数据表。",
                }
            )
        for wind_id in wind_ids:
            for row in load_rows:
                writer.writerow(
                    {
                        "component": "wind",
                        "component_id": wind_id,
                        "hour": row.get("hour") or "",
                        "lower_multiplier": 0.8,
                        "upper_multiplier": 1.2,
                        "source_type_cn": "论文规则重建",
                        "paper_basis_cn": "按论文材料中的 +/-20% 风电不确定性整理为结构化边界。",
                        "needs_user_confirm": "否",
                        "source_note": "由阶段二对话生成，可用于回填 data/uncertainty_bounds.csv 前人工复核。",
                    }
                )

    repair_path = output_dir / "nasri_function_repair_patch.py"
    repair_path.write_text(
        '''"""Nasri 2016 dialogue-generated validation helper draft.

This patch is intentionally placed in dialogue_outputs. It shows how the
multi-turn LLM workspace can turn a user request into a reviewable script
without overwriting the formal reproduction implementation.
"""
from __future__ import annotations

import csv
from pathlib import Path


OPTIONAL_TABLES = {
    "reserves.csv": "The implemented baseline can run without explicit reserves, but reserve assumptions should be reviewed for paper-style reporting.",
    "uncertainty_bounds.csv": "Wind/load uncertainty bounds can be regenerated from paper rules before final AC-Benders experiments.",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_table(data_dir: str | Path, filename: str) -> dict[str, object]:
    path = Path(data_dir) / filename
    rows = read_rows(path)
    return {
        "file": filename,
        "rows": len(rows),
        "status": "optional_candidate_needed" if not rows and filename in OPTIONAL_TABLES else ("ok" if rows else "missing_or_empty"),
        "note": OPTIONAL_TABLES.get(filename, ""),
    }


def validate_optional_assumptions(data_dir: str | Path) -> list[dict[str, object]]:
    """Return review items that the UI can show before finalizing assumptions."""
    return [summarize_table(data_dir, filename) for filename in OPTIONAL_TABLES]
''',
        encoding="utf-8",
    )

    metrics_path = output_dir / "nasri_showcase_metrics.json"
    metrics = {
        "target_id": target_id,
        "title": target.get("title") or target_id,
        "dialogue_rounds": 4,
        "complete_data_files": validation.get("complete_files"),
        "optional_empty_files_before_dialogue": validation.get("optional_empty_files"),
        "candidate_files_generated": 2,
        "repair_patch_generated": True,
        "figures_available": len(figures),
        "comparison_metrics": len(comparison_rows),
        "generated_files": [
            "dialogue_outputs/nasri_candidate_reserves.csv",
            "dialogue_outputs/nasri_candidate_uncertainty_bounds.csv",
            "dialogue_outputs/nasri_function_repair_patch.py",
            "dialogue_outputs/nasri_effect_improvement_summary.md",
        ],
    }
    save_json(metrics_path, metrics)

    dialogue_path = output_dir / "nasri_dialogue_rounds.md"
    dialogue_path.write_text(
        f"""# Nasri 2016 多轮对话生成记录

这份记录用于展示阶段二如何把用户的个性化需求转成可检查的本地文件。演示按钮会稳定回放这组对话并落盘产物；正式“发送给大模型”按钮会调用当前环境配置的大模型接口。

| 轮次 | 用户意图 | 大模型处理 | 落地产物 |
| --- | --- | --- | --- |
| 1 | 检查 Nasri 工作区还有哪些可补齐数据 | 读取 `data_validation.json`，发现正式数据已可运行，但 `reserves.csv` 与 `uncertainty_bounds.csv` 属于可补充的假设表 | 明确补齐对象和是否覆盖正式数据 |
| 2 | 生成可展示的数据补全文件 | 使用 `load_profile.csv` 和风电场编号构造候选备用与不确定性边界，并标注来源分类 | `nasri_candidate_reserves.csv`、`nasri_candidate_uncertainty_bounds.csv` |
| 3 | 修复“空表看起来像错误”的展示/校验问题 | 生成一个校验辅助脚本，把可选空表解释成“候选假设待确认”，避免误解为脚本失败 | `nasri_function_repair_patch.py` |
| 4 | 准备展示说明 | 汇总原文对齐、候选文件、现有图表和展示话术 | `nasri_effect_improvement_summary.md`、`nasri_showcase_metrics.json` |

## 本轮结论

- 阶段一已经把论文拆解成数据需求、模型结构和环境配置。
- 阶段二的价值体现在：不用重新跑完整优化模型，也能围绕当前材料补齐候选数据、生成修复脚本、解释展示效果。
- 这些文件都放在 `runs/{target_id}/dialogue_outputs/`，不会覆盖正式 `data/` 或 `src/`。
""",
        encoding="utf-8",
    )

    effect_path = output_dir / "nasri_effect_improvement_summary.md"
    effect_path.write_text(
        f"""# Nasri 2016 阶段二效果展示说明

## 对话产出了什么

本轮多轮对话生成了 4 类材料：

| 类型 | 文件 | 作用 |
| --- | --- | --- |
| 数据补全 | `nasri_candidate_reserves.csv` | 基于负荷曲线生成备用需求候选表，展示如何把“可选空表”变成可审查的假设数据 |
| 数据补全 | `nasri_candidate_uncertainty_bounds.csv` | 将风电 +/-20% 与负荷 +/-3% 边界整理成结构化候选表 |
| 功能修复 | `nasri_function_repair_patch.py` | 生成校验辅助脚本，修复展示中“可选空表被误解为失败”的问题 |
| 展示材料 | `nasri_showcase_metrics.json` | 汇总当前数据、图表和候选产物，便于页面或汇报使用 |

## 展示时怎么讲

1. 阶段一先说明：系统已从论文中拆解出 IEEE RTS 网络、机组参数、负荷曲线、风电场景、AC-UC 模型和求解器环境。
2. 阶段二演示：用户提出“补齐数据/修复展示/完善效果”的自然语言需求，工作台生成本地文件。
3. 打开 `nasri_candidate_reserves.csv` 和 `nasri_candidate_uncertainty_bounds.csv`，强调这些是候选假设，不会伪装成论文原始数据。
4. 打开 `nasri_function_repair_patch.py`，说明大模型不只是聊天，还能把功能修复需求转成代码草稿。
5. 最后切到已有图表，说明对话产物服务于后续图表和差距解释，而不是孤立文件。

## 当前效果锚点

- 数据校验：完整数据表 {validation.get("complete_files")} 个，可选空表 {validation.get("optional_empty_files")} 个。
- 图表产物：{len(figures)} 个。
- 原文对齐指标：{len(comparison_rows)} 项。
- 本轮新增候选文件：2 个 CSV、1 个 Python 修复草稿、2 个说明/指标文件。
""",
        encoding="utf-8",
    )

    return [
        {
            "label": "Nasri 多轮对话记录",
            "relative": "dialogue_outputs/nasri_dialogue_rounds.md",
            "kind": "markdown",
            "url": repro_file_url(target_id, "dialogue_outputs/nasri_dialogue_rounds.md"),
        },
        {
            "label": "候选备用需求表",
            "relative": "dialogue_outputs/nasri_candidate_reserves.csv",
            "kind": "csv",
            "url": repro_file_url(target_id, "dialogue_outputs/nasri_candidate_reserves.csv"),
        },
        {
            "label": "候选不确定性边界表",
            "relative": "dialogue_outputs/nasri_candidate_uncertainty_bounds.csv",
            "kind": "csv",
            "url": repro_file_url(target_id, "dialogue_outputs/nasri_candidate_uncertainty_bounds.csv"),
        },
        {
            "label": "功能修复脚本草稿",
            "relative": "dialogue_outputs/nasri_function_repair_patch.py",
            "kind": "python",
            "url": repro_file_url(target_id, "dialogue_outputs/nasri_function_repair_patch.py"),
        },
        {
            "label": "效果展示说明",
            "relative": "dialogue_outputs/nasri_effect_improvement_summary.md",
            "kind": "markdown",
            "url": repro_file_url(target_id, "dialogue_outputs/nasri_effect_improvement_summary.md"),
        },
        {
            "label": "展示指标 JSON",
            "relative": "dialogue_outputs/nasri_showcase_metrics.json",
            "kind": "json",
            "url": repro_file_url(target_id, "dialogue_outputs/nasri_showcase_metrics.json"),
        },
    ]


def write_repro_demo_outputs(target_id: str, target: dict[str, Any]) -> list[dict[str, str]]:
    if target_id == "nasri_2016_ac_uc_benders":
        return write_nasri_demo_outputs(target_id, target)

    run_path = repro_run_path(target_id)
    output_dir = run_path / "dialogue_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_plan_path = output_dir / "demo_data_completion_plan.csv"
    with data_plan_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "purpose_cn", "source_strategy_cn", "next_action_cn", "needs_user_confirm"],
        )
        writer.writeheader()
        for row in [
            {
                "file": "buses.csv",
                "purpose_cn": "建立节点集合、负荷分配和基准电压信息",
                "source_strategy_cn": "优先使用公开 IEEE 118 / MATPOWER 基准，再记录论文改动",
                "next_action_cn": "导入节点编号、区域、负荷占比，并标记需确认的论文调整",
                "needs_user_confirm": "是",
            },
            {
                "file": "generators.csv",
                "purpose_cn": "建立机组出力、爬坡、启停和最小开停机约束",
                "source_strategy_cn": "追踪论文引用的 UC benchmark；缺失字段写入假设文件",
                "next_action_cn": "补齐 p_min、p_max、ramp、startup_cost、min_up/down",
                "needs_user_confirm": "是",
            },
            {
                "file": "load_profile.csv",
                "purpose_cn": "构造 24 小时负荷序列，驱动调度与市场出清",
                "source_strategy_cn": "从论文图表或公开负荷曲线缩放到论文峰值",
                "next_action_cn": "生成 hour,total_load_mw,source_note 三列并记录缩放方法",
                "needs_user_confirm": "是",
            },
            {
                "file": "market_prices.csv",
                "purpose_cn": "支撑收益分解和调频/备用市场收入计算",
                "source_strategy_cn": "使用论文市场参数、PJM/ISO-NE 公开数据或明确假设",
                "next_action_cn": "补齐 energy/reserve/regulation 价格序列",
                "needs_user_confirm": "否",
            },
        ]:
            writer.writerow(row)

    code_path = output_dir / "demo_model_interfaces_patch.py"
    code_path.write_text(
        '''"""Example code generated from the reproduction dialogue.

This file is intentionally a patch draft. It shows how a user request can
be turned into a concrete data-loading implementation before editing the
main src/model_interfaces.py file.
"""
from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "buses.csv": ["bus_id", "base_kv", "pd_fraction"],
    "generators.csv": ["gen_id", "bus_id", "p_min_mw", "p_max_mw"],
    "load_profile.csv": ["hour", "total_load_mw"],
    "market_prices.csv": ["hour", "energy_price"],
}


def read_csv_checked(data_dir: str | Path, filename: str) -> list[dict[str, str]]:
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"missing reproduction data file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = [name for name in REQUIRED_COLUMNS.get(filename, []) if name not in (rows[0].keys() if rows else [])]
    if missing:
        raise ValueError(f"{filename} missing columns: {missing}")
    return rows


def load_minimum_case_data(data_dir: str | Path) -> dict[str, list[dict[str, str]]]:
    return {
        "buses": read_csv_checked(data_dir, "buses.csv"),
        "generators": read_csv_checked(data_dir, "generators.csv"),
        "load_profile": read_csv_checked(data_dir, "load_profile.csv"),
        "market_prices": read_csv_checked(data_dir, "market_prices.csv"),
    }
''',
        encoding="utf-8",
    )

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        f"""# 多轮对话生成产物

这组文件由阶段二“运行示例对话”生成，用于展示用户与大模型交互后可以落到本地工作区的结果。

- `demo_data_completion_plan.csv`：把对话中的数据补齐建议转成可分配任务。
- `demo_model_interfaces_patch.py`：把“请生成数据读取函数”的需求转成代码草稿。

这些文件不会直接覆盖正式复现代码。展示时可以说明：多轮对话先生成可检查的草稿，用户确认后再合并到 `src/model_interfaces.py` 或回填 `data/`。

目标论文：{target.get('title') or target_id}
""",
        encoding="utf-8",
    )

    return [
        {
            "label": "示例数据补齐计划 CSV",
            "relative": "dialogue_outputs/demo_data_completion_plan.csv",
            "kind": "csv",
            "url": repro_file_url(target_id, "dialogue_outputs/demo_data_completion_plan.csv"),
        },
        {
            "label": "示例数据读取代码草稿",
            "relative": "dialogue_outputs/demo_model_interfaces_patch.py",
            "kind": "python",
            "url": repro_file_url(target_id, "dialogue_outputs/demo_model_interfaces_patch.py"),
        },
        {
            "label": "生成产物说明",
            "relative": "dialogue_outputs/README.md",
            "kind": "markdown",
            "url": repro_file_url(target_id, "dialogue_outputs/README.md"),
        },
    ]


def build_repro_demo_answer(target_id: str, target: dict[str, Any], message: str) -> tuple[str, list[dict[str, str]]]:
    validation = target.get("data_validation") if isinstance(target.get("data_validation"), dict) else {}
    model_spec = target.get("model_spec") if isinstance(target.get("model_spec"), dict) else {}
    materials = target.get("materials") if isinstance(target.get("materials"), dict) else {}
    data_files = materials.get("data_files", []) if isinstance(materials, dict) else []
    empty_files = int(validation.get("empty_files") or 0)
    missing_files = int(validation.get("missing_files") or 0)
    complete_files = int(validation.get("complete_files") or 0)
    candidate_files = [
        str(item.get("relative") or item.get("name") or "")
        for item in data_files
        if item.get("relative") or item.get("name")
    ][:8]
    parameters = (model_spec.get("parameters") or [])[:5] if isinstance(model_spec, dict) else []
    constraints = (model_spec.get("constraints") or [])[:4] if isinstance(model_spec, dict) else []
    file_text = "\n".join(f"- `{name}`" for name in candidate_files) or "- 暂无可枚举数据文件"
    parameter_text = "\n".join(f"- {item}" for item in parameters) or "- 暂无参数摘要"
    constraint_text = "\n".join(f"- {item}" for item in constraints) or "- 暂无约束摘要"
    artifacts = write_repro_demo_outputs(target_id, target)
    artifact_text = "\n".join(f"- {item['label']}：`{item['relative']}`" for item in artifacts)
    if target_id == "nasri_2016_ac_uc_benders":
        optional_empty = int(validation.get("optional_empty_files") or 0)
        answer = f"""已运行 Nasri 2016 专属的阶段二示例对话。这个示例把“数据补全、功能修复、效果展示”三件事一次串起来，并把结果写入本地工作区。

用户需求：
{message}

当前工作区状态：
- 正式数据表已完整：{complete_files} 个。
- 必需空表/缺失表：{empty_files + missing_files} 个。
- 可选待确认表：{optional_empty} 个，主要是 `reserves.csv` 和 `uncertainty_bounds.csv` 这类假设/边界表。

本轮对话的处理逻辑：
1. 先读数据校验结果，判断这不是“跑不通”的问题，而是“展示时需要把可选假设讲清楚”的问题。
2. 基于 `load_profile.csv` 和 `wind_farms.csv` 生成两个候选补数表，用于展示如何把空表变成可审查、可确认的结构化数据。
3. 生成一个校验辅助脚本草稿，让页面/报告能把可选空表解释为“候选假设待确认”，而不是误判成失败。
4. 生成展示说明和指标 JSON，方便汇报时解释多轮对话模块到底提升了什么。

本轮已经生成本地文件：
{artifact_text}

展示时建议这样讲：
- 阶段一完成论文拆解：明确了 IEEE RTS 网络、机组参数、风电场景、AC-UC 模型和求解器环境。
- 阶段二接收个性化需求：用户要求补齐数据、修复展示或完善效果。
- 多轮对话的价值：把自然语言需求落成 `dialogue_outputs/` 里的 CSV、Python 草稿和展示说明；这些文件不覆盖正式复现代码，适合先预览、讨论，再决定是否合并。"""
        return answer, artifacts

    answer = f"""已运行一段定制化示例对话。我的判断是：当前卡点主要在**数据补齐 + 脚本落地**，不是界面展示本身。

用户需求：
{message}

当前工作区状态：
- 已完整数据表：{complete_files} 个。
- 待补齐或待确认数据表：{empty_files + missing_files} 个，其中空表 {empty_files} 个，缺失表 {missing_files} 个。
- 当前目标论文：{target.get('title') or target.get('id')}。

建议的定制化推进方案：
1. 先锁定可直接复用的公开基准数据。若论文使用 IEEE 118、市场价格、储能参数或 AGC/负荷曲线，应把“论文原始数据”“公开基准”“明确假设”分列记录，避免把重构数据误写成原文数据。
2. 优先补齐能让脚本跑通的最小数据闭环：节点/线路、机组参数、负荷曲线、储能参数、市场价格、求解器配置。
3. 为 `src/model_interfaces.py` 生成读取函数，先读取 CSV 并做字段校验；为 `src/run_reproduction.py` 生成一个最小实验入口，输出调度表和图表所需的中间结果。
4. 对仍无法从论文确认的字段，在 `configs/reproduction_assumptions.json` 中声明来源和假设，再让用户确认是否接受。

本轮可以加入上下文的文件：
{file_text}

模型抽取中最该对齐的参数：
{parameter_text}

脚本生成时需要覆盖的约束：
{constraint_text}

下一条你可以继续问：
“请基于这些文件，为 `model_interfaces.py` 写一个只做读取和校验的第一版函数，并指出哪些 CSV 字段目前需要人工确认。”

本轮已经生成本地示例产物：
{artifact_text}"""
    return answer, artifacts


def build_repro_chat_prompt(target: dict[str, Any], mode: str, message: str, history: list[dict[str, str]]) -> str:
    mode_labels = {
        "data": "补齐数据、寻找数据来源、生成 CSV 回填方案",
        "code": "编写或修改复现代码，尤其是 model_interfaces.py 和 run_reproduction.py",
        "feature": "调整网页或工具链功能",
        "gap": "解释当前复现效果与完整论文之间的差距",
        "general": "综合协作",
    }
    history_text = "\n".join(
        f"{item.get('role', 'user')}: {str(item.get('content', ''))[:1000]}"
        for item in history[-6:]
        if item.get("content")
    )
    return f"""你是论文复现工具链中的 AI 协作助手。请基于给定复现上下文回答用户的个性化需求。

交互意图：{mode_labels.get(mode, mode_labels['general'])}

当前复现上下文：
{compact_repro_chat_context(target)}

最近对话：
{history_text or '暂无'}

用户新需求：
{message}

回答要求：
1. 先判断当前卡点属于数据、模型、代码、前端展示还是实验结果。
2. 给出可执行的下一步，尽量引用具体文件名、CSV 字段、命令或代码位置。
3. 如果需要补数据，不要编造论文原始数据；请区分“论文证据”“公开基准”“明确假设”。
4. 如果生成代码，请给出小而清晰的代码片段，并说明应写入哪个文件。
5. 如果需要用户确认，请列出最少的问题，不要泛泛而谈。
"""


def quote_yaml(value: Any) -> str:
    text = str(value or "").replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def normalize_literature_pdf_path(raw_path: str, run_path: Path | None = None) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.exists():
        return candidate.resolve()
    normalized = text.replace("\\", "/")
    marker = "/output/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        mapped = PROJECT_ROOT / "output" / suffix
        if mapped.exists():
            return mapped.resolve()
    if run_path:
        filename = Path(normalized).name
        if filename:
            mapped = run_path / "01_discovery" / "pdfs" / filename
            if mapped.exists():
                return mapped.resolve()
    return None


def generated_repro_target_id(source_run_id: str, paper: dict[str, Any]) -> str:
    seed = "|".join(
        str(paper.get(key) or "")
        for key in ["paper_id", "candidate_id", "title", "title_cn", "doi"]
    )
    digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:10]
    label = re.sub(r"[^A-Za-z0-9]+", "_", str(paper.get("candidate_id") or paper.get("paper_id") or "paper")).strip("_").lower()
    if not label:
        label = "paper"
    run_digest = hashlib.sha1(str(source_run_id).encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"lit_{run_digest}_{label}_{digest}".lower()


def load_assigned_paper_index(run_path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for root in [run_path / "01_discovery" / "directions", run_path / "02_reviews" / "directions"]:
        if not root.exists():
            continue
        for assigned_path in root.glob("*/assigned_papers.json"):
            data = load_json(assigned_path, {})
            if not isinstance(data, dict):
                continue
            direction_id = str(data.get("direction_id") or assigned_path.parent.name.split("_", 1)[0])
            for paper in data.get("papers") or []:
                if not isinstance(paper, dict):
                    continue
                record = {**paper, "_direction_id": direction_id}
                for key in [paper.get("paper_id"), paper.get("candidate_id")]:
                    if key:
                        index[str(key)] = record
    return index


def generated_target_path(target_id: str) -> Path:
    return REPO_ROOT / "config" / "targets" / "generated" / f"{target_id}.yaml"


def repro_run_path(target_id: str) -> Path:
    return REPRO_RUN_ROOT / target_id


def repro_status_for_paper(source_run_id: str, paper: dict[str, Any], pdf_path: Path | None = None) -> dict[str, Any]:
    target_id = generated_repro_target_id(source_run_id, paper)
    run_path = repro_run_path(target_id)
    target_path = generated_target_path(target_id)
    audit = load_json(run_path / "audits" / "reproducibility_audit.json", {})
    model_spec = load_json(run_path / "artifacts" / "model_spec.json", {})
    validation = load_json(run_path / "reports" / "data_validation.json", {})
    status = "not_started"
    if run_path.exists():
        status = "prepared"
    if isinstance(audit, dict) and audit.get("scores"):
        status = "audited"
    if isinstance(model_spec, dict) and model_spec.get("parameters"):
        status = "model_spec_ready"
    if isinstance(validation, dict) and validation.get("checks"):
        status = "workspace_ready"
    if not pdf_path:
        status = "missing_pdf"
    return {
        "target_id": target_id,
        "target_config": str(target_path),
        "run_dir": str(run_path),
        "status": status,
        "pdf_available": bool(pdf_path),
        "pdf_name": pdf_path.name if pdf_path else "",
        "scores": audit.get("scores") if isinstance(audit, dict) else {},
        "recommended_role": audit.get("recommended_role") if isinstance(audit, dict) else "",
        "blockers": (audit.get("blockers") or [])[:4] if isinstance(audit, dict) else [],
        "model_spec_counts": {
            "sets": len(model_spec.get("sets") or []) if isinstance(model_spec, dict) else 0,
            "parameters": len(model_spec.get("parameters") or []) if isinstance(model_spec, dict) else 0,
            "variables": len(model_spec.get("variables") or []) if isinstance(model_spec, dict) else 0,
            "constraints": len(model_spec.get("constraints") or []) if isinstance(model_spec, dict) else 0,
        },
        "links": {
            "audit": repro_file_url(target_id, "audits/reproducibility_audit.md") if (run_path / "audits" / "reproducibility_audit.md").exists() else "",
            "model_spec": repro_file_url(target_id, "artifacts/model_spec.md") if (run_path / "artifacts" / "model_spec.md").exists() else "",
            "data_validation": repro_file_url(target_id, "reports/data_validation.md") if (run_path / "reports" / "data_validation.md").exists() else "",
        },
    }


def enrich_showcase_with_reproduction(data: dict[str, Any], run_path: Path, source_run_id: str) -> dict[str, Any]:
    assigned_index = load_assigned_paper_index(run_path)
    for direction in data.get("directions") or []:
        if not isinstance(direction, dict):
            continue
        for paper in direction.get("papers") or []:
            if not isinstance(paper, dict):
                continue
            assigned = assigned_index.get(str(paper.get("candidate_id") or "")) or assigned_index.get(str(paper.get("id") or "")) or {}
            pdf_path = normalize_literature_pdf_path(
                str(paper.get("source_pdf") or assigned.get("pdf_path") or assigned.get("_pdf_path") or ""),
                run_path,
            )
            if pdf_path:
                paper["source_pdf"] = str(pdf_path)
            if assigned.get("txt_path"):
                paper["source_text"] = str(assigned.get("txt_path"))
            paper["reproduction"] = repro_status_for_paper(source_run_id, {**assigned, **paper}, pdf_path)
    return data


def find_literature_paper_context(source_run_id: str, direction_id: str, paper_id: str) -> dict[str, Any]:
    run_path = safe_child(OUTPUT_ROOT, source_run_id)
    data = load_showcase_data(source_run_id)
    assigned_index = load_assigned_paper_index(run_path)
    for direction in data.get("directions") or []:
        if str(direction.get("id")) != str(direction_id):
            continue
        for paper in direction.get("papers") or []:
            if str(paper.get("id")) != str(paper_id):
                continue
            assigned = assigned_index.get(str(paper.get("candidate_id") or "")) or assigned_index.get(str(paper.get("id") or "")) or {}
            merged = {**assigned, **paper}
            pdf_path = normalize_literature_pdf_path(
                str(merged.get("source_pdf") or merged.get("pdf_path") or merged.get("_pdf_path") or ""),
                run_path,
            )
            return {"run_path": run_path, "direction": direction, "paper": merged, "pdf_path": pdf_path}
    abort(404, description="paper not found")


def write_generated_target(source_run_id: str, direction_id: str, paper: dict[str, Any], pdf_path: Path) -> Path:
    target_id = generated_repro_target_id(source_run_id, paper)
    target_path = generated_target_path(target_id)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    authors = paper.get("authors")
    if isinstance(authors, str):
        authors_list = [item.strip() for item in re.split(r",|;| and ", authors) if item.strip()]
    elif isinstance(authors, list):
        authors_list = [str(item) for item in authors if str(item).strip()]
    else:
        authors_list = []
    origin_note = (
        f"Generated from literature showcase run {source_run_id}, "
        f"direction {direction_id}, paper {paper.get('id') or paper.get('paper_id')}."
    )
    lines = [
        f"id: {quote_yaml(target_id)}",
        f"title: {quote_yaml(paper.get('title') or paper.get('title_cn') or target_id)}",
        "authors:",
    ]
    lines.extend([f"  - {quote_yaml(author)}" for author in (authors_list or ["Unknown"])])
    lines.extend(
        [
            f"year: {quote_yaml(paper.get('year') or '')}",
            f"venue: {quote_yaml(paper.get('venue') or '')}",
            f"doi: {quote_yaml(paper.get('doi') or '')}",
            f"role: {quote_yaml('literature_agent_selected_paper')}",
            f"source_pdf: {quote_yaml(str(pdf_path))}",
            f"run_dir: {quote_yaml(str(repro_run_path(target_id).relative_to(REPO_ROOT)))}",
            "notes:",
            f"  - {quote_yaml(origin_note)}",
            f"  - {quote_yaml('Created for file-level reproducibility audit and toolchain handoff.')}",
        ]
    )
    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_path


def collect_stage_files(run_path: Path) -> dict[str, list[dict[str, Any]]]:
    stages: dict[str, list[dict[str, Any]]] = {}
    for stage in ["01_discovery", "02_reviews", "logs", "time_records"]:
        root = run_path / stage
        items: list[dict[str, Any]] = []
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    items.append(
                        {
                            "name": path.name,
                            "relative": str(path.relative_to(run_path)),
                            "size": path.stat().st_size,
                        }
                    )
        stages[stage] = items
    return stages


def collect_directions(run_id: str, run_path: Path) -> list[dict[str, Any]]:
    review_root = run_path / "02_reviews" / "directions"
    discovery_root = run_path / "01_discovery" / "directions"
    directions: list[dict[str, Any]] = []

    names = set()
    for root in [review_root, discovery_root]:
        if root.exists():
            names.update(path.name for path in root.iterdir() if path.is_dir())

    for name in sorted(names):
        assigned = load_json(review_root / name / "assigned_papers.json", None)
        if assigned is None:
            assigned = load_json(discovery_root / name / "assigned_papers.json", {})
        summary = load_json(review_root / name / "direction_review_summary.json", {})
        cards = []
        cards_dir = review_root / name / "paper_cards"
        if cards_dir.exists():
            for card_path in sorted(cards_dir.glob("*.json")):
                card = load_json(card_path, {})
                if isinstance(card, dict):
                    card["_file"] = str(card_path.relative_to(run_path))
                    cards.append(card)

        review_rel = Path("02_reviews") / "directions" / name / "direction_review.md"
        directions.append(
            {
                "folder": name,
                "direction_id": assigned.get("direction_id") or summary.get("direction_id") or name.split("_")[0],
                "name_cn": assigned.get("direction_name_cn") or summary.get("direction_name_cn") or name,
                "name_en": assigned.get("direction_name_en") or "",
                "summary": assigned.get("direction_summary_cn") or summary.get("summary_cn") or "",
                "papers": assigned.get("papers", []),
                "paper_cards": cards,
                "review_md": read_text(run_path / review_rel),
                "review_url": file_url(run_id, str(review_rel)) if (run_path / review_rel).exists() else "",
            }
        )
    return directions


def collect_run(run_id: str) -> dict[str, Any]:
    run_path = safe_child(OUTPUT_ROOT, run_id)
    if not run_path.exists() or not run_path.is_dir():
        abort(404)

    report = load_json(run_path / "unified_run_report.json", {})
    input_mode = load_json(run_path / "01_discovery" / "input_mode.json", {})
    if not isinstance(report, dict):
        report = {}
    if not isinstance(input_mode, dict):
        input_mode = {}
    candidate_directions = load_json(run_path / "01_discovery" / "candidate_directions.json", [])
    selected_candidates = load_json(run_path / "01_discovery" / "selected_candidates.json", [])
    paper_table = collect_csv(run_path / "01_discovery" / "paper_table.csv")
    corpus_review = read_text(run_path / "02_reviews" / "corpus_literature_review.md")
    corpus_summary = load_json(run_path / "02_reviews" / "corpus_review_summary.json", {})
    time_records = []
    time_root = run_path / "time_records"
    if time_root.exists():
        latest = sorted(time_root.glob("run_*.json"))
        if latest:
            payload = load_json(latest[-1], [])
            time_records = payload if isinstance(payload, list) else payload.get("records", [])

    return {
        "run_id": run_id,
        "path": str(run_path),
        "title": str(report.get("topic") or input_mode.get("topic") or run_id),
        "report": report,
        "input_mode": input_mode,
        "candidate_directions": candidate_directions if isinstance(candidate_directions, list) else [],
        "selected_candidates": selected_candidates if isinstance(selected_candidates, list) else [],
        "paper_table": paper_table,
        "corpus_review": corpus_review,
        "corpus_summary": corpus_summary if isinstance(corpus_summary, dict) else {},
        "directions": collect_directions(run_id, run_path),
        "stage_files": collect_stage_files(run_path),
        "time_records": time_records,
    }


def pdf_batch_rows(root: Path, source_label: str) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    folders = [root]
    folders.extend(path for path in sorted(root.rglob("*")) if path.is_dir())
    rows: list[dict[str, Any]] = []
    for folder in folders:
        pdfs = sorted(folder.glob("*.pdf"))
        if not pdfs:
            continue
        size_mb = round(sum(path.stat().st_size for path in pdfs) / 1024 / 1024, 2)
        relative = folder.relative_to(PROJECT_ROOT).as_posix()
        rows.append(
            {
                "source": source_label,
                "name": folder.name if folder != root else "根目录",
                "relative": relative,
                "count": len(pdfs),
                "size_mb": size_mb,
                "sample": [path.name for path in pdfs[:3]],
            }
        )
    return rows


def list_pdf_sources() -> list[dict[str, Any]]:
    rows = []
    rows.extend(pdf_batch_rows(INPUT_PDF_DIR, "input_pdfs"))
    rows.extend(pdf_batch_rows(LIBRARY_PDF_DIR, "library/pdfs"))
    return sorted(rows, key=lambda item: (item["source"], item["relative"]))


def resolve_pdf_dir(value: str) -> Path:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return INPUT_PDF_DIR
    candidate = (PROJECT_ROOT / text).resolve()
    allowed_roots = [INPUT_PDF_DIR.resolve(), LIBRARY_PDF_DIR.resolve()]
    if any(candidate == root or root in candidate.parents for root in allowed_roots):
        if candidate.exists() and candidate.is_dir():
            return candidate
    abort(400, description="invalid pdf_dir")


def run_pipeline_job(job_id: str, args: list[str]) -> None:
    job = jobs[job_id]
    job["status"] = "running"
    job["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    command = [str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable), str(PIPELINE_SCRIPT), *args]
    job["command"] = command
    persist_job(job)
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        job["pid"] = process.pid
        persist_job(job)
        lines: list[str] = []
        last_persisted = time.time()
        if process.stdout:
            for line in process.stdout:
                text = line.rstrip()
                if not text:
                    continue
                lines.append(text)
                lines = lines[-120:]
                job["last_message"] = text
                job["log_tail"] = "\n".join(lines[-24:])
                job["stdout"] = "\n".join(lines)[-12000:]
                if time.time() - last_persisted >= 3:
                    persist_job(job)
                    last_persisted = time.time()
        returncode = process.wait()
        job["returncode"] = returncode
        job["stdout"] = "\n".join(lines)[-12000:]
        job["stderr"] = ""
        job["status"] = "completed" if returncode == 0 else "failed"
    except Exception as exc:
        job["status"] = "failed"
        job["stderr"] = str(exc)
    finally:
        job["progress"] = summarize_job_progress(job)
        job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        persist_job(job)


def repro_command_plan(target_path: Path, stages: list[str], offline: bool) -> list[list[str]]:
    python_exe = REPRO_PYTHON_EXE if REPRO_PYTHON_EXE.exists() else Path(sys.executable)
    base = [str(python_exe), "-m", "tools.repro_cli"]
    commands = [[*base, "init-target", "--target", str(target_path)], [*base, "extract-pdf", "--target", str(target_path)]]
    if "audit" in stages:
        command = [*base, "audit", "--target", str(target_path)]
        if offline:
            command.append("--offline")
        commands.append(command)
    if "model-spec" in stages:
        command = [*base, "model-spec", "--target", str(target_path)]
        if offline:
            command.append("--offline")
        commands.append(command)
    if "prepare-repro" in stages:
        commands.append([*base, "prepare-repro", "--target", str(target_path)])
    if "write-obsidian" in stages:
        commands.append([*base, "write-obsidian", "--target", str(target_path)])
    return commands


def summarize_repro_job(job: dict[str, Any]) -> dict[str, Any]:
    target_id = str(job.get("target_id") or "")
    run_path = repro_run_path(target_id) if target_id else Path()
    audit = load_json(run_path / "audits" / "reproducibility_audit.json", {})
    model_spec = load_json(run_path / "artifacts" / "model_spec.json", {})
    validation = load_json(run_path / "reports" / "data_validation.json", {})
    total = max(1, int(job.get("total_steps") or 1))
    completed = int(job.get("completed_steps") or 0)
    if job.get("status") == "completed":
        percent = 100
    elif job.get("status") == "failed":
        percent = max(5, min(95, round(completed / total * 100)))
    else:
        percent = max(4, min(96, round(completed / total * 100)))
    return {
        "status": job.get("status") or "queued",
        "stage_label": job.get("stage_label") or "等待启动复现工具链",
        "percent": percent,
        "target_id": target_id,
        "run_dir": str(run_path) if target_id else "",
        "completed_steps": completed,
        "total_steps": total,
        "scores": audit.get("scores") if isinstance(audit, dict) else {},
        "model_spec_counts": {
            "sets": len(model_spec.get("sets") or []) if isinstance(model_spec, dict) else 0,
            "parameters": len(model_spec.get("parameters") or []) if isinstance(model_spec, dict) else 0,
            "variables": len(model_spec.get("variables") or []) if isinstance(model_spec, dict) else 0,
            "constraints": len(model_spec.get("constraints") or []) if isinstance(model_spec, dict) else 0,
        },
        "data_validation": {
            "complete_files": validation.get("complete_files") if isinstance(validation, dict) else "",
            "missing_files": validation.get("missing_files") if isinstance(validation, dict) else "",
            "empty_files": validation.get("empty_files") if isinstance(validation, dict) else "",
        },
        "links": {
            "audit": repro_file_url(target_id, "audits/reproducibility_audit.md") if (run_path / "audits" / "reproducibility_audit.md").exists() else "",
            "model_spec": repro_file_url(target_id, "artifacts/model_spec.md") if (run_path / "artifacts" / "model_spec.md").exists() else "",
            "data_validation": repro_file_url(target_id, "reports/data_validation.md") if (run_path / "reports" / "data_validation.md").exists() else "",
        },
    }


def public_repro_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in job.items()
        if key not in {"commands", "stdout", "stderr"}
    }
    payload["progress"] = summarize_repro_job(job)
    if job.get("status") == "failed":
        payload["error"] = str(job.get("stderr") or job.get("last_message") or "未知错误")[:800]
    return payload


def run_repro_job(job_id: str, commands: list[list[str]]) -> None:
    job = repro_jobs[job_id]
    job["status"] = "running"
    job["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    job["total_steps"] = len(commands)
    lines: list[str] = []
    try:
        labels = {
            "init-target": "初始化复现目标",
            "extract-pdf": "抽取 PDF 正文与证据",
            "audit": "生成复现潜力评估",
            "model-spec": "抽取模型参数与变量",
            "prepare-repro": "生成复现工作区与数据模板",
            "write-obsidian": "写入知识库归档",
        }
        for index, command in enumerate(commands, start=1):
            command_name = command[3] if len(command) > 3 else "step"
            job["stage_label"] = labels.get(command_name, command_name)
            job["current_command"] = command_name
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            )
            job["pid"] = process.pid
            if process.stdout:
                for line in process.stdout:
                    text = line.rstrip()
                    if not text:
                        continue
                    lines.append(text)
                    lines = lines[-160:]
                    job["last_message"] = text
                    job["log_tail"] = "\n".join(lines[-30:])
                    job["stdout"] = "\n".join(lines)[-16000:]
            returncode = process.wait()
            job["returncode"] = returncode
            if returncode != 0:
                job["status"] = "failed"
                job["stderr"] = "\n".join(lines[-30:])
                return
            job["completed_steps"] = index
        job["status"] = "completed"
        job["stage_label"] = "复现工具链已生成"
        job["stderr"] = ""
    except Exception as exc:
        job["status"] = "failed"
        job["stderr"] = str(exc)
    finally:
        job["progress"] = summarize_repro_job(job)
        job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


@app.get("/")
def index():
    return render_showcase({"layer": "overview"})


@app.get("/direction/<direction_id>")
def direction_page(direction_id: str):
    return render_showcase({"layer": "direction", "direction_id": direction_id})


@app.get("/paper/<direction_id>/<paper_id>")
def paper_page(direction_id: str, paper_id: str):
    return render_showcase({"layer": "paper", "direction_id": direction_id, "paper_id": paper_id})


def render_showcase(initial_view: dict[str, str]):
    runs = iter_runs()
    selected_id = request.args.get("run") or select_default_run(runs)
    selected = collect_run(selected_id) if selected_id else None
    showcase_data = load_showcase_data(selected_id)
    run_cards = [
        {
            "id": run.run_id,
            "title": run_title(run),
            "modified": run.modified,
            "has_showcase": has_showcase_data(run),
            **run_card_meta(run),
        }
        for run in runs
    ]
    return render_template(
        "index.html",
        runs=run_cards,
        selected=selected,
        pdf_sources=list_pdf_sources(),
        showcase_data=showcase_data,
        reproduction_data=collect_reproduction_showcase(),
        query_meta=load_query_meta(selected_id) if selected_id else {},
        initial_view=initial_view,
    )


@app.get("/api/showcase-data")
def api_showcase_data():
    return jsonify(load_showcase_data(str(request.args.get("run") or "")))


@app.get("/api/quality-report")
def api_quality_report():
    return jsonify(load_quality_report(str(request.args.get("run") or "")))


@app.get("/api/reproduction")
def api_reproduction():
    return jsonify(collect_reproduction_showcase())


@app.post("/api/repro-chat")
def api_repro_chat():
    payload = request.get_json(silent=True) or {}
    target_id = str(payload.get("target_id") or "").strip()
    message = str(payload.get("message") or "").strip()
    mode = str(payload.get("mode") or "general").strip() or "general"
    history = payload.get("history") if isinstance(payload.get("history"), list) else []
    if not target_id or not message:
        return jsonify({"error": "target_id and message are required"}), 400
    run_path = safe_child(REPRO_RUN_ROOT, target_id)
    if not run_path.exists() or not (run_path / "target.yaml").exists():
        return jsonify({"error": "reproduction target not found"}), 404
    target = collect_repro_target(run_path)
    if bool(payload.get("demo")):
        answer, artifacts = build_repro_demo_answer(target_id, target, message)
        return jsonify(
            {
                "target_id": target_id,
                "mode": mode,
                "answer": answer,
                "artifacts": artifacts,
                "demo": True,
                "context_summary": {
                    "complete_files": target.get("data_validation", {}).get("complete_files"),
                    "empty_files": target.get("data_validation", {}).get("empty_files"),
                    "data_tasks": len(target.get("data_completion", {}).get("tasks", [])),
                },
            }
        )
    if call_openai_text is None:
        return jsonify({"error": "LLM client is unavailable in this environment."}), 500
    prompt = build_repro_chat_prompt(target, mode, message, history)
    system = (
        "You are a research reproducibility copilot inside a local literature-reproduction app. "
        "Answer in Chinese. Be concrete, cite filenames and columns, and distinguish evidence from assumptions."
    )
    try:
        answer = call_openai_text(prompt=prompt, system=system)
    except LLMError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify(
        {
            "target_id": target_id,
            "mode": mode,
            "answer": answer,
            "context_summary": {
                "complete_files": target.get("data_validation", {}).get("complete_files"),
                "empty_files": target.get("data_validation", {}).get("empty_files"),
                "data_tasks": len(target.get("data_completion", {}).get("tasks", [])),
            },
        }
    )


@app.get("/api/reproduction/paper")
def api_reproduction_paper():
    source_run_id = str(request.args.get("run") or "")
    direction_id = str(request.args.get("direction") or "")
    paper_id = str(request.args.get("paper") or "")
    if not source_run_id or not direction_id or not paper_id:
        return jsonify({"error": "run, direction and paper are required"}), 400
    context = find_literature_paper_context(source_run_id, direction_id, paper_id)
    paper = context["paper"]
    pdf_path = context["pdf_path"]
    return jsonify(
        {
            "paper": {
                "id": paper.get("id") or paper.get("paper_id"),
                "candidate_id": paper.get("candidate_id"),
                "title": paper.get("title"),
                "title_cn": paper.get("title_cn"),
                "doi": paper.get("doi"),
            },
            "reproduction": repro_status_for_paper(source_run_id, paper, pdf_path),
        }
    )


@app.get("/api/runs/<run_id>")
def api_run(run_id: str):
    return jsonify(collect_run(run_id))


@app.post("/api/jobs")
def create_job():
    payload = request.get_json(silent=True) or {}
    topic = str(payload.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "topic is required"}), 400

    mode = str(payload.get("mode") or "pdf_only")
    max_papers = int(payload.get("max_papers") or 5)
    max_results = int(payload.get("max_results") or max_papers)
    year_from = int(payload["year_from"]) if payload.get("year_from") else None
    year_to = int(payload["year_to"]) if payload.get("year_to") else None
    if year_from and year_to and year_from > year_to:
        return jsonify({"error": "year_from must be <= year_to"}), 400
    downloadable_limit = max(max_results, max_papers)
    source_fetch_limit = max(10, max_papers * 4, downloadable_limit)
    requested_parts = str(payload.get("run_parts") or "discovery,reviews")
    allowed_parts = {"discovery", "reviews"}
    parts = [part.strip() for part in requested_parts.split(",") if part.strip() in allowed_parts]
    run_parts = ",".join(parts or ["discovery", "reviews"])
    pipeline_args: list[str] = []
    pipeline_args.extend(
        [
            "--max-results",
            str(source_fetch_limit),
            "--max-papers",
            str(max_papers),
            "--run-parts",
            run_parts,
            "--extract-figures-tables",
        ]
    )
    if mode == "pdf_only":
        pdf_dir = resolve_pdf_dir(str(payload.get("pdf_dir") or ""))
        pipeline_args.extend(["--input-mode", "local", "--pdf-dir", str(pdf_dir)])
        if bool(payload.get("all_papers")):
            pipeline_args.append("--all-papers")
    else:
        pipeline_args.extend(
            [
                "--input-mode",
                "online",
                "--candidate-multiplier",
                "3",
                "--max-candidates",
                str(downloadable_limit),
                "--require-pdf",
                "true",
                "--compare-sources",
            ]
        )
        if year_from:
            pipeline_args.extend(["--year-from", str(year_from)])
        if year_to:
            pipeline_args.extend(["--year-to", str(year_to)])
    and_terms: list[str] = []
    filter_and = str(payload.get("filter_and") or "").strip()
    if filter_and:
        and_terms.append(filter_and)
        pipeline_args.extend(["--filter-and", filter_and])
    for item in payload.get("topic_clauses", []):
        logic = str(item.get("logic") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        if logic == "and" and text:
            and_terms.append(text)
            pipeline_args.extend(["--filter-and", text])
        elif logic == "or" and text:
            pipeline_args.extend(["--filter-or", text])
        elif logic == "not" and text:
            pipeline_args.extend(["--filter-not", text])

    pipeline_topic = topic
    if mode != "pdf_only" and and_terms:
        joined_terms = " ".join(dict.fromkeys(and_terms))
        pipeline_topic = f"{topic} {joined_terms}".strip()
    output_dir = OUTPUT_ROOT / f"{time.strftime('%Y%m%d_%H%M')}_{safe_output_name(pipeline_topic)}"
    args = ["--topic", pipeline_topic, "--overwrite", "--output-dir", str(output_dir), *pipeline_args]

    client_job_id = str(payload.get("client_job_id") or "").strip()
    job_id = client_job_id if re.fullmatch(r"[A-Za-z0-9_-]{8,80}", client_job_id) else time.strftime("%Y%m%d_%H%M%S")
    if job_id in jobs:
        job_id = f"{job_id}_{int(time.time() * 1000) % 100000}"
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "topic": pipeline_topic,
        "base_topic": topic,
        "output_dir": str(output_dir),
        "max_papers": max_papers,
        "downloadable_limit": downloadable_limit,
        "run_parts": run_parts,
        "year_from": year_from,
        "year_to": year_to,
    }
    persist_job(jobs[job_id])
    thread = threading.Thread(target=run_pipeline_job, args=(job_id, args), daemon=True)
    thread.start()
    return jsonify(public_job_payload(jobs[job_id]))


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    job = jobs.get(job_id) or load_persisted_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(public_job_payload(job))


@app.get("/api/jobs/by-run/<run_id>")
def get_job_by_run(run_id: str):
    run_path = safe_child(OUTPUT_ROOT, run_id)
    if not run_path.exists():
        return jsonify({"error": "job not found"}), 404
    report = load_json(run_path / "unified_run_report.json", {})
    report_status = str(report.get("status") or "") if isinstance(report, dict) else ""
    if report_status in {"completed", "failed"}:
        status = report_status
    else:
        for job in jobs.values():
            output_dir = Path(str(job.get("output_dir") or ""))
            if output_dir.name == run_id:
                return jsonify(public_job_payload(job))
        persisted = load_json(run_path / WEB_JOB_STATUS_NAME, {})
        if isinstance(persisted, dict) and persisted:
            persisted.setdefault("output_dir", str(run_path))
            return jsonify(public_job_payload(persisted))
        status = "running"
    payload = {
        "id": f"run:{run_id}",
        "status": status,
        "topic": report.get("topic") if isinstance(report, dict) else run_id,
        "output_dir": str(run_path),
        "max_papers": report.get("max_papers") if isinstance(report, dict) else "",
    }
    return jsonify(public_job_payload(payload))


@app.post("/api/repro-jobs")
def create_repro_job():
    payload = request.get_json(silent=True) or {}
    source_run_id = str(payload.get("run_id") or "").strip()
    direction_id = str(payload.get("direction_id") or "").strip()
    paper_id = str(payload.get("paper_id") or "").strip()
    if not source_run_id or not direction_id or not paper_id:
        return jsonify({"error": "run_id, direction_id and paper_id are required"}), 400
    context = find_literature_paper_context(source_run_id, direction_id, paper_id)
    paper = context["paper"]
    pdf_path = context["pdf_path"]
    if not pdf_path:
        return jsonify({"error": "当前单篇文献没有可访问的 PDF，无法启动复现工具链。"}), 400
    requested_stages = payload.get("stages") or ["audit", "model-spec", "prepare-repro"]
    stages = [str(item).strip() for item in requested_stages if str(item).strip()]
    allowed = {"audit", "model-spec", "prepare-repro", "write-obsidian"}
    stages = [stage for stage in stages if stage in allowed] or ["audit", "model-spec", "prepare-repro"]
    offline = bool(payload.get("offline", True))
    target_path = write_generated_target(source_run_id, direction_id, paper, pdf_path)
    target_id = target_path.stem
    commands = repro_command_plan(target_path, stages, offline)
    job_id = f"repro-{time.strftime('%Y%m%d_%H%M%S')}"
    repro_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "source_run_id": source_run_id,
        "direction_id": direction_id,
        "paper_id": paper.get("id") or paper.get("paper_id") or paper_id,
        "target_id": target_id,
        "title": paper.get("title_cn") or paper.get("title") or target_id,
        "pdf_path": str(pdf_path),
        "target_config": str(target_path),
        "stages": stages,
        "offline": offline,
        "commands": commands,
        "completed_steps": 0,
        "total_steps": len(commands),
    }
    thread = threading.Thread(target=run_repro_job, args=(job_id, commands), daemon=True)
    thread.start()
    return jsonify(public_repro_job_payload(repro_jobs[job_id]))


@app.get("/api/repro-jobs/<job_id>")
def get_repro_job(job_id: str):
    job = repro_jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(public_repro_job_payload(job))


@app.get("/runs/<run_id>/files/<path:relative_path>")
def run_file(run_id: str, relative_path: str):
    run_path = safe_child(OUTPUT_ROOT, run_id)
    target = safe_child(run_path, relative_path)
    if not target.exists() or not target.is_file():
        abort(404)
    return send_from_directory(target.parent, target.name)


@app.get("/repro-runs/<run_id>/files/<path:relative_path>")
def repro_run_file(run_id: str, relative_path: str):
    run_path = safe_child(REPRO_RUN_ROOT, run_id)
    target = safe_child(run_path, relative_path)
    if not target.exists() or not target.is_file():
        abort(404)
    return send_from_directory(target.parent, target.name)


@app.get("/repo-files/<path:relative_path>")
def repo_file(relative_path: str):
    target = safe_child(REPO_ROOT, relative_path)
    if not target.exists() or not target.is_file():
        abort(404)
    return send_from_directory(target.parent, target.name)


if __name__ == "__main__":
    host = str(os.environ.get("LITERATURE_SHOWCASE_HOST") or "127.0.0.1")
    port = int(os.environ.get("LITERATURE_SHOWCASE_PORT") or "8051")
    app.run(host=host, port=port, debug=False)

