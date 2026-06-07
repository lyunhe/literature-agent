from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from analysis_pipeline.core.common import ensure_dir, save_json


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _round(value: Any) -> float:
    try:
        return round(float(value), 3)
    except Exception:
        return 0.0


def _latest_time_payload(time_dir: Path) -> dict[str, Any]:
    latest = _load_json(time_dir / "latest_time_record.json", {})
    if isinstance(latest, dict) and latest:
        return latest
    candidates = sorted(time_dir.glob("run_*.json"))
    if not candidates:
        return {}
    payload = _load_json(candidates[-1], {})
    return payload if isinstance(payload, dict) else {}


def write_timing_summary(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    time_dir = ensure_dir(run_dir / "time_records")
    report = _load_json(run_dir / "unified_run_report.json", {})
    if not isinstance(report, dict):
        report = {}
    payload = _latest_time_payload(time_dir)
    records = payload.get("records", []) if isinstance(payload.get("records"), list) else []

    rows: list[dict[str, Any]] = []
    order = 1

    for step in report.get("steps", []) if isinstance(report.get("steps"), list) else []:
        rows.append(
            {
                "order": order,
                "record_type": "pipeline_step",
                "stage": step.get("name", ""),
                "item": "",
                "status": step.get("status", ""),
                "started_at": step.get("start_time", ""),
                "elapsed_seconds": _round(step.get("elapsed_seconds")),
                "parent_stage": "",
                "error": step.get("reason", ""),
                "note": f"step_index={step.get('index', '')}; log_file={step.get('log_file', '')}",
            }
        )
        order += 1

    stage_totals: dict[str, float] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        stage = str(record.get("stage") or "")
        elapsed = _round(record.get("elapsed_seconds"))
        stage_totals[stage] = round(stage_totals.get(stage, 0.0) + elapsed, 3)
        rows.append(
            {
                "order": order,
                "record_type": "paper_or_review_detail",
                "stage": stage,
                "item": record.get("item", ""),
                "status": record.get("status", ""),
                "started_at": record.get("started_at", ""),
                "elapsed_seconds": elapsed,
                "parent_stage": "reviews",
                "error": record.get("error", ""),
                "note": "single_paper_lit_card 表示单篇文献信息卡耗时",
            }
        )
        order += 1

    for stage, elapsed in sorted(stage_totals.items()):
        rows.append(
            {
                "order": order,
                "record_type": "stage_total",
                "stage": stage,
                "item": "TOTAL",
                "status": "ok",
                "started_at": "",
                "elapsed_seconds": elapsed,
                "parent_stage": "reviews",
                "error": "",
                "note": "由 paper_or_review_detail 聚合",
            }
        )
        order += 1

    step_total = round(sum(_round(step.get("elapsed_seconds")) for step in report.get("steps", []) if isinstance(step, dict)), 3)
    rows.append(
        {
            "order": order,
            "record_type": "pipeline_total",
            "stage": "all_pipeline_steps",
            "item": "TOTAL",
            "status": report.get("status", ""),
            "started_at": report.get("started_at", ""),
            "elapsed_seconds": step_total,
            "parent_stage": "",
            "error": report.get("failure", ""),
            "note": "由 unified_run_report.steps 聚合，包含 discovery/reviews/图表截取等环节",
        }
    )

    csv_path = time_dir / "timing_summary.csv"
    fieldnames = [
        "order",
        "record_type",
        "stage",
        "item",
        "status",
        "started_at",
        "elapsed_seconds",
        "parent_stage",
        "error",
        "note",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_json(time_dir / "timing_summary.json", {"rows": rows})
    return csv_path
