from __future__ import annotations

import contextlib
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from analysis_pipeline._bootstrap import PROJECT_ROOT
from analysis_pipeline.core.common import ensure_dir, save_json
from analysis_pipeline.core.run_context import now_text, safe_output_name


class TeeWriter:
    def __init__(self, *writers: Any) -> None:
        self.writers = writers

    def write(self, data: str) -> int:
        for writer in self.writers:
            try:
                writer.write(data)
            except UnicodeEncodeError:
                encoding = getattr(writer, "encoding", None) or "utf-8"
                writer.write(data.encode(encoding, errors="replace").decode(encoding))
            writer.flush()
        return len(data)

    def flush(self) -> None:
        for writer in self.writers:
            writer.flush()


def make_step_log_path(logs_dir: Path, index: int, name: str) -> Path:
    return logs_dir / f"{index:02d}_{safe_output_name(name)}.log"


def write_step_records(logs_dir: Path, steps: list[dict[str, Any]]) -> None:
    ensure_dir(logs_dir)
    (logs_dir / "step_records.jsonl").write_text(
        "".join(json.dumps(step, ensure_ascii=False) + "\n" for step in steps),
        encoding="utf-8",
    )
    fieldnames = ["index", "name", "status", "returncode", "elapsed_seconds", "start_time", "end_time", "log_file", "reason"]
    with (logs_dir / "step_records.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for step in steps:
            writer.writerow({key: step.get(key, "") for key in fieldnames})


def start_step(ctx: Any, name: str) -> tuple[dict[str, Any], Path]:
    index = len(ctx.report["steps"]) + 1
    log_path = make_step_log_path(ctx.logs_dir, index, name)
    step = {
        "index": index,
        "name": name,
        "status": "running",
        "start_time": now_text(),
        "log_file": str(log_path.resolve()),
    }
    ctx.report["current_step"] = step
    ctx.save_report()
    save_json(ctx.logs_dir / "current_step.json", step)
    return step, log_path


def finish_step(ctx: Any, step: dict[str, Any], *, status: str, started: float, reason: str | None = None) -> None:
    step["status"] = status
    step["end_time"] = now_text()
    step["elapsed_seconds"] = round(time.time() - started, 3)
    step["returncode"] = 0 if status == "completed" else ""
    if reason:
        step["reason"] = reason
    ctx.report.pop("current_step", None)
    ctx.report["steps"].append(step)
    ctx.report["updated_at"] = now_text()
    ctx.save_report()
    save_json(ctx.logs_dir / "latest_step.json", step)
    write_step_records(ctx.logs_dir, ctx.report["steps"])


def run_tracked_block(ctx: Any, name: str, callback: Any) -> Any:
    started = time.time()
    step, log_path = start_step(ctx, name)
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
            finish_step(ctx, step, status="failed", started=started, reason=str(exc))
            ctx.report["status"] = "failed"
            ctx.report["failed_at"] = now_text()
            ctx.report["failure"] = str(exc)
            ctx.save_report()
            raise
        log.write(f"\n结束时间：{now_text()}\n")
    finish_step(ctx, step, status="completed", started=started)
    return result


def add_skipped_step(ctx: Any, name: str, reason: str) -> None:
    started = time.time()
    step, log_path = start_step(ctx, name)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"=== {name} ===\n")
        log.write(f"开始时间：{step['start_time']}\n")
        log.write(f"跳过原因：{reason}\n")
    print(f"[跳过] {name}: {reason}")
    finish_step(ctx, step, status="skipped", started=started, reason=reason)


def run_existing_script(ctx: Any, name: str, script_path: Path, args: list[str], required: bool = True) -> bool:
    started = time.time()
    step, log_path = start_step(ctx, name)
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
    finish_step(ctx, step, status="completed" if returncode == 0 else "failed", started=started, reason="" if returncode == 0 else f"退出码 {returncode}")
    if returncode != 0 and required:
        raise RuntimeError(f"{name} 失败，退出码：{returncode}")
    return returncode == 0
