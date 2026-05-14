from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request, send_from_directory


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
RUNS_ROOT = PROJECT_ROOT / "output"
PIPELINE_SCRIPT = PIPELINE_DIR / "unified_literature_pipeline.py"

app = Flask(__name__)
jobs: dict[str, dict[str, Any]] = {}


PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>文献综述图生成器</title>
  <style>
    :root {
      --ink: #16211f;
      --muted: #66736f;
      --line: #d8e0dc;
      --paper: #f8faf7;
      --panel: #ffffff;
      --green: #286a52;
      --green-2: #e3f1eb;
      --amber: #9b631d;
      --red: #b23a3a;
      --shadow: 0 18px 40px rgba(30, 48, 42, .10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 8%, rgba(40, 106, 82, .10), transparent 28rem),
        linear-gradient(135deg, #f4f7f2 0%, #eef5f3 48%, #f9f6ef 100%);
      font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    }
    .shell {
      width: min(1180px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 34px 0 56px;
    }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 24px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 34px;
      font-weight: 800;
      letter-spacing: 0;
    }
    .sub {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
      max-width: 720px;
    }
    .badge {
      border: 1px solid var(--line);
      background: rgba(255,255,255,.72);
      color: var(--green);
      padding: 10px 14px;
      border-radius: 999px;
      white-space: nowrap;
      font-size: 14px;
    }
    .layout {
      display: grid;
      grid-template-columns: 390px 1fr;
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: rgba(255,255,255,.86);
      border: 1px solid rgba(216, 224, 220, .9);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .form-panel { padding: 20px; position: sticky; top: 18px; }
    label {
      display: block;
      font-size: 14px;
      color: var(--muted);
      margin-bottom: 8px;
    }
    textarea, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      outline: none;
    }
    textarea {
      min-height: 116px;
      resize: vertical;
      padding: 12px;
      line-height: 1.6;
    }
    input { padding: 10px 12px; }
    textarea:focus, input:focus {
      border-color: var(--green);
      box-shadow: 0 0 0 3px rgba(40, 106, 82, .12);
    }
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 14px;
    }
    button {
      width: 100%;
      margin-top: 18px;
      border: 0;
      border-radius: 6px;
      background: var(--green);
      color: white;
      font: inherit;
      font-weight: 700;
      padding: 12px 14px;
      cursor: pointer;
    }
    button:disabled { opacity: .55; cursor: wait; }
    .hint {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      margin: 12px 0 0;
    }
    .status {
      padding: 18px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      border-bottom: 1px solid var(--line);
    }
    .state {
      color: var(--green);
      font-weight: 800;
    }
    .state.failed { color: var(--red); }
    .state.running { color: var(--amber); }
    .content { padding: 18px 20px 20px; }
    .empty {
      min-height: 320px;
      display: grid;
      place-items: center;
      color: var(--muted);
      text-align: center;
      line-height: 1.8;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,.45);
    }
    .images {
      display: grid;
      gap: 18px;
    }
    figure {
      margin: 0;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    figcaption {
      padding: 10px 12px;
      color: var(--muted);
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }
    figure img {
      display: block;
      width: 100%;
      background: white;
    }
    pre {
      margin: 18px 0 0;
      max-height: 260px;
      overflow: auto;
      padding: 14px;
      border-radius: 8px;
      background: #17211e;
      color: #dce9e4;
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-wrap;
    }
    @media (max-width: 880px) {
      header { display: block; }
      .badge { display: inline-block; margin-top: 16px; }
      .layout { grid-template-columns: 1fr; }
      .form-panel { position: static; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>文献综述图生成器</h1>
        <p class="sub">输入关键研究领域，系统会检索文献、生成正文结构化结果，并在网页中展示综述 SVG 图。</p>
      </div>
      <div class="badge">DeepSeek + 本地流水线</div>
    </header>

    <section class="layout">
      <form class="panel form-panel" id="jobForm">
        <label for="topic">关键研究领域</label>
        <textarea id="topic" name="topic">储能参与电力市场报价方式</textarea>
        <div class="grid-2">
          <div>
            <label for="maxPapers">处理文献数</label>
            <input id="maxPapers" name="max_papers" type="number" min="1" max="8" value="3">
          </div>
          <div>
            <label for="maxResults">检索条数</label>
            <input id="maxResults" name="max_results" type="number" min="2" max="12" value="5">
          </div>
        </div>
        <button id="submitBtn" type="submit">生成综述图</button>
        <p class="hint">生成过程可能需要数分钟。页面会自动刷新状态；日志中能看到当前执行到哪一步。</p>
      </form>

      <section class="panel">
        <div class="status">
          <span id="jobTitle">尚未开始</span>
          <span id="jobState" class="state">待运行</span>
        </div>
        <div class="content">
          <div id="empty" class="empty">提交研究领域后，生成的 SVG 图会显示在这里。</div>
          <div id="images" class="images"></div>
          <pre id="log" hidden></pre>
        </div>
      </section>
    </section>
  </main>

  <script>
    const form = document.getElementById("jobForm");
    const button = document.getElementById("submitBtn");
    const title = document.getElementById("jobTitle");
    const state = document.getElementById("jobState");
    const log = document.getElementById("log");
    const empty = document.getElementById("empty");
    const images = document.getElementById("images");
    let timer = null;

    function setState(text, cls) {
      state.textContent = text;
      state.className = "state " + (cls || "");
    }

    function renderJob(job) {
      title.textContent = job.topic || "任务运行中";
      const label = job.status === "completed" ? "已完成" : job.status === "failed" ? "失败" : "运行中";
      setState(label, job.status);
      log.hidden = false;
      log.textContent = (job.log || []).join("");
      images.innerHTML = "";
      if (job.images && job.images.length) {
        empty.hidden = true;
        for (const image of job.images) {
          const fig = document.createElement("figure");
          const cap = document.createElement("figcaption");
          cap.textContent = image.name;
          const img = document.createElement("img");
          img.src = image.url + "?t=" + Date.now();
          img.alt = image.name;
          fig.appendChild(cap);
          fig.appendChild(img);
          images.appendChild(fig);
        }
      } else {
        empty.hidden = false;
      }
      if (job.status === "completed" || job.status === "failed") {
        clearInterval(timer);
        button.disabled = false;
      }
    }

    async function poll(jobId) {
      const response = await fetch(`/api/jobs/${jobId}`);
      renderJob(await response.json());
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      clearInterval(timer);
      button.disabled = true;
      images.innerHTML = "";
      empty.hidden = false;
      log.hidden = false;
      log.textContent = "任务已提交，正在启动流水线...\n";
      setState("运行中", "running");
      const payload = {
        topic: document.getElementById("topic").value,
        max_papers: Number(document.getElementById("maxPapers").value || 3),
        max_results: Number(document.getElementById("maxResults").value || 5)
      };
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const job = await response.json();
      title.textContent = job.topic;
      timer = setInterval(() => poll(job.id), 3500);
      poll(job.id);
    });
  </script>
</body>
</html>
"""


def ensure_runs_root() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def safe_output_name(topic: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", topic.strip())
    cleaned = re.sub(r"\s+", "_", cleaned, flags=re.UNICODE).strip("._ ")
    return (cleaned or "关键研究领域")[:80]


def next_job_dir(topic: str) -> tuple[str, Path]:
    base = time.strftime("%Y%m%d_%H%M_") + safe_output_name(topic)
    job_dir = RUNS_ROOT / base
    if not job_dir.exists():
        return base, job_dir
    index = 2
    while True:
        job_id = f"{base}_{index}"
        job_dir = RUNS_ROOT / job_id
        if not job_dir.exists():
            return job_id, job_dir
        index += 1


def image_payload(job_id: str, job_dir: Path) -> list[dict[str, str]]:
    review_dir = job_dir / "review_figures"
    if not review_dir.exists():
        return []
    return [
        {
            "name": path.name,
            "url": f"/runs/{job_id}/review_figures/{path.name}",
        }
        for path in sorted(review_dir.glob("*.svg"))
    ]


def append_log(job: dict[str, Any], text: str) -> None:
    job.setdefault("log", []).append(text)
    if len(job["log"]) > 600:
        job["log"] = job["log"][-600:]


def run_job(job_id: str) -> None:
    job = jobs[job_id]
    job_dir = Path(job["output_dir"])
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        job["topic"],
        "--max-results",
        str(job["max_results"]),
        "--max-papers",
        str(job["max_papers"]),
        "--output-dir",
        str(job_dir),
        "--overwrite",
    ]
    job["command"] = command
    job["status"] = "running"
    append_log(job, "启动命令：\n" + " ".join(command) + "\n\n")

    started = time.time()
    try:
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
            append_log(job, line)
        return_code = process.wait()
        job["returncode"] = return_code
        job["elapsed_seconds"] = round(time.time() - started, 3)
        job["images"] = image_payload(job_id, job_dir)
        job["status"] = "completed" if return_code == 0 else "failed"
        if return_code != 0:
            append_log(job, f"\n流程失败，退出码：{return_code}\n")
    except Exception as exc:
        job["status"] = "failed"
        job["elapsed_seconds"] = round(time.time() - started, 3)
        append_log(job, f"\n任务异常：{exc}\n")

    (job_dir / "web_job.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/api/jobs")
def create_job():
    ensure_runs_root()
    payload = request.get_json(force=True) or {}
    topic = str(payload.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "请输入关键研究领域"}), 400
    max_papers = max(1, min(int(payload.get("max_papers") or 3), 8))
    max_results = max(2, min(int(payload.get("max_results") or 5), 12))
    job_id, job_dir = next_job_dir(topic)
    job_dir.mkdir(parents=True, exist_ok=True)
    jobs[job_id] = {
        "id": job_id,
        "topic": topic,
        "status": "queued",
        "max_papers": max_papers,
        "max_results": max_results,
        "output_dir": str(job_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log": [],
        "images": [],
    }
    thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    thread.start()
    return jsonify({"id": job_id, "topic": topic})


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "任务不存在"}), 404
    if job["status"] == "completed":
        job["images"] = image_payload(job_id, Path(job["output_dir"]))
    return jsonify(job)


@app.get("/runs/<job_id>/review_figures/<filename>")
def get_review_figure(job_id: str, filename: str):
    job = jobs.get(job_id)
    if not job:
        return "任务不存在", 404
    review_dir = Path(job["output_dir"]) / "review_figures"
    return send_from_directory(review_dir, filename)


def main() -> None:
    ensure_runs_root()
    try:
        from waitress import serve

        serve(app, host="127.0.0.1", port=5000, threads=4)
    except ImportError:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
