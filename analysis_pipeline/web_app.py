from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    from flask import Flask, jsonify, render_template_string, request, send_from_directory
except ModuleNotFoundError:
    Flask = None  # type: ignore[assignment]
    jsonify = render_template_string = request = send_from_directory = None  # type: ignore[assignment]


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
RUNS_ROOT = PROJECT_ROOT / "output"
PIPELINE_SCRIPT = PIPELINE_DIR / "unified_literature_pipeline.py"

class _MissingFlaskApp:
    def get(self, *args: Any, **kwargs: Any):
        def decorator(func):
            return func
        return decorator

    def post(self, *args: Any, **kwargs: Any):
        def decorator(func):
            return func
        return decorator


app = Flask(__name__) if Flask is not None else _MissingFlaskApp()
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
    .topic-builder {
      display: grid;
      gap: 10px;
    }
    .topic-main textarea {
      min-height: 96px;
      margin-bottom: 10px;
    }
    .topic-spacer {
      height: 18px;
    }
    .topic-row {
      display: grid;
      grid-template-columns: 82px 1fr 34px;
      gap: 8px;
      align-items: start;
    }
    select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      padding: 10px 8px;
      outline: none;
    }
    .topic-row input {
      min-height: 42px;
    }
    .icon-btn {
      width: 34px;
      height: 34px;
      margin: 4px 0 0;
      display: inline-grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--green);
      font-size: 22px;
      line-height: 1;
      padding: 0;
      cursor: pointer;
    }
    .icon-btn:hover {
      border-color: var(--green);
      background: var(--green-2);
    }
    .add-topic {
      justify-self: center;
      margin-top: 2px;
    }
    .remove-topic {
      color: var(--red);
      font-size: 18px;
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
    .direction-section { display: grid; gap: 14px; }
    .direction-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 2px;
    }
    .direction-actions p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 14px;
    }
    .continue-btn {
      width: auto;
      min-width: 160px;
      margin: 0;
      flex: 0 0 auto;
    }
    .direction-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }
    .direction-card {
      position: relative;
      min-height: 230px;
      max-height: 360px;
      overflow: auto;
      padding: 14px;
      border: 1px solid #1f2725;
      border-radius: 4px;
      background: #fff;
      color: #111;
      cursor: pointer;
      box-shadow: none;
      font-family: SimSun, serif;
    }
    .direction-card.excluded {
      background: #eeeeee;
      color: #5f6663;
      border-color: #9ca5a1;
    }
    .direction-card.excluded::after {
      content: "";
      position: absolute;
      left: 8px;
      right: 8px;
      top: 50%;
      border-top: 2px solid #4f5653;
      pointer-events: none;
    }
    .direction-title {
      margin: 0 0 6px;
      font-size: 18px;
      line-height: 1.35;
      font-family: SimSun, serif;
    }
    .direction-title .en {
      display: block;
      margin-top: 2px;
      font-family: "Times New Roman", Times, serif;
      font-size: 15px;
      font-weight: 600;
    }
    .direction-summary {
      margin: 0 0 10px;
      color: inherit;
      font-size: 13px;
      line-height: 1.55;
    }
    .paper-list {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }
    .paper-item {
      padding-top: 8px;
      border-top: 1px solid #d6ddd9;
      font-size: 12px;
      line-height: 1.45;
    }
    .paper-cn {
      font-family: SimSun, serif;
      font-weight: 600;
    }
    .paper-en {
      margin-top: 3px;
      font-family: "Times New Roman", Times, serif;
      font-size: 12px;
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
        <div class="topic-builder">
          <div class="topic-main">
            <label for="topic">关键研究领域</label>
            <textarea id="topic" name="topic">储能参与电力市场报价方式</textarea>
            <div class="topic-spacer"></div>
            <button class="icon-btn add-topic" id="addTopicBtn" type="button" title="添加主题条件" aria-label="添加主题条件">+</button>
          </div>
          <div id="topicRows"></div>
        </div>
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
        <button id="submitBtn" type="submit">开始文献调研</button>
        <p class="hint">生成过程可能需要数分钟。页面会自动刷新状态；日志中能看到当前执行到哪一步。</p>
      </form>

      <section class="panel">
        <div class="status">
          <span id="jobTitle">尚未开始</span>
          <span id="jobState" class="state">待运行</span>
        </div>
        <div class="content">
          <div id="empty" class="empty">提交研究领域后，生成的 SVG 图会显示在这里。</div>
          <div id="directions" class="direction-section" hidden>
            <div class="direction-actions">
              <p>点击不需要的方向，将其排除；再次点击可恢复。确认后继续下载 PDF 并分析全文。</p>
              <button class="continue-btn" id="continueBtn" type="button">继续下载并分析</button>
            </div>
            <div id="directionGrid" class="direction-grid"></div>
          </div>
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
    const directions = document.getElementById("directions");
    const directionGrid = document.getElementById("directionGrid");
    const continueBtn = document.getElementById("continueBtn");
    const addTopicBtn = document.getElementById("addTopicBtn");
    const topicRows = document.getElementById("topicRows");
    let timer = null;
    let currentJobId = null;
    let excludedDirections = new Set();

    function setState(text, cls) {
      state.textContent = text;
      state.className = "state " + (cls || "");
    }

    function addTopicRow(initialLogic, initialText) {
      const row = document.createElement("div");
      row.className = "topic-row";
      const select = document.createElement("select");
      select.name = "topic_logic";
      for (const [value, label] of [["and", "且"], ["or", "或"], ["not", "非"]]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        if (value === initialLogic) option.selected = true;
        select.appendChild(option);
      }
      const input = document.createElement("input");
      input.name = "topic_text";
      input.type = "text";
      input.placeholder = "填写主题或关键词，多个同义词可用逗号分隔";
      input.value = initialText || "";
      const remove = document.createElement("button");
      remove.className = "icon-btn remove-topic";
      remove.type = "button";
      remove.title = "删除主题条件";
      remove.setAttribute("aria-label", "删除主题条件");
      remove.textContent = "x";
      remove.addEventListener("click", () => row.remove());
      row.appendChild(select);
      row.appendChild(input);
      row.appendChild(remove);
      topicRows.appendChild(row);
      input.focus();
    }

    function collectTopicClauses() {
      return Array.from(topicRows.querySelectorAll(".topic-row"))
        .map(row => ({
          logic: row.querySelector("select").value,
          text: row.querySelector("input").value.trim()
        }))
        .filter(item => item.text);
    }

    function renderJob(job) {
      currentJobId = job.id || currentJobId;
      title.textContent = job.topic || "任务运行中";
      const label = job.status === "completed" ? "已完成" : job.status === "failed" ? "失败" : job.status === "awaiting_selection" ? "等待选择方向" : "运行中";
      setState(label, job.status);
      log.hidden = false;
      log.textContent = (job.log || []).join("");
      images.innerHTML = "";
      directionGrid.innerHTML = "";
      directions.hidden = true;
      continueBtn.disabled = false;
      if (job.status === "awaiting_selection" && job.directions && job.directions.length) {
        clearInterval(timer);
        button.disabled = false;
        empty.hidden = true;
        directions.hidden = false;
        for (const direction of job.directions) {
          const card = document.createElement("article");
          card.className = "direction-card" + (excludedDirections.has(direction.direction_id) ? " excluded" : "");
          card.dataset.directionId = direction.direction_id;
          const heading = document.createElement("h2");
          heading.className = "direction-title";
          heading.textContent = `${direction.direction_id} ${direction.direction_name_cn || ""}`;
          if (direction.direction_name_en) {
            const en = document.createElement("span");
            en.className = "en";
            en.textContent = direction.direction_name_en;
            heading.appendChild(en);
          }
          const summary = document.createElement("p");
          summary.className = "direction-summary";
          summary.textContent = direction.direction_summary_cn || "";
          const list = document.createElement("div");
          list.className = "paper-list";
          for (const paper of direction.papers || []) {
            const item = document.createElement("div");
            item.className = "paper-item";
            const cn = document.createElement("div");
            cn.className = "paper-cn";
            cn.textContent = paper.title_cn || paper.title || "";
            const en = document.createElement("div");
            en.className = "paper-en";
            en.textContent = paper.title || "";
            item.appendChild(cn);
            item.appendChild(en);
            list.appendChild(item);
          }
          card.appendChild(heading);
          card.appendChild(summary);
          card.appendChild(list);
          card.addEventListener("click", () => {
            const id = card.dataset.directionId;
            if (excludedDirections.has(id)) {
              excludedDirections.delete(id);
              card.classList.remove("excluded");
            } else {
              excludedDirections.add(id);
              card.classList.add("excluded");
            }
          });
          directionGrid.appendChild(card);
        }
        return;
      }
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
      excludedDirections = new Set();
      images.innerHTML = "";
      directions.hidden = true;
      directionGrid.innerHTML = "";
      empty.hidden = false;
      log.hidden = false;
      log.textContent = "任务已提交，正在启动流水线...\n";
      setState("运行中", "running");
      const payload = {
        topic: document.getElementById("topic").value,
        topic_clauses: collectTopicClauses(),
        max_papers: Number(document.getElementById("maxPapers").value || 3),
        max_results: Number(document.getElementById("maxResults").value || 5)
      };
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const job = await response.json();
      currentJobId = job.id;
      title.textContent = job.topic;
      timer = setInterval(() => poll(job.id), 3500);
      poll(job.id);
    });

    addTopicBtn.addEventListener("click", () => addTopicRow("and", ""));

    continueBtn.addEventListener("click", async () => {
      if (!currentJobId) return;
      const cards = Array.from(directionGrid.querySelectorAll(".direction-card"));
      const selected = cards
        .map(card => card.dataset.directionId)
        .filter(id => !excludedDirections.has(id));
      if (!selected.length) {
        alert("请至少保留一个方向。");
        return;
      }
      continueBtn.disabled = true;
      button.disabled = true;
      log.textContent += "\n已提交方向选择，继续下载并分析...\n";
      const response = await fetch(`/api/jobs/${currentJobId}/selection`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({selected_directions: selected})
      });
      const job = await response.json();
      renderJob(job);
      timer = setInterval(() => poll(currentJobId), 3500);
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


def load_direction_payload(job_dir: Path) -> list[dict[str, Any]]:
    path = job_dir / "download" / "candidate_directions.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def run_process(job: dict[str, Any], command: list[str], fail_message: str) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    job["command"] = command
    append_log(job, "启动命令：\n" + " ".join(command) + "\n\n")
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
    if return_code != 0:
        append_log(job, f"\n{fail_message}，退出码：{return_code}\n")
    return return_code


def topic_clause_args(job: dict[str, Any]) -> list[str]:
    flag_by_logic = {
        "and": "--filter-and",
        "or": "--filter-or",
        "not": "--filter-not",
    }
    args: list[str] = []
    for item in job.get("topic_clauses", []):
        logic = str(item.get("logic") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        flag = flag_by_logic.get(logic)
        if flag and text:
            args.extend([flag, text])
    return args


def run_screening_job(job_id: str) -> None:
    job = jobs[job_id]
    job_dir = Path(job["output_dir"])
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
        "--screen-only",
        "--overwrite",
    ]
    command.extend(topic_clause_args(job))
    job["status"] = "screening"

    started = time.time()
    try:
        return_code = run_process(job, command, "方向预筛失败")
        job["elapsed_seconds"] = round(time.time() - started, 3)
        if return_code == 0:
            job["directions"] = load_direction_payload(job_dir)
            job["status"] = "awaiting_selection"
            append_log(job, "\n方向预筛完成，请在网页中选择要保留的方向。\n")
        else:
            job["status"] = "failed"
    except Exception as exc:
        job["status"] = "failed"
        job["elapsed_seconds"] = round(time.time() - started, 3)
        append_log(job, f"\n任务异常：{exc}\n")

    (job_dir / "web_job.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_analysis_job(job_id: str, selected_directions: list[str]) -> None:
    job = jobs[job_id]
    job_dir = Path(job["output_dir"])
    state_path = job_dir / "download" / "screening_state.json"
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
        "--screening-state",
        str(state_path),
        "--selected-directions",
        ",".join(selected_directions),
        "--overwrite",
    ]
    command.extend(topic_clause_args(job))
    job["status"] = "running"
    job["selected_directions"] = selected_directions
    started = time.time()
    try:
        return_code = run_process(job, command, "流程失败")
        job["returncode"] = return_code
        job["elapsed_seconds"] = round(time.time() - started, 3)
        job["images"] = image_payload(job_id, job_dir)
        job["status"] = "completed" if return_code == 0 else "failed"
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
    topic_clauses = []
    for item in payload.get("topic_clauses", []):
        logic = str(item.get("logic") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        if logic in {"and", "or", "not"} and text:
            topic_clauses.append({"logic": logic, "text": text})
    job_id, job_dir = next_job_dir(topic)
    job_dir.mkdir(parents=True, exist_ok=True)
    jobs[job_id] = {
        "id": job_id,
        "topic": topic,
        "status": "queued",
        "max_papers": max_papers,
        "max_results": max_results,
        "topic_clauses": topic_clauses,
        "output_dir": str(job_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "log": [],
        "images": [],
    }
    thread = threading.Thread(target=run_screening_job, args=(job_id,), daemon=True)
    thread.start()
    return jsonify({"id": job_id, "topic": topic})


@app.post("/api/jobs/<job_id>/selection")
def submit_selection(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "任务不存在"}), 404
    if job.get("status") != "awaiting_selection":
        return jsonify({"error": "当前任务不在方向选择阶段"}), 400
    payload = request.get_json(force=True) or {}
    selected = [str(item).strip() for item in payload.get("selected_directions", []) if str(item).strip()]
    if not selected:
        return jsonify({"error": "请至少保留一个方向"}), 400
    thread = threading.Thread(target=run_analysis_job, args=(job_id, selected), daemon=True)
    thread.start()
    return jsonify(job)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动文献综述图生成网页")
    parser.add_argument("--host", default="127.0.0.1", help="网页监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=5000, help="网页监听端口，默认 5000")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Flask is None:
        raise RuntimeError(
            "未安装 Flask。请先运行：.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        )
    ensure_runs_root()
    try:
        from waitress import serve

        print(f"网页已启动：http://{args.host}:{args.port}")
        serve(app, host=args.host, port=args.port, threads=4)
    except ImportError:
        print(f"网页已启动：http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
