# Literature Agent 文献综述图生成项目

本项目用于从一个关键研究领域出发，自动完成文献检索、PDF 下载、正文结构化、图表/公式提取，并生成中文综述可视化图。当前示例主题可使用：

```text
储能参与电力市场报价方式
```

## 项目结构

```text
literature-agent-main/
├─ analysis_pipeline/          # 分析与可视化主流程代码
│  ├─ unified_literature_pipeline.py
│  ├─ multi_paper_structured_pipeline_v2.py
│  ├─ extract_pdf_figures_tables.py
│  ├─ extract_pdf_formula_regions_v2.py
│  ├─ ocr_formula_images_pix2tex.py
│  ├─ generate_review_figures.py
│  ├─ generate_plot_ready_structures.py
│  ├─ render_plot_ready_figures.py
│  ├─ web_app.py
│  ├─ main.py / cli.py         # 旧 Agent 命令行入口，已归入分析代码目录
│  └─ _bootstrap.py
├─ literature_download/        # 文献检索与 PDF 下载代码
│  ├─ workflow.py              # 检索、去重、下载、保存检索结果
│  ├─ arxiv_search.py
│  ├─ openalex_search.py
│  ├─ crossref_search.py
│  └─ ieee_search.py
├─ backend/                    # LLM、数据库、本地文献库、图谱等支撑模块
├─ library/
│  ├─ pdfs/                    # 下载和待分析的 PDF
│  └─ library.db               # SQLite 文献库
├─ output/                     # 所有运行输出
├─ requirements.txt
├─ requirement.txt             # 兼容入口，实际依赖维护在 requirements.txt
├─ .env.example
└─ env.example.yaml
```

根目录不再存放主要功能脚本；功能代码集中在 `analysis_pipeline/` 和 `literature_download/`。

## 输出目录约定

统一流程默认输出到：

```text
output/YYYYMMDD_HHMM_关键研究领域/
```

每次运行内部按类别组织：

```text
output/YYYYMMDD_HHMM_关键研究领域/
├─ download/
│  ├─ search_results.json      # 检索候选文献
│  ├─ selected_source_pdfs.json # 下载/本地库中的原始 PDF 路径
│  └─ selected_pdfs.json       # 本次输出目录中的 PDF 路径
├─ pdfs/                       # 本次运行实际使用的 PDF 副本
├─ logs/
│  ├─ current_step.json        # 当前正在执行的步骤
│  ├─ latest_step.json         # 最近完成的步骤
│  ├─ step_records.jsonl       # 每步状态、日志路径和耗时记录
│  ├─ step_records.csv         # 每步状态、日志路径和耗时表格
│  └─ 01_xxx.log               # 每个步骤的过程日志
├─ analysis/
│  ├─ txt_output/              # PDF 正文 TXT
│  ├─ single_paper_structures/ # 单篇正文结构化 JSON
│  ├─ directions/
│  ├─ direction_schemas/
│  ├─ direction_records/
│  ├─ comparisons/
│  └─ adaptive_structured_output_bundle.json
├─ figures_tables/             # 可选：图表截图、表格 CSV、manifest.json
├─ formulas/
│  ├─ regions/                 # 可选：带编号公式截图
│  └─ ocr/                     # 可选：公式 LaTeX/Markdown
├─ review_figures/             # 综述 SVG 图
└─ unified_run_report.json     # 运行报告
```

历史散落输出已归入 `output/legacy_root_output/`。

## 环境安装

建议使用 Python 3.11。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果习惯使用 `requirement.txt`，也可以运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirement.txt
```

如果还没有虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## API 配置

复制 `.env.example` 为 `.env`，填写 DeepSeek/OpenAI-compatible 配置：

```env
DEEPSEEK_API_KEY=你的key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_REASONING_EFFORT=high
DEEPSEEK_ENABLE_THINKING=true
```

`.env` 已加入 `.gitignore`，不要提交真实密钥。

## 一键完整流程

输入一个中文研究领域，系统会自动扩展为英文检索词，搜索 OpenAlex/arXiv，下载 PDF，并完成正文结构化和综述图生成。默认一键流程不提取图表和公式：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py "储能参与电力市场报价方式" --max-results 5 --max-papers 3 --overwrite
```

如需在一键流程中额外提取图表或公式，显式添加参数：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py "储能参与电力市场报价方式" --max-results 5 --max-papers 3 --extract-figures-tables --extract-formulas --overwrite
```

过程会输出类似信息：

```text
[检索] 研究主题：储能参与电力市场报价方式
[检索] arXiv: energy storage bidding strategies in electricity markets -> 5 条
[下载完成] ... -> library/pdfs/xxxx.pdf
[PDF归档] xxxx.pdf -> output/20260512_1946_储能参与电力市场报价方式/pdfs/xxxx.pdf
正在提取 PDF 文本：xxxx.pdf
已生成 TXT：...
[跳过] 图表提取：默认一键流程不提取，如需提取请添加 --extract-figures-tables
[跳过] 公式提取与 OCR：默认一键流程不提取，如需提取请添加 --extract-formulas
```

## 分步骤运行

只进行正文结构化：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\multi_paper_structured_pipeline_v2.py --pdf-dir library\pdfs --file 2402.19110v1.pdf --output-dir output\demo_analysis --topic "储能参与电力市场报价方式。请主要使用中文输出，保留必要英文术语。" --single-only --overwrite
```

提取图表：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\extract_pdf_figures_tables.py --pdf library\pdfs\2402.19110v1.pdf --output-dir output\demo_analysis\figures_tables
```

提取带编号公式：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\extract_pdf_formula_regions_v2.py --pdf library\pdfs\2402.19110v1.pdf --output-dir output\demo_analysis\formulas\regions --overwrite
```

生成公式 Markdown：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\ocr_formula_images_pix2tex.py --input-dir output\demo_analysis\formulas\regions --output-dir output\demo_analysis\formulas\ocr --overwrite
```

生成综述 SVG：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\generate_review_figures.py --input-dir output\demo_analysis --output-dir output\demo_analysis\review_figures
```

## Web 页面

启动网页：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\web_app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

网页支持输入关键研究领域、设置文献数，并展示生成后的 SVG 综述图。网页运行输出同样保存在 `output/YYYYMMDD_HHMM_关键研究领域/` 目录中。

## 主要文件说明

- `literature_download/workflow.py`：统一检索、扩展英文检索词、去重、下载 PDF、保存检索清单。
- `analysis_pipeline/unified_literature_pipeline.py`：统一调度入口，串联下载、结构化和综述图生成；图表/公式为可选步骤。
- `analysis_pipeline/multi_paper_structured_pipeline_v2.py`：PDF 正文抽取和 LLM 结构化主流程。
- `analysis_pipeline/extract_pdf_figures_tables.py`：从 PDF 中提取图、表、caption 和截图。
- `analysis_pipeline/extract_pdf_formula_regions_v2.py`：提取带编号公式区域截图。
- `analysis_pipeline/ocr_formula_images_pix2tex.py`：生成公式 OCR JSON 和 Markdown。
- `analysis_pipeline/generate_review_figures.py`：根据结构化结果生成综述 SVG。
- `analysis_pipeline/web_app.py`：本地交互网页。
- `backend/`：LLM 客户端、SQLite 文献库、Agent 工具、文献图谱等支撑功能。

## 注意事项

- PDF 原始下载会进入 `library/pdfs/`，每次运行会同时复制一份到对应的 `output/YYYYMMDD_HHMM_关键研究领域/pdfs/`。
- 分析结果统一存放在 `output/`。
- IEEE 检索需要单独配置 IEEE API key；默认流程主要使用 OpenAlex 和 arXiv。
- 公式 OCR 如果本机没有 `pix2tex`，会退化为文本层提示，不会阻塞主流程。
