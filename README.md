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
│  ├─ topic_filter.py          # AND/OR/NOT 主题关键词过滤器
│  ├─ prescreen.py             # 下载前方向筛选、期刊打分和排序
│  ├─ paper_table.py           # 文献汇总表格与标题批量翻译
│  ├─ arxiv_search.py
│  ├─ openalex_search.py
│  ├─ crossref_search.py
│  └─ ieee_search.py
├─ backend/                    # LLM、数据库、本地文献库、图谱等支撑模块
├─ docs/
│  └─ prompts/                 # 全部 19 个 AI Prompt 文档 + 流程图
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
│  ├─ filter_config.json       # 主题过滤配置（如有）
│  ├─ filtered_results.json    # 过滤明细（通过/排除）
│  ├─ screening_state.json     # 下载前方向筛选状态
│  ├─ candidate_directions.json # 下载前候选方向与论文列表
│  ├─ scored_candidates.json   # 相关度/期刊水平加权排序结果
│  ├─ selected_candidates.json # 最终进入 PDF 下载的候选文献
│  ├─ paper_table.json         # 文献汇总表（JSON）
│  ├─ paper_table.csv          # 文献汇总表（CSV，Excel 可打开）
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
DEEPSEEK_FLASH_MODEL=deepseek-v4-flash
DEEPSEEK_REASONING_EFFORT=high
DEEPSEEK_ENABLE_THINKING=true
```

`.env` 已加入 `.gitignore`，不要提交真实密钥。

## 主题过滤（AND/OR/NOT）

支持对检索结果进行多主题关键词过滤，确保论文**同时包含多个主题**，并排除不相关方向。

`literature_download/topic_filter.py` 是过滤器模块，不是单独处理检索结果文件的完整命令行工具。直接运行它只会执行内置自测：

```powershell
.\.venv\Scripts\python.exe -m literature_download.topic_filter
```

日常使用主题过滤时，应在统一流程入口 `analysis_pipeline/unified_literature_pipeline.py` 中添加 `--filter-and`、`--filter-or`、`--filter-not` 或 `--filter-config` 参数。过滤发生在“检索完成”和“PDF 下载”之间，输出会写入本次运行目录的 `download/filter_config.json` 和 `download/filtered_results.json`。

### 过滤逻辑

- **组内 OR**：一个 `--filter-and` 内的多个关键词，命中任意一个即满足
- **组间 AND**：多个 `--filter-and` 必须全部满足
- **可选 OR**：存在 `--filter-or` 时，论文还必须至少命中一个 OR 组
- **排除 NOT**：`--filter-not` 中的关键词命中即排除

```powershell
# 必须同时包含 "多模态/大模型" 和 "风电预测"，排除综述类
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py "风电场功率预测" `
  --filter-and "multimodal,LLM,large language model,foundation model,transformer,deep learning" `
  --filter-and "wind power,wind farm,wind speed,wind energy,wind turbine" `
  --filter-or "forecast,prediction,probabilistic forecasting" `
  --filter-not "review,survey" `
  --max-results 10 --max-papers 10
```

也可使用 JSON 配置文件：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py "风电场功率预测" `
  --filter-config filters/wind_multimodal.json --max-results 10 --max-papers 10
```

JSON 配置格式（`filters/wind_multimodal.json`）：

```json
{
  "groups": [
    {"logic": "AND", "keywords": ["multimodal", "LLM", "transformer", "foundation model"]},
    {"logic": "AND", "keywords": ["wind power", "wind farm", "wind speed", "wind turbine"]},
    {"logic": "OR", "keywords": ["forecast", "prediction", "probabilistic forecasting"]},
    {"logic": "NOT", "keywords": ["review", "survey"]}
  ]
}
```

也可以在 Python 中直接作为模块使用：

```python
from literature_download.topic_filter import TopicFilter

papers = [
    {
        "title": "Battery storage bidding in electricity markets",
        "abstract": "We study optimal bidding strategies.",
        "concepts": ["Energy storage", "Electricity market"],
    }
]

topic_filter = TopicFilter.from_cli_args(
    and_groups=[
        ["energy storage", "battery"],
        ["electricity market", "bidding"],
    ],
    not_groups=[["review", "survey"]],
)

accepted, rejected = topic_filter.filter_papers(papers)
passed, matched_keywords = topic_filter.evaluate_with_matches(papers[0])
```

### 文献汇总表格

每次运行自动在 `download/` 目录生成文献汇总表，包含下载前筛选和排序后的论文：

| 列名 | 说明 |
|------|------|
| 是否下载 | `true` / `false` |
| rank | 下载前排序名次 |
| direction_id | 下载前 AI 初步归纳的方向 ID |
| final_score | 最终排序分 |
| relevance_score | 基于标题、摘要等元数据的主题相关度分 |
| journal_level_score | 本地期刊分区表匹配到的期刊水平分 |
| journal_level | 本地期刊分区表中的等级说明 |
| 文献名 | 原始标题 |
| 文献名中文翻译 | 由 LLM 批量翻译（使用 `DEEPSEEK_FLASH_MODEL`） |
| 关键词 | 命中的过滤关键词 |
| DOI/链接 | 论文 DOI 或 arXiv 链接 |

输出文件：`paper_table.json`（程序读取）+ `paper_table.csv`（Excel 可直接打开）。

### 模型分离

- **复杂任务**（PDF 结构化、综述生成）→ `DEEPSEEK_MODEL`（默认 `deepseek-v4-pro`）
- **简单任务**（标题翻译）→ `DEEPSEEK_FLASH_MODEL`（默认 `deepseek-v4-flash`）

在 `.env` 中配置：

```env
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_FLASH_MODEL=deepseek-v4-flash
```

## 下载前方向筛选与期刊打分

默认一键流程会先检索元数据，再做下载前筛选和排序：

```text
检索元数据 -> 关键词过滤 -> AI 初步分方向 -> 相关度打分 -> 期刊分区加权排序 -> 下载 Top N PDF -> 全文结构化
```

下载前只使用标题、摘要、作者、年份、期刊/来源、DOI/链接、OpenAlex concepts 和引用量等元数据；不会把未下载的 PDF 交给模型。

打分规则：

- `relevance_score`：0-10，由 `DEEPSEEK_FLASH_MODEL` 基于标题、摘要、方向和方法整体判断主题相关度。
- `journal_level_score`：0-10，仅来自本地期刊分区 CSV。
- 匹配到期刊分：`final_score = 0.7 * relevance_score + 0.3 * journal_level_score`
- 没有期刊字段、没有 CSV、或 CSV 未匹配：`final_score = relevance_score`

默认读取项目根目录的 `journal_levels.csv`。也可以指定路径：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py "储能参与电力市场报价方式" `
  --journal-levels data\journal_levels.csv --max-results 10 --max-papers 5
```

CSV 格式：

```csv
venue,aliases,level,score
IEEE Transactions on Power Systems,IEEE TPWRS|IEEE Trans. Power Syst.,JCR_Q1/CAS_Q1,10
Applied Energy,,JCR_Q1/CAS_Q1,9
```

如果只想生成下载前方向筛选结果、不下载 PDF：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py "储能参与电力市场报价方式" `
  --screen-only --max-results 10 --max-papers 5
```

继续运行某些方向：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py "储能参与电力市场报价方式" `
  --screening-state output\某次运行\download\screening_state.json `
  --selected-directions D1,D3 --max-papers 5
```

如需恢复旧流程，跳过下载前 AI 预筛：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py "储能参与电力市场报价方式" `
  --skip-ai-prescreen --max-results 5 --max-papers 3
```

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

默认端口是 `5000`，浏览器打开 `http://127.0.0.1:5000`。如果端口被占用，可改端口：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\web_app.py --port 5001
```

浏览器打开：

```text
http://127.0.0.1:5000
```

网页支持输入关键研究领域、追加“且/或/非”主题条件、设置文献数；点击“开始文献调研”后会先展示下载前候选方向方框。追加条件会映射为统一流程的 `--filter-and`、`--filter-or`、`--filter-not`；右侧主题框也可以用英文逗号写多个同义词。点击不需要的方向会将方框置灰并加横线，再点击可恢复；确认后继续下载 PDF、全文结构化并展示生成后的 SVG 综述图。网页运行输出同样保存在 `output/YYYYMMDD_HHMM_关键研究领域/` 目录中。

## 主要文件说明

- `literature_download/workflow.py`：统一检索、扩展英文检索词、去重、下载 PDF、保存检索清单；集成主题过滤和表格生成。
- `literature_download/topic_filter.py`：AND/OR/NOT 主题关键词过滤器模块，供统一流程的 CLI 参数和 JSON 配置文件调用。
- `literature_download/prescreen.py`：下载前候选方向归纳、相关度评分、期刊分区 CSV 匹配和排序。
- `literature_download/paper_table.py`：文献汇总表格生成（JSON + CSV），标题批量翻译。
- `analysis_pipeline/unified_literature_pipeline.py`：统一调度入口，串联下载、过滤、结构化和综述图生成；图表/公式为可选步骤。
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
