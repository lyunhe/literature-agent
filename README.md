# Literature Agent 文献综述图生成项目

本项目用于从研究主题出发，完成文献检索、下载前方向筛选、PDF 下载或本地 PDF 读取、方向内富化分析、方向综述、单方向 SVG、跨方向总综述和总 SVG。

当前版本为 `pdf_postprocess_v3`。核心规则是：

```text
方向划分只做一次：10 是唯一方向来源。
PDF 全文只用于方向内深度分析，不再用于二次分方向。
```

## 新版流程图（文本版）

### 总体调用链

```text
输入层
  研究主题 topic
  过滤关键词 --filter-and / --filter-or / --filter-not
  API 配置 .env / env.yaml
  直接输入 PDF library/pdfs/ 或 --pdf-dir

检索与规划
  01 研究馆员 Agent
  02 生成查询变体
  03 规划搜索策略
  04 单篇相关性评分（可选）
  05 批量相关性评分
  06 优化查询（可选）
  07 优化检索计划（可选）
  08 Flash AI 检索词扩展（FLASH）
  输出：download/search_results.json

规则过滤
  TopicFilter（无 LLM）
  输入：search_results.json + AND/OR/NOT
  输出：download/filtered_results.json

下载前方向筛选与排序
  09 标题批量翻译（FLASH）
  10 下载前方向归纳 + 相关度评分 + fast_check（FLASH）
  网页或 CLI 方向选择（无 LLM）
  期刊 CSV 加权排序（无 LLM）
  输出：
    download/candidate_directions.json
    download/screening_state.json
    download/scored_candidates.json
    download/selected_candidates.json

PDF 下载与归档
  download_papers()（无 LLM）
  PDF 归档到本次 output/.../pdfs/

PDF 正文与图表提取
  PDF 正文提取（无 LLM）
  可选 extract_pdf_figures_tables.py（无 LLM）
  方向工作区 build_direction_workspace()（无 LLM）
  输出：
    analysis/txt_output/*.txt
    figures_tables/manifest.json
    analysis/directions/D*/assigned_papers.json

每个方向的 PDF 后处理
  11 方向内富化单篇（FLASH，并发）
  12 方向 records（PRO）
  13 单方向综述 Markdown（PRO）
  14 单方向作图结构化（PRO）
  SVG Canvas 渲染（无 LLM）
  输出：
    analysis/directions/D*/enriched_single_papers/*.json
    analysis/directions/D*/direction_records.json
    analysis/directions/D*/literature_review.md
    analysis/directions/D*/plot_ready.json
    analysis/directions/D*/single_direction_overview.svg

跨方向总综述与总图
  15 跨方向总综述 Markdown（PRO）
  16 跨方向总 SVG 作图结构化（PRO）
  跨方向 SVG 渲染（无 LLM）
  输出：
    analysis/corpus_literature_review.md
    analysis/cross_direction_plot_ready.json
    review_figures/corpus_overview.svg

可选兜底
  17 JSON 局部修复：json.loads 或 schema validator 失败时触发
  18 作图文本局部修复：文本过长、裸英文缩写、变量未解释时触发

运行报告
  unified_run_report.json
```

### 两种入口

```text
入口 A：从检索开始
  研究主题
  -> 01-08 检索与查询优化
  -> search_results.json
  -> TopicFilter 规则过滤
  -> filtered_results.json
  -> 10 标题翻译 + 10 下载前方向筛选
  -> candidate_directions.json / scored_candidates.json
  -> 用户保留或排除方向
  -> selected_candidates.json
  -> 下载 PDF
  -> PDF 正文 TXT + 可选图表 manifest
  -> 按 10 方向结果组织 analysis/directions/D*/
  -> 11 -> 12 -> 13 -> 14 方向内处理
  -> 15 -> 16 跨方向总综述与总图

入口 B：直接从 PDF 开始
  已有 PDF
  -> 提取 PDF 元数据：标题 / 摘要 / DOI / 年份 / 期刊
  -> 复用 10 做轻量方向划分
  -> pdf_metadata_direction_mapping.json
  -> 可选：用户保留或排除方向
  -> PDF 正文 TXT + 可选图表 manifest
  -> 按 10 方向结果组织 analysis/directions/D*/
  -> 11 -> 12 -> 13 -> 14 方向内处理
  -> 15 -> 16 跨方向总综述与总图
```

### 中间文件关系

```text
download/candidate_directions.json
download/selected_candidates.json
download/selected_source_pdfs.json
analysis/txt_output/*.txt
figures_tables/manifest.json
  -> build_direction_workspace()
  -> analysis/directions/D*/assigned_papers.json

analysis/directions/D*/assigned_papers.json
analysis/txt_output/{paper}.txt
figures_tables/manifest.json
  -> 11-enriched-single-paper-by-direction.txt
  -> analysis/directions/D*/enriched_single_papers/{paper_id}.json

analysis/directions/D*/assigned_papers.json
analysis/directions/D*/enriched_single_papers/*.json
  -> 12-direction-records.txt
  -> analysis/directions/D*/direction_records.json

analysis/directions/D*/direction_records.json
analysis/directions/D*/enriched_single_papers/*.json
  -> 13-single-direction-review-md.txt
  -> analysis/directions/D*/literature_review.md

analysis/directions/D*/direction_records.json
analysis/directions/D*/literature_review.md
  -> 14-single-direction-plot.txt
  -> analysis/directions/D*/plot_ready.json
  -> analysis/directions/D*/single_direction_overview.svg

analysis/directions/*/direction_records.json
analysis/directions/*/literature_review.md
  -> 15-cross-direction-review-md.txt
  -> analysis/corpus_literature_review.md
  -> 16-cross-direction-plot.txt
  -> analysis/cross_direction_plot_ready.json
  -> review_figures/corpus_overview.svg
```

## 项目结构

```text
analysis_pipeline/
├─ unified_literature_pipeline.py   # 统一入口：检索/PDF-only -> 方向工作区 -> 方向综述 -> 总综述
├─ single_direction_analysis.py     # 单方向入口，复用同一套方向内 pipeline
├─ direction_workspace.py           # 生成 analysis/directions/D*/assigned_papers.json
├─ direction_pipeline.py            # 方向内 11-14 + 跨方向 15-16
├─ render_review_figures_v3.py      # 新版单方向/跨方向 SVG 渲染
├─ prompt_loader.py                 # 读取 docs/prompts 中的新 prompt
├─ pipeline_common.py               # PDF 文本、LLM、JSON、计时等公共工具
├─ extract_pdf_figures_tables.py    # 可选图表提取
└─ web_app.py                       # 本地网页

literature_download/
├─ workflow.py                      # 检索、过滤、10 预筛、下载
├─ prescreen.py                     # 10 方向归纳、相关度评分、期刊加权
├─ topic_filter.py                  # AND/OR/NOT 主题过滤
└─ paper_table.py                   # 文献表格与标题翻译

docs/prompts/
├─ 01-agent-system.txt
├─ 02-generate-query-variations.txt
├─ ...
├─ 09-batch-title-translation.txt
├─ 10-download-prescreen-improved.txt
├─ 11-enriched-single-paper-by-direction.txt
├─ 12-direction-records.txt
├─ 13-single-direction-review-md.txt
├─ 14-single-direction-plot.txt
├─ 15-cross-direction-review-md.txt
├─ 16-cross-direction-plot.txt
├─ 17-json-local-repair.txt
├─ 18-plot-text-repair.txt
├─ flowchart.md
└─ upgrade_plan.md
```

旧的 PDF 后二次分方向、方向 schema、`corpus_synthesis`、旧全量 repair 和旧综述图脚本已经从默认流程中移除。

## 环境安装

建议使用 Python 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

配置 `.env`：

```env
DEEPSEEK_API_KEY=你的key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_FLASH_MODEL=deepseek-v4-flash
DEEPSEEK_REASONING_EFFORT=high
DEEPSEEK_ENABLE_THINKING=true
```

## 一键流程

检索并运行完整新版后处理：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py "高空风能" `
  --filter-and "轨迹控制,trajectory control,path control,flight path control" `
  --max-results 5 --max-papers 3 --overwrite
```

只生成下载前方向筛选，供网页或 CLI 选择方向：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py "高空风能" `
  --filter-and "轨迹控制,trajectory control,path control,flight path control" `
  --max-results 5 --max-papers 3 --screen-only --overwrite
```

基于已有筛选状态继续：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py "高空风能" `
  --screening-state output\某次运行\download\screening_state.json `
  --selected-directions D1,D3 --max-papers 3 --overwrite
```

## PDF-only 模式

直接从本地 PDF 开始，仍然会用 PDF 元数据复用 10 做轻量方向划分：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py "高空风能" `
  --from-pdf-only --pdf-dir library\pdfs --max-papers 3 --overwrite
```

如果你明确所有 PDF 属于同一方向：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py "高空风能轨迹控制" `
  --from-pdf-only --pdf-dir library\pdfs --single-direction-only --max-papers 3 --overwrite
```

也可以直接运行单方向入口：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\single_direction_analysis.py `
  --pdf-dir library\pdfs `
  --topic "高空风能轨迹控制" `
  --output-dir output\awe_single_direction `
  --parallel-papers 3 --overwrite
```

## 输出目录

统一流程默认输出到：

```text
output/YYYYMMDD_HHMM_研究主题/
├─ download/
│  ├─ search_results.json
│  ├─ filtered_results.json
│  ├─ screening_state.json
│  ├─ candidate_directions.json
│  ├─ scored_candidates.json
│  ├─ selected_candidates.json
│  └─ paper_table.json / paper_table.csv
├─ pdfs/
├─ figures_tables/
├─ analysis/
│  ├─ txt_output/
│  ├─ pdf_metadata_direction_mapping.json
│  ├─ direction_workspace_manifest.json
│  ├─ directions/
│  │  └─ D*_方向名/
│  │     ├─ assigned_papers.json
│  │     ├─ enriched_single_papers/
│  │     ├─ direction_records.json
│  │     ├─ literature_review.md
│  │     ├─ plot_ready.json
│  │     └─ single_direction_overview.svg
│  ├─ corpus_literature_review.md
│  └─ cross_direction_plot_ready.json
├─ review_figures/
│  └─ corpus_overview.svg
├─ logs/
└─ unified_run_report.json
```

## 方向工作区

`analysis/directions/D*/assigned_papers.json` 是 10 方向结果和 PDF 后文件之间的桥梁。它把以下信息合并到同一个方向目录中：

- 10 的 `direction_id`、方向名、方向说明
- 候选论文元数据和预筛理由
- `relevance_score`、`direction_role`、`assignment_confidence`
- 本次运行中的 PDF 路径
- PDF 正文 TXT 路径
- 可选图表 `manifest.json` 路径

后续 11-14 prompt 只读取这个方向工作区，不再重新分方向。

## Web 页面

启动网页：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\web_app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

网页流程：

```text
输入主题和 AND/OR/NOT 条件
-> 运行 --screen-only
-> 显示 10 候选方向方框
-> 用户选择保留方向
-> 运行新版方向内 pipeline
-> 展示每方向 SVG 和总 SVG
```

## Prompt 策略

| 阶段 | Prompt | 模型 |
|---|---|---|
| 检索与查询优化 | 01-07 | PRO |
| Flash 检索词扩展 | 08 | FLASH |
| 标题翻译 | 09 | FLASH |
| 下载前方向筛选与相关度评分 | 10 | FLASH |
| 方向内富化单篇 | 11 | FLASH |
| 方向 records | 12 | PRO |
| 单方向综述 | 13 | PRO |
| 单方向作图 JSON | 14 | PRO |
| 跨方向总综述 | 15 | PRO |
| 跨方向总图 JSON | 16 | PRO |
| JSON 局部修复 | 17 | FLASH/PRO |
| 作图文本局部修复 | 18 | FLASH/PRO |

## 注意事项

- `.env` 不要提交真实密钥。
- 默认一键流程不提取图表；需要图表时添加 `--extract-figures-tables`。
- `--skip-ai-prescreen` 已不适用于新版流程，因为 10 是唯一方向来源。
- Markdown 综述中的公式使用 `$...$` 和 `$$...$$`。
