# 文献综述与论文复现智能体

这是一个以 SPARK 为品牌标识的自包含课程作业演示包，目标是把“文献综述流水线、Flask 展示页、Nasri 2016 论文复现工作区”放在同一个本地项目中，演示从文献获取、PDF 处理、结构化综述，到单篇论文复现材料组织的完整链路。

当前网页名称为“文献综述与论文复现智能体”，左上角使用 SPARK 火焰标识。默认本地端口统一为 `8051`。

## 项目定位

本项目适合：

- 本地运行三层递进式文献综述页面。
- 从本地 PDF 或在线检索结果生成结构化综述。
- 展示历史文献检索、下载、筛选、质量检查和综述结果。
- 展示 Nasri 2016 AC unit commitment under uncertainty 论文复现的脚本、数据、报告和结果。
- 作为继续扩展文献智能体与论文复现工具链的代码基础。

本项目不承诺：

- 内置在线检索、LLM 调用或论文下载所需的外部账号额度。
- 所有参考 target 都能直接复现；部分参考 target 的 PDF 未随包分发。
- Nasri 2016 复现实验与原论文数值完全一致，尤其是原文未公开的风电时序、场景细节和求解器设置。

## 目录结构

```text
Spark/
├─ analysis_pipeline/
│  ├─ unified_literature_pipeline.py      # 文献流水线命令行主入口
│  ├─ core/                               # LLM 配置、prompt registry、IO、日志、运行上下文
│  └─ stages/
│     ├─ discovery/                       # 检索、关键词扩展、筛选、下载、本地 PDF、图表截取
│     ├─ reviews/                         # 单篇卡片、方向综述、总体综述
│     └─ showcase_export.py               # 导出 three_stage_review.json 与 quality_report.json
├─ literature_showcase/
│  ├─ app.py                              # Flask 后端与 API
│  ├─ templates/index.html                # 单页工作台
│  ├─ static/showcase.js                  # 前端交互与渲染
│  ├─ static/styles.css                   # 页面样式
│  ├─ static/assets/spark_uploaded_logo.png
│  ├─ data/sample_three_stage_review.json # 无输出时的示例数据
│  ├─ run_web.bat                         # Windows 一键启动脚本
│  ├─ run_web.ps1                         # PowerShell 启动脚本
│  └─ test_showcase.ps1                   # 网页 smoke test
├─ prompts/
│  ├─ 01_discovery/                       # 检索、过滤关键词扩展、方向预筛、标题翻译等 prompt
│  ├─ 02_reviews/                         # 单篇、方向、总体综述 prompt
│  ├─ repair/                             # JSON 修复 prompt
│  └─ system/                             # 通用系统 prompt 与严格 JSON 约束
├─ tools/                                 # 论文复现审计、模型抽取、数据模板、CLI
├─ config/                                # schema、复现 prompt、target 配置
├─ input_pdfs/                            # 本地 PDF 输入批次
├─ output/                                # 文献流水线历史运行结果
├─ runs/nasri_2016_ac_uc_benders/         # Nasri 2016 复现工作区
├─ .env.example
├─ requirements.txt
├─ start.py
└─ start.sh
```

## 快速启动

推荐在 Windows PowerShell 中使用项目根目录下的虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe start.py
```

启动后访问：

```text
http://127.0.0.1:8051/?view=reproduction#repro-nasri_2016_ac_uc_benders
```

也可以使用展示页自带的 Windows 启动脚本：

```powershell
literature_showcase\run_web.bat
```

或直接使用 PowerShell 脚本：

```powershell
powershell -ExecutionPolicy Bypass -File literature_showcase\run_web.ps1
```

在 bash 环境中可使用：

```bash
bash start.sh
```

端口可通过环境变量覆盖：

```powershell
$env:LITERATURE_SHOWCASE_PORT = "8060"
.\.venv\Scripts\python.exe start.py
```

## 环境变量

复制 `.env.example` 为 `.env` 后填写本地密钥。`.env` 已加入 `.gitignore`，不要提交真实密钥。

优先使用：

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`
- `LLM_FLASH_MODEL`
- `LLM_REASONING_EFFORT`
- `LLM_ENABLE_THINKING`

兼容变量：

- `OPENAI_*`
- `DEEPSEEK_*`

读取优先级以代码为准：`LLM_*` 优先，其次兼容变量。没有 API key 时，页面仍能展示已有数据；启动新的在线 LLM 流水线、标题翻译、方向预筛、综述生成或真实大模型对话会失败。

## 工作流

```text
01_discovery
  在线检索或读取本地 PDF
  -> LLM 扩展英文检索 query
  -> 内置词表 + LLM 扩展 AND/OR/NOT 过滤关键词
  -> 主题过滤与复合主题概念过滤
  -> 可下载候选验证
  -> LLM/规则相关性排序
  -> PDF 下载或本地 PDF 归档
  -> PDF 正文提取
  -> LLM/PDF 正文最终方向分类
  -> 可选图表截取
  -> 方向工作区构建

02_reviews
  单篇论文结构化卡片
  -> 单方向文献综述
  -> 跨方向总体综述

showcase_export
  导出 three_stage_review.json
  -> 生成 quality_report.json
  -> Flask 页面读取 output/<run_id>/ 展示三层综述
```

### 阶段说明

- **Discovery**：负责检索、去重、关键词扩展、主题过滤、PDF 下载/复制、正文提取、最终方向分类、图表截取和方向工作区构建。在线模式会先在可下载候选池中进行相关性排序，再选择最终下载论文。
- **Reviews**：负责生成单篇论文卡片、方向综述和总体综述。单篇卡片会抽取研究问题、方法、公式、变量、结论、局限和网页展示字段。
- **Showcase Export**：负责把流水线产物整理为网页统一数据接口，并生成质量报告。
- **网页工作台**：读取 `three_stage_review.json`，按照“总主题层 -> 方向层 -> 单篇论文层”展示结果，同时提供论文复现入口。

## 文献流水线命令行

主入口：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py --help
```

### 在线检索并生成网页展示数据

示例：围绕“储能参与电力市场”检索并分析 20 篇文献，同时提取图表用于第三层展示。

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py `
  --input-mode online `
  --topic "储能参与电力市场" `
  --sources openalex,arxiv `
  --max-results 40 `
  --max-papers 20 `
  --candidate-multiplier 3 `
  --max-candidates 60 `
  --require-pdf true `
  --compare-sources `
  --run-parts discovery,reviews `
  --extract-figures-tables `
  --parallel-papers 5 `
  --overwrite
```

### 使用 AND/OR/NOT 主题过滤

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py `
  --input-mode online `
  --topic "电力系统 调度" `
  --filter-and "调度" `
  --sources openalex,arxiv `
  --year-from 2022 `
  --max-results 160 `
  --max-papers 40 `
  --max-candidates 40 `
  --run-parts discovery,reviews `
  --overwrite
```

过滤逻辑说明：

- `--filter-and`：每个 AND 组至少命中一个关键词，所有 AND 组都必须满足。
- `--filter-or`：存在 OR 组时，至少一个 OR 组命中。
- `--filter-not`：NOT 组命中则排除。
- 中文关键词会先通过内置双语词表扩展，再调用 LLM 根据研究主题生成领域英文同义词。
- LLM 扩展结果写入 `01_discovery/filter_keyword_expansion.json`，最终过滤词表写入 `01_discovery/filter_config.json`。

例如“调度”会扩展到 `dispatch`、`economic dispatch`、`unit commitment`、`scheduling` 等，并可继续由 LLM 补充 `SCED`、`security constrained dispatch` 这类领域词。

### 本地 PDF 分析

示例：分析 `input_pdfs/frequency` 中的全部 PDF。

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py `
  --input-mode local `
  --topic "高比例新能源电力系统频率稳定与频率响应控制" `
  --pdf-dir input_pdfs\frequency `
  --all-papers `
  --run-parts discovery,reviews `
  --extract-figures-tables `
  --parallel-papers 5 `
  --overwrite
```

### 只运行发现阶段

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py `
  --input-mode online `
  --topic "电力市场中的容量市场" `
  --max-results 20 `
  --max-papers 10 `
  --run-parts discovery `
  --screen-only `
  --overwrite
```

### 复用已有 discovery 继续生成 reviews

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py `
  --topic "电力市场中的容量市场" `
  --run-parts reviews `
  --discovery-dir output\<run_id>\01_discovery `
  --parallel-papers 5 `
  --overwrite
```

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `topic` | 内置示例主题 | 位置参数形式的研究主题。 |
| `--topic` | `None` | 显式指定研究主题，会覆盖位置参数。 |
| `--input-mode` | `None` | `online` 在线检索下载；`local` 从 `--pdf-dir` 读取本地 PDF。 |
| `--sources` | `openalex,arxiv` | 在线检索源，支持 `openalex`、`arxiv`、`ieee`。 |
| `--max-results` | `5` | 每个查询词在每个来源返回的候选上限。 |
| `--max-papers` | `1` | 最终进入分析的 PDF 数量上限。 |
| `--year-from` / `--year-to` | `None` | 在线检索的年份范围。 |
| `--candidate-multiplier` | `2` | 在线模式下，候选池目标数为 `max_papers * candidate_multiplier`。 |
| `--max-candidates` | `None` | 在线模式下验证可下载 PDF 的最大候选数。 |
| `--require-pdf` | `true` | 在线模式是否只保留已验证可直接下载的 PDF。 |
| `--compare-sources` | 关闭 | 输出不同检索源的候选、可下载和入选统计。 |
| `--all-papers` | 关闭 | 处理全部可用或已下载 PDF，常用于本地 PDF 目录。 |
| `--pdf-dir` | `input_pdfs` | 本地 PDF 目录。 |
| `--from-pdf-only` / `--skip-search` | 关闭 | 跳过在线检索，从本地 PDF 开始。 |
| `--pdf-metadata-path` | `None` | 本地 PDF 模式下可选的 metadata JSON。 |
| `--single-direction-only` / `--single-only` | 关闭 | 将所有 PDF 视为同一方向。 |
| `--run-parts` | `discovery,reviews` | 要运行的阶段，可选 `discovery`、`reviews`，逗号分隔。 |
| `--output-dir` | 自动生成 | 手动指定输出目录。 |
| `--discovery-dir` | `None` | 复用已有 `01_discovery` 目录继续后续阶段。 |
| `--reviews-dir` | `None` | 复用已有 `02_reviews` 目录。 |
| `--extract-figures-tables` | 关闭 | 在 discovery 阶段截取 PDF 图表并生成 manifest。 |
| `--screen-only` | 关闭 | 只运行到方向预筛，便于人工检查方向划分。 |
| `--screening-state` | `None` | 复用已有 `screening_state.json`。 |
| `--selected-directions` | 空字符串 | 只保留指定方向，例如 `D1,D3`。 |
| `--journal-levels` | `journal_levels.csv` | 期刊等级 CSV，用于辅助排序。 |
| `--skip-ai-prescreen` | 关闭 | 已禁用；新流程要求方向预筛。 |
| `--parallel-papers` | `7` | 单方向内并发生成单篇论文卡片的线程数。 |
| `--filter-and` | `None` | AND 关键词组，可重复使用，组内用逗号分隔。 |
| `--filter-or` | `None` | OR 关键词组，可重复使用，组内用逗号分隔。 |
| `--filter-not` | `None` | NOT 关键词组，可重复使用，组内用逗号分隔。 |
| `--filter-config` | `None` | JSON 格式主题过滤配置。 |
| `--overwrite` | 关闭 | 覆盖已有中间输出。 |

## 输出目录

每次运行会在 `output/` 下生成独立目录，命名格式通常为：

```text
YYYYMMDD_HHMM_<topic>/
```

典型结构：

```text
output/<run_id>/
├─ 01_discovery/
│  ├─ input_mode.json
│  ├─ raw_candidates.json
│  ├─ filter_keyword_expansion.json
│  ├─ filter_config.json
│  ├─ filtered_results.json
│  ├─ downloadable_candidates.json
│  ├─ skipped_candidates.json
│  ├─ selected_candidates.json
│  ├─ selected_pdfs.json
│  ├─ scored_candidates.json
│  ├─ paper_table.csv
│  ├─ paper_table.json
│  ├─ pdfs/
│  ├─ txt_output/
│  ├─ figures_tables/
│  └─ directions/D*/assigned_papers.json
├─ 02_reviews/
│  ├─ directions/D*/
│  │  ├─ assigned_papers.json
│  │  ├─ paper_cards/*.json
│  │  ├─ direction_review.md
│  │  └─ direction_review_summary.json
│  ├─ corpus_literature_review.md
│  ├─ corpus_review_summary.json
│  └─ reviews_manifest.json
├─ time_records/
│  └─ timing_summary.csv
├─ logs/
├─ three_stage_review.json
├─ quality_report.json
└─ unified_run_report.json
```

关键文件：

- `three_stage_review.json`：网页主数据接口，包含 corpus、directions、papers、methods_distribution、evidence、visual_assets 等字段。
- `quality_report.json`：质量检查报告，检查论文数量、方向数量、D1/D2、公式字段、方法步骤公式引用等。疑似无关论文只作为人工复核提示，不影响整体质量状态。
- `filter_keyword_expansion.json`：LLM 对过滤关键词的扩展记录，便于追踪为什么某些英文术语参与过滤。
- `01_discovery/paper_table.csv`：候选与入选论文表，适合人工检查标题、来源、下载状态和相关性。
- `01_discovery/figures_tables/`：PDF 图表截取结果，第三层页面会按 caption 自动挑选关键图表辅助讲解。
- `time_records/timing_summary.csv`：各阶段耗时汇总。
- `unified_run_report.json`：整次运行的状态、阶段日志、失败原因和输出路径。

## 网页工作台

启动：

```powershell
literature_showcase\run_web.bat
```

访问：

```text
http://127.0.0.1:8051
```

推荐演示入口：

```text
http://127.0.0.1:8051/?view=reproduction#repro-nasri_2016_ac_uc_benders
```

左侧工作台保留交互入口：主题输入、AND/OR/NOT 条件、本地 PDF/在线检索模式、论文数量、可下载候选上限、运行阶段、历史运行选择、本地文献批次和复现目标。右侧展示：

1. 总主题层：总体概览、方向卡片、共同问题、差异问题、gap 和证据链。
2. 方向层：方向总结、方向对比表、知识卡片、论文筛选、方向内文献对比和方法分布。
3. 单篇论文层：元数据、研究问题、方法、结论、局限、相似文献、方法流程、公式变量解释和关键图表。
4. 论文复现工作台：目标论文审计、数据来源、模型参数、对话产物、代码产物、复现图表和原文差距。

更多前端说明见 [frontend_web.md](frontend_web.md)。

## 本地网页接口

| 接口 | 说明 |
| --- | --- |
| `/` | 展示最新或指定 run 的三层综述页面。 |
| `/direction/<direction_id>` | 打开第二层方向页面。 |
| `/paper/<direction_id>/<paper_id>` | 打开第三层单篇论文页面。 |
| `/?view=reproduction` | 打开论文复现工作台。 |
| `/?view=reproduction#repro-<target_id>` | 打开指定复现目标详情。 |
| `/api/showcase-data` | 返回当前 run 的 `three_stage_review.json`。 |
| `/api/quality-report` | 返回当前 run 的 `quality_report.json`。 |
| `/api/runs` | 返回可选择的历史运行。 |
| `/api/jobs` | 从网页启动文献流水线后台任务。 |
| `/api/reproduction` | 汇总 `runs/*/target.yaml` 复现目标。 |
| `/api/repro-chat` | 复现阶段二对话接口。 |
| `/api/repro-jobs` | 从单篇论文页启动复现辅助工具链。 |
| `/runs/<run_id>/files/<path>` | 读取 run 目录内的图表、PDF 等静态资源。 |
| `/repo-files/<path>` | 读取包内可展示文件。 |

指定某次运行：

```text
http://127.0.0.1:8051/?run=<run_id>
```

## Prompt 管理

所有文献流水线 LLM prompt 统一放在 `prompts/` 下，并通过 `analysis_pipeline/core/prompts.py` 的注册表加载。新增或调整 prompt 时，应先修改 `prompts/` 中的文本文件，再在注册表中补充编号、别名和必填变量检查。

当前分组：

- `prompts/01_discovery/`：检索策略、query 扩展、过滤关键词扩展、相关性评分、方向预筛、标题翻译、下载前评分。
- `prompts/02_reviews/`：单篇论文卡片、方向综述、总体综述。
- `prompts/repair/`：JSON 局部修复。
- `prompts/system/`：通用系统提示词、严格 JSON 输出约束和翻译约束。

## Nasri 2016 复现工作区

默认复现目标：

```text
runs/nasri_2016_ac_uc_benders/target.yaml
```

主要内容：

- `pdfs/12_Amin_UC.pdf`：目标论文 PDF。
- `data/`：机组、线路、负荷、风电、场景概率、参数和可选假设表。
- `src/`：转录、校验、调度筛选、Benders/AC 评估等脚本。
- `results/`：历史求解结果、图表和对比数据。
- `reports/`：阶段报告、展示材料和 Overleaf 包。
- `obsidian/`：复现过程中的数据、论文和对话材料。

常用 CLI：

```powershell
.\.venv\Scripts\python.exe -m tools.repro_cli --help
.\.venv\Scripts\python.exe -m tools.repro_cli validate-data --target runs\nasri_2016_ac_uc_benders\target.yaml
```

参考 target：

- `config/targets/bertsimas_2013.yaml`
- `config/targets/lee_2014.yaml`
- `config/targets/gourtani_2016.yaml`

这些 target 用于方法参考或 sanity check；对应 PDF 未随当前包分发，运行前需自行补齐 PDF 并更新 `source_pdf`。

## 测试与验证

基础测试：

```powershell
.\.venv\Scripts\python.exe -m pytest analysis_pipeline\tests -q
```

核心脚本编译检查：

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  literature_showcase\app.py `
  analysis_pipeline\unified_literature_pipeline.py `
  analysis_pipeline\stages\showcase_export.py `
  tools\repro_cli.py
```

网页 smoke test：

```powershell
powershell -ExecutionPolicy Bypass -File literature_showcase\test_showcase.ps1
```

`test_showcase.ps1` 会检查：

- Flask 服务是否可访问。
- 最新 `three_stage_review.json` 和 `quality_report.json` 是否为合法 JSON。
- 首页、方向页、单篇页是否返回 200。
- `/api/showcase-data` 和 `/api/quality-report` 是否正常。
- 静态资源版本是否可访问。

## 数据与缓存说明

`output/`、`input_pdfs/`、`runs/nasri_2016_ac_uc_benders/results/` 都是演示与复现证据的一部分，不按缓存删除。

可以安全清理的运行缓存包括：

- `.pytest_cache/`
- 项目源码目录下的 `__pycache__/`
- `*.pyc`
- `literature_showcase/showcase_server.*.log`

`.venv/` 是本地虚拟环境，不随发布包提交，也不作为项目源码缓存清理。

## 常见问题

### 为什么某个中文主题在线检索不到文献？

通常不是“检索不到”，而是后续过滤过严或 PDF 直链不足。建议检查 `output/<run_id>/logs/`、`raw_candidates.json`、`filtered_results.json`、`downloadable_candidates.json`。当前流程已支持：

- LLM 扩展英文检索 query。
- 内置双语词表扩展中文过滤词。
- LLM 根据主题动态扩展 AND/OR/NOT 过滤关键词。

如果 `raw_candidates.json` 很多但 `filtered_results.json` 全被排除，优先检查 AND 条件是否过窄。

### 为什么候选很多但没有可处理 PDF？

在线模式默认 `--require-pdf true`，只处理已验证可直接下载的 PDF。OpenAlex/arXiv 返回的候选可能没有稳定 PDF 直链。可提高 `--max-results`、`--max-candidates`、`--candidate-multiplier`，或先把 PDF 放入 `input_pdfs/` 使用本地模式。

### 网页没有显示最新改动怎么办？

确认服务端口是 `8051`，并重启：

```powershell
literature_showcase\run_web.bat
```

静态资源带有版本号，当前页面使用 `spark-brand-3`。

### 质量报告里的“疑似无关论文”为什么是 pass？

该项现在作为人工复核提示，不影响质量状态。报告仍会保留疑似无关论文列表和数量，但状态显示为 pass。

## 已知限制

- 在线检索依赖外部网络、论文源 API、PDF 可下载性和 LLM API 配置。
- LLM 生成内容需要人工复核，尤其是公式、变量解释、方向命名和相关性判断。
- Nasri 工作区部分历史 JSON/CSV 记录保留了打包前机器上的绝对路径；这些记录用于结果追溯，不影响页面读取当前包内相对路径。
- AC/NLP 和 Benders 相关实验依赖求解器环境。当前包主要保留结果、脚本和报告，不保证商业或本机求解器开箱即用。

## 配套文档

- [CODE_SUMMARY.md](CODE_SUMMARY.md)：代码结构和模块职责。
- [DATA_INVENTORY.md](DATA_INVENTORY.md)：数据、输出和复现工作区清单。
- [frontend_web.md](frontend_web.md)：前端页面设计、接口和维护说明。
- [RELEASE_MANIFEST.md](RELEASE_MANIFEST.md)：发布包内容和边界。
