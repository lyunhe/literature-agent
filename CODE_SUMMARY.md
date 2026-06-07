# 代码总结

## 总体架构

项目由三条链路组成：

| 链路 | 主要目录 | 说明 |
| --- | --- | --- |
| Web 展示链路 | `literature_showcase/` | Flask 后端、Jinja 单页、前端交互脚本、样式、静态资源和 smoke test。 |
| 文献综述链路 | `analysis_pipeline/`、`prompts/`、`output/` | 从研究主题、本地 PDF 或在线检索结果生成 `three_stage_review.json`、`quality_report.json` 和阶段产物。 |
| 论文复现链路 | `tools/`、`config/`、`runs/nasri_2016_ac_uc_benders/` | 单篇论文审计、模型抽取、数据模板、Nasri 2016 复现实验脚本、结果和报告。 |

展示服务启动后会扫描：

- `output/*`：已有文献综述运行。
- `input_pdfs/*`：本地 PDF 批次。
- `runs/*/target.yaml`：可展示的复现目标。

## Web 展示层

核心文件：

- `literature_showcase/app.py`
- `literature_showcase/templates/index.html`
- `literature_showcase/static/showcase.js`
- `literature_showcase/static/styles.css`
- `literature_showcase/static/assets/spark_uploaded_logo.png`

主要路由：

| 路由 | 作用 |
| --- | --- |
| `/` | 文献综述与论文复现智能体首页。 |
| `/direction/<direction_id>` | 指定研究方向页面。 |
| `/paper/<direction_id>/<paper_id>` | 单篇论文页面。 |
| `/?view=reproduction` | 论文复现工作台。 |
| `/api/showcase-data` | 返回展示用三层综述 JSON。 |
| `/api/quality-report` | 返回质量检查 JSON。 |
| `/api/reproduction` | 汇总复现目标、数据表、材料、图表和报告索引。 |
| `/api/repro-chat` | 复现阶段二对话接口；示例模式会生成本地可预览产物。 |
| `/api/jobs` | 从网页启动文献流水线后台任务。 |
| `/api/repro-jobs` | 从单篇论文页启动复现辅助工具链。 |

`start.py`、`literature_showcase/run_web.bat` 和 `literature_showcase/run_web.ps1` 默认使用 `8051` 端口。

## 文献综述流水线

入口：

```powershell
.\.venv\Scripts\python.exe analysis_pipeline\unified_literature_pipeline.py --help
```

核心模块：

| 模块 | 作用 |
| --- | --- |
| `analysis_pipeline/core/` | LLM 配置、JSON/文本处理、日志、prompt registry、运行上下文和耗时统计。 |
| `analysis_pipeline/stages/discovery/` | 查询拓展、过滤关键词拓展、在线检索、PDF 下载、本地 PDF 接入、正文抽取、图表截取和方向预筛。 |
| `analysis_pipeline/stages/reviews/` | 单篇论文卡片、方向综述和总综述生成。 |
| `analysis_pipeline/stages/showcase_export.py` | 将流水线产物转换为网页使用的 `three_stage_review.json` 和 `quality_report.json`。 |
| `analysis_pipeline/tests/` | prompt registry、搜索重试、下载约束、方向分配、主题过滤和展示导出的回归测试。 |

`prompts/` 是流水线运行时依赖，当前包含：

- `01_discovery/`：13 个检索、过滤关键词拓展、筛选、翻译和方向预筛 prompt。
- `02_reviews/`：3 个单篇/方向/总体综述 prompt。
- `repair/`：1 个 JSON 修复 prompt。
- `system/`：4 个系统提示词和严格 JSON 输出提示词。

## 论文复现工具链

核心文件：

- `tools/repro_cli.py`
- `tools/pdf_extract.py`
- `tools/audit.py`
- `tools/model_spec.py`
- `tools/repro_scaffold.py`
- `tools/traces.py`

常用命令：

```powershell
.\.venv\Scripts\python.exe -m tools.repro_cli --help
.\.venv\Scripts\python.exe -m tools.repro_cli validate-data --target runs\nasri_2016_ac_uc_benders\target.yaml
```

复现 CLI 读取 target YAML，并生成或更新 `extracted_text/`、`audits/`、`artifacts/`、`reports/`、`data/`、`src/` 和 `configs/` 等材料。

## Nasri 2016 工作区

路径：`runs/nasri_2016_ac_uc_benders/`

| 子目录 | 文件数 | 说明 |
| --- | ---: | --- |
| `pdfs/` | 1 | 目标论文 PDF。 |
| `data/` | 13 | 可运行 CSV 数据表。 |
| `src/` | 20 | 数据构造、DC UC、AC 子问题、Benders loop 和图表渲染脚本。 |
| `results/` | 655 | 求解输出、图表数据、中间结果和论文风格图表。 |
| `reports/` | 58 | 阶段报告、展示说明、LaTeX/PDF 和 Overleaf 包。 |
| `artifacts/` | 16 | 模型规范、来源追踪、算法追踪和图表清单。 |
| `dialogue_outputs/` | 6 | 复现阶段二示例对话生成的本地文件。 |

## 发布边界

- `output/`、`input_pdfs/` 和 `runs/nasri_2016_ac_uc_benders/results/` 是展示与复现证据，不作为缓存删除。
- `config/targets/bertsimas_2013.yaml`、`lee_2014.yaml`、`gourtani_2016.yaml` 是参考目标；对应 PDF 未打包，运行前需要自行补齐。
- `.env`、`.venv/`、`.pytest_cache/`、`__pycache__/`、`*.pyc` 和本地日志不属于提交内容。
