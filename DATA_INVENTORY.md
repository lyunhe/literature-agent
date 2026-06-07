# 数据清单

本项目是本地自包含课程作业演示包。网页和复现工作台默认读取包内数据，不要求首次启动时重新联网生成。

## 本地 PDF

| 批次 | 目录 | 数量 | 大小 | 用途 |
| --- | --- | ---: | ---: | --- |
| 容量市场 | `input_pdfs/capacity_market/` | 20 | 20.16 MB | 文献综述本地 PDF 批次。 |
| 频率安全 | `input_pdfs/frequency/` | 5 | 17.44 MB | 文献综述本地 PDF 批次。 |

页面启动后会扫描 `input_pdfs/`，在左侧“本地文献”区域展示批次名称、文件数量和样例文件名。

## 历史文献运行

| 运行目录 | 文件数 | 大小 | 状态 |
| --- | ---: | ---: | --- |
| `output/20260531_1824_储能_电力市场_报价/` | 1100 | 310.74 MB | 已完成，质量状态 pass。 |
| `output/20260531_1905_容量市场/` | 345 | 57.88 MB | 已完成，质量状态 pass。 |
| `output/20260531_1930_储能_电力市场/` | 787 | 203.93 MB | 已完成，质量状态 pass。 |
| `output/20260531_2008_电力系统_调度/` | 581 | 141.67 MB | 已完成，质量状态 pass。 |
| `output/20260531_2032_电力系统_大语言模型/` | 382 | 113.11 MB | 已完成，质量状态 pass。 |

典型文件：

- `unified_run_report.json`
- `three_stage_review.json`
- `quality_report.json`
- `01_discovery/`
- `02_reviews/`
- `logs/`
- `time_records/`

`01_discovery/figures_tables/` 中包含较大的 PDF 图表截取结果，用于追溯和网页展示，不按缓存删除。

## Prompt 资源

| 目录 | 数量 | 用途 |
| --- | ---: | --- |
| `prompts/01_discovery/` | 13 | 查询拓展、过滤关键词拓展、标题翻译、相关性评分、方向预筛和下载相关性判断。 |
| `prompts/02_reviews/` | 3 | 单篇论文卡、方向综述和总体综述。 |
| `prompts/repair/` | 1 | JSON 输出修复。 |
| `prompts/system/` | 4 | 系统提示词和严格 JSON 输出约束。 |

这些 prompt 是 `analysis_pipeline.core.prompts` 的运行时依赖。

## Nasri 2016 复现工作区

路径：`runs/nasri_2016_ac_uc_benders/`

| 类型 | 目录 | 数量 | 大小 | 说明 |
| --- | --- | ---: | ---: | --- |
| 目标论文 | `pdfs/` | 1 | 1.77 MB | `12_Amin_UC.pdf`。 |
| 可运行数据 | `data/` | 13 | 0.42 MB | 机组、线路、负荷、风电、概率、参数和可选假设表。 |
| 源代码 | `src/` | 20 | 0.21 MB | 数据构造、求解、Benders 迭代和图表渲染。 |
| 结果 | `results/` | 655 | 193.82 MB | DC/AC 子问题、Benders 多轮实验、图表数据和中间结果。 |
| 报告 | `reports/` | 58 | 8.95 MB | 阶段报告、展示文档、PDF 和 Overleaf 包。 |
| 追踪材料 | `artifacts/` | 16 | 4.57 MB | 模型规范、算法追踪、数据来源登记和图表清单。 |
| 对话产物 | `dialogue_outputs/` | 6 | 0.03 MB | 复现阶段二示例对话生成文件。 |
| Obsidian 材料 | `obsidian/` | 118 | 5.78 MB | 复现过程中的论文、数据和对话整理材料。 |

## 缓存与非发布数据

可以清理：

- `.pytest_cache/`
- 项目源码目录下的 `__pycache__/`
- `*.pyc`
- `literature_showcase/showcase_server.*.log`
- 根目录中临时生成的文件索引或草稿文档

本轮保留：

- `output/`
- `input_pdfs/`
- `runs/nasri_2016_ac_uc_benders/results/`
- `runs/nasri_2016_ac_uc_benders/reports/`
- `.venv/`
- `.env`

`.env` 只保留在本地，包含密钥时不得分发。
