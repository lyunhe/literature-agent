# 前端网页说明

本文档描述 `literature_showcase` 的页面定位、数据接口、运行方式和维护规则。当前页面不是静态落地页，而是一个本地研究工作台：左侧配置文献流水线和运行记录，右侧展示三层文献综述或论文复现工作台。

## 产品目标

页面服务两类任务：

1. 文献综述：围绕研究主题，从本地 PDF 或在线检索结果生成三层结构化综述。
2. 论文复现：围绕单篇论文，展示数据来源、模型结构、脚本、结果、报告和多轮对话辅助材料。

页面名称统一为“文献综述与论文复现智能体”，品牌区显示 `SPARK` 与上传的火焰 logo。推荐演示入口：

```text
http://127.0.0.1:8051/?view=reproduction#repro-nasri_2016_ac_uc_benders
```

## 入口文件

```text
literature_showcase/
├─ app.py
├─ templates/index.html
├─ static/showcase.js
├─ static/styles.css
├─ static/assets/spark_uploaded_logo.png
├─ data/sample_three_stage_review.json
├─ run_web.ps1
├─ run_web.bat
└─ test_showcase.ps1
```

启动方式：

```powershell
.\.venv\Scripts\python.exe start.py
```

`start.py`、`literature_showcase/run_web.bat` 和 `literature_showcase/run_web.ps1` 默认端口均为 `8051`，主要用于页面开发、课程演示和 smoke test。

## 页面与接口

核心页面路由：

| 路由 | 功能 |
| --- | --- |
| `/` | 文献综述总览。 |
| `/direction/<direction_id>` | 方向分析页。 |
| `/paper/<direction_id>/<paper_id>` | 单篇论文页。 |
| `/?view=reproduction` | 论文复现工作台。 |
| `/?view=reproduction#repro-<target_id>` | 指定复现目标详情。 |

核心 API：

| 接口 | 功能 |
| --- | --- |
| `/api/showcase-data` | 读取或导出当前运行的 `three_stage_review.json`。 |
| `/api/quality-report` | 返回 `quality_report.json`。 |
| `/api/runs/<run_id>` | 返回指定历史运行详情。 |
| `/api/jobs` | 启动文献流水线后台任务。 |
| `/api/reproduction` | 汇总 `runs/*/target.yaml` 复现目标。 |
| `/api/repro-chat` | 复现阶段二对话接口。 |
| `/api/repro-jobs` | 从单篇论文页启动复现辅助工具链。 |

## 布局

页面采用固定左侧工作台 + 右侧内容区：

```text
Left Control Panel
  - 工作台模式切换
  - 主题与查询条件
  - 本地 PDF / 在线检索配置
  - 运行进度
  - 已有运行
  - 本地文献
  - 复现目标

Right Workspace
  - 文献综述三层视图
  - 或论文复现工作台
```

左侧强调稳定、紧凑、可扫描；右侧根据当前视图渲染综述内容或复现材料。

## 文献综述视图

三层结构：

1. Corpus Overview：整体主题、方向、共同问题、差异、gap 和证据链。
2. Direction Analysis：方向对比、横向知识卡片、方法分布、方向内论文筛选。
3. Paper Detail：元数据、方法概括、方法流程、公式、结论、局限和复现入口。

数据来源：

- `output/<run_id>/three_stage_review.json`
- `output/<run_id>/quality_report.json`
- `output/<run_id>/unified_run_report.json`
- `output/<run_id>/01_discovery/`
- `output/<run_id>/02_reviews/`

公式要求：

- 行内公式使用 `\( ... \)`。
- 块级公式使用 `\[ ... \]`。
- 前端插入动态内容后触发 MathJax typeset。
- 不直接向用户暴露未渲染的 `$...$` 或 `\begin{}` 文本。

## 论文复现视图

论文复现工作台默认扫描 `runs/*/target.yaml`。当前主示例为：

```text
runs/nasri_2016_ac_uc_benders/target.yaml
```

页面分两阶段：

1. 阶段一：展示论文拆解、数据准备、模型与环境摘要。
2. 阶段二：展示多轮对话工作台、工作文件缓存、提示词、复现脚本、结果图表和差距说明。

Nasri 示例对话产物位于：

```text
runs/nasri_2016_ac_uc_benders/dialogue_outputs/
```

重要展示材料：

- `data/`：CSV 数据层。
- `src/`：复现脚本。
- `results/paper_style_results/`：论文风格结果表和对比图数据。
- `reports/stage_20_feature_showcase_and_demo_cn.md`：中文展示说明。
- `artifacts/model_spec.md`：模型规范。
- `audits/reproducibility_audit.md`：复现审计。

## 数据契约

`three_stage_review.json` 顶层应包含：

- `corpus`
- `directions`
- `directions[].papers`

单篇论文对象建议包含：

- `id`
- `title`
- `title_cn`
- `authors`
- `year`
- `doi`
- `abstract`
- `research_problem`
- `method`
- `method_detail`
- `method_flow`
- `formula_items`
- `scenario`
- `conclusion`
- `limitation`
- `innovation`
- `evidence`

`quality_report.json` 用于检查：

- 论文数量是否达标。
- 方向数量是否合理。
- 公式字段是否可渲染。
- 疑似无关论文是否需要人工复核。
- 是否存在引用不存在方向或论文的链接。

疑似无关论文只作为人工复核提示，不影响总状态。

## 视觉规则

- 左侧是工作台，不做装饰性大卡片。
- 右侧以信息层级为主，减少纯装饰元素。
- 方向卡片可点击，但链接语义要清晰。
- 第二层知识卡片横向排列，移动端可换行。
- 第三层内容同时支持快速扫读和展开细读。
- 文件预览弹窗支持 Markdown、JSON、CSV、Python 和纯文本。
- 所有动态内容必须避免文本溢出和元素重叠。

## 测试清单

每次改动前端、后端接口或展示 JSON 后运行：

```powershell
.\.venv\Scripts\python.exe -m pytest analysis_pipeline\tests -q
.\.venv\Scripts\python.exe -m py_compile literature_showcase\app.py analysis_pipeline\unified_literature_pipeline.py analysis_pipeline\stages\showcase_export.py tools\repro_cli.py
powershell -ExecutionPolicy Bypass -File literature_showcase\test_showcase.ps1
```

手工检查：

- 首页可访问。
- 方向页可访问。
- 单篇论文页可访问。
- 复现工作台可访问。
- `/api/showcase-data` 返回有效 JSON。
- `/api/quality-report` 返回有效 JSON。
- `/api/reproduction` 返回 Nasri target。
- 文件预览能打开 Markdown/JSON/CSV。

## 已知限制

- 页面可展示已有 `output/` 数据；启动新的在线流水线需要 `.env` 中有有效 LLM/API 配置和网络。
- 复现辅助工具可以生成 target、数据骨架和代码草稿；完整 AC/NLP 求解依赖本机求解器环境。
- 历史结果中保留的绝对路径是生成时记录，不应作为当前包的运行入口。
