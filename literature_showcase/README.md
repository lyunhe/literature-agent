# 文献综述与论文复现智能体网页工作台

`literature_showcase` 是项目的 Flask 展示应用，负责读取文献流水线生成的 `three_stage_review.json`、`quality_report.json`，并把 Nasri 2016 论文复现工作区整理成可浏览的本地网页。

默认页面名称为“文献综述与论文复现智能体”，左上角使用 `static/assets/spark_uploaded_logo.png` 中的 SPARK 火焰标识，默认端口为 `8051`。

## 功能

- 左侧工作台：配置研究主题、AND/OR/NOT 条件、本地 PDF/在线检索模式、论文数量、候选数量、运行阶段和历史运行选择。
- 文献综述视图：展示总体主题层、方向层和单篇论文层，支持方法分布、证据链、公式、图表和质量报告。
- 论文复现视图：扫描 `runs/*/target.yaml`，展示目标论文、数据来源、模型参数、复现脚本、结果图表、报告和差距说明。
- 复现阶段二对话：保留“运行示例对话”入口，用于生成可预览的 CSV、Python 草稿和说明文件；这些产物不会覆盖正式复现代码。
- 质量接口：`/api/quality-report` 返回论文数量、方向数量、公式字段、引用完整性和疑似无关论文等检查结果；疑似无关论文现在仅作为人工复核提示，不影响总状态。

## 启动

推荐在项目根目录运行：

```powershell
literature_showcase\run_web.bat
```

也可以直接使用 PowerShell 脚本：

```powershell
powershell -ExecutionPolicy Bypass -File literature_showcase\run_web.ps1
```

或使用根目录入口：

```powershell
.\.venv\Scripts\python.exe start.py
```

访问地址：

```text
http://127.0.0.1:8051
http://127.0.0.1:8051/?view=reproduction#repro-nasri_2016_ac_uc_benders
```

## 测试

```powershell
powershell -ExecutionPolicy Bypass -File literature_showcase\test_showcase.ps1
```

该脚本会检查服务可访问性、首页/方向页/单篇论文页状态、主要 API 返回、静态资源版本和质量报告。
