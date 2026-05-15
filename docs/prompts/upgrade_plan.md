# Literature Agent PDF 后处理流程升级计划

> 本文档说明如何从旧版 `flowchart(2).md` 对应的流程，升级到新版 `flowchart.md`。重点不是重写检索和下载模块，而是重构 **PDF 下载之后的文献研究产出链路**：方向划分前移到 10，PDF 后处理不再重复分方向，每个方向独立生成综述和 SVG，最后生成总综述 Markdown 和总 SVG。

---

## 1. 升级目标

### 1.1 解决旧流程的问题

旧流程中，PDF 下载后还会运行：

```text
11 单篇自适应结构化
→ 12 研究方向划分
→ 13 方向 Schema
→ 14 方向 records
→ 15 跨方向比较
→ 16 corpus_synthesis
→ 17 repair
→ 18 plot_ready / repair
```

主要问题是：

1. **方向划分重复**：下载前 10 已经基于标题、摘要和元数据完成初步方向划分，PDF 后又用全文再分一次方向，既慢又可能与用户保留/排除方向结果冲突。
2. **单篇结构化过重**：旧 11 同时承担相关性判断、背景、任务、输入、方法、输出、评价、结论、方向提示，输出大而慢。
3. **schema 和 corpus_synthesis 太重**：旧 13 和旧 16 增加了大量中间结构，但很多字段并不直接服务后续综述和作图。
4. **输出重复**：单篇结构化、方向 records、corpus_synthesis、plot_ready 里会重复输出相似的背景、方法和结论。
5. **repair 依赖过强**：因为大 JSON 容易不一致，所以旧流程需要全量 repair，耗时且容易改坏已有有效信息。

### 1.2 新流程目标

新版流程遵循以下原则：

```text
方向划分只做一次；全文只用于方向内深度分析；每一步只输出下一步需要的信息。
```

具体目标：

1. 从检索开始时，沿用下载前 `10-download-prescreen-improved.txt` 的方向划分结果。
2. 直接 PDF 开始时，也只用标题、摘要、期刊、年份、DOI 等元数据复用 10 做轻量方向划分，不用全文分方向。
3. PDF 全文只用于方向内富化单篇。
4. 每个方向生成：`enriched_single_papers/`、`direction_records.json`、`literature_review.md`、`single_direction_overview.svg`。
5. 所有方向完成后生成：`corpus_literature_review.md` 和 `corpus_overview.svg`。
6. 取消默认文献关系图。
7. 取消全量 corpus_synthesis 和全量 repair，改为 validator + 局部修复。

---

## 2. 新旧 prompt 对照

### 2.1 保留旧 prompt

这些 prompt 不属于 PDF 后处理重构核心，可以保留：

| 旧编号 | 文件 | 处理 |
|---|---|---|
| 01 | `01-agent-system.txt` | 保留 |
| 02 | `02-generate-query-variations.txt` | 保留 |
| 03 | `03-plan-search-strategy.txt` | 保留 |
| 04 | `04-score-relevance.txt` | 保留，可作为检索评分或调试 |
| 05 | `05-batch-score-papers.txt` | 保留 |
| 06 | `06-refine-query.txt` | 保留，可选 |
| 07 | `07-refine-search-plan.txt` | 保留，可选 |
| 09 | `09-batch-title-translation.txt` | 保留 |

### 2.2 替换旧 prompt

| 旧编号 | 旧 prompt | 新 prompt | 改法 |
|---|---|---|---|
| 10 | `10-download-prescreen.txt` | `10-download-prescreen-improved.txt` | 增加唯一分配检查、confidence、direction_role、fast_check |
| 14 | `14-direction-record.txt` | `12-direction-records.txt` | 输入改为方向内 enriched JSON，输出比较 records |
| 15 | `15-cross-direction-comparison.txt` | `15-cross-direction-review-md.txt` + `16-cross-direction-plot.txt` | 跨方向比较改成总综述 MD 和总图 JSON |
| 17 | `17-corpus-repair.txt` | `17-json-local-repair.txt` | 全量 repair 改局部 JSON repair |
| 18 | `18-plot-ready-structure.txt` | `14-single-direction-plot.txt` + `16-cross-direction-plot.txt` + `18-plot-text-repair.txt` | 分成单方向图、总图和局部文本修复 |
| 19 | `19-enriched-single-paper.txt` | `11-enriched-single-paper-by-direction.txt` | 增加 direction 上下文，去掉方向划分职责 |
| 20 | `20-literature-review.txt` | `13-single-direction-review-md.txt` + `15-cross-direction-review-md.txt` | 单方向和跨方向综述分开 |
| 21 | `21-single-direction-plot.txt` | `14-single-direction-plot.txt` | 输入改为 direction_records + review.md |

### 2.3 删除或归档旧 prompt

| 旧编号 | 文件 | 原因 |
|---|---|---|
| 11 | `11-single-paper-structure.txt` | 过重；与新 11 方向内富化单篇重复 |
| 12 | `12-direction-discovery.txt` | PDF 后不再重复分方向；方向划分前移到 10 |
| 13 | `13-direction-schema.txt` | 独立 schema 层过重；合并到新 12 的 comparison_axes |
| 16 | `16-corpus-synthesis.txt` | 四合一综合过重；改为 direction review + corpus review |

### 2.4 暂停默认调用

| 旧编号 | 文件 | 原因 |
|---|---|---|
| legacy-pdf-relation | `pdf-relation-classify.txt` | 暂时不要文献关系图 |
| legacy-llm-relation | `llm-relation-classify.txt` | 暂时不要文献关系图 |

---

## 3. 代码改造总览

建议按模块分 8 步改造。

```text
Step 1  更新 prompt 文件与 prompt 索引
Step 2  改造下载前 10 预筛输出
Step 3  新增直接 PDF 模式的元数据轻量分方向
Step 4  新增方向工作区构建函数
Step 5  改造单方向快速通道为“每个方向都可调用”
Step 6  新增跨方向总综述与总图生成
Step 7  改造 validator + 局部 repair
Step 8  更新 CLI、Web 页面和输出目录
```

---

## 4. Step 1：更新 prompt 文件与索引

### 4.1 新增 prompt 文件

建议放到：

```text
docs/prompts_new/
├─ 10-download-prescreen-improved.txt
├─ 11-enriched-single-paper-by-direction.txt
├─ 12-direction-records.txt
├─ 13-single-direction-review-md.txt
├─ 14-single-direction-plot.txt
├─ 15-cross-direction-review-md.txt
└─ 16-cross-direction-plot.txt


docs/prompts_new_optional/
├─ 17-json-local-repair.txt
└─ 18-plot-text-repair.txt
```

### 4.2 修改 prompt 加载逻辑

如果项目中已有类似 `load_prompt()` 的函数，建议增加一层 prompt registry：

```python
PROMPT_REGISTRY = {
    "download_prescreen": "docs/prompts_new/10-download-prescreen-improved.txt",
    "enriched_single_by_direction": "docs/prompts_new/11-enriched-single-paper-by-direction.txt",
    "direction_records": "docs/prompts_new/12-direction-records.txt",
    "single_direction_review": "docs/prompts_new/13-single-direction-review-md.txt",
    "single_direction_plot": "docs/prompts_new/14-single-direction-plot.txt",
    "cross_direction_review": "docs/prompts_new/15-cross-direction-review-md.txt",
    "cross_direction_plot": "docs/prompts_new/16-cross-direction-plot.txt",
    "json_local_repair": "docs/prompts_new_optional/17-json-local-repair.txt",
    "plot_text_repair": "docs/prompts_new_optional/18-plot-text-repair.txt",
}
```

---

## 5. Step 2：改造下载前 10 预筛输出

### 5.1 涉及文件

```text
literature_download/prescreen.py
analysis_pipeline/unified_literature_pipeline.py
analysis_pipeline/web_app.py
```

### 5.2 当前逻辑

旧 10 已经会生成：

```text
candidate_directions.json
assignments
relevance_score
selected_candidates.json
```

### 5.3 新增字段

在 `infer_candidate_directions()` 或相关函数中，让 LLM 输出并保存：

```json
{
  "directions": [],
  "assignments": [],
  "relevance_scores": [],
  "fast_check": {
    "all_papers_assigned_once": true,
    "empty_directions": [],
    "duplicated_paper_ids": [],
    "unassigned_paper_ids": [],
    "notes": ""
  }
}
```

每条 assignment 增加：

```json
{
  "candidate_id": "C001",
  "direction_id": "D1",
  "direction_role": "main / method / background / boundary / exclude",
  "assignment_confidence": 0.85,
  "method_summary_cn": "",
  "assignment_reason_cn": ""
}
```

### 5.4 增加程序 validator

不要依赖 LLM 的 `fast_check`。应在代码里做真实检查：

```python
def validate_prescreen_directions(candidates, directions, assignments):
    candidate_ids = {p["candidate_id"] for p in candidates}
    assigned = [a["candidate_id"] for a in assignments]

    errors = []
    if set(assigned) != candidate_ids:
        errors.append({"type": "coverage", "missing": list(candidate_ids - set(assigned))})
    duplicates = [pid for pid in set(assigned) if assigned.count(pid) > 1]
    if duplicates:
        errors.append({"type": "duplicates", "paper_ids": duplicates})

    direction_ids = {d["direction_id"] for d in directions}
    for d in directions:
        if not d.get("paper_ids"):
            errors.append({"type": "empty_direction", "direction_id": d["direction_id"]})

    for a in assignments:
        if a["direction_id"] not in direction_ids:
            errors.append({"type": "unknown_direction", "assignment": a})

    return errors
```

失败时有两种策略：

1. 简单错误直接程序修复，例如删除空方向、同步 paper_ids。
2. 复杂错误才调用 `17-json-local-repair.txt`。

---

## 6. Step 3：新增直接 PDF 模式的元数据轻量分方向

### 6.1 背景

如果用户跳过检索下载，直接给 `--pdf-dir`，就没有下载前 10 的方向结果。此时需要补做轻量方向划分，但不能用 PDF 全文。

### 6.2 新增函数

建议新增：

```python
def extract_pdf_metadata_for_prescreen(pdf_dir):
    """从 PDF 或已有 txt / DOI 信息中提取标题、摘要、年份、期刊等元数据。"""
```

输出：

```json
{
  "candidate_id": "P001",
  "title": "",
  "title_cn": "",
  "abstract": "",
  "authors": [],
  "year": "",
  "venue": "",
  "doi": "",
  "concepts": []
}
```

### 6.3 调用 10

直接复用：

```python
run_download_prescreen_improved(topic, pdf_metadata_candidates)
```

输出：

```text
analysis/pdf_metadata_direction_mapping.json
```

### 6.4 CLI 建议

新增参数：

```text
--from-pdf-only
--pdf-metadata-path 可选
--skip-direction-prescreen 可选，仅当用户明确所有 PDF 属于同一方向
```

逻辑：

```text
如果存在 download/candidate_directions.json：直接使用
否则如果 from_pdf_only：提取 PDF 元数据并调用 10
否则报错或提示缺少方向映射
```

---

## 7. Step 4：新增方向工作区构建函数

### 7.1 目标

将下载前方向划分结果、PDF 路径、TXT 路径和图表 manifest 连接起来。

### 7.2 建议新增函数

放在：

```text
analysis_pipeline/direction_workspace.py
```

或放到 `unified_literature_pipeline.py` 中。

```python
def build_direction_workspace(
    output_dir,
    direction_mapping_path,
    selected_candidates_path,
    pdf_dir,
    txt_dir,
    figures_manifest_path=None,
):
    """生成 analysis/directions/D*/assigned_papers.json。"""
```

### 7.3 输出目录

```text
analysis/directions/
├─ D1_xxx/
│  ├─ assigned_papers.json
│  ├─ enriched_single_papers/
│  ├─ direction_records.json
│  ├─ literature_review.md
│  ├─ plot_ready.json
│  └─ single_direction_overview.svg
├─ D2_xxx/
└─ ...
```

### 7.4 assigned_papers.json 格式

```json
{
  "direction_id": "D1",
  "direction_name_cn": "",
  "direction_summary_cn": "",
  "papers": [
    {
      "paper_id": "P001",
      "candidate_id": "C001",
      "title": "",
      "title_cn": "",
      "abstract": "",
      "year": "",
      "venue": "",
      "doi": "",
      "pdf_path": "",
      "txt_path": "",
      "figures_tables_manifest_path": "",
      "prescreen": {
        "direction_role": "main",
        "relevance_score": 8.5,
        "assignment_reason_cn": ""
      }
    }
  ]
}
```

---

## 8. Step 5：改造单方向快速通道为每个方向都可调用

### 8.1 涉及文件

```text
analysis_pipeline/single_direction_analysis.py
analysis_pipeline/multi_paper_structured_pipeline_v2.py
analysis_pipeline/unified_literature_pipeline.py
```

### 8.2 当前单方向快速通道

旧快速通道大致是：

```text
PDF → 19 enriched_single_papers → 20 literature_review.md → 21 single_direction_overview.svg
```

### 8.3 新快速通道

改为：

```text
assigned_papers.json
  ↓
11 enriched_single_paper_by_direction
  ↓
12 direction_records
  ↓
13 single_direction_review_md
  ↓
14 single_direction_plot
  ↓
SVG 渲染
```

### 8.4 建议新增主函数

```python
def run_direction_pipeline(
    direction_dir,
    topic,
    model_config,
    parallel_papers=3,
    overwrite=False,
):
    assigned = load_json(direction_dir / "assigned_papers.json")
    run_enriched_single_papers_by_direction(assigned, topic, parallel_papers)
    run_direction_records(direction_dir, topic)
    run_single_direction_review(direction_dir, topic)
    run_single_direction_plot(direction_dir, topic)
    render_single_direction_svg(direction_dir)
```

### 8.5 改造富化单篇函数

旧函数可能是：

```python
build_enriched_single_paper_prompt(paper_name, paper_text, topic, figures_tables_text)
```

建议改为：

```python
build_enriched_single_paper_by_direction_prompt(
    topic,
    direction,
    paper_meta,
    paper_text,
    figures_tables_text,
)
```

重点增加：

```text
direction_id
direction_name_cn
direction_summary_cn
prescreen.assignment_reason_cn
prescreen.direction_role
```

---

## 9. Step 6：新增 direction_records 生成

### 9.1 目标

`direction_records.json` 是后续综述和作图的核心中间层。它不是给读者看的成品，而是给模型和程序用的比较表。

### 9.2 建议函数

```python
def run_direction_records(direction_dir, topic):
    enriched = load_all_json(direction_dir / "enriched_single_papers")
    assigned = load_json(direction_dir / "assigned_papers.json")
    prompt = build_direction_records_prompt(topic, assigned, enriched)
    result = call_api_json(prompt, model=DEEPSEEK_MODEL)
    save_json(result, direction_dir / "direction_records.json")
```

### 9.3 不再单独生成 direction_schema

旧：

```text
13 direction_schema → 14 direction_record
```

新：

```text
12 direction_records 同时生成 comparison_axes + records
```

即：

```json
{
  "comparison_axes": [],
  "records": [],
  "within_direction_summary": {}
}
```

---

## 10. Step 7：新增单方向综述和单方向作图

### 10.1 单方向综述

函数：

```python
def run_single_direction_review(direction_dir, topic):
    records = load_json(direction_dir / "direction_records.json")
    enriched_compact = load_enriched_key_formulas_figures(direction_dir)
    prompt = build_single_direction_review_md_prompt(topic, records, enriched_compact)
    md = call_api_text(prompt, model=DEEPSEEK_MODEL)
    write_text(direction_dir / "literature_review.md", md)
```

输入应以 `direction_records.json` 为主，`enriched_single_papers` 只提供公式、图表和证据补充，避免逐篇罗列。

### 10.2 单方向作图

函数：

```python
def run_single_direction_plot(direction_dir, topic):
    records = load_json(direction_dir / "direction_records.json")
    review_md = read_text(direction_dir / "literature_review.md")
    prompt = build_single_direction_plot_prompt(topic, records, review_md)
    plot_ready = call_api_json(prompt, model=DEEPSEEK_MODEL)
    validate_plot_text(plot_ready)
    save_json(plot_ready, direction_dir / "plot_ready.json")
```

作图 prompt 只输出短句 JSON，不要重复长篇综述。

---

## 11. Step 8：新增跨方向总综述和总图

### 11.1 总综述 Markdown

函数：

```python
def run_cross_direction_review(output_dir, topic):
    direction_records = collect_files("analysis/directions/*/direction_records.json")
    direction_reviews = collect_files("analysis/directions/*/literature_review.md")
    prompt = build_cross_direction_review_md_prompt(topic, direction_records, direction_reviews)
    md = call_api_text(prompt, model=DEEPSEEK_MODEL)
    write_text(output_dir / "analysis/corpus_literature_review.md", md)
```

### 11.2 总图 JSON 和 SVG

函数：

```python
def run_cross_direction_plot(output_dir, topic):
    corpus_review = read_text(output_dir / "analysis/corpus_literature_review.md")
    direction_plot_ready = collect_files("analysis/directions/*/plot_ready.json")
    records = collect_files("analysis/directions/*/direction_records.json")
    prompt = build_cross_direction_plot_prompt(topic, corpus_review, records, direction_plot_ready)
    plot_ready = call_api_json(prompt, model=DEEPSEEK_MODEL)
    save_json(plot_ready, output_dir / "analysis/cross_direction_plot_ready.json")
    render_cross_direction_svg(plot_ready, output_dir / "review_figures/corpus_overview.svg")
```

---

## 12. Step 9：改造 validator 和 repair

### 12.1 JSON validator

建议每个 JSON prompt 后都做：

```python
try:
    data = json.loads(raw)
except JSONDecodeError as e:
    data = repair_json_locally(raw, error=str(e), schema=schema)
```

再做 schema 检查：

```python
validate_required_fields(data, required_schema)
```

失败才调用：

```text
17-json-local-repair.txt
```

### 12.2 Plot text validator

对 `plot_ready.json` 和 `cross_direction_plot_ready.json` 做：

```python
def validate_plot_text(plot_ready):
    checks = [
        check_max_items_per_box,
        check_max_chars_per_sentence,
        check_unexplained_english_abbreviations,
        check_bare_symbols,
        check_empty_required_boxes,
    ]
```

失败才调用：

```text
18-plot-text-repair.txt
```

### 12.3 不再默认调用全量 repair

删除旧逻辑：

```text
16 corpus_synthesis → 17 corpus_repair
```

改为：

```text
每个 JSON 输出 → validator → 局部 repair only if failed
```

---

## 13. Step 10：修改 unified_literature_pipeline.py

### 13.1 旧流程示意

旧流程可能类似：

```python
run_search()
run_filter()
run_prescreen()
download_pdfs()
extract_texts()
run_multi_paper_structured_pipeline_v2()
generate_review_figures()
```

### 13.2 新流程示意

建议改成：

```python
def run_unified_pipeline(args):
    if not args.from_pdf_only:
        search_results = run_search_and_filter(args)
        prescreen = run_download_prescreen_improved(search_results, args.topic)
        selected = run_user_or_cli_direction_selection(prescreen, args)
        pdfs = download_selected_pdfs(selected)
    else:
        pdfs = collect_existing_pdfs(args.pdf_dir)
        metadata = extract_pdf_metadata_for_prescreen(pdfs)
        prescreen = run_download_prescreen_improved(metadata, args.topic)
        selected = run_user_or_cli_direction_selection(prescreen, args)

    txts = extract_pdf_texts(pdfs)
    figures = extract_figures_tables_if_enabled(pdfs, args)

    direction_dirs = build_direction_workspace(
        output_dir=args.output_dir,
        direction_mapping=prescreen,
        selected_candidates=selected,
        pdfs=pdfs,
        txts=txts,
        figures=figures,
    )

    for direction_dir in direction_dirs:
        run_direction_pipeline(direction_dir, args.topic, args)

    run_cross_direction_review(args.output_dir, args.topic)
    run_cross_direction_plot(args.output_dir, args.topic)
    write_unified_run_report()
```

### 13.3 暂时不要调用

```python
run_literature_graph()
run_pdf_relation_classification()
run_old_corpus_synthesis()
run_old_direction_discovery_after_pdf()
```

---

## 14. Step 11：修改 single_direction_analysis.py

### 14.1 保留单方向直接分析能力

如果用户明确所有 PDF 属于同一方向，可以继续提供快速通道：

```text
--single-direction-only
```

此时不需要 10 分方向，直接构造一个虚拟方向：

```json
{
  "direction_id": "D1",
  "direction_name_cn": "用户指定方向",
  "direction_summary_cn": "由用户 topic 指定",
  "papers": [...]
}
```

然后调用同一个：

```python
run_direction_pipeline(direction_dir, topic)
```

### 14.2 避免维护两套逻辑

旧 `single_direction_analysis.py` 中的：

```text
19 → 20 → 21
```

应改成复用：

```text
11 → 12 → 13 → 14
```

这样多方向和单方向共用同一套方向内处理函数。

---

## 15. Step 12：修改输出目录结构

建议统一为：

```text
output/YYYYMMDD_HHMM_topic/
├─ download/
│  ├─ search_results.json
│  ├─ filtered_results.json
│  ├─ candidate_directions.json
│  ├─ scored_candidates.json
│  ├─ selected_directions.json
│  ├─ selected_candidates.json
│  └─ paper_table.json/csv
├─ pdfs/
├─ figures_tables/
├─ analysis/
│  ├─ txt_output/
│  ├─ pdf_metadata_direction_mapping.json       # 仅直接 PDF 模式需要
│  ├─ directions/
│  │  ├─ D1_xxx/
│  │  │  ├─ assigned_papers.json
│  │  │  ├─ enriched_single_papers/
│  │  │  ├─ direction_records.json
│  │  │  ├─ literature_review.md
│  │  │  ├─ plot_ready.json
│  │  │  └─ single_direction_overview.svg
│  │  └─ D2_xxx/
│  ├─ corpus_literature_review.md
│  └─ cross_direction_plot_ready.json
├─ review_figures/
│  └─ corpus_overview.svg
├─ logs/
└─ unified_run_report.json
```

### 15.1 兼容旧输出

可以保留旧目录但不再默认写入：

```text
analysis/single_paper_structures/
analysis/direction_schemas/
analysis/comparisons/
analysis/adaptive_structured_output_bundle.json
review_figures/传统多方向图
```

建议在报告中标注：

```json
{
  "legacy_outputs_skipped": [
    "single_paper_structures",
    "direction_schemas",
    "comparisons",
    "adaptive_structured_output_bundle"
  ]
}
```

---

## 16. Step 13：修改 Web 页面

### 16.1 保留现有方向方框交互

网页方向筛选仍然发生在下载前：

```text
10 candidate_directions → 用户点击保留/排除 → selected_candidates
```

### 16.2 新增运行结果展示

建议新增：

```text
每个方向：
- literature_review.md 下载按钮
- single_direction_overview.svg 预览
- direction_records.json 下载按钮

总结果：
- corpus_literature_review.md 下载按钮
- corpus_overview.svg 预览
```

### 16.3 直接 PDF 模式

如果 Web 支持直接上传 PDF，可以加入：

```text
上传 PDF → 提取元数据 → 10 轻量方向划分 → 方向方框 → 后处理
```

---

## 17. Step 14：修改运行报告

`unified_run_report.json` 建议新增：

```json
{
  "pipeline_version": "pdf_postprocess_v3",
  "entry_mode": "search_to_pdf / pdf_only / single_direction_only",
  "direction_source": "download_prescreen_10 / pdf_metadata_10 / user_single_direction",
  "directions": [
    {
      "direction_id": "D1",
      "direction_name_cn": "",
      "paper_count": 0,
      "outputs": {
        "assigned_papers": "",
        "direction_records": "",
        "literature_review_md": "",
        "single_direction_svg": ""
      }
    }
  ],
  "corpus_outputs": {
    "corpus_literature_review_md": "",
    "corpus_overview_svg": ""
  },
  "legacy_steps_skipped": [],
  "repair_events": []
}
```

---

## 18. 建议实施顺序

### 第一阶段：不动检索，只改 PDF 后处理

1. 添加新 prompt 文件。
2. 新增 `build_direction_workspace()`。
3. 改 `single_direction_analysis.py`，使其支持 `assigned_papers.json` 输入。
4. 实现新 11、12、13、14。
5. 跑一个已有 PDF 目录测试。

### 第二阶段：接入下载前 10

1. 改 `prescreen.py` 的 10 输出。
2. 加 validator。
3. 将 `candidate_directions.json` 接到 `build_direction_workspace()`。
4. 跑完整流程测试。

### 第三阶段：跨方向总综述和总图

1. 实现新 15。
2. 实现新 16。
3. 渲染 `corpus_overview.svg`。
4. 更新 Web 页面展示。

### 第四阶段：清理旧流程

1. 停用旧 11、12、13、16。
2. 停用旧 corpus_synthesis 与 corpus_repair。
3. 将旧多方向作图入口标记 legacy。
4. 更新 README 和 flowchart。

---

## 19. 最小可行改造版本

如果只想先快速跑通，最低限度只需要实现：

```text
10-download-prescreen-improved.txt
11-enriched-single-paper-by-direction.txt
12-direction-records.txt
13-single-direction-review-md.txt
14-single-direction-plot.txt
```

跨方向总综述和总图可以稍后加。

最小数据流：

```text
已有方向结果 / 直接 PDF 轻量方向结果
  ↓
assigned_papers.json
  ↓
11 富化单篇
  ↓
12 direction_records
  ↓
13 单方向综述 md
  ↓
14 单方向 plot_ready
  ↓
SVG
```

---

## 20. 最终判断

这次改造不是推倒重写，而是把旧流程中已经存在的能力重新编排：

```text
旧能力：
检索、过滤、下载、10 预筛、PDF 文本提取、图表提取、单方向快速通道、SVG 渲染

新编排：
10 方向结果作为唯一方向来源
每个方向独立调用增强版单方向快速通道
最后再做跨方向总综述和总图
```

完成后，项目的数据流会更短、更稳定、更适合论文综述产出：

```text
标题/摘要决定方向，PDF 全文支撑方向内深度分析，direction_records 支撑综述和作图，总综述只读取各方向产物。
```
