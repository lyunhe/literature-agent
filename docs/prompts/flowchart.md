# Literature Agent — AI 调用流程图

```mermaid
flowchart TB
    subgraph INPUT[输入层]
        T1["研究主题<br/>(中文/英文)"]
        T2["PDF 文献<br/>(library/pdfs/)"]
        T3["过滤关键词<br/>(--filter-and/or/not)"]
        T4["API 配置<br/>(.env / env.yaml)"]
    end

    subgraph SEARCH[文献检索模块]
        direction TB
        S1["02 生成查询变体<br/>generate_query_variations()<br/>📄 docs/prompts/02"]
        S2["03 规划搜索策略<br/>plan_search_strategy()<br/>📄 docs/prompts/03"]
        S3["04 相关性评分<br/>score_relevance()<br/>📄 docs/prompts/04"]
        S4["05 批量评分<br/>batch_score_papers()<br/>📄 docs/prompts/05"]
        S5["06 优化查询<br/>refine_query_from_results()<br/>📄 docs/prompts/06"]
        S6["07 优化计划<br/>refine_search_plan()<br/>📄 docs/prompts/07"]

        S1 --> S2 --> S4 --> S6 --> S5
        S4 --> S3
    end

    subgraph AGENT[智能体模块]
        A1["01 研究馆员 Agent<br/>research()<br/>📄 docs/prompts/01"]
        A1 --> SEARCH
    end

    subgraph FILTER[主题过滤模块]
        F1["关键词 AND/OR/NOT<br/>TopicFilter<br/>(无 LLM 调用)"]
    end

    subgraph PRESCREEN[下载前方向筛选与排序]
        PS1["候选标题翻译<br/>translate_titles()<br/>模型: DEEPSEEK_FLASH_MODEL"]
        PS2["AI 初步分方向<br/>标题/摘要/期刊元数据<br/>模型: DEEPSEEK_FLASH_MODEL"]
        PS3["网页方向筛选<br/>用户点击保留/排除方向"]
        PS4["相关度评分 + 期刊 CSV 加权<br/>70%相关度 + 30%期刊水平"]
        PS1 --> PS2 --> PS3 --> PS4
    end

    subgraph DOWNLOAD[下载与翻译模块]
        D1["文献下载<br/>download_papers()<br/>(无 LLM 调用)"]
        D2["10 批量标题翻译<br/>translate_titles()<br/>📄 docs/prompts/10<br/>模型: DEEPSEEK_FLASH_MODEL"]
    end

    subgraph STRUCTURE[正文结构化模块]
        direction TB
        P1["11 单篇自适应结构化<br/>build_single_paper_prompt()<br/>📄 docs/prompts/11<br/>模型: DEEPSEEK_MODEL"]
        P2["12 研究方向划分<br/>build_direction_discovery_prompt()<br/>📄 docs/prompts/12<br/>模型: DEEPSEEK_MODEL"]
        P3["13 方向 Schema 设计<br/>build_direction_schema_prompt()<br/>📄 docs/prompts/13<br/>模型: DEEPSEEK_MODEL"]
        P4["14 方向内规整<br/>build_direction_record_prompt()<br/>📄 docs/prompts/14<br/>模型: DEEPSEEK_MODEL"]
        P5["15 跨方向比较<br/>build_cross_direction_comparison_prompt()<br/>📄 docs/prompts/15<br/>模型: DEEPSEEK_MODEL"]
        P6["16 综合结构化<br/>build_corpus_synthesis_prompt()<br/>📄 docs/prompts/16<br/>模型: DEEPSEEK_MODEL"]
        P7["17 结构化修复<br/>build_corpus_repair_prompt()<br/>📄 docs/prompts/17<br/>模型: DEEPSEEK_MODEL"]

        P1 --> P2
        P2 --> P3 --> P4
        P4 --> P5
        P1 --> P6 --> P7
    end

    subgraph PLOT[综述作图模块]
        direction TB
        V1["18 综述作图结构化<br/>build_plot_ready_prompt()<br/>📄 docs/prompts/18<br/>模型: DEEPSEEK_MODEL"]
        V2["18 作图修复<br/>build_repair_prompt()<br/>📄 docs/prompts/18<br/>模型: DEEPSEEK_MODEL"]
        V1 --> V2
    end

    subgraph GRAPH[文献关系图模块]
        direction TB
        G1["08 PDF 关系分类<br/>analyze_pdfs()<br/>📄 docs/prompts/08<br/>模型: OpenAI 响应 API"]
        G2["09 LLM 关系分类<br/>_infer_relations_with_llm()<br/>📄 docs/prompts/09<br/>模型: DEEPSEEK_MODEL"]
    end

    subgraph OUTPUT[输出层]
        O1["search_results.json<br/>检索候选文献"]
        O2["paper_table.json/csv<br/>文献汇总表"]
        O3["single_paper_structures/<br/>单篇结构化 JSON"]
        O4["directions/<br/>方向划分 JSON"]
        O5["direction_schemas/<br/>方向 Schema JSON"]
        O6["direction_records/<br/>方向规整 JSON"]
        O7["comparisons/<br/>跨方向比较 JSON"]
        O8["review_figures/<br/>综述 SVG 图"]
        O9["unified_run_report.json<br/>运行报告"]
    end

    T1 --> AGENT
    T1 --> SEARCH
    T1 --> FILTER
    T2 --> D1
    T3 --> FILTER
    T4 --> AGENT
    T4 --> SEARCH
    T4 --> PRESCREEN
    T4 --> D2
    T4 --> STRUCTURE
    T4 --> PLOT
    T4 --> GRAPH

    SEARCH --> FILTER
    FILTER --> PRESCREEN
    PRESCREEN --> D1
    D1 --> D2
    D2 --> O2

    D1 --> P1
    P1 --> O3
    P2 --> O4
    P3 --> O5
    P4 --> O6
    P5 --> O7

    P4 --> V1
    P5 --> V1
    V1 --> V2 --> O8

    D1 --> G1
    G1 --> G2
```

## 数据流总览

```
用户输入主题
  │
  ├─ [智能体 Agent] → 调度工具 → 检索/下载/保存
  │
  ├─ [高级检索] → 查询生成 → 策略规划 → 评分 → 优化 → 检索结果 JSON
  │
  ├─ [主题过滤] → 关键词 AND/OR/NOT 筛选 → 过滤后论文列表
  │
  ├─ [下载前方向筛选] → 标题/摘要/期刊元数据分方向 → 用户保留/排除方向
  │
  ├─ [下载前排序] → 相关度 70% + 期刊 CSV 水平 30% → selected_candidates.json
  │
  ├─ [PDF 下载] → library/pdfs/
  │
  ├─ [标题翻译] → paper_table.json / paper_table.csv
  │
  ├─ [PDF 正文结构化]
  │   ├─ 单篇结构化 → single_paper_structures/
  │   ├─ 方向划分   → directions/
  │   ├─ Schema     → direction_schemas/
  │   ├─ 规整       → direction_records/
  │   └─ 跨方向比较 → comparisons/
  │
  ├─ [综述作图] → 中文结构化 → 修复 → review_figures/*.svg
  │
  └─ [文献关系图] → PDF 分析 → LLM 关系分类 → 文献图谱
```

## Prompt 文件索引

| # | 文件 | 模块 | 模型 | 角色 |
|---|------|------|------|------|
| 01 | `01-agent-system.txt` | 智能体 | 默认 | system |
| 02 | `02-generate-query-variations.txt` | 高级检索 | 默认 | user |
| 03 | `03-plan-search-strategy.txt` | 高级检索 | 默认 | user |
| 04 | `04-score-relevance.txt` | 高级检索 | 默认 | user |
| 05 | `05-batch-score-papers.txt` | 高级检索 | 默认 | user |
| 06 | `06-refine-query.txt` | 高级检索 | 默认 | user |
| 07 | `07-refine-search-plan.txt` | 高级检索 | 默认 | user |
| 08 | `08-pdf-relation-classify.txt` | 关系图 | OpenAI 响应API | user |
| 09 | `09-llm-relation-classify.txt` | 关系图 | 默认 | system+user |
| 10 | `10-batch-title-translation.txt` | 翻译 | **FLASH** | system+user |
| 10A | `10A-download-prescreen.txt` | 预筛 | **FLASH** | system+user |
| 11 | `11-single-paper-structure.txt` | 结构化 | 默认 | user |
| 12 | `12-direction-discovery.txt` | 结构化 | 默认 | user |
| 13 | `13-direction-schema.txt` | 结构化 | 默认 | user |
| 14 | `14-direction-record.txt` | 结构化 | 默认 | user |
| 15 | `15-cross-direction-comparison.txt` | 结构化 | 默认 | user |
| 16 | `16-corpus-synthesis.txt` | 结构化 | 默认 | user |
| 17 | `17-corpus-repair.txt` | 结构化 | 默认 | user |
| 18 | `18-plot-ready-structure.txt` | 作图 | 默认 | user |

> **模型说明**：`默认` = DEEPSEEK_MODEL (deepseek-v4-pro)；`FLASH` = DEEPSEEK_FLASH_MODEL (deepseek-v4-flash)
