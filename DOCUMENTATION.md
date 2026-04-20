# Literature Agent 功能说明文档

## 项目概述

Literature Agent 是一个基于大语言模型的学术文献智能助手，支持从多个学术数据源检索、下载和管理论文，并通过 LLM Agent 循环实现自然语言交互式的文献调研。

---

## 项目结构

```
literature-agent/
├── main.py                        # CLI 入口（argparse 子命令）
├── cli.py                         # （备用 CLI 文件）
├── env.yaml                       # API 密钥与模型配置
├── requirements.txt               # Python 依赖
├── library/
│   ├── library.db                 # SQLite 本地文献库
│   └── pdfs/                      # 下载的 PDF 文件
└── backend/
    ├── __init__.py                # Agent 主循环（research 函数）
    ├── config.py                  # 配置加载
    ├── db.py                      # SQLite 数据库操作
    ├── llm_client.py              # 统一 LLM 客户端（Claude / OpenAI）
    ├── lit_tools.py               # 工具定义与实现（Tool Specs + Implementations）
    ├── advanced_search.py         # 多轮智能检索引擎
    └── search/
        ├── arxiv_search.py        # arXiv 检索与下载
        ├── crossref_search.py     # CrossRef DOI 解析
        └── ieee_search.py         # IEEE Xplore 检索与下载
```

---

## 核心功能

### 1. LLM Agent 主循环

**文件：** [backend/__init__.py](backend/__init__.py)  
**入口函数：** `research(prompt)`

用户输入自然语言 prompt，Agent 进入工具调用循环：
1. 将用户消息追加到历史对话
2. 调用 LLM（携带完整工具列表）
3. 若 LLM 返回工具调用 → 执行对应工具函数 → 将结果反馈给 LLM → 继续循环
4. 若 LLM 返回文本 → 结束循环，输出最终答案

系统提示引导 LLM 优先使用 `advanced_search` 进行深度文献调研，并鼓励下载 PDF 和保存到本地库。

---

### 2. 多轮智能检索（Advanced Search）

**文件：** [backend/advanced_search.py](backend/advanced_search.py)  
**核心函数：** `multi_round_search(user_query, ...)`

这是项目最核心的检索能力，分两轮执行：

#### 第一轮：规划与检索
- `plan_search_strategy()`：LLM 将用户查询扩展为结构化检索计划，包含：
  - 主题摘要、子主题、核心关键词、方法关键词、应用关键词
  - 目标期刊/会议、目标作者
  - 多条不同角度的检索查询（指定数据源）
- 执行计划中的所有查询，去重合并结果

#### 第二轮：评分与精炼
- `batch_score_papers()`：LLM 批量评分候选论文（0–10分），返回相关性判定（high/medium/low）、理由和后续搜索词
- `refine_search_plan()`：基于高分论文推断遗漏方向，生成第二轮精炼查询
- 合并两轮结果，再次评分排序，返回 Top-N 论文

**领域词汇表：** 内置电力系统领域中英文术语对照（SCUC、Benders分解、并行计算等），辅助查询扩展。

---

### 3. 数据源接入

#### arXiv（[backend/search/arxiv_search.py](backend/search/arxiv_search.py)）
- 基于 `arxiv` Python 包，无需 API Key
- `search(query, max_results)`：按关键词检索，按相关性排序
- `download_pdf(arxiv_id, output_dir)`：下载指定论文 PDF
- `get_info(arxiv_id)`：获取单篇论文详细元数据

#### IEEE Xplore（[backend/search/ieee_search.py](backend/search/ieee_search.py)）
- 通过 IEEE REST API 检索，**需要机构订阅的 API Key**
- `search(query, max_results)`：关键词检索
- `download_pdf(ieee_id, output_path)`：下载论文 PDF
- `get_info(ieee_id)`：获取单篇论文详细元数据

#### CrossRef DOI 解析（[backend/search/crossref_search.py](backend/search/crossref_search.py)）
- 无需认证，通过 CrossRef 公开 API 解析 DOI
- `resolve_doi(doi)`：DOI → 论文元数据（标题、作者、摘要、年份、arXiv ID）

---

### 4. 本地文献库

**文件：** [backend/db.py](backend/db.py)  
**存储：** SQLite（`library/library.db`）+ PDF 文件（`library/pdfs/`）

数据表 `papers` 字段：
| 字段 | 说明 |
|------|------|
| `title` | 论文标题（唯一键） |
| `authors` | 作者列表 |
| `abstract` | 摘要 |
| `arxiv_id` | arXiv ID |
| `doi` | DOI |
| `ieee_id` | IEEE 文章编号 |
| `source` | 来源（arxiv/ieee/crossref） |
| `pdf_path` | 本地 PDF 路径 |
| `year` | 发表年份 |
| `date_added` | 入库时间 |

支持 FTS5 全文搜索（标题、摘要、作者），回退到 LIKE 查询。

**主要操作：**
- `add_paper(paper)`：插入或忽略重复论文
- `list_papers(limit)`：按入库时间倒序列出
- `search_papers(keyword)`：全文关键词搜索
- `get_paper(identifier, source)`：按 ID 查询单篇
- `update_pdf_path(paper_id, pdf_path)`：更新 PDF 路径

---

### 5. 统一 LLM 客户端

**文件：** [backend/llm_client.py](backend/llm_client.py)  
**核心函数：** `llm_request(messages, model, tools, ...)`

自动识别模型类型并路由：
- **Claude 模型**（`claude-*`）：使用 `anthropic` SDK，将 OpenAI 格式的消息和工具定义转换为 Claude API 格式，并将响应包装为与 OpenAI 兼容的鸭子类型对象（`ClaudeResp / ClaudeChoice / ClaudeMsg`）
- **OpenAI 模型**：使用 `openai` SDK 直接调用

---

### 6. 工具列表（Agent 可调用）

**文件：** [backend/lit_tools.py](backend/lit_tools.py)

| 工具名 | 功能 |
|--------|------|
| `advanced_search` | 多轮 LLM 规划+评分+精炼检索（推荐用于深度调研） |
| `search_arxiv` | 基础 arXiv 关键词检索 |
| `search_ieee` | IEEE Xplore 关键词检索 |
| `resolve_doi` | DOI 解析为论文元数据 |
| `download_pdf` | 下载论文 PDF（支持 arxiv/doi/ieee） |
| `get_paper_info` | 获取指定论文详细元数据 |
| `list_library` | 列出本地库中所有论文 |
| `save_to_library` | 将论文元数据保存到本地 SQLite 库 |
| `search_library` | 在本地库中按关键词搜索 |

---

## CLI 命令

通过 `python main.py <command>` 使用：

| 命令 | 说明 | 示例 |
|------|------|------|
| `search <query>` | 通过 LLM Agent 执行检索任务 | `python main.py search "SCUC parallel computing"` |
| `search --source arxiv/ieee` | 指定优先数据源 | `python main.py search "unit commitment" --source arxiv` |
| `search --max N` | 限制返回结果数 | `python main.py search "MILP" --max 10` |
| `search --download` | 检索后自动下载 PDF | `python main.py search "Benders decomposition" --download` |
| `list` | 列出本地文献库中的论文 | `python main.py list --limit 20` |
| `info <identifier>` | 查看论文详细信息（arXiv ID/DOI/IEEE ID） | `python main.py info 2301.00001` |
| `download <identifier>` | 下载论文 PDF | `python main.py download 2301.00001 --source arxiv` |
| `run <prompt>` | 完整多轮 Agent 会话 | `python main.py run "帮我找关于安全约束机组组合的最新论文"` |

---

## 配置说明

**文件：** [env.yaml](env.yaml)

```yaml
api_keys:
  openai_key: "sk-..."          # OpenAI 兼容 API 密钥（必填）
  ieee_xplore: ""               # IEEE API Key（可选，需机构订阅）
  semantic_scholar: ""          # 预留，当前未使用

openai:
  base_url: "https://..."       # 自定义 API Base URL（支持第三方代理）
  model: "gpt-5.4"              # 使用的模型名（claude-* 自动切换到 Anthropic SDK）

llm:
  temperature: 0.6
  max_tokens: 4096
  time_out: 600                 # 请求超时（秒）
  max_retries: 5
```

**环境变量（优先级高于 env.yaml）：**
- `OPENAI_API_KEY`
- `OPENAI_API_BASE_URL`

---

## 依赖

```
openai>=1.0.0
anthropic>=0.34.0
arxiv>=1.4.0
PyYAML>=6.0
requests>=2.28.0
```

Python 版本要求：3.11+（env.yaml 指定）
