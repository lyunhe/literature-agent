# 发布包清单

版本：`spark_literature_reproduction_agent_20260531`

## 核心内容

| 路径 | 文件数 | 大小 | 说明 |
| --- | ---: | ---: | --- |
| `literature_showcase/` | 11 | 0.39 MB | Flask 展示应用、页面、样式、脚本和 SPARK logo。 |
| `analysis_pipeline/` | 50 | 0.33 MB | 文献综述流水线和测试。 |
| `prompts/` | 21 | 0.03 MB | 流水线 LLM prompt registry。 |
| `tools/` | 12 | 0.07 MB | 论文复现辅助 CLI 和工具函数。 |
| `config/` | 12 | 0.01 MB | schema、复现 prompt、target 配置。 |
| `input_pdfs/` | 25 | 37.60 MB | 展示用本地 PDF 批次。 |
| `output/` | 3195 | 827.32 MB | 历史文献流水线运行结果。 |
| `runs/nasri_2016_ac_uc_benders/` | 897 | 215.70 MB | Nasri 2016 复现工作区。 |

`.venv/`、`.env`、`.pytest_cache/`、`__pycache__/`、`*.pyc` 和本地日志不属于发布内容。

## 启动入口

| 文件 | 说明 |
| --- | --- |
| `start.py` | 根目录启动脚本，默认端口 `8051`。 |
| `start.sh` | bash 启动脚本，会创建/使用 `.venv` 并安装依赖，默认端口 `8051`。 |
| `literature_showcase/run_web.bat` | Windows 一键启动脚本，默认端口 `8051`。 |
| `literature_showcase/run_web.ps1` | Windows PowerShell 展示页启动脚本，默认端口 `8051`。 |
| `literature_showcase/test_showcase.ps1` | 展示页 smoke test。 |
| `requirements.txt` | Python 依赖列表。 |

推荐入口：

```text
http://127.0.0.1:8051/?view=reproduction#repro-nasri_2016_ac_uc_benders
```

## 关键文档

- `README.md`
- `CODE_SUMMARY.md`
- `DATA_INVENTORY.md`
- `frontend_web.md`
- `RELEASE_MANIFEST.md`
- `runs/nasri_2016_ac_uc_benders/reports/stage_20_feature_showcase_and_demo_cn.md`
- `runs/nasri_2016_ac_uc_benders/reports/stage_20_feature_showcase_and_demo_cn.pdf`
- `runs/nasri_2016_ac_uc_benders/reports/stage_20_feature_showcase_overleaf.zip`

注：上述 Nasri 报告文件名中保留历史生成时的 `demo` 字样，是既有复现实验产物的一部分；文档和页面统一使用“课程作业演示/示例”口径。

## 本地文献与历史运行

| 路径 | 数量 | 页面用途 |
| --- | ---: | --- |
| `input_pdfs/frequency/` | 5 篇 PDF | 左侧“本地文献”频率安全批次。 |
| `input_pdfs/capacity_market/` | 20 篇 PDF | 左侧“本地文献”容量市场批次。 |
| `output/` | 5 个运行目录 | 左侧“已有运行”、三层综述、质量报告和下载论文信息。 |

## Nasri 工作区边界

Nasri 复现工作区包含完整展示材料和大量历史求解结果。`results/` 中的 CSV/JSON/PNG/SVG 是复现实验记录，不是缓存。部分历史结果内部仍保留原机器绝对路径字符串，用于记录当时生成位置；页面和当前 target 已使用包内相对路径。

## 参考 Target

`config/targets/bertsimas_2013.yaml`、`lee_2014.yaml`、`gourtani_2016.yaml` 是方法参考或 sanity check target。对应 PDF 未随当前发布包分发，运行 CLI 前需要自行补齐 PDF 并更新 `source_pdf`。
