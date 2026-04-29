## 项目概述

本项目用于处理 `PDF` 文献，当前保留四条功能线：

- 多文献正文结构化主流程
- 单篇图表提取
- 基于公式编号的单篇公式截图提取
- 综述可视化 SVG 生成

## 当前目录说明

- `文献/`
  - 输入文献目录，放待处理的 `PDF`
- `output/`
  - 主流程输出目录
- `pdf_figures_tables_output/`
  - 图表提取输出目录
- `pdf_formula_regions_output/`
  - 公式截图提取输出目录
- `pdf_formula_regions_output_v2/`
  - `extract_pdf_formula_regions_v2.py` 的输出目录
- `formula_ocr_pix2tex_output/`
  - `ocr_formula_images_pix2tex.py` 的输出目录
- `time_records/`
  - 计时记录
- `test.pdf`
  - 当前仓库内的测试 PDF
- `.env`
  - 模型接口配置

## 代码地图

### `multi_paper_structured_pipeline_v2.py`

- 作用
  - 多文献正文抽取与结构化主流程
- 读取什么
  - `文献/*.pdf`
  - `.env` 中的模型配置
- 使用什么
  - `PyMuPDF` 抽取正文文本
  - `openai` 兼容客户端访问 `DeepSeek / OpenAI`
- 输出什么
  - `output/txt_output/*.txt`
  - `output/single_paper_structures/*.json`
  - `output/directions/direction_mapping.json`
  - `output/direction_schemas/*.json`
  - `output/direction_records/*.json`
  - `output/comparisons/cross_direction_comparison.json`
  - `output/adaptive_structured_output_bundle.json`
- 如何运行

```bash
python multi_paper_structured_pipeline_v2.py
python multi_paper_structured_pipeline_v2.py --file "some-paper.pdf"
python multi_paper_structured_pipeline_v2.py --single-only
python multi_paper_structured_pipeline_v2.py --overwrite
```

### `extract_pdf_figures_tables.py`

- 作用
  - 从 PDF 中独立提取图、表和 caption 对应区域
- 读取什么
  - 单个 `PDF` 或整个 `PDF` 目录
- 使用什么
  - `PyMuPDF`
  - 基于 caption、版面和候选区域规则提取图表
- 输出什么
  - `pdf_figures_tables_output/*/manifest.json`
  - `pdf_figures_tables_output/*/figures/*.png`
  - `pdf_figures_tables_output/*/tables/*.png`
  - 部分表格会同时输出 `*.csv`
- 如何运行

```bash
python extract_pdf_figures_tables.py --pdf 文献/some-paper.pdf
python extract_pdf_figures_tables.py --pdf-dir 文献 --output-dir pdf_figures_tables_output
```

### `extract_pdf_formula_regions.py`

- 作用
  - 根据 `(1)`、`(1.1)`、`(A1)` 这类公式编号定位并裁剪公式图片
- 读取什么
  - 单个 `PDF` 或整个 `PDF` 目录
- 使用什么
  - `PyMuPDF`
  - 复用 `extract_pdf_figures_tables.py` 的版面检测、列判断和图片裁剪逻辑
  - 用公式编号行做锚点，再向左/向上合并邻近公式块
- 输出什么
  - `pdf_formula_regions_output/*/manifest.json`
  - `pdf_formula_regions_output/*/formula_crops/*.png`
  - `pdf_formula_regions_output/run_summary.json`
- 如何运行

```bash
python extract_pdf_formula_regions.py --pdf test.pdf
python extract_pdf_formula_regions.py --pdf-dir 文献 --output-dir pdf_formula_regions_output --overwrite
```

### `extract_pdf_formula_regions_v2.py`

- 作用
  - `extract_pdf_formula_regions.py` 的宽松锚点识别版本，优先用于补抓被严格右对齐阈值漏掉的带编号公式
- 读取什么
  - 单个 `PDF` 或整个 `PDF` 目录
- 使用什么
  - `PyMuPDF`
  - 复用 `extract_pdf_figures_tables.py` 的版面检测、列判断和图片裁剪逻辑
  - 对编号行使用更宽松的编号模式、右侧几何打分和邻近公式上下文辅助判断
- 输出什么
  - `pdf_formula_regions_output_v2/*/manifest.json`
  - `pdf_formula_regions_output_v2/*/formula_crops/*.png`
  - `pdf_formula_regions_output_v2/run_summary.json`
- 如何运行

```bash
python extract_pdf_formula_regions_v2.py --pdf test.pdf --overwrite
python extract_pdf_formula_regions_v2.py --pdf-dir 文献 --output-dir pdf_formula_regions_output_v2 --overwrite
```

### `ocr_formula_images_pix2tex.py`

- 作用
  - 读取公式截图 `manifest.json`，调用 `pix2tex` 将每张公式图片识别为 LaTeX，并生成 `json + md` 文档
- 读取什么
  - 默认读取 `pdf_formula_regions_output_v2/*/manifest.json`
  - 也可通过 `--manifest` 直接指定单个或多个 `manifest.json`
- 使用什么
  - `pix2tex` Python API
  - 复用公式截图阶段生成的 `png_path`、编号、页码和检测状态
  - 导出前按公式编号去重，并按编号顺序重排
- 输出什么
  - `formula_ocr_pix2tex_output/*/formula_ocr.json`
  - `formula_ocr_pix2tex_output/*/formulas.md`
    - 简洁版 Markdown，只保留带编号公式，格式为 `$$ ... \tag{n} $$`
  - `formula_ocr_pix2tex_output/run_summary.json`
- 如何运行

```bash
python ocr_formula_images_pix2tex.py --overwrite
python ocr_formula_images_pix2tex.py --manifest pdf_formula_regions_output_v2/test/manifest.json --overwrite
python ocr_formula_images_pix2tex.py --manifest pdf_formula_regions_output_v2/test/manifest.json --reuse-json --overwrite
```

### `generate_review_figures.py`

- 作用
  - 根据主流程结构化结果生成综述 SVG 图
- 读取什么
  - `output/directions/direction_mapping.json`
  - `output/comparisons/cross_direction_comparison.json`
  - `output/direction_records/*.json`
  - `output/single_paper_structures/*.json`
- 使用什么
  - 直接拼接 `SVG`
- 输出什么
  - `output/review_figures/*.svg`
- 如何运行

```bash
python generate_review_figures.py --input-dir output
```

## 推荐使用顺序

1. 运行 `multi_paper_structured_pipeline_v2.py` 生成正文结构化结果
2. 运行 `extract_pdf_figures_tables.py` 提取单篇图表
3. 优先运行 `extract_pdf_formula_regions_v2.py` 提取带编号的公式截图
4. 运行 `ocr_formula_images_pix2tex.py` 生成公式 LaTeX/Markdown 文档
5. 运行 `generate_review_figures.py` 生成综述可视化图

## 注意事项

- `multi_paper_structured_pipeline_v2.py` 依赖 `.env` 中的模型配置
- `extract_pdf_formula_regions.py` 当前只输出公式截图，不输出公式 `md` 或公式语义结构化
- `extract_pdf_formula_regions_v2.py` 当前仍只抓“带编号公式”，不覆盖无编号公式、行内公式和编号缺失公式
- `ocr_formula_images_pix2tex.py` 默认会重新执行 OCR；如果只想调整 Markdown 排版或去重排序，可使用 `--reuse-json` 直接复用已有 `formula_ocr.json`
- 公式截图路线依赖 PDF 文本层中存在可识别的公式编号；没有编号的公式目前不会被抓到
- 当前所有路径默认都在项目目录内部
