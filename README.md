### `extract_pdf_figures_tables.py`
- 读取什么
  - 单个 PDF 或一个 PDF 目录。
- 使用什么工具 / 基本原理
  - `PyMuPDF` 解析页面内容、文本行和几何区域。
  - 基于 caption、页面布局、候选区域合并和表格内容判断提取图表。
- 输出什么
  - 每篇 PDF 一个输出子目录。
  - 裁剪后的图像 PNG。
  - 表格 CSV。
  - `manifest.json`。
- 如何运行
```bash
python extract_pdf_figures_tables.py --pdf 文献/some-paper.pdf
python extract_pdf_figures_tables.py --pdf-dir 文献 --output-dir pdf_figures_tables_output
