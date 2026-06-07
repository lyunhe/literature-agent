# Overleaf 使用说明

这个目录是 Nasri 2016 论文复现功能展示文档的 Overleaf 项目。

## 文件结构

- `main.tex`：主 LaTeX 文件。
- `figures/`：文档截图图片。
- `main.pdf`：本地编译生成的 PDF。

## Overleaf 编译设置

推荐编译器：

- XeLaTeX

主文件：

- `main.tex`

如果 Overleaf 提示字体问题，请确认导言区使用了：

```tex
\documentclass[UTF8,fontset=fandol,11pt]{ctexart}
```

该设置使用 Overleaf/TeX Live 常见的 Fandol 中文字体，适合多人协作。
