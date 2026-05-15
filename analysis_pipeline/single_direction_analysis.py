from __future__ import annotations

import argparse
from pathlib import Path

try:
    from analysis_pipeline._bootstrap import PROJECT_ROOT  # noqa: F401
except ModuleNotFoundError:
    from _bootstrap import PROJECT_ROOT  # noqa: F401

from analysis_pipeline.direction_pipeline import run_direction_pipeline
from analysis_pipeline.direction_workspace import (
    build_direction_workspace,
    build_local_pdf_candidates,
    build_virtual_single_direction_state,
)
from analysis_pipeline.pipeline_common import (
    TimeRecorder,
    build_client,
    ensure_dir,
    extract_text_from_pdf,
    resolve_llm_config,
    safe_output_stem,
    save_json,
)


def convert_pdf_dir_to_txt(pdf_dir: Path, txt_dir: Path, overwrite: bool) -> list[Path]:
    ensure_dir(txt_dir)
    txt_paths: list[Path] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        txt_path = txt_dir / f"{safe_output_stem(pdf_path.stem)}.txt"
        txt_paths.append(txt_path)
        if txt_path.exists() and not overwrite:
            print(f"[TXT] 复用已有文本：{txt_path.name}")
            continue
        text = extract_text_from_pdf(pdf_path, add_page_mark=True)
        txt_path.write_text(text + "\n", encoding="utf-8")
        print(f"[TXT] 已生成：{txt_path}")
    return txt_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单方向文献分析 v3：复用方向内 11-14 pipeline")
    parser.add_argument("--assigned-papers", type=Path, default=None, help="方向工作区 assigned_papers.json")
    parser.add_argument("--pdf-dir", type=Path, default=None, help="旧兼容入口：PDF 目录")
    parser.add_argument("--topic", required=True, help="研究主题（中文）")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录或方向目录")
    parser.add_argument("--figures-dir", type=Path, default=None, help="图表提取输出目录（含 manifest.json）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有结果")
    parser.add_argument("--model", default=None, help="LLM 主模型名")
    parser.add_argument("--flash-model", default=None, help="LLM flash 模型名")
    parser.add_argument("--parallel-papers", type=int, default=1, help="并发处理方向内单篇富化数量")
    return parser.parse_args()


def prepare_direction_dir(args: argparse.Namespace) -> Path:
    if args.assigned_papers is not None:
        if not args.assigned_papers.exists():
            raise FileNotFoundError(f"assigned_papers.json 不存在：{args.assigned_papers}")
        return args.assigned_papers.parent
    if args.pdf_dir is None:
        raise ValueError("必须提供 --assigned-papers 或 --pdf-dir")
    pdf_files = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"未在目录中找到 PDF：{args.pdf_dir}")
    output_dir = ensure_dir(args.output_dir)
    txt_dir = ensure_dir(output_dir / "analysis" / "txt_output")
    convert_pdf_dir_to_txt(args.pdf_dir, txt_dir, args.overwrite)
    candidates = build_local_pdf_candidates([path.resolve() for path in pdf_files])
    state = build_virtual_single_direction_state(args.topic, candidates)
    selected_candidates = list(state["papers"])
    direction_dirs = build_direction_workspace(
        output_dir=output_dir,
        screening_state=state,
        selected_candidates=selected_candidates,
        pdf_dir=args.pdf_dir,
        txt_dir=txt_dir,
        figures_dir=args.figures_dir,
    )
    if not direction_dirs:
        raise RuntimeError("未能构建单方向工作区")
    save_json(output_dir / "download" / "screening_state.json", state)
    save_json(output_dir / "download" / "selected_candidates.json", selected_candidates)
    return direction_dirs[0]


def main() -> None:
    args = parse_args()
    direction_dir = prepare_direction_dir(args)
    config = resolve_llm_config()
    client = build_client(config)
    timer = TimeRecorder()
    result = run_direction_pipeline(
        direction_dir=direction_dir,
        topic=f"{args.topic}。请主要使用中文输出，保留必要英文术语。",
        client=client,
        model=args.model or config.model,
        flash_model=args.flash_model or config.flash_model,
        overwrite=args.overwrite,
        parallel_papers=args.parallel_papers,
        timer=timer,
    )
    timer.save(direction_dir / "time_records")
    print("单方向文献分析完成。")
    print(f"方向目录：{direction_dir}")
    print(f"综述 Markdown：{result['outputs']['literature_review_md']}")
    print(f"综述 SVG：{result['outputs']['single_direction_svg']}")


if __name__ == "__main__":
    main()
