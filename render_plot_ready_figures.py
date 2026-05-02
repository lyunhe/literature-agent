import argparse
import json
from pathlib import Path
from typing import Any

from generate_review_figures import (
    COLOR_MUTED,
    COLOR_TEXT,
    DIRECTION_COLORS,
    MARGIN,
    SvgCanvas,
    draw_citation_box,
    draw_flow_box,
    draw_header,
    draw_panel,
    draw_paragraph_panel,
    estimate_citation_box_height,
    estimate_flow_box_height,
    estimate_panel_height,
    estimate_paragraph_panel_height,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "output_awe_20260426_2212" / "plot_ready_structures"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output_awe_20260426_2212" / "review_figures_plot_ready"
PAGE_WIDTH = 1840
COLUMN_GAP = 28
SECTION_GAP = 34
ROW_GAP = 18


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_glossary_items(payload: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for entry in payload.get("symbol_glossary_cn", []):
        symbol = str(entry.get("symbol", "")).strip()
        meaning = str(entry.get("meaning_cn", "")).strip()
        if symbol and meaning:
            items.append(f"{symbol}：{meaning}")
    return items


def render_direction_plot_ready(payload: dict[str, Any], output_dir: Path) -> Path:
    direction_id = str(payload.get("direction_id", "")).strip() or "DX"
    direction_name = str(payload.get("direction_name", "")).strip() or "未命名方向"
    accent = DIRECTION_COLORS.get(direction_id, "#2E86DE")
    title = f"{direction_id}｜{direction_name}"
    subtitle = "单方向可视化总结：输入 / 方法 / 输出 + 方向内差异"

    baseline = payload.get("baseline_paper", {})
    comparison_rows = payload.get("comparison_rows", [])
    glossary_items = make_glossary_items(payload)

    header_bottom_est = MARGIN + max(
        120,
        int(
            24
            + 34
            + (len(title) // 40 + 1) * 34 * 1.16
            + 14
            + 19
            + 24
        ),
    ) + 20

    problem_h = estimate_paragraph_panel_height(
        "核心问题",
        str(payload.get("core_problem_cn", "")).strip(),
        PAGE_WIDTH - 2 * MARGIN,
        22,
    )

    citation_w = 190
    slot_w = (PAGE_WIDTH - 2 * MARGIN - citation_w - COLUMN_GAP * 3) / 3

    baseline_h = max(
        estimate_flow_box_height("输入 / 条件", list(baseline.get("input_box_cn", [])), slot_w, 19),
        estimate_flow_box_height("方法 / 模型", list(baseline.get("method_box_cn", [])), slot_w, 19),
        estimate_flow_box_height("输出 / 结果", list(baseline.get("output_box_cn", [])), slot_w, 19),
        estimate_citation_box_height(str(baseline.get("citation_cn", "")).strip(), citation_w, 20),
    )

    diff_heights: list[int] = []
    for row in comparison_rows:
        row_h = max(
            estimate_flow_box_height("输入差异", list(row.get("input_diff_cn", [])), slot_w, 18),
            estimate_flow_box_height("方法差异", list(row.get("method_diff_cn", [])), slot_w, 18),
            estimate_flow_box_height("输出差异", list(row.get("output_diff_cn", [])), slot_w, 18),
            estimate_citation_box_height(str(row.get("citation_cn", "")).strip(), citation_w, 19),
        )
        diff_heights.append(row_h)

    glossary_h = 0
    if glossary_items:
        glossary_h = estimate_panel_height("符号说明", glossary_items, PAGE_WIDTH - 2 * MARGIN, 19)

    total_height = header_bottom_est + problem_h + SECTION_GAP + 34 + baseline_h + 42
    if comparison_rows:
        total_height += 36 + sum(diff_heights) + ROW_GAP * (len(diff_heights) - 1)
    if glossary_h:
        total_height += SECTION_GAP + glossary_h
    total_height += 46

    canvas = SvgCanvas(PAGE_WIDTH, int(total_height))
    header_bottom = draw_header(canvas, title, subtitle, accent, PAGE_WIDTH, "方向图")

    current_y = header_bottom
    draw_paragraph_panel(
        canvas,
        MARGIN,
        current_y,
        PAGE_WIDTH - 2 * MARGIN,
        "核心问题",
        str(payload.get("core_problem_cn", "")).strip(),
        accent,
        font_size=22,
    )
    current_y += problem_h + SECTION_GAP

    canvas.text(MARGIN, current_y, ["典型文献主线"], font_size=28, fill=COLOR_TEXT, weight=800)
    current_y += 18

    input_x = MARGIN
    method_x = input_x + slot_w + COLUMN_GAP
    output_x = method_x + slot_w + COLUMN_GAP
    cite_x = output_x + slot_w + COLUMN_GAP
    row_y = current_y + 18

    draw_flow_box(canvas, input_x, row_y, slot_w, baseline_h, "输入 / 条件", list(baseline.get("input_box_cn", [])), "#DCE7FF", "#3159C7", body_font_size=19)
    draw_flow_box(canvas, method_x, row_y, slot_w, baseline_h, "方法 / 模型", list(baseline.get("method_box_cn", [])), "#D9F2E8", "#1E6F5C", body_font_size=19)
    draw_flow_box(canvas, output_x, row_y, slot_w, baseline_h, "输出 / 结果", list(baseline.get("output_box_cn", [])), "#FCE9C9", "#C97912", body_font_size=19)
    draw_citation_box(canvas, cite_x, row_y, citation_w, baseline_h, str(baseline.get("citation_cn", "")).strip(), accent)
    mid_y = row_y + baseline_h / 2
    canvas.arrow(input_x + slot_w + 8, mid_y, method_x - 8, mid_y, "#8AA7B6", 2.4)
    canvas.arrow(method_x + slot_w + 8, mid_y, output_x - 8, mid_y, "#8AA7B6", 2.4)

    current_y = row_y + baseline_h + 40

    if comparison_rows:
        canvas.text(MARGIN, current_y, ["其余文献相对主线的差异"], font_size=28, fill=COLOR_TEXT, weight=800)
        current_y += 18
        for row, row_h in zip(comparison_rows, diff_heights):
            row_y = current_y + 18
            draw_flow_box(canvas, input_x, row_y, slot_w, row_h, "输入差异", list(row.get("input_diff_cn", [])), "#DCE7FF", "#3159C7", body_font_size=18)
            draw_flow_box(canvas, method_x, row_y, slot_w, row_h, "方法差异", list(row.get("method_diff_cn", [])), "#D9F2E8", "#1E6F5C", body_font_size=18)
            draw_flow_box(canvas, output_x, row_y, slot_w, row_h, "输出差异", list(row.get("output_diff_cn", [])), "#FCE9C9", "#C97912", body_font_size=18)
            draw_citation_box(canvas, cite_x, row_y, citation_w, row_h, str(row.get("citation_cn", "")).strip(), accent)
            mid_y = row_y + row_h / 2
            canvas.arrow(input_x + slot_w + 8, mid_y, method_x - 8, mid_y, "#8AA7B6", 2.0)
            canvas.arrow(method_x + slot_w + 8, mid_y, output_x - 8, mid_y, "#8AA7B6", 2.0)
            current_y = row_y + row_h + ROW_GAP

    if glossary_h:
        current_y += SECTION_GAP - ROW_GAP
        draw_panel(canvas, MARGIN, current_y, PAGE_WIDTH - 2 * MARGIN, "符号说明", glossary_items, accent, font_size=19)
        current_y += glossary_h

    canvas.text(MARGIN, int(total_height) - 22, ["数据来源：plot_ready_structures/*.json"], font_size=16, fill=COLOR_MUTED)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{direction_id}_plot_ready_visual.svg"
    canvas.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="把画图专用结构化 JSON 渲染成 SVG")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--direction-id", action="append", help="只渲染指定方向，如 --direction-id D1")
    args = parser.parse_args()

    selected_ids = set(args.direction_id or [])
    files = sorted(args.input_dir.glob("*_plot_ready.json"))
    if selected_ids:
        files = [path for path in files if path.name.split("_", 1)[0] in selected_ids]
    if not files:
        raise FileNotFoundError("未找到可渲染的 plot_ready JSON")

    created: list[Path] = []
    for path in files:
        payload = load_json(path)
        created.append(render_direction_plot_ready(payload, args.output_dir))

    print("已生成 SVG：")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
