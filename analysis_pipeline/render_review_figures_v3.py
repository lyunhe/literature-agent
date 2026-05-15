from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis_pipeline.svg_utils import (
    COLOR_AXIS,
    COLOR_MUTED,
    COLOR_TEXT,
    DIRECTION_COLORS,
    MARGIN,
    SvgCanvas,
    clamp_items,
    draw_citation_box,
    draw_flow_box,
    draw_header,
    draw_panel,
    draw_paragraph_panel,
    estimate_citation_box_height,
    estimate_flow_box_height,
    estimate_panel_height,
    estimate_paragraph_panel_height,
    estimate_text_block_height,
)


PAGE_WIDTH = 1840
COLUMN_GAP = 28
ROW_GAP = 18
SECTION_GAP = 34


def _items(value: Any, limit: int = 3) -> list[str]:
    return clamp_items(value, limit)


def _glossary_items(payload: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for entry in payload.get("symbol_glossary_cn", []):
        symbol = str(entry.get("symbol", "")).strip()
        meaning = str(entry.get("meaning_cn", "")).strip()
        if symbol and meaning:
            result.append(f"{symbol}: {meaning}")
    return result


def render_single_direction_svg(payload: dict[str, Any], output_path: Path) -> Path:
    direction_id = str(payload.get("direction_id") or "D1")
    direction_name = str(payload.get("direction_name_cn") or payload.get("direction_name") or "研究方向")
    accent = DIRECTION_COLORS.get(direction_id, "#1F8A70")
    title = str(payload.get("figure_title_cn") or f"{direction_id}｜{direction_name}")
    subtitle = "单方向综述图：核心问题 / 主线文献 / 方向内差异"

    baseline = payload.get("baseline_or_mainline") or payload.get("baseline_paper") or {}
    core_problem_text = "；".join(_items(payload.get("core_problem_box"), 3)) or str(payload.get("core_problem_cn", ""))
    comparison_rows = list(payload.get("comparison_rows", []))
    glossary_items = _glossary_items(payload)
    formula_items = [
        f"{item.get('formula_id', '')}: {item.get('meaning_cn', '')}"
        for item in payload.get("formula_boxes", [])
        if item.get("meaning_cn") or item.get("formula_id")
    ][:3]
    gap_items = _items(payload.get("research_gap_box"), 3)
    evolution_items = _items(payload.get("method_evolution_box"), 3)

    header_est = MARGIN + 140
    problem_h = estimate_paragraph_panel_height("核心问题", core_problem_text, PAGE_WIDTH - 2 * MARGIN, 22)
    citation_w = 190
    slot_w = (PAGE_WIDTH - 2 * MARGIN - citation_w - COLUMN_GAP * 3) / 3
    baseline_h = max(
        estimate_flow_box_height("输入 / 条件", _items(baseline.get("input_box_cn"), 3), slot_w, 19),
        estimate_flow_box_height("方法 / 模型", _items(baseline.get("method_box_cn"), 3), slot_w, 19),
        estimate_flow_box_height("输出 / 结果", _items(baseline.get("output_box_cn"), 3), slot_w, 19),
        estimate_citation_box_height(str(baseline.get("citation_cn", "")), citation_w, 20),
    )
    diff_heights = [
        max(
            estimate_flow_box_height("输入差异", _items(row.get("input_diff_cn"), 2), slot_w, 18),
            estimate_flow_box_height("方法差异", _items(row.get("method_diff_cn"), 2), slot_w, 18),
            estimate_flow_box_height("输出差异", _items(row.get("output_diff_cn"), 2), slot_w, 18),
            estimate_citation_box_height(str(row.get("citation_cn", "")), citation_w, 19),
        )
        for row in comparison_rows
    ]
    side_panels: list[tuple[str, list[str]]] = []
    if evolution_items:
        side_panels.append(("方法演进", evolution_items))
    if formula_items:
        side_panels.append(("关键公式", formula_items))
    if gap_items:
        side_panels.append(("研究空白", gap_items))
    if glossary_items:
        side_panels.append(("符号说明", glossary_items))
    side_heights = [estimate_panel_height(title, items, PAGE_WIDTH - 2 * MARGIN, 19) for title, items in side_panels]

    total_height = header_est + problem_h + SECTION_GAP + 34 + baseline_h + 42
    if comparison_rows:
        total_height += 36 + sum(diff_heights) + ROW_GAP * max(0, len(diff_heights) - 1)
    if side_heights:
        total_height += SECTION_GAP + sum(side_heights) + ROW_GAP * max(0, len(side_heights) - 1)
    total_height += 46

    canvas = SvgCanvas(PAGE_WIDTH, int(total_height))
    current_y = draw_header(canvas, title, subtitle, accent, PAGE_WIDTH, "方向图")
    draw_paragraph_panel(canvas, MARGIN, current_y, PAGE_WIDTH - 2 * MARGIN, "核心问题", core_problem_text, accent, font_size=22)
    current_y += problem_h + SECTION_GAP
    canvas.text(MARGIN, current_y, ["典型文献主线"], font_size=28, fill=COLOR_TEXT, weight=800)
    current_y += 18

    input_x = MARGIN
    method_x = input_x + slot_w + COLUMN_GAP
    output_x = method_x + slot_w + COLUMN_GAP
    cite_x = output_x + slot_w + COLUMN_GAP
    row_y = current_y + 18
    draw_flow_box(canvas, input_x, row_y, slot_w, baseline_h, "输入 / 条件", _items(baseline.get("input_box_cn"), 3), "#DCE7FF", "#3159C7", body_font_size=19)
    draw_flow_box(canvas, method_x, row_y, slot_w, baseline_h, "方法 / 模型", _items(baseline.get("method_box_cn"), 3), "#D9F2E8", "#1E6F5C", body_font_size=19)
    draw_flow_box(canvas, output_x, row_y, slot_w, baseline_h, "输出 / 结果", _items(baseline.get("output_box_cn"), 3), "#FCE9C9", "#C97912", body_font_size=19)
    draw_citation_box(canvas, cite_x, row_y, citation_w, baseline_h, str(baseline.get("citation_cn", "")), accent)
    mid_y = row_y + baseline_h / 2
    canvas.arrow(input_x + slot_w + 8, mid_y, method_x - 8, mid_y, COLOR_AXIS, 2.4)
    canvas.arrow(method_x + slot_w + 8, mid_y, output_x - 8, mid_y, COLOR_AXIS, 2.4)
    current_y = row_y + baseline_h + 40

    if comparison_rows:
        canvas.text(MARGIN, current_y, ["其余文献相对主线的差异"], font_size=28, fill=COLOR_TEXT, weight=800)
        current_y += 18
        for row, row_h in zip(comparison_rows, diff_heights):
            row_y = current_y + 18
            draw_flow_box(canvas, input_x, row_y, slot_w, row_h, "输入差异", _items(row.get("input_diff_cn"), 2), "#DCE7FF", "#3159C7", body_font_size=18)
            draw_flow_box(canvas, method_x, row_y, slot_w, row_h, "方法差异", _items(row.get("method_diff_cn"), 2), "#D9F2E8", "#1E6F5C", body_font_size=18)
            draw_flow_box(canvas, output_x, row_y, slot_w, row_h, "输出差异", _items(row.get("output_diff_cn"), 2), "#FCE9C9", "#C97912", body_font_size=18)
            draw_citation_box(canvas, cite_x, row_y, citation_w, row_h, str(row.get("citation_cn", "")), accent)
            mid_y = row_y + row_h / 2
            canvas.arrow(input_x + slot_w + 8, mid_y, method_x - 8, mid_y, COLOR_AXIS, 2.0)
            canvas.arrow(method_x + slot_w + 8, mid_y, output_x - 8, mid_y, COLOR_AXIS, 2.0)
            current_y = row_y + row_h + ROW_GAP

    for (panel_title, panel_items), panel_h in zip(side_panels, side_heights):
        current_y += SECTION_GAP if panel_title == side_panels[0][0] else ROW_GAP
        draw_panel(canvas, MARGIN, current_y, PAGE_WIDTH - 2 * MARGIN, panel_title, panel_items, accent, font_size=19)
        current_y += panel_h

    canvas.text(MARGIN, int(total_height) - 22, ["数据来源：direction_records.json + plot_ready.json"], font_size=16, fill=COLOR_MUTED)
    canvas.save(output_path)
    return output_path


def render_cross_direction_svg(payload: dict[str, Any], output_path: Path) -> Path:
    accent = "#2E86DE"
    title = str(payload.get("figure_title_cn") or f"{payload.get('topic', '研究主题')} 总综述图")
    core = str(payload.get("global_core_problem_cn") or "")
    direction_blocks = list(payload.get("direction_blocks", []))
    comparisons = list(payload.get("cross_direction_comparison", []))[:6]
    gaps = list(payload.get("research_gap_blocks", []))[:4]
    storyline = _items(payload.get("storyline_cn"), 4)
    block_w = (PAGE_WIDTH - 2 * MARGIN - COLUMN_GAP * 2) / 3

    direction_heights = []
    for block in direction_blocks:
        items = [
            block.get("main_problem_cn", ""),
            "方法：" + "、".join(_items(block.get("method_keywords_cn"), 3)),
            "输出：" + "、".join(_items(block.get("main_outputs_cn"), 2)),
            "局限：" + "、".join(_items(block.get("limitations_cn"), 2)),
        ]
        direction_heights.append(estimate_panel_height(str(block.get("direction_name_cn", "")), [x for x in items if x], block_w, 18))
    rows = (len(direction_blocks) + 2) // 3
    direction_grid_h = 0
    for row in range(rows):
        direction_grid_h += max(direction_heights[row * 3 : row * 3 + 3] or [0])
    direction_grid_h += ROW_GAP * max(0, rows - 1)
    core_h = estimate_paragraph_panel_height("总核心问题", core, PAGE_WIDTH - 2 * MARGIN, 22)
    story_h = estimate_panel_height("综述主线", storyline, PAGE_WIDTH - 2 * MARGIN, 20) if storyline else 0
    comp_items = [
        f"{item.get('comparison_axis_cn', '')}: "
        + "；".join(f"{k}={v}" for k, v in dict(item.get("direction_values", {})).items())
        for item in comparisons
    ]
    comp_h = estimate_panel_height("跨方向比较", comp_items, PAGE_WIDTH - 2 * MARGIN, 18) if comp_items else 0
    gap_items = [
        f"{item.get('gap_name_cn', '')}: {item.get('gap_description_cn', '')}；切入点：{item.get('possible_entry_point_cn', '')}"
        for item in gaps
    ]
    gap_h = estimate_panel_height("研究空白与切入点", gap_items, PAGE_WIDTH - 2 * MARGIN, 18) if gap_items else 0
    total_height = MARGIN + 150 + core_h + SECTION_GAP + direction_grid_h + SECTION_GAP + story_h + SECTION_GAP + comp_h + SECTION_GAP + gap_h + 60

    canvas = SvgCanvas(PAGE_WIDTH, int(total_height))
    current_y = draw_header(canvas, title, "跨方向总览：方向版图 / 比较维度 / 后续切入点", accent, PAGE_WIDTH, "总图")
    draw_paragraph_panel(canvas, MARGIN, current_y, PAGE_WIDTH - 2 * MARGIN, "总核心问题", core, accent, font_size=22)
    current_y += core_h + SECTION_GAP

    x_positions = [MARGIN, MARGIN + block_w + COLUMN_GAP, MARGIN + (block_w + COLUMN_GAP) * 2]
    index = 0
    for row in range(rows):
        row_heights = direction_heights[row * 3 : row * 3 + 3]
        row_h = max(row_heights or [0])
        for col in range(3):
            if index >= len(direction_blocks):
                break
            block = direction_blocks[index]
            direction_id = str(block.get("direction_id") or f"D{index + 1}")
            color = DIRECTION_COLORS.get(direction_id, accent)
            items = [
                block.get("main_problem_cn", ""),
                "方法：" + "、".join(_items(block.get("method_keywords_cn"), 3)),
                "输出：" + "、".join(_items(block.get("main_outputs_cn"), 2)),
                "局限：" + "、".join(_items(block.get("limitations_cn"), 2)),
            ]
            draw_panel(canvas, x_positions[col], current_y, block_w, f"{direction_id} {block.get('direction_name_cn', '')}", [x for x in items if x], color, font_size=18)
            index += 1
        current_y += row_h + ROW_GAP

    current_y += SECTION_GAP - ROW_GAP
    if storyline:
        draw_panel(canvas, MARGIN, current_y, PAGE_WIDTH - 2 * MARGIN, "综述主线", storyline, accent, font_size=20)
        current_y += story_h + SECTION_GAP
    if comp_items:
        draw_panel(canvas, MARGIN, current_y, PAGE_WIDTH - 2 * MARGIN, "跨方向比较", comp_items, accent, font_size=18)
        current_y += comp_h + SECTION_GAP
    if gap_items:
        draw_panel(canvas, MARGIN, current_y, PAGE_WIDTH - 2 * MARGIN, "研究空白与切入点", gap_items, accent, font_size=18)

    canvas.text(MARGIN, int(total_height) - 22, ["数据来源：各方向 direction_records + corpus_literature_review"], font_size=16, fill=COLOR_MUTED)
    canvas.save(output_path)
    return output_path
