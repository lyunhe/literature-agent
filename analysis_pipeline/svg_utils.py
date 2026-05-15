from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


DEFAULT_FONT_STACK = "Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif"
PAGE_WIDTH = 1760
MARGIN = 60
LINE_GAP = 1.42
TITLE_GAP = 1.2

COLOR_BG = "#F7FBFC"
COLOR_TEXT = "#14313D"
COLOR_MUTED = "#5D7785"
COLOR_BORDER = "#D3E3EA"
COLOR_PANEL = "#FFFFFF"
COLOR_SUBTLE = "#EEF6F8"
COLOR_AXIS = "#8AA7B6"

DIRECTION_COLORS = {
    "D1": "#1F8A70",
    "D2": "#2E86DE",
    "D3": "#E67E22",
    "D4": "#D35454",
    "D5": "#4C9A5F",
    "D6": "#8E5DB7",
}


def is_wide_char(ch: str) -> bool:
    return ord(ch) > 127


def char_units(ch: str) -> float:
    if ch == " ":
        return 0.35
    if ch in ".,:;!|/()[]{}-_":
        return 0.45
    if is_wide_char(ch):
        return 1.0
    return 0.62


def text_units(text: str) -> float:
    return sum(char_units(ch) for ch in text)


def wrap_text(text: Any, width: float, font_size: int) -> list[str]:
    raw = "" if text is None else str(text)
    if not raw:
        return [""]
    unit_limit = max(width / (font_size * 0.93), 1.0)
    lines: list[str] = []
    for paragraph in raw.split("\n"):
        para = paragraph.strip()
        if not para:
            lines.append("")
            continue
        current = ""
        current_units = 0.0
        for ch in para:
            units = char_units(ch)
            if current and current_units + units > unit_limit:
                lines.append(current)
                current = ch
                current_units = units
            else:
                current += ch
                current_units += units
        if current:
            lines.append(current)
    return lines or [""]


def normalize_items(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [", ".join(f"{k}: {v}" for k, v in value.items() if v)]
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            if not item:
                continue
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(", ".join(f"{k}: {v}" for k, v in item.items() if v))
            else:
                result.append(str(item))
        return result
    return [str(value)]


def clamp_items(items: Any, limit: int) -> list[str]:
    normalized = normalize_items(items)
    return normalized[:limit]


def wrap_bullets(items: Any, width: float, font_size: int) -> list[str]:
    lines: list[str] = []
    indent = "  "
    for item in normalize_items(items):
        wrapped = wrap_text(item, width - 8, font_size)
        if not wrapped:
            continue
        lines.append(f"- {wrapped[0]}")
        for extra in wrapped[1:]:
            lines.append(f"{indent}{extra}")
    return lines or [""]


def clean_label(value: Any, fallback: str = "") -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text or fallback


class SvgCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.elements: list[str] = []

    def add(self, raw: str) -> None:
        self.elements.append(raw)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: str,
        stroke: str = COLOR_BORDER,
        stroke_width: float = 1.0,
        radius: float = 18.0,
    ) -> None:
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
            f'rx="{radius:.1f}" ry="{radius:.1f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.1f}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str,
        stroke_width: float = 2.0,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.1f}" stroke-linecap="round"{dash_attr}/>'
        )

    def arrow(self, x1: float, y1: float, x2: float, y2: float, stroke: str, stroke_width: float = 2.0) -> None:
        self.line(x1, y1, x2, y2, stroke, stroke_width)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 10
        ax = x2 - size * math.cos(angle - math.pi / 6)
        ay = y2 - size * math.sin(angle - math.pi / 6)
        bx = x2 - size * math.cos(angle + math.pi / 6)
        by = y2 - size * math.sin(angle + math.pi / 6)
        self.add(f'<polygon points="{x2:.1f},{y2:.1f} {ax:.1f},{ay:.1f} {bx:.1f},{by:.1f}" fill="{stroke}"/>')

    def text(
        self,
        x: float,
        y: float,
        lines: list[str],
        font_size: int = 26,
        fill: str = COLOR_TEXT,
        weight: int = 400,
        anchor: str = "start",
        line_gap: float = 1.35,
    ) -> None:
        text_anchor = {"start": "start", "middle": "middle", "end": "end"}[anchor]
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{DEFAULT_FONT_STACK}" '
            f'font-size="{font_size}" fill="{fill}" font-weight="{weight}" text-anchor="{text_anchor}">'
        )
        for idx, line in enumerate(lines):
            dy = 0 if idx == 0 else font_size * line_gap
            dy_attr = ' dy="0"' if idx == 0 else f' dy="{dy:.1f}"'
            self.add(f'<tspan x="{x:.1f}"{dy_attr}>{escape(str(line))}</tspan>')
        self.add("</text>")

    def save(self, path: Path) -> None:
        body = "\n".join(self.elements)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'<rect width="{self.width}" height="{self.height}" fill="{COLOR_BG}"/>\n'
            f"{body}\n</svg>\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg, encoding="utf-8")


def estimate_text_block_height(line_count: int, font_size: int, line_gap: float = LINE_GAP) -> float:
    if line_count <= 0:
        return 0
    return font_size + (line_count - 1) * font_size * line_gap


def estimate_panel_height(title: str, items: list[str], width: float, body_font_size: int, title_font_size: int = 24) -> int:
    title_lines = wrap_text(title, width - 36, title_font_size)
    body_lines = wrap_bullets(items, width - 34, body_font_size)
    height = 24
    height += estimate_text_block_height(len(title_lines), title_font_size, TITLE_GAP)
    height += 20
    height += estimate_text_block_height(len(body_lines), body_font_size, LINE_GAP)
    height += 28
    return int(height)


def estimate_paragraph_panel_height(title: str, text: str, width: float, body_font_size: int, title_font_size: int = 24) -> int:
    title_lines = wrap_text(title, width - 36, title_font_size)
    body_lines = wrap_text(text, width - 34, body_font_size)
    height = 24
    height += estimate_text_block_height(len(title_lines), title_font_size, TITLE_GAP)
    height += 20
    height += estimate_text_block_height(len(body_lines), body_font_size, LINE_GAP)
    height += 28
    return int(height)


def estimate_flow_box_height(title: str, items: list[str], width: float, body_font_size: int = 19, title_font_size: int = 20) -> int:
    title_lines = wrap_text(title, width - 34, title_font_size)
    body_lines = wrap_bullets(items, width - 32, body_font_size)
    height = 22
    height += estimate_text_block_height(len(title_lines), title_font_size, TITLE_GAP)
    height += 18
    height += estimate_text_block_height(len(body_lines), body_font_size, LINE_GAP)
    height += 24
    return int(height)


def estimate_citation_box_height(citation: str, width: float, font_size: int = 20) -> int:
    lines = wrap_text(citation, width - 24, font_size)
    return int(30 + estimate_text_block_height(len(lines), font_size, 1.2) + 30)


def draw_panel(canvas: SvgCanvas, x: float, y: float, width: float, title: str, items: list[str], accent: str, font_size: int = 22) -> int:
    title_lines = wrap_text(title, width - 36, 24)
    lines = wrap_bullets(items, width - 34, font_size)
    height = estimate_panel_height(title, items, width, font_size)
    canvas.rect(x, y, width, height, COLOR_PANEL)
    canvas.rect(x, y, width, 12, accent, accent, 0, 16)
    title_y = y + 34
    canvas.text(x + 18, title_y, title_lines, font_size=24, fill=accent, weight=700, line_gap=TITLE_GAP)
    body_y = title_y + estimate_text_block_height(len(title_lines), 24, TITLE_GAP) + 20
    canvas.text(x + 18, body_y, lines, font_size=font_size, fill=COLOR_TEXT, weight=400, line_gap=LINE_GAP)
    return height


def draw_paragraph_panel(canvas: SvgCanvas, x: float, y: float, width: float, title: str, text: str, accent: str, font_size: int = 22) -> int:
    title_lines = wrap_text(title, width - 36, 24)
    lines = wrap_text(text, width - 34, font_size)
    height = estimate_paragraph_panel_height(title, text, width, font_size)
    canvas.rect(x, y, width, height, COLOR_PANEL)
    canvas.rect(x, y, width, 12, accent, accent, 0, 16)
    title_y = y + 34
    canvas.text(x + 18, title_y, title_lines, font_size=24, fill=accent, weight=700, line_gap=TITLE_GAP)
    body_y = title_y + estimate_text_block_height(len(title_lines), 24, TITLE_GAP) + 20
    canvas.text(x + 18, body_y, lines, font_size=font_size, fill=COLOR_TEXT, weight=400, line_gap=LINE_GAP)
    return height


def draw_header(canvas: SvgCanvas, title: str, subtitle: str, accent: str, width: int, page_type: str) -> int:
    y = MARGIN
    tag_width = 144
    title_lines = wrap_text(title, width - 2 * MARGIN - tag_width - 92, 34)
    subtitle_lines = wrap_text(subtitle, width - 2 * MARGIN - 48, 19)
    inner_height = 24
    inner_height += estimate_text_block_height(len(title_lines), 34, 1.16)
    inner_height += 14
    inner_height += estimate_text_block_height(len(subtitle_lines), 19, 1.28)
    inner_height += 24
    box_height = max(120, int(inner_height))
    canvas.rect(MARGIN, y, width - 2 * MARGIN, box_height, COLOR_PANEL, stroke=accent, stroke_width=2.0, radius=24)
    title_y = y + 40
    canvas.text(MARGIN + 24, title_y, title_lines, font_size=34, fill=COLOR_TEXT, weight=800, line_gap=1.16)
    subtitle_y = title_y + estimate_text_block_height(len(title_lines), 34, 1.16) + 16
    canvas.text(MARGIN + 24, subtitle_y, subtitle_lines, font_size=19, fill=COLOR_MUTED, line_gap=1.28)
    canvas.rect(width - MARGIN - tag_width - 20, y + 22, tag_width, 42, accent, stroke=accent, stroke_width=0, radius=18)
    canvas.text(width - MARGIN - tag_width / 2 - 20, y + 50, [page_type], font_size=20, fill="#FFFFFF", weight=700, anchor="middle")
    return y + box_height + 20


def draw_flow_box(
    canvas: SvgCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    items: list[str],
    accent: str,
    title_fill: str,
    body_font_size: int = 19,
) -> None:
    canvas.rect(x, y, width, height, COLOR_PANEL, stroke=accent, stroke_width=1.8, radius=20)
    canvas.rect(x, y, width, 12, title_fill, stroke=title_fill, stroke_width=0, radius=18)
    title_lines = wrap_text(title, width - 34, 20)
    title_y = y + 32
    canvas.text(x + 16, title_y, title_lines, font_size=20, fill=title_fill, weight=800, line_gap=TITLE_GAP)
    body_y = title_y + estimate_text_block_height(len(title_lines), 20, TITLE_GAP) + 18
    canvas.text(x + 16, body_y, wrap_bullets(items, width - 32, body_font_size), font_size=body_font_size, fill=COLOR_TEXT, line_gap=LINE_GAP)


def draw_citation_box(canvas: SvgCanvas, x: float, y: float, width: float, height: float, citation: str, accent: str) -> None:
    canvas.rect(x, y, width, height, COLOR_SUBTLE, stroke=accent, stroke_width=1.6, radius=18)
    lines = wrap_text(citation, width - 24, 20)
    content_h = estimate_text_block_height(len(lines), 20, 1.2)
    text_y = y + (height - content_h) / 2 + 18
    canvas.text(x + width / 2, text_y, lines, font_size=20, fill=accent, weight=800, anchor="middle", line_gap=1.2)
