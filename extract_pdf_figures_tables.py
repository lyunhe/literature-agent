import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF


DEFAULT_OUTPUT_DIR = Path("pdf_figures_tables_output")
CAPTION_RE = re.compile(
    r"^(?P<label>(?:Fig|Figure|FIG|FIGURE|Table|TABLE))\.?\s*(?P<number>(?:\d+|[IVXLCDMivxlcdm]+)(?:[A-Za-z])?)\b"
)


@dataclass
class PageLayout:
    page: int
    layout_type: str
    page_rect: fitz.Rect
    gutter_x0: float | None = None
    gutter_x1: float | None = None
    left_rect: fitz.Rect | None = None
    right_rect: fitz.Rect | None = None


@dataclass
class TextLine:
    page: int
    rect: fitz.Rect
    text: str
    font_size: float
    fonts: tuple[str, ...]
    flags: tuple[int, ...]
    block_index: int
    line_index: int
    column_id: str


@dataclass
class Caption:
    kind: str
    page: int
    rect: fitz.Rect
    text: str
    number_token: str | None
    column_id: str
    layout_type: str
    font_size: float
    label_id: str | None = None
    order: int = 0


@dataclass
class CandidateRegion:
    kind: str
    page: int
    rect: fitz.Rect
    column_id: str
    source: str
    rows: list[list[str | None]] | None = None
    sources: list[str] = field(default_factory=list)


@dataclass
class ExtractedItem:
    id: str
    kind: str
    page: int
    caption: str | None
    caption_bbox: tuple[float, float, float, float] | None
    content_bbox: tuple[float, float, float, float] | None
    crop_bbox: tuple[float, float, float, float]
    number_token: str | None
    png_path: str
    data_path: str | None
    body_source: list[str]
    layout_type: str
    column_id: str
    confidence: float
    status: str
    needs_review: bool


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "pdf"


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())


def rect_to_bbox(rect: fitz.Rect | None) -> tuple[float, float, float, float] | None:
    if rect is None:
        return None
    return (round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2))


def union_rects(rects: Iterable[fitz.Rect]) -> fitz.Rect:
    rects = list(rects)
    if not rects:
        raise ValueError("No rectangles to union.")
    current = fitz.Rect(rects[0])
    for rect in rects[1:]:
        current |= rect
    return current


def clip_rect_to_page(rect: fitz.Rect, page_rect: fitz.Rect, padding: float = 4.0) -> fitz.Rect:
    expanded = fitz.Rect(rect.x0 - padding, rect.y0 - padding, rect.x1 + padding, rect.y1 + padding)
    return expanded & page_rect


def rect_is_valid(rect: fitz.Rect, min_width: float, min_height: float, min_area: float, max_area: float) -> bool:
    area = rect.get_area()
    if rect.is_empty or rect.width < min_width or rect.height < min_height:
        return False
    if area < min_area or area > max_area:
        return False
    return True


def horizontal_overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    if overlap <= 0:
        return 0.0
    base = min(a.width, b.width)
    if base <= 0:
        return 0.0
    return overlap / base


def vertical_overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    if overlap <= 0:
        return 0.0
    base = min(a.height, b.height)
    if base <= 0:
        return 0.0
    return overlap / base


def rect_horizontal_gap(a: fitz.Rect, b: fitz.Rect) -> float:
    if a.x1 < b.x0:
        return b.x0 - a.x1
    if b.x1 < a.x0:
        return a.x0 - b.x1
    return 0.0


def rect_vertical_gap(a: fitz.Rect, b: fitz.Rect) -> float:
    if a.y1 < b.y0:
        return b.y0 - a.y1
    if b.y1 < a.y0:
        return a.y0 - b.y1
    return 0.0


def max_overlap_ratio(candidate: fitz.Rect, others: Iterable[fitz.Rect]) -> float:
    candidate_area = candidate.get_area()
    if candidate_area <= 0:
        return 1.0
    best = 0.0
    for other in others:
        inter = candidate & other
        if inter.is_empty:
            continue
        best = max(best, inter.get_area() / candidate_area)
    return best


def render_clip(page: fitz.Page, rect: fitz.Rect, out_path: Path, dpi: int) -> None:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
    pix.save(out_path)


def write_table_csv(rows: list[list[str | None]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(["" if cell is None else str(cell).strip() for cell in row])


def table_has_content(rows: list[list[str | None]]) -> bool:
    for row in rows:
        for cell in row:
            if cell is not None and str(cell).strip():
                return True
    return False


def parse_roman_numeral(token: str) -> int | None:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    token = token.upper()
    if not token or any(ch not in values for ch in token):
        return None
    total = 0
    prev = 0
    for ch in reversed(token):
        value = values[ch]
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total


def normalize_number_token(token: str | None) -> str | None:
    if not token:
        return None
    token = token.strip()
    if not token:
        return None

    match = re.fullmatch(r"(\d+)([A-Za-z]?)", token)
    if match:
        number = str(int(match.group(1)))
        suffix = match.group(2).lower()
        return f"{number}{suffix}"

    match = re.fullmatch(r"([IVXLCDM]+)([A-Za-z]?)", token, re.IGNORECASE)
    if match:
        roman_value = parse_roman_numeral(match.group(1))
        if roman_value is not None:
            suffix = match.group(2).lower()
            return f"{roman_value}{suffix}"

    return re.sub(r"\s+", "", token).lower()


def detect_page_layout(page: fitz.Page, page_number: int) -> PageLayout:
    page_rect = fitz.Rect(page.rect)
    raw = page.get_text("rawdict")
    anchor_candidates: list[tuple[float, float, float]] = []

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join("".join(char.get("c", "") for char in span.get("chars", [])) for span in spans)
            text = normalize_text(text)
            if len(text) < 12:
                continue
            rect = fitz.Rect(line.get("bbox", block.get("bbox")))
            width = rect.width
            if width < page_rect.width * 0.18 or width > page_rect.width * 0.62:
                continue
            anchor_candidates.append((rect.x0, rect.x1, rect.y0))

    anchor_candidates.sort()
    groups: list[list[tuple[float, float, float]]] = []
    for item in anchor_candidates:
        if not groups:
            groups.append([item])
            continue
        current_mean = sum(value[0] for value in groups[-1]) / len(groups[-1])
        if abs(item[0] - current_mean) <= 24:
            groups[-1].append(item)
        else:
            groups.append([item])

    dense_groups = [group for group in groups if len(group) >= 6]
    if len(dense_groups) >= 2:
        dense_groups.sort(key=lambda group: len(group), reverse=True)
        picked: list[list[tuple[float, float, float]]] = []
        for group in dense_groups:
            if not picked:
                picked.append(group)
                continue
            mean_gap = abs(sum(value[0] for value in group) / len(group) - sum(value[0] for value in picked[0]) / len(picked[0]))
            if mean_gap >= page_rect.width * 0.18:
                picked.append(group)
                break

        if len(picked) == 2:
            picked.sort(key=lambda group: sum(value[0] for value in group) / len(group))
            left_group, right_group = picked
            left_edges = sorted(value[1] for value in left_group)
            right_edges = sorted(value[0] for value in right_group)
            left_edge = left_edges[len(left_edges) // 2]
            right_edge = right_edges[len(right_edges) // 2]
            if right_edge - left_edge < 8:
                left_anchor = sum(value[0] for value in left_group) / len(left_group)
                right_anchor = sum(value[0] for value in right_group) / len(right_group)
                gutter_center = (left_anchor + right_anchor) / 2
                gutter_half = max(8.0, page_rect.width * 0.015)
            else:
                gutter_center = (left_edge + right_edge) / 2
                gutter_half = max(6.0, (right_edge - left_edge) / 2)
            left_rect = fitz.Rect(page_rect.x0, page_rect.y0, gutter_center - gutter_half, page_rect.y1)
            right_rect = fitz.Rect(gutter_center + gutter_half, page_rect.y0, page_rect.x1, page_rect.y1)
            return PageLayout(
                page=page_number,
                layout_type="two_column",
                page_rect=page_rect,
                gutter_x0=gutter_center - gutter_half,
                gutter_x1=gutter_center + gutter_half,
                left_rect=left_rect,
                right_rect=right_rect,
            )

    words = page.get_text("words")
    bins = 120
    occupancy = [0] * bins

    for word in words:
        text = str(word[4]).strip()
        if not text:
            continue
        x0 = max(page_rect.x0, min(page_rect.x1, float(word[0])))
        x1 = max(page_rect.x0, min(page_rect.x1, float(word[2])))
        if x1 <= x0:
            continue
        start = max(0, min(bins - 1, int((x0 - page_rect.x0) / page_rect.width * bins)))
        end = max(0, min(bins - 1, int((x1 - page_rect.x0) / page_rect.width * bins)))
        for index in range(start, end + 1):
            occupancy[index] += 1

    sparse_threshold = max(1, int(len(words) / 800))
    mid_start = int(bins * 0.25)
    mid_end = int(bins * 0.75)
    best_segment: tuple[int, int] | None = None
    current_start: int | None = None

    for index in range(mid_start, mid_end):
        if occupancy[index] <= sparse_threshold:
            if current_start is None:
                current_start = index
        elif current_start is not None:
            segment = (current_start, index - 1)
            if best_segment is None or segment[1] - segment[0] > best_segment[1] - best_segment[0]:
                best_segment = segment
            current_start = None
    if current_start is not None:
        segment = (current_start, mid_end - 1)
        if best_segment is None or segment[1] - segment[0] > best_segment[1] - best_segment[0]:
            best_segment = segment

    if best_segment is None:
        return PageLayout(page=page_number, layout_type="single", page_rect=page_rect)

    gutter_x0 = page_rect.x0 + page_rect.width * best_segment[0] / bins
    gutter_x1 = page_rect.x0 + page_rect.width * (best_segment[1] + 1) / bins
    gutter_width = gutter_x1 - gutter_x0
    gutter_center = (gutter_x0 + gutter_x1) / 2

    left_words = [word for word in words if float(word[2]) <= gutter_center]
    right_words = [word for word in words if float(word[0]) >= gutter_center]
    if gutter_width < page_rect.width * 0.035 or len(left_words) < 25 or len(right_words) < 25:
        return PageLayout(page=page_number, layout_type="single", page_rect=page_rect)

    left_rect = fitz.Rect(
        min(float(word[0]) for word in left_words),
        page_rect.y0,
        max(float(word[2]) for word in left_words),
        page_rect.y1,
    )
    right_rect = fitz.Rect(
        min(float(word[0]) for word in right_words),
        page_rect.y0,
        max(float(word[2]) for word in right_words),
        page_rect.y1,
    )
    if right_rect.x0 - left_rect.x1 < page_rect.width * 0.03:
        return PageLayout(page=page_number, layout_type="single", page_rect=page_rect)

    return PageLayout(
        page=page_number,
        layout_type="two_column",
        page_rect=page_rect,
        gutter_x0=gutter_x0,
        gutter_x1=gutter_x1,
        left_rect=left_rect,
        right_rect=right_rect,
    )


def classify_rect_column(rect: fitz.Rect, layout: PageLayout) -> str:
    if layout.layout_type != "two_column" or layout.left_rect is None or layout.right_rect is None:
        return "full"

    if rect.width >= layout.page_rect.width * 0.82:
        return "full"

    gutter_center = ((layout.gutter_x0 or 0.0) + (layout.gutter_x1 or 0.0)) / 2
    left_overlap = horizontal_overlap_ratio(rect, layout.left_rect)
    right_overlap = horizontal_overlap_ratio(rect, layout.right_rect)

    if rect.x1 <= gutter_center + 6 and left_overlap >= max(0.28, right_overlap):
        return "left"
    if rect.x0 >= gutter_center - 6 and right_overlap >= max(0.28, left_overlap):
        return "right"
    if left_overlap >= 0.55 and right_overlap < 0.2:
        return "left"
    if right_overlap >= 0.55 and left_overlap < 0.2:
        return "right"
    return "full"


def column_bounds(layout: PageLayout, column_id: str) -> tuple[float, float]:
    if layout.layout_type != "two_column":
        return (layout.page_rect.x0 + 8, layout.page_rect.x1 - 8)
    if column_id == "left" and layout.left_rect is not None:
        return (max(layout.page_rect.x0 + 8, layout.left_rect.x0 - 8), min(layout.page_rect.x1 - 8, layout.left_rect.x1 + 8))
    if column_id == "right" and layout.right_rect is not None:
        return (max(layout.page_rect.x0 + 8, layout.right_rect.x0 - 8), min(layout.page_rect.x1 - 8, layout.right_rect.x1 + 8))
    return (layout.page_rect.x0 + 8, layout.page_rect.x1 - 8)


def collect_text_lines(page: fitz.Page, page_number: int, layout: PageLayout) -> list[TextLine]:
    raw = page.get_text("rawdict")
    lines: list[TextLine] = []
    for block_index, block in enumerate(raw.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line_index, line in enumerate(block.get("lines", [])):
            spans = line.get("spans", [])
            pieces: list[str] = []
            fonts: list[str] = []
            flags: list[int] = []
            sizes: list[float] = []
            for span in spans:
                chars = span.get("chars", [])
                text = "".join(char.get("c", "") for char in chars)
                if text:
                    pieces.append(text)
                fonts.append(str(span.get("font", "")))
                flags.append(int(span.get("flags", 0)))
                sizes.append(float(span.get("size", 0.0)))
            normalized = normalize_text("".join(pieces))
            if not normalized:
                continue
            rect = fitz.Rect(line.get("bbox", block.get("bbox")))
            lines.append(
                TextLine(
                    page=page_number,
                    rect=rect,
                    text=normalized,
                    font_size=sum(sizes) / max(len(sizes), 1),
                    fonts=tuple(fonts),
                    flags=tuple(flags),
                    block_index=block_index,
                    line_index=line_index,
                    column_id=classify_rect_column(rect, layout),
                )
            )
    lines.sort(key=lambda item: (item.rect.y0, item.rect.x0))
    return lines


def is_caption_start_candidate(line: TextLine, previous_same_column: TextLine | None) -> re.Match[str] | None:
    match = CAPTION_RE.match(line.text)
    if match is None:
        return None
    label = match.group("label")
    if label.islower():
        return None
    if previous_same_column is not None:
        gap = line.rect.y0 - previous_same_column.rect.y1
        if gap < max(1.8, line.font_size * 0.45) and horizontal_overlap_ratio(line.rect, previous_same_column.rect) > 0.6:
            return None
    return match


def should_extend_caption(start_line: TextLine, previous_line: TextLine, next_line: TextLine) -> bool:
    if CAPTION_RE.match(next_line.text):
        return False
    if next_line.column_id != start_line.column_id:
        return False
    if next_line.page != start_line.page:
        return False
    gap = next_line.rect.y0 - previous_line.rect.y1
    if gap > max(5.0, max(start_line.font_size, previous_line.font_size) * 1.45):
        return False
    if abs(next_line.font_size - start_line.font_size) > 1.6:
        return False
    x_aligned = abs(next_line.rect.x0 - start_line.rect.x0) <= 40 or horizontal_overlap_ratio(start_line.rect, next_line.rect) >= 0.25
    return x_aligned


def collect_captions(lines: list[TextLine], layout: PageLayout) -> list[Caption]:
    captions: list[Caption] = []
    used_indices: set[int] = set()

    for index, line in enumerate(lines):
        if index in used_indices:
            continue

        previous_same_column = None
        for prev_index in range(index - 1, -1, -1):
            candidate = lines[prev_index]
            if candidate.column_id == line.column_id:
                previous_same_column = candidate
                break

        match = is_caption_start_candidate(line, previous_same_column)
        if match is None:
            continue

        caption_lines = [line]
        used_indices.add(index)
        last_line = line

        for next_index in range(index + 1, len(lines)):
            if next_index in used_indices:
                continue
            next_line = lines[next_index]
            if next_line.page != line.page:
                break
            if next_line.column_id != line.column_id:
                continue
            if should_extend_caption(line, last_line, next_line):
                caption_lines.append(next_line)
                used_indices.add(next_index)
                last_line = next_line
                if len(caption_lines) >= 8:
                    break
            elif next_line.rect.y0 - last_line.rect.y1 > max(8.0, line.font_size * 1.8):
                break

        caption_text = " ".join(item.text for item in caption_lines)
        captions.append(
            Caption(
                kind="figure" if match.group("label").lower().startswith("fig") else "table",
                page=line.page,
                rect=union_rects(item.rect for item in caption_lines),
                text=caption_text,
                number_token=normalize_number_token(match.group("number")),
                column_id=line.column_id,
                layout_type=layout.layout_type,
                font_size=sum(item.font_size for item in caption_lines) / len(caption_lines),
            )
        )

    captions.sort(key=lambda item: (item.page, item.rect.y0, item.rect.x0))
    return captions


def extract_table_candidates(
    page: fitz.Page,
    page_number: int,
    layout: PageLayout,
    min_width: float,
    min_height: float,
    min_area: float,
    max_area: float,
) -> list[CandidateRegion]:
    candidates: list[CandidateRegion] = []
    if not hasattr(page, "find_tables"):
        return candidates

    try:
        finder = page.find_tables()
    except Exception:
        return candidates

    for table in getattr(finder, "tables", []):
        rect = clip_rect_to_page(fitz.Rect(table.bbox), page.rect)
        if not rect_is_valid(rect, min_width, min_height, min_area, max_area):
            continue
        try:
            rows = table.extract()
        except Exception:
            continue
        if not rows or not table_has_content(rows):
            continue
        candidates.append(
            CandidateRegion(
                kind="table",
                page=page_number,
                rect=rect,
                column_id=classify_rect_column(rect, layout),
                source="page.find_tables",
                rows=rows,
                sources=["page.find_tables"],
            )
        )
    return candidates


def extract_figure_primitives(
    page: fitz.Page,
    page_number: int,
    layout: PageLayout,
    min_width: float,
    min_height: float,
    min_area: float,
    max_area: float,
    blocked_rects: Iterable[fitz.Rect],
    detect_vector_figures: bool,
) -> list[CandidateRegion]:
    primitives: list[CandidateRegion] = []
    seen: set[tuple[float, float, float, float]] = set()

    for image_info in page.get_images(full=True):
        xref = image_info[0]
        try:
            image_rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in image_rects:
            clipped = clip_rect_to_page(rect, page.rect)
            bbox_key = rect_to_bbox(clipped)
            if bbox_key in seen:
                continue
            seen.add(bbox_key)
            if not rect_is_valid(clipped, min_width, min_height, min_area, max_area):
                continue
            if max_overlap_ratio(clipped, blocked_rects) > 0.75:
                continue
            primitives.append(
                CandidateRegion(
                    kind="figure",
                    page=page_number,
                    rect=clipped,
                    column_id=classify_rect_column(clipped, layout),
                    source=f"embedded_image:xref={xref}",
                    sources=[f"embedded_image:xref={xref}"],
                )
            )

    if detect_vector_figures and hasattr(page, "cluster_drawings"):
        try:
            clusters = page.cluster_drawings()
        except Exception:
            clusters = []
        for cluster in clusters:
            clipped = clip_rect_to_page(fitz.Rect(cluster), page.rect)
            if not rect_is_valid(clipped, min_width, min_height, min_area, max_area):
                continue
            if max_overlap_ratio(clipped, blocked_rects) > 0.45:
                continue
            primitives.append(
                CandidateRegion(
                    kind="figure",
                    page=page_number,
                    rect=clipped,
                    column_id=classify_rect_column(clipped, layout),
                    source="page.cluster_drawings",
                    sources=["page.cluster_drawings"],
                )
            )
    return primitives


def should_merge_primitives(a: CandidateRegion, b: CandidateRegion) -> bool:
    if a.page != b.page:
        return False
    if a.column_id != b.column_id and a.column_id != "full" and b.column_id != "full":
        return False
    if vertical_overlap_ratio(a.rect, b.rect) > 0.32 and rect_horizontal_gap(a.rect, b.rect) <= 48:
        return True
    if horizontal_overlap_ratio(a.rect, b.rect) > 0.32 and rect_vertical_gap(a.rect, b.rect) <= 14:
        return True
    expanded = fitz.Rect(a.rect.x0 - 14, a.rect.y0 - 14, a.rect.x1 + 14, a.rect.y1 + 14)
    return not (expanded & b.rect).is_empty


def merge_figure_primitives(primitives: list[CandidateRegion]) -> list[CandidateRegion]:
    if not primitives:
        return []
    groups: list[list[CandidateRegion]] = []
    for primitive in sorted(primitives, key=lambda item: (item.rect.y0, item.rect.x0)):
        attached = False
        for group in groups:
            group_region = CandidateRegion(
                kind="figure",
                page=group[0].page,
                rect=union_rects(item.rect for item in group),
                column_id="full" if any(item.column_id == "full" for item in group) else group[0].column_id,
                source="",
            )
            if should_merge_primitives(group_region, primitive):
                group.append(primitive)
                attached = True
                break
        if not attached:
            groups.append([primitive])

    merged: list[CandidateRegion] = []
    for group in groups:
        merged.append(
            CandidateRegion(
                kind="figure",
                page=group[0].page,
                rect=union_rects(item.rect for item in group),
                column_id="full" if any(item.column_id == "full" for item in group) else group[0].column_id,
                source=" + ".join(dict.fromkeys(src for item in group for src in item.sources)),
                sources=list(dict.fromkeys(src for item in group for src in item.sources)),
            )
        )
    return merged


def compatible_columns(caption: Caption, candidate: CandidateRegion) -> bool:
    if caption.column_id == "full" or candidate.column_id == "full":
        return True
    return caption.column_id == candidate.column_id


def anchor_score(caption: Caption, candidate: CandidateRegion, page_rect: fitz.Rect) -> float | None:
    if candidate.page != caption.page or not compatible_columns(caption, candidate):
        return None

    preferred_above = caption.kind == "figure"
    center_dx = abs((candidate.rect.x0 + candidate.rect.x1) / 2 - (caption.rect.x0 + caption.rect.x1) / 2)
    center_penalty = center_dx / max(page_rect.width, 1.0) * 24.0
    x_overlap = horizontal_overlap_ratio(caption.rect, candidate.rect)

    if preferred_above:
        if candidate.rect.y1 <= caption.rect.y0 + 8:
            gap = caption.rect.y0 - candidate.rect.y1
            direction_bonus = 1.0
        elif candidate.rect.y0 >= caption.rect.y1 - 8:
            gap = candidate.rect.y0 - caption.rect.y1
            direction_bonus = 0.45
        else:
            gap = 0.0
            direction_bonus = 0.25
    else:
        if candidate.rect.y0 >= caption.rect.y1 - 8:
            gap = candidate.rect.y0 - caption.rect.y1
            direction_bonus = 1.0
        elif candidate.rect.y1 <= caption.rect.y0 + 8:
            gap = caption.rect.y0 - candidate.rect.y1
            direction_bonus = 0.45
        else:
            gap = 0.0
            direction_bonus = 0.25

    max_gap = page_rect.height * (0.52 if preferred_above else 0.4)
    if gap > max_gap:
        return None

    if caption.column_id != "full" and x_overlap < 0.05 and center_dx > page_rect.width * 0.22:
        return None

    size_bonus = min(candidate.rect.get_area() / max(page_rect.get_area(), 1.0), 0.30) * 50.0
    return direction_bonus * 100.0 + x_overlap * 28.0 + size_bonus - gap * 0.20 - center_penalty


def merge_candidate_group(items: list[CandidateRegion]) -> CandidateRegion:
    return CandidateRegion(
        kind=items[0].kind,
        page=items[0].page,
        rect=union_rects(item.rect for item in items),
        column_id="full" if any(item.column_id == "full" for item in items) else items[0].column_id,
        source=" + ".join(dict.fromkeys(src for item in items for src in item.sources)),
        sources=list(dict.fromkeys(src for item in items for src in item.sources)),
        rows=items[0].rows if items[0].kind == "table" else None,
    )


def expand_figure_group(
    caption: Caption,
    anchor_index: int,
    candidates: list[CandidateRegion],
    used_indices: set[int],
) -> tuple[CandidateRegion, list[int]]:
    group_indices = [anchor_index]
    group_rect = fitz.Rect(candidates[anchor_index].rect)
    anchor = candidates[anchor_index]
    preferred_above = anchor.rect.y1 <= caption.rect.y0 + 8

    changed = True
    while changed:
        changed = False
        for index, candidate in enumerate(candidates):
            if index == anchor_index or index in used_indices or index in group_indices:
                continue
            if not compatible_columns(caption, candidate):
                continue
            if preferred_above and candidate.rect.y1 > caption.rect.y0 + 12:
                continue
            if not preferred_above and candidate.rect.y0 < caption.rect.y1 - 12:
                continue

            same_row = vertical_overlap_ratio(candidate.rect, group_rect) >= 0.35
            row_aligned = abs(candidate.rect.y0 - anchor.rect.y0) <= 56 or abs(candidate.rect.y1 - anchor.rect.y1) <= 56
            col_near = rect_horizontal_gap(candidate.rect, group_rect) <= (180 if caption.column_id == "full" else 56)
            stack_near = rect_vertical_gap(candidate.rect, group_rect) <= 36 and horizontal_overlap_ratio(candidate.rect, group_rect) >= 0.18
            if (same_row and col_near) or row_aligned or stack_near:
                group_indices.append(index)
                group_rect |= candidate.rect
                changed = True

    items = [candidates[index] for index in group_indices]
    return merge_candidate_group(items), group_indices


def select_candidate_group(
    caption: Caption,
    candidates: list[CandidateRegion],
    used_indices: set[int],
    page_rect: fitz.Rect,
) -> tuple[CandidateRegion | None, list[int], float]:
    scored: list[tuple[float, int]] = []
    for index, candidate in enumerate(candidates):
        if index in used_indices:
            continue
        score = anchor_score(caption, candidate, page_rect)
        if score is None:
            continue
        scored.append((score, index))

    if not scored:
        return None, [], 0.0

    scored.sort(reverse=True)
    best_score, best_index = scored[0]
    if caption.kind == "figure":
        merged, indices = expand_figure_group(caption, best_index, candidates, used_indices)
    else:
        merged = candidates[best_index]
        indices = [best_index]

    confidence = max(0.0, min(1.0, best_score / 115.0))
    return merged, indices, confidence


def assign_caption_ids(captions: list[Caption], prefix: str) -> None:
    used: dict[str, int] = {}
    counter = 1
    for order, caption in enumerate(captions, start=1):
        caption.order = order
        if caption.number_token:
            candidate_id = f"{prefix}{caption.number_token}"
        else:
            candidate_id = f"{prefix}{counter}"
            counter += 1
        if candidate_id in used:
            used[candidate_id] += 1
            candidate_id = f"{candidate_id}_{used[candidate_id]}"
        else:
            used[candidate_id] = 1
        caption.label_id = candidate_id


def find_neighbor_caption_bound(caption: Caption, page_captions: list[Caption], direction: str) -> float | None:
    same_lane = [item for item in page_captions if item is not caption and (item.column_id == caption.column_id or item.column_id == "full" or caption.column_id == "full")]
    if direction == "above":
        candidates = [item.rect.y1 for item in same_lane if item.rect.y1 <= caption.rect.y0]
        return max(candidates) if candidates else None
    candidates = [item.rect.y0 for item in same_lane if item.rect.y0 >= caption.rect.y1]
    return min(candidates) if candidates else None


def build_fallback_content_rect(caption: Caption, layout: PageLayout, page_captions: list[Caption]) -> fitz.Rect:
    x0, x1 = column_bounds(layout, caption.column_id)
    page_rect = layout.page_rect
    if caption.kind == "figure":
        upper_bound = find_neighbor_caption_bound(caption, page_captions, "above")
        y1 = max(page_rect.y0 + 24, caption.rect.y0 - 4)
        y0 = max(page_rect.y0 + 8, y1 - min(page_rect.height * 0.40, 260))
        if upper_bound is not None:
            y0 = max(y0, upper_bound + 4)
    else:
        lower_bound = find_neighbor_caption_bound(caption, page_captions, "below")
        y0 = min(page_rect.y1 - 28, caption.rect.y1 + 4)
        y1 = min(page_rect.y1 - 8, y0 + min(page_rect.height * 0.34, 230))
        if lower_bound is not None:
            y1 = min(y1, lower_bound - 4)
    rect = fitz.Rect(x0, y0, x1, y1) & page_rect
    if rect.height < max(72.0, page_rect.height * 0.12):
        if caption.kind == "figure":
            rect = fitz.Rect(x0, max(page_rect.y0 + 8, caption.rect.y0 - 140), x1, max(page_rect.y0 + 44, caption.rect.y0 - 4)) & page_rect
        else:
            rect = fitz.Rect(x0, min(page_rect.y1 - 44, caption.rect.y1 + 4), x1, min(page_rect.y1 - 8, caption.rect.y1 + 140)) & page_rect
    return rect


def build_item_from_caption(
    caption: Caption,
    candidate: CandidateRegion | None,
    page: fitz.Page,
    layout: PageLayout,
    page_captions: list[Caption],
    item_dir: Path,
    dpi: int,
    confidence: float,
) -> ExtractedItem:
    assert caption.label_id is not None

    if candidate is None:
        content_rect = build_fallback_content_rect(caption, layout, page_captions)
        body_sources = ["fallback_band"]
        status = "fallback"
        confidence = min(confidence, 0.25)
    else:
        content_rect = clip_rect_to_page(candidate.rect, page.rect, padding=2.0)
        body_sources = candidate.sources or [candidate.source]
        status = "matched"

    crop_rect = clip_rect_to_page(union_rects([caption.rect, content_rect]), page.rect, padding=4.0)
    png_path = item_dir / f"{caption.label_id}.png"
    render_clip(page, crop_rect, png_path, dpi=dpi)

    data_path: str | None = None
    if caption.kind == "table" and candidate is not None and candidate.rows:
        csv_path = item_dir / f"{caption.label_id}.csv"
        write_table_csv(candidate.rows, csv_path)
        data_path = str(csv_path)

    needs_review = status != "matched" or confidence < 0.60
    return ExtractedItem(
        id=caption.label_id,
        kind=caption.kind,
        page=caption.page,
        caption=caption.text,
        caption_bbox=rect_to_bbox(caption.rect),
        content_bbox=rect_to_bbox(content_rect),
        crop_bbox=rect_to_bbox(crop_rect) or (0.0, 0.0, 0.0, 0.0),
        number_token=caption.number_token,
        png_path=str(png_path),
        data_path=data_path,
        body_source=body_sources,
        layout_type=caption.layout_type,
        column_id=caption.column_id,
        confidence=round(confidence, 3),
        status=status,
        needs_review=needs_review,
    )


def item_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    label = str(item.get("id", ""))
    match = re.fullmatch(r"[a-z]+(\d+)(.*)", label, re.IGNORECASE)
    if match:
        return (int(match.group(1)), match.group(2))
    return (10**9, label)


def extract_from_pdf(
    pdf_path: Path,
    output_root: Path,
    dpi: int,
    min_width: float,
    min_height: float,
    min_area_ratio: float,
    max_area_ratio: float,
    detect_vector_figures: bool,
) -> Path:
    pdf_name = safe_stem(pdf_path.stem)
    pdf_output_dir = ensure_dir(output_root / pdf_name)
    figures_dir = ensure_dir(pdf_output_dir / "figures")
    tables_dir = ensure_dir(pdf_output_dir / "tables")

    manifest: dict[str, Any] = {
        "pdf": str(pdf_path),
        "output_dir": str(pdf_output_dir),
        "pages": [],
        "figures": [],
        "tables": [],
        "orphans": [],
    }

    with fitz.open(pdf_path) as doc:
        layouts: dict[int, PageLayout] = {}
        text_lines_by_page: dict[int, list[TextLine]] = {}
        captions_by_page: dict[int, list[Caption]] = {}
        table_candidates_by_page: dict[int, list[CandidateRegion]] = {}
        figure_candidates_by_page: dict[int, list[CandidateRegion]] = {}
        all_figure_captions: list[Caption] = []
        all_table_captions: list[Caption] = []

        for page_number, page in enumerate(doc, start=1):
            page_area = page.rect.get_area()
            min_area = page_area * min_area_ratio
            max_area = page_area * max_area_ratio

            layout = detect_page_layout(page, page_number)
            lines = collect_text_lines(page, page_number, layout)
            captions = collect_captions(lines, layout)
            table_candidates = extract_table_candidates(
                page=page,
                page_number=page_number,
                layout=layout,
                min_width=min_width,
                min_height=min_height,
                min_area=min_area,
                max_area=max_area,
            )
            figure_primitives = extract_figure_primitives(
                page=page,
                page_number=page_number,
                layout=layout,
                min_width=min_width,
                min_height=min_height,
                min_area=min_area,
                max_area=max_area,
                blocked_rects=[candidate.rect for candidate in table_candidates],
                detect_vector_figures=detect_vector_figures,
            )
            figure_candidates = merge_figure_primitives(figure_primitives)

            layouts[page_number] = layout
            text_lines_by_page[page_number] = lines
            captions_by_page[page_number] = captions
            table_candidates_by_page[page_number] = table_candidates
            figure_candidates_by_page[page_number] = figure_candidates

            manifest["pages"].append(
                {
                    "page": page_number,
                    "layout_type": layout.layout_type,
                    "gutter_bbox": rect_to_bbox(fitz.Rect(layout.gutter_x0, page.rect.y0, layout.gutter_x1, page.rect.y1)) if layout.gutter_x0 is not None and layout.gutter_x1 is not None else None,
                    "caption_count": len(captions),
                    "figure_candidate_count": len(figure_candidates),
                    "table_candidate_count": len(table_candidates),
                }
            )

            for caption in captions:
                if caption.kind == "figure":
                    all_figure_captions.append(caption)
                else:
                    all_table_captions.append(caption)

        assign_caption_ids(all_figure_captions, "fig")
        assign_caption_ids(all_table_captions, "tab")

        used_figure_candidate_indices: dict[int, set[int]] = {page: set() for page in figure_candidates_by_page}
        used_table_candidate_indices: dict[int, set[int]] = {page: set() for page in table_candidates_by_page}

        for caption in all_figure_captions:
            page = doc[caption.page - 1]
            layout = layouts[caption.page]
            candidate, indices, confidence = select_candidate_group(
                caption=caption,
                candidates=figure_candidates_by_page.get(caption.page, []),
                used_indices=used_figure_candidate_indices[caption.page],
                page_rect=page.rect,
            )
            for index in indices:
                used_figure_candidate_indices[caption.page].add(index)
            item = build_item_from_caption(
                caption=caption,
                candidate=candidate,
                page=page,
                layout=layout,
                page_captions=captions_by_page[caption.page],
                item_dir=figures_dir,
                dpi=dpi,
                confidence=confidence,
            )
            manifest["figures"].append(asdict(item))

        for caption in all_table_captions:
            page = doc[caption.page - 1]
            layout = layouts[caption.page]
            candidate, indices, confidence = select_candidate_group(
                caption=caption,
                candidates=table_candidates_by_page.get(caption.page, []),
                used_indices=used_table_candidate_indices[caption.page],
                page_rect=page.rect,
            )
            for index in indices:
                used_table_candidate_indices[caption.page].add(index)
            item = build_item_from_caption(
                caption=caption,
                candidate=candidate,
                page=page,
                layout=layout,
                page_captions=captions_by_page[caption.page],
                item_dir=tables_dir,
                dpi=dpi,
                confidence=confidence,
            )
            manifest["tables"].append(asdict(item))

        for page_number, candidates in figure_candidates_by_page.items():
            for index, candidate in enumerate(candidates):
                if index in used_figure_candidate_indices[page_number]:
                    continue
                manifest["orphans"].append(
                    {
                        "kind": candidate.kind,
                        "page": candidate.page,
                        "bbox": rect_to_bbox(candidate.rect),
                        "column_id": candidate.column_id,
                        "source": candidate.sources or [candidate.source],
                    }
                )

        for page_number, candidates in table_candidates_by_page.items():
            for index, candidate in enumerate(candidates):
                if index in used_table_candidate_indices[page_number]:
                    continue
                manifest["orphans"].append(
                    {
                        "kind": candidate.kind,
                        "page": candidate.page,
                        "bbox": rect_to_bbox(candidate.rect),
                        "column_id": candidate.column_id,
                        "source": candidate.sources or [candidate.source],
                    }
                )

    manifest["figures"].sort(key=item_sort_key)
    manifest["tables"].sort(key=item_sort_key)
    manifest["orphans"].sort(key=lambda item: (item["page"], item["kind"], item["bbox"][1], item["bbox"][0]))

    manifest_path = pdf_output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def collect_pdf_files(single_pdf: Path | None, pdf_dir: Path | None) -> list[Path]:
    pdf_files: list[Path] = []
    if single_pdf is not None:
        pdf_files.append(single_pdf)
    if pdf_dir is not None:
        pdf_files.extend(sorted(pdf_dir.glob("*.pdf")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in pdf_files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return [path for path in unique if path.exists()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract figures and tables from PDF files with caption-aware naming such as fig1 and tab1."
    )
    parser.add_argument("--pdf", type=Path, default=None, help="A single PDF file to process.")
    parser.add_argument("--pdf-dir", type=Path, default=None, help="A directory that contains PDF files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220, help="Render DPI for PNG output.")
    parser.add_argument("--min-width", type=float, default=80.0, help="Minimum region width in PDF points.")
    parser.add_argument("--min-height", type=float, default=60.0, help="Minimum region height in PDF points.")
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.01,
        help="Minimum region area as a fraction of page area.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.70,
        help="Maximum region area as a fraction of page area.",
    )
    parser.add_argument(
        "--no-vector-figures",
        action="store_true",
        help="Disable heuristic extraction for vector charts and drawings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_files = collect_pdf_files(args.pdf, args.pdf_dir)
    if not pdf_files:
        raise SystemExit("No PDF files found. Use --pdf or --pdf-dir.")

    output_root = ensure_dir(args.output_dir)
    for pdf_path in pdf_files:
        manifest_path = extract_from_pdf(
            pdf_path=pdf_path,
            output_root=output_root,
            dpi=args.dpi,
            min_width=args.min_width,
            min_height=args.min_height,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            detect_vector_figures=not args.no_vector_figures,
        )
        print(f"Done: {pdf_path}")
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
