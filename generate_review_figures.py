import argparse
import json
import math
import re
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "output_awe_20260426_2212"
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
}

KNOWN_TITLE_TO_CITATION = {
    "An Iterative Learning Approach for Online Flight Path Optimization for Tethered Energy Systems Undergoing Cyclic Spooling Motion": "Cobb 等（2019）",
    "Iterative Learning-Based Path Optimization for Repetitive Path Planning, With Application to 3-D Crosswind Flight of Airborne Wind Energy Systems": "Cobb 等（2020）",
    "Iterative Learning-Based Waypoint Optimization for Repetitive Path Planning, with Application to Airborne Wind Energy Systems": "Cobb 等（2017）",
    "Online Energy Maximization of an Airborne Wind Energy Turbine in Simulated Periodic Flight": "Kehs 等（2018）",
    "Optimizing airborne wind energy with reinforcement learning": "Orzan 等（2023）",
    "Waypoint Optimization using Reinforcement Learning for Maximizing Energy Harvesting of High Altitude Airborne Wind Energy Systems": "Selje 等（2024）",
    "Tension Control using Reinforcement Learning for Airborne Wind Energy Systems": "Selje 等（2026）",
    "Navigation and Flight Control for Airborne Wind Energy Kite": "Zhu 等（2024）",
    "Crosswind Flight Control of an Airborne Wind Energy Kite": "Zhu 等（2026）",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(text: str) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "figure"


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


def wrap_text(text: str, width: float, font_size: int) -> list[str]:
    if not text:
        return [""]
    unit_limit = max(width / (font_size * 0.93), 1.0)
    paragraphs = str(text).split("\n")
    lines: list[str] = []
    for paragraph in paragraphs:
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


def wrap_bullets(items: list[str], width: float, font_size: int) -> list[str]:
    lines: list[str] = []
    indent = "　"
    for item in items:
        wrapped = wrap_text(item, width - 8, font_size)
        if not wrapped:
            continue
        lines.append(f"• {wrapped[0]}")
        for extra in wrapped[1:]:
            lines.append(f"{indent}{extra}")
    return lines


def clamp_items(items: list[str], limit: int) -> list[str]:
    if len(items) <= limit:
        return items
    return items[:limit] + [f"其余 {len(items) - limit} 项见结构化结果"]


def flatten_difference_items(
    value: Any,
    paper_lookup: dict[str, dict[str, str]],
) -> list[str]:
    if not value:
        return []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [str(item) for item in value]

    items: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                paper_id = str(item.get("paper_id", "")).strip()
                prefix = paper_lookup.get(paper_id, {}).get("citation") or paper_id or "文献"
                for diff in item.get("differences", []):
                    items.append(f"{prefix}：{diff}")
            elif isinstance(item, str):
                items.append(item)
    return items


def format_citations(paper_ids: list[str], paper_lookup: dict[str, dict[str, str]]) -> list[str]:
    citations = []
    for paper_id in paper_ids:
        info = paper_lookup.get(str(paper_id), {})
        citations.append(info.get("citation") or info.get("title") or str(paper_id))
    return citations


def build_paper_lookup(input_dir: Path, mapping: dict[str, Any]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    single_dir = input_dir / "single_paper_structures"
    year_by_title: dict[str, str] = {}
    for path in single_dir.glob("*.json"):
        data = load_json(path)
        title = str(data.get("bibliography", {}).get("title", "")).strip()
        year = data.get("bibliography", {}).get("year")
        if title:
            year_by_title[title] = str(year) if year not in (None, "") else ""

    for assignment in mapping.get("paper_assignments", []):
        paper_id = str(assignment.get("paper_id", "")).strip()
        title = str(assignment.get("title", "")).strip()
        year = year_by_title.get(title, "")
        citation = KNOWN_TITLE_TO_CITATION.get(title)
        if not citation:
            short_title = title if len(title) <= 20 else title[:18] + "…"
            citation = f"{short_title}（{year}）" if year else short_title
        lookup[paper_id] = {"title": title, "year": year, "citation": citation}
    return lookup


def build_paper_lookup(input_dir: Path, mapping: dict[str, Any]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    single_dir = input_dir / "single_paper_structures"
    year_by_title: dict[str, str] = {}
    for path in single_dir.glob("*.json"):
        data = load_json(path)
        title = str(data.get("bibliography", {}).get("title", "")).strip()
        year = data.get("bibliography", {}).get("year")
        if title:
            year_by_title[title] = str(year) if year not in (None, "") else ""

    for assignment in mapping.get("paper_assignments", []):
        paper_id = str(assignment.get("paper_id", "")).strip()
        title = str(assignment.get("title", "")).strip()
        year = year_by_title.get(title, "")
        citation = KNOWN_TITLE_TO_CITATION.get(title)
        if not citation:
            short_title = title if len(title) <= 20 else title[:18] + "…"
            citation = f"{short_title}（{year}）" if year else short_title
        lookup[paper_id] = {"title": title, "year": year, "citation": citation}
    return lookup


class SvgCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
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
        stroke_width: float = 1.5,
        radius: float = 18,
    ) -> None:
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
            f'rx="{radius:.1f}" ry="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}"/>'
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

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str,
        stroke_width: float = 2.0,
    ) -> None:
        self.line(x1, y1, x2, y2, stroke, stroke_width)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 10
        ax = x2 - size * math.cos(angle - math.pi / 6)
        ay = y2 - size * math.sin(angle - math.pi / 6)
        bx = x2 - size * math.cos(angle + math.pi / 6)
        by = y2 - size * math.sin(angle + math.pi / 6)
        self.add(
            f'<polygon points="{x2:.1f},{y2:.1f} {ax:.1f},{ay:.1f} {bx:.1f},{by:.1f}" fill="{stroke}"/>'
        )

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
        safe_lines = [escape(str(line)) for line in lines]
        text_anchor = {"start": "start", "middle": "middle", "end": "end"}[anchor]
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{DEFAULT_FONT_STACK}" '
            f'font-size="{font_size}" fill="{fill}" font-weight="{weight}" text-anchor="{text_anchor}">'
        )
        for idx, line in enumerate(safe_lines):
            dy = 0 if idx == 0 else font_size * line_gap
            dy_attr = ' dy="0"' if idx == 0 else f' dy="{dy:.1f}"'
            self.add(f'<tspan x="{x:.1f}"{dy_attr}>{line}</tspan>')
        self.add("</text>")

    def save(self, path: Path) -> None:
        body = "\n".join(self.elements)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'<rect width="{self.width}" height="{self.height}" fill="{COLOR_BG}"/>\n'
            f"{body}\n"
            "</svg>\n"
        )
        path.write_text(svg, encoding="utf-8")


def estimate_text_block_height(line_count: int, font_size: int, line_gap: float = LINE_GAP) -> float:
    if line_count <= 0:
        return 0
    return font_size + (line_count - 1) * font_size * line_gap


def estimate_panel_height(
    title: str,
    items: list[str],
    width: float,
    body_font_size: int,
    title_font_size: int = 24,
) -> int:
    title_lines = wrap_text(title, width - 36, title_font_size)
    body_lines = wrap_bullets(items, width - 34, body_font_size)
    height = 24
    height += estimate_text_block_height(len(title_lines), title_font_size, TITLE_GAP)
    height += 20
    height += estimate_text_block_height(len(body_lines), body_font_size, LINE_GAP)
    height += 28
    return int(height)


def estimate_paragraph_panel_height(
    title: str,
    text: str,
    width: float,
    body_font_size: int,
    title_font_size: int = 24,
) -> int:
    title_lines = wrap_text(title, width - 36, title_font_size)
    body_lines = wrap_text(text, width - 34, body_font_size)
    height = 24
    height += estimate_text_block_height(len(title_lines), title_font_size, TITLE_GAP)
    height += 20
    height += estimate_text_block_height(len(body_lines), body_font_size, LINE_GAP)
    height += 28
    return int(height)


def draw_panel(
    canvas: SvgCanvas,
    x: float,
    y: float,
    width: float,
    title: str,
    items: list[str],
    accent: str,
    font_size: int = 22,
) -> int:
    items = [item for item in items if item]
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


def draw_paragraph_panel(
    canvas: SvgCanvas,
    x: float,
    y: float,
    width: float,
    title: str,
    text: str,
    accent: str,
    font_size: int = 22,
) -> int:
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


def draw_header(
    canvas: SvgCanvas,
    title: str,
    subtitle: str,
    accent: str,
    width: int,
    page_type: str,
) -> int:
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


COMMON_TERM_MAP = {
    "lap_to_lap_closed_path": "按圈闭合航迹",
    "iteration_to_iteration": "跨循环迭代",
    "low_level_attitude_or_actuation": "低层姿态/操纵",
    "high_level_trajectory_parameter": "高层航迹参数",
    "pumping_cycle": "抽水循环",
    "towing": "牵引工况",
    "uniform_constant": "恒定均匀风",
    "turbulent": "湍流风场",
    "online_interaction": "在线交互训练",
    "offline_from_simulator_dataset": "仿真数据离线训练",
    "value_based": "值函数",
    "discrete_choice": "离散动作",
    "discrete_increment_pair": "离散增量动作",
    "reactive_partial_observation": "部分可观测反应式状态",
    "continuous_vector": "连续状态向量",
    "discretized_bins": "离散分箱状态",
    "full_system_pumping_cycle": "完整抽水循环系统",
    "crosswind_control_subsystem_with_optional_planner": "crosswind 子系统",
    "reel-out": "放缆发电阶段",
    "predefined_figure_8": "预定义 figure-8 航迹",
    "wind_lidar_for_ground_truth_only": "风激光仅作真值对照",
}

METRIC_TERM_MAP = {
    "autonomous_duration_61_minutes": "连续自主运行 61 min",
    "net_positive_energy": "实现正净能量输出",
    "trajectory_smoothness_qualitative": "航迹更平滑",
    "azimuth_tracking_hesitation_reduction": "方位跟踪犹豫减弱",
    "wind_alignment_vs_lidar_qualitative": "航迹方位可随风向对齐",
    "action_match_rate_vs_oracle": "与 oracle 的动作匹配率",
    "MSE_vs_oracle": "相对 oracle 的 MSE",
    "training_loss_curve": "训练损失收敛",
    "episode_return_proxy_energy": "回报/能量代理提升",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u0000", "")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pretty_term(value: Any) -> str:
    text = clean_text(value)
    if not text or text.lower() in {"unknown", "not_applicable", "not applicable"}:
        return ""
    if text in COMMON_TERM_MAP:
        return COMMON_TERM_MAP[text]
    if text in METRIC_TERM_MAP:
        return METRIC_TERM_MAP[text]
    text = text.replace("figure_8", "figure-8").replace("figure8", "figure-8")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def shorten_text(text: str, limit: int = 86) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def unique_terms(items: list[str], limit: int = 3, text_limit: int = 78) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = shorten_text(pretty_term(item), text_limit)
        key = re.sub(r"[\W_]+", "", value.lower())
        if not value or not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def join_terms(items: list[str], limit: int = 3, text_limit: int = 30, sep: str = "、") -> str:
    values = unique_terms(items, limit=limit, text_limit=text_limit)
    return sep.join(values)


def build_single_paper_lookup(input_dir: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in (input_dir / "single_paper_structures").glob("*.json"):
        data = load_json(path)
        paper_id = str(data.get("paper_id", "")).strip()
        if paper_id:
            lookup[paper_id] = data
    return lookup


def extract_record_meta(record: dict[str, Any]) -> tuple[str, str, Any]:
    if "paper" in record and isinstance(record["paper"], dict):
        paper = record["paper"]
        return str(paper.get("paper_id", "")).strip(), str(paper.get("title", "")).strip(), paper.get("year")
    if "meta" in record and isinstance(record["meta"], dict):
        meta = record["meta"]
        return str(meta.get("paper_id", "")).strip(), str(meta.get("title", "")).strip(), meta.get("year")
    return str(record.get("paper_id", "")).strip(), str(record.get("title", "")).strip(), record.get("year")


def extract_problem_text(raw_paper: dict[str, Any], fallback: str) -> str:
    if raw_paper:
        problem = clean_text(raw_paper.get("problem_context", {}).get("problem_to_solve", ""))
        if problem:
            return shorten_text(problem, 180)
        task = clean_text(raw_paper.get("task_object", {}).get("research_task", ""))
        if task:
            return shorten_text(task, 180)
    return shorten_text(fallback, 180)


def normalize_diff_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", clean_text(text).lower())


def relative_items(current: list[str], baseline: list[str], limit: int = 3) -> list[str]:
    baseline_keys = {normalize_diff_key(item) for item in baseline if item}
    result = [item for item in current if normalize_diff_key(item) not in baseline_keys]
    result = [item for item in result if item]
    return result[:limit] if result else ["同主线"]


def estimate_flow_box_height(title: str, items: list[str], width: float, body_font_size: int = 19, title_font_size: int = 20) -> int:
    title_lines = wrap_text(title, width - 34, title_font_size)
    body_lines = wrap_bullets(items, width - 32, body_font_size)
    height = 22
    height += estimate_text_block_height(len(title_lines), title_font_size, TITLE_GAP)
    height += 18
    height += estimate_text_block_height(len(body_lines), body_font_size, LINE_GAP)
    height += 24
    return int(height)


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


def estimate_citation_box_height(citation: str, width: float, font_size: int = 20) -> int:
    lines = wrap_text(citation, width - 24, font_size)
    return int(30 + estimate_text_block_height(len(lines), font_size, 1.2) + 30)


def draw_citation_box(canvas: SvgCanvas, x: float, y: float, width: float, height: float, citation: str, accent: str) -> None:
    canvas.rect(x, y, width, height, COLOR_SUBTLE, stroke=accent, stroke_width=1.6, radius=18)
    lines = wrap_text(citation, width - 24, 20)
    content_h = estimate_text_block_height(len(lines), 20, 1.2)
    text_y = y + (height - content_h) / 2 + 18
    canvas.text(x + width / 2, text_y, lines, font_size=20, fill=accent, weight=800, anchor="middle", line_gap=1.2)


def fallback_slot_items(raw_paper: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    inputs = []
    methods = []
    outputs = []
    if raw_paper:
        input_vars = raw_paper.get("inputs", {}).get("input_variables", [])
        data_sources = raw_paper.get("inputs", {}).get("data_sources", [])
        method_family = raw_paper.get("methods", {}).get("method_family", [])
        mechanisms = raw_paper.get("methods", {}).get("key_mechanisms", [])
        model_outputs = raw_paper.get("outputs", {}).get("model_outputs", [])
        key_results = raw_paper.get("evaluation", {}).get("key_results", [])
        if input_vars:
            inputs.append(f"变量：{join_terms(input_vars, limit=3, text_limit=22)}")
        if data_sources:
            inputs.append(f"来源：{join_terms(data_sources, limit=2, text_limit=26)}")
        if method_family:
            methods.append(f"方法：{join_terms(method_family, limit=2, text_limit=24)}")
        if mechanisms:
            methods.append(shorten_text(pretty_term(mechanisms[0]), 72))
        if model_outputs:
            outputs.append(f"输出：{join_terms(model_outputs, limit=2, text_limit=26)}")
        if key_results:
            if isinstance(key_results[0], str):
                outputs.append(f"结果：{shorten_text(pretty_term(key_results[0]), 72)}")
    return inputs[:3], methods[:3], outputs[:3]


def extract_d1_slots(record: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    paper = record["paper"]
    iter_unit = pretty_term(paper.get("directional_summary", {}).get("iteration_unit", ""))
    iter_meas = join_terms(paper.get("inputs_directional", {}).get("iteration_level_measurements", []), limit=2, text_limit=18)
    within = join_terms(paper.get("inputs_directional", {}).get("within_iteration_signals_for_J", []), limit=3, text_limit=18)
    params = join_terms(paper.get("path_parameterization", {}).get("basis_parameters", []), limit=2, text_limit=16) or join_terms(
        paper.get("inputs_directional", {}).get("path_parameter_history", []), limit=2, text_limit=22
    )
    estimator = pretty_term(paper.get("learning_model", {}).get("estimator", {}).get("type", ""))
    response = clean_text(paper.get("learning_model", {}).get("response_surface", {}).get("form", ""))
    path_following = pretty_term(paper.get("lower_level_control", {}).get("path_following", {}).get("method", ""))
    actuation = join_terms(paper.get("lower_level_control", {}).get("actuation_and_inner_loops", []), limit=2, text_limit=18)
    updated = join_terms(paper.get("outputs_directional", {}).get("updated_path_parameters", []), limit=2, text_limit=18)
    artifacts = join_terms(paper.get("outputs_directional", {}).get("learned_model_artifacts", []), limit=2, text_limit=18)
    result = shorten_text(clean_text(paper.get("evaluation", {}).get("key_results_text", "")), 72)
    inputs = [
        f"条件：{iter_unit}" if iter_unit else "",
        f"反馈：{iter_meas}；圈内信号：{within}" if iter_meas or within else "",
        f"待调参数：{params}" if params else "",
    ]
    methods = [
        f"响应面：{estimator}+{shorten_text(response, 34)}" if estimator or response else "",
        f"更新：{shorten_text(pretty_term(paper.get('update_law', {}).get('update_type', '')), 36)}",
        f"下层控制：{path_following}" + (f" + {actuation}" if actuation else "") if path_following or actuation else "",
    ]
    outputs = [
        f"输出参数：{updated}" if updated else "",
        f"学习产物：{artifacts}" if artifacts else "",
        f"结果：{result}" if result else "",
    ]
    return unique_terms(inputs, limit=3, text_limit=78), unique_terms(methods, limit=3, text_limit=78), unique_terms(outputs, limit=3, text_limit=78)


def extract_d2_slots(record: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    wind = record.get("inputs", {}).get("wind_measurement", {})
    bias = wind.get("bias_model", {}).get("value")
    perf = pretty_term(record.get("inputs", {}).get("performance_feedback_signal", {}).get("signal_name", ""))
    index_var = pretty_term(record.get("offline_bank", {}).get("index_variable_and_grid", {}).get("index_variable", ""))
    outputs_bank = join_terms(record.get("outputs", {}).get("selected_setpoint_trajectory", {}).get("representation", "").split("_plus_"), limit=2, text_limit=20, sep="+")
    result = ""
    for item in record.get("evaluation", {}).get("key_results", []):
        if isinstance(item, dict):
            value = item.get("improvement_percent", {}).get("over_stationary")
            if isinstance(value, (int, float)):
                result = f"相对 stationary 净功率提升 {value:g}%"
                break
    inputs = [
        f"测量：{pretty_term(wind.get('measured_variable', ''))}" + (f"，偏差约 {bias:g} m/s" if isinstance(bias, (int, float)) else ""),
        f"反馈：{perf}" if perf else "",
        f"轨迹库索引：{index_var}" if index_var else "",
    ]
    methods = [
        "离线生成 Fourier 轨迹库与周期 T",
        "在线 ES 修正有效风速 w_eff 并插值选轨",
        "饱和约束 + stationary/adaptive/nonadaptive 模式切换",
    ]
    outputs = [
        f"输出设定：{outputs_bank}" if outputs_bank else "输出设定：roll 轨迹 + 周期 T",
        "在线量：w_eff / w_corr / 当前模式",
        f"结果：{result}" if result else "",
    ]
    return unique_terms(inputs, limit=3, text_limit=78), unique_terms(methods, limit=3, text_limit=78), unique_terms(outputs, limit=3, text_limit=78)


def extract_d3_slots(record: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    state_def = record.get("mdp", {}).get("state", {}).get("definition", {})
    observables = state_def.get("observables", [])
    state_text = ""
    if isinstance(state_def.get("dimension"), int) and state_def.get("dimension", 0) >= 20:
        state_text = f"{state_def.get('dimension')}维状态：{join_terms(observables, limit=3, text_limit=18)}"
    else:
        state_text = join_terms(observables, limit=3, text_limit=22)
    wind_type = pretty_term(record.get("environment", {}).get("wind_model", {}).get("type", ""))
    control_level = pretty_term(record.get("task", {}).get("control_level", ""))
    reward = pretty_term(record.get("mdp", {}).get("reward", {}).get("definition", {}).get("primary_signal", ""))
    algo = pretty_term(record.get("rl", {}).get("algorithm", {}).get("instance", ""))
    targets = join_terms(record.get("mdp", {}).get("action", {}).get("definition", {}).get("control_targets", []), limit=2, text_limit=20)
    regime = pretty_term(record.get("training", {}).get("regime", ""))
    policy = pretty_term(record.get("outputs", {}).get("policy_representation", ""))
    action_semantics = shorten_text(pretty_term(record.get("outputs", {}).get("action_semantics", "")), 50)
    key_results = record.get("evaluation", {}).get("key_results", [])
    result = shorten_text(pretty_term(key_results[0]), 64) if key_results else ""
    inputs = [
        f"状态：{state_text}" if state_text else "",
        f"环境：{wind_type}；层级：{control_level}" if wind_type or control_level else "",
        f"目标：{reward}" if reward else "",
    ]
    methods = [
        f"算法：{algo}" if algo else "",
        f"动作：{targets}" + (f"（{action_semantics}）" if action_semantics else "") if targets or action_semantics else "",
        f"训练：{regime}" if regime else "",
    ]
    outputs = [
        f"策略产物：{policy}" if policy else "",
        f"控制输出：{action_semantics}" if action_semantics else "",
        f"结果：{result}" if result else "",
    ]
    return unique_terms(inputs, limit=3, text_limit=78), unique_terms(methods, limit=3, text_limit=78), unique_terms(outputs, limit=3, text_limit=78)


def extract_d4_slots(record: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    paper = record["paper"]
    mdp = paper.get("mdp", {})
    state = mdp.get("state_definition", {})
    action = mdp.get("action_definition", {})
    wind = paper.get("wind_handling", {})
    methods_data = paper.get("method", {})
    training = paper.get("training", {})
    output = paper.get("output", {})
    dim = state.get("state_dim")
    waypoints = state.get("num_waypoints")
    action_levels = action.get("action_levels_N", [])
    low = min(action_levels) if action_levels else None
    high = max(action_levels) if action_levels else None
    algos = " / ".join([pretty_term(item.get("name", "")) for item in methods_data.get("algorithms", []) if item.get("name")])
    results = [
        item.get("percent_reduction_vs_optimal")
        for item in paper.get("evaluation", {}).get("results_by_wind", [])
        if item.get("algorithm") == "A2C" and isinstance(item.get("percent_reduction_vs_optimal"), (int, float))
    ]
    result_text = ""
    if results:
        result_text = f"A2C 相对穷举最优损失 {min(results):.2f}%–{max(results):.2f}%"
    inputs = [
        f"状态：{waypoints} 航点聚合的 {dim} 维向量" if isinstance(dim, int) and isinstance(waypoints, int) else "",
        f"动作：离散张力 {low:g}–{high:g} N" if isinstance(low, (int, float)) and isinstance(high, (int, float)) else "",
        "条件：reel-out + 预定义 figure-8 + 无显式风测量" if wind.get("wind_measurement_in_state") is False else "",
    ]
    methods = [
        "离线 replay buffer + 全动作/多风速采样",
        f"批量强化学习：{algos}" if algos else "",
        f"训练：{paper.get('data', {}).get('offline_buffer', {}).get('collection_policy', '')}" if paper.get("data", {}).get("offline_buffer", {}).get("collection_policy") else "",
    ]
    outputs = [
        f"策略产物：{join_terms(list(output.get('policy_artifacts', {}).keys()), limit=2, text_limit=12, sep=' / ')}" if output.get("policy_artifacts") else "",
        "控制输出：期望系绳张力序列",
        f"结果：{result_text}" if result_text else "",
    ]
    return unique_terms(inputs, limit=3, text_limit=78), unique_terms(methods, limit=3, text_limit=78), unique_terms(outputs, limit=3, text_limit=78)


def extract_d5_slots(record: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    paper = record["paper"]
    sensors = join_terms(record.get("inputs", {}).get("required_sensors_minimal_set", []), limit=3, text_limit=16)
    tether = join_terms(record.get("inputs", {}).get("tether_and_ground_signals", []), limit=2, text_limit=16)
    scope = pretty_term(paper.get("core_setting", {}).get("system_scope", ""))
    wind_usage = pretty_term(record.get("inputs", {}).get("wind_data_usage", ""))
    controller = pretty_term(record.get("control", {}).get("inner_loop_type_and_controlled_variable", {}).get("controller_form", ""))
    controlled = pretty_term(record.get("control", {}).get("inner_loop_type_and_controlled_variable", {}).get("controlled_variable", ""))
    robust = join_terms(record.get("methods", {}).get("robustness_mechanisms", []), limit=3, text_limit=22)
    wind_enabled = record.get("wind_adaptation", {}).get("enabled")
    delta_phi = pretty_term(record.get("wind_adaptation", {}).get("update_target", {}).get("updated_variable", ""))
    primary = join_terms(record.get("outputs", {}).get("primary_commands", []), limit=3, text_limit=20)
    extra = join_terms(record.get("outputs", {}).get("supervisory_and_winch_commands", []), limit=2, text_limit=18)
    metric_terms = join_terms(record.get("evaluation", {}).get("metrics_direction_specific", []), limit=2, text_limit=28)
    methods = [
        "导航/引导：极坐标航迹量 + waypoint/heading 参考",
        f"控制：{controller} {controlled}" if controller or controlled else "",
        (
            f"监督：风向自适应 {delta_phi}"
            if wind_enabled
            else "监督：混合状态机协同绞盘/俯仰"
        ),
    ]
    inputs = [
        f"传感：{sensors}" + (f"；地面/系绳信号：{tether}" if tether else "") if sensors or tether else "",
        f"任务：{scope}" if scope else "",
        f"条件：{wind_usage}" if wind_usage else "",
    ]
    outputs = [
        f"输出命令：{primary}" if primary else "",
        f"系统输出：{extra}" if extra else (f"风向对齐输出：{delta_phi}" if delta_phi else ""),
        f"结果：{metric_terms}" if metric_terms else "",
    ]
    if robust and len(methods) < 3:
        methods.append(f"鲁棒机制：{robust}")
    return unique_terms(inputs, limit=3, text_limit=78), unique_terms(methods, limit=3, text_limit=78), unique_terms(outputs, limit=3, text_limit=78)


def normalize_direction_paper(
    record: dict[str, Any],
    direction_definition: str,
    paper_lookup: dict[str, dict[str, str]],
    single_papers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    paper_id, title, year = extract_record_meta(record)
    raw_paper = single_papers.get(paper_id, {})
    citation = paper_lookup.get(paper_id, {}).get("citation") or (f"{title}（{year}）" if year else title)
    problem = extract_problem_text(raw_paper, direction_definition)
    if "meta" in record and "offline_bank" in record:
        inputs, methods, outputs = extract_d2_slots(record)
    elif "task" in record and "rl" in record and "mdp" in record:
        inputs, methods, outputs = extract_d3_slots(record)
    elif "paper" in record and isinstance(record["paper"], dict) and "directional_summary" in record["paper"]:
        inputs, methods, outputs = extract_d1_slots(record)
    elif "paper" in record and ("inputs" in record or "methods" in record or "outputs" in record):
        inputs, methods, outputs = extract_d5_slots(record)
    elif "paper" in record and isinstance(record["paper"], dict):
        inputs, methods, outputs = extract_d4_slots(record)
    else:
        inputs, methods, outputs = fallback_slot_items(raw_paper)
    if not inputs or not methods or not outputs:
        fb_inputs, fb_methods, fb_outputs = fallback_slot_items(raw_paper)
        inputs = inputs or fb_inputs
        methods = methods or fb_methods
        outputs = outputs or fb_outputs
    return {
        "paper_id": paper_id,
        "title": title,
        "year": year,
        "citation": citation,
        "problem": problem,
        "inputs": inputs[:3],
        "methods": methods[:3],
        "outputs": outputs[:3],
    }


def build_direction_figure(
    output_dir: Path,
    direction: dict[str, Any],
    record: dict[str, Any],
    paper_lookup: dict[str, dict[str, str]],
    single_papers: dict[str, dict[str, Any]],
) -> Path:
    direction_id = str(direction["direction_id"])
    accent = DIRECTION_COLORS.get(direction_id, "#2E86DE")
    figure_title = f"{direction_id}｜{direction['direction_name']}"
    header_bottom_est = MARGIN + max(
        120,
        int(
            24
            + estimate_text_block_height(len(wrap_text(figure_title, PAGE_WIDTH - 2 * MARGIN - 236, 34)), 34, 1.16)
            + 14
            + estimate_text_block_height(len(wrap_text("单方向文献主线：输入 / 方法 / 输出", PAGE_WIDTH - 2 * MARGIN - 48, 19)), 19, 1.28)
            + 24
        ),
    ) + 20

    record_by_id: dict[str, dict[str, Any]] = {}
    for item in record.get("records", []):
        paper_id, _, _ = extract_record_meta(item)
        if paper_id:
            record_by_id[paper_id] = item

    included_ids = [str(paper_id) for paper_id in direction.get("included_paper_ids", []) if str(paper_id) in record_by_id]
    baseline_id = ""
    for paper_id in direction.get("representative_paper_ids", []):
        if str(paper_id) in record_by_id:
            baseline_id = str(paper_id)
            break
    if not baseline_id and included_ids:
        baseline_id = included_ids[0]
    ordered_ids = [baseline_id] + [paper_id for paper_id in included_ids if paper_id != baseline_id] if baseline_id else included_ids
    normalized_papers = [
        normalize_direction_paper(record_by_id[paper_id], direction.get("direction_definition", ""), paper_lookup, single_papers)
        for paper_id in ordered_ids
    ]
    if not normalized_papers:
        raise ValueError(f"{direction_id} 未找到可绘制的方向记录")

    baseline = normalized_papers[0]
    others = normalized_papers[1:]

    citation_w = 170
    column_gap = 26
    slot_w = (PAGE_WIDTH - 2 * MARGIN - citation_w - column_gap * 3) / 3
    problem_h = estimate_paragraph_panel_height("核心问题", baseline["problem"], PAGE_WIDTH - 2 * MARGIN, 21)

    base_titles = ["输入 / 条件", "方法 / 模型", "输出 / 结果"]
    base_heights = [
        estimate_flow_box_height(base_titles[0], baseline["inputs"], slot_w, 19),
        estimate_flow_box_height(base_titles[1], baseline["methods"], slot_w, 19),
        estimate_flow_box_height(base_titles[2], baseline["outputs"], slot_w, 19),
        estimate_citation_box_height(baseline["citation"], citation_w, 20),
    ]
    baseline_row_h = max(base_heights)

    diff_rows: list[dict[str, Any]] = []
    for paper in others:
        diff_inputs = relative_items(paper["inputs"], baseline["inputs"], limit=3)
        diff_methods = relative_items(paper["methods"], baseline["methods"], limit=3)
        diff_outputs = relative_items(paper["outputs"], baseline["outputs"], limit=3)
        row_heights = [
            estimate_flow_box_height("输入差异", diff_inputs, slot_w, 18),
            estimate_flow_box_height("方法差异", diff_methods, slot_w, 18),
            estimate_flow_box_height("输出差异", diff_outputs, slot_w, 18),
            estimate_citation_box_height(paper["citation"], citation_w, 19),
        ]
        diff_rows.append(
            {
                "citation": paper["citation"],
                "inputs": diff_inputs,
                "methods": diff_methods,
                "outputs": diff_outputs,
                "height": max(row_heights),
            }
        )

    gap = 30
    section_gap = 18
    total_height = header_bottom_est + problem_h + gap + 36 + baseline_row_h + 46
    if diff_rows:
        total_height += 42
        total_height += sum(row["height"] for row in diff_rows) + section_gap * (len(diff_rows) - 1) + 30

    canvas = SvgCanvas(PAGE_WIDTH, int(total_height))
    header_bottom = draw_header(
        canvas,
        figure_title,
        "单方向文献主线：输入 / 方法 / 输出",
        accent,
        PAGE_WIDTH,
        "方向图",
    )

    current_y = header_bottom
    draw_paragraph_panel(canvas, MARGIN, current_y, PAGE_WIDTH - 2 * MARGIN, "核心问题", baseline["problem"], accent, font_size=21)
    current_y += problem_h + gap

    canvas.text(MARGIN, current_y, ["典型文献主线"], font_size=27, fill=COLOR_TEXT, weight=800)
    current_y += 18

    input_x = MARGIN
    method_x = input_x + slot_w + column_gap
    output_x = method_x + slot_w + column_gap
    cite_x = output_x + slot_w + column_gap
    row_y = current_y + 18

    draw_flow_box(canvas, input_x, row_y, slot_w, baseline_row_h, "输入 / 条件", baseline["inputs"], "#7DA0FA", "#3159C7", body_font_size=19)
    draw_flow_box(canvas, method_x, row_y, slot_w, baseline_row_h, "方法 / 模型", baseline["methods"], "#76C9AE", "#1E6F5C", body_font_size=19)
    draw_flow_box(canvas, output_x, row_y, slot_w, baseline_row_h, "输出 / 结果", baseline["outputs"], "#F4C57C", "#C97912", body_font_size=19)
    draw_citation_box(canvas, cite_x, row_y, citation_w, baseline_row_h, baseline["citation"], accent)
    mid_y = row_y + baseline_row_h / 2
    canvas.arrow(input_x + slot_w + 8, mid_y, method_x - 8, mid_y, COLOR_AXIS, 2.4)
    canvas.arrow(method_x + slot_w + 8, mid_y, output_x - 8, mid_y, COLOR_AXIS, 2.4)

    current_y = row_y + baseline_row_h + 36

    if diff_rows:
        canvas.text(MARGIN, current_y, ["其余文献相对主线的差异"], font_size=27, fill=COLOR_TEXT, weight=800)
        current_y += 22
        for row in diff_rows:
            row_y = current_y + 16
            row_h = int(row["height"])
            draw_flow_box(canvas, input_x, row_y, slot_w, row_h, "输入差异", row["inputs"], "#DCE7FF", "#3159C7", body_font_size=18)
            draw_flow_box(canvas, method_x, row_y, slot_w, row_h, "方法差异", row["methods"], "#D9F2E8", "#1E6F5C", body_font_size=18)
            draw_flow_box(canvas, output_x, row_y, slot_w, row_h, "输出差异", row["outputs"], "#FCE9C9", "#C97912", body_font_size=18)
            draw_citation_box(canvas, cite_x, row_y, citation_w, row_h, row["citation"], accent)
            mid_y = row_y + row_h / 2
            canvas.arrow(input_x + slot_w + 8, mid_y, method_x - 8, mid_y, COLOR_AXIS, 2.0)
            canvas.arrow(method_x + slot_w + 8, mid_y, output_x - 8, mid_y, COLOR_AXIS, 2.0)
            current_y = row_y + row_h + section_gap

    canvas.text(MARGIN, int(total_height) - 24, ["数据来源：single_paper_structures + direction_records/*.json"], font_size=16, fill=COLOR_MUTED)

    path = output_dir / f"{direction_id}_{slugify(direction['direction_name'])}.svg"
    canvas.save(path)
    return path


def build_cross_figure(
    output_dir: Path,
    mapping: dict[str, Any],
    comparison: dict[str, Any],
    paper_lookup: dict[str, dict[str, str]],
) -> Path:
    width = 1980
    accent = "#1E6F5C"
    title = "跨方向比较｜系留风筝系统控制文献谱系"
    subtitle = "基于五个研究方向的共同点、时间尺度、决策对象与工程落地性对比"

    header_bottom_est = MARGIN + max(
        120,
        int(
            24
            + estimate_text_block_height(len(wrap_text(title, width - 2 * MARGIN - 236, 34)), 34, 1.16)
            + 14
            + estimate_text_block_height(len(wrap_text(subtitle, width - 2 * MARGIN - 48, 19)), 19, 1.28)
            + 24
        ),
    ) + 20

    common_x = MARGIN
    common_y = header_bottom_est
    common_w = 1220
    legend_w = width - 2 * MARGIN - common_w - 24
    common_items = clamp_items(comparison.get("cross_direction_commonalities", []), 5)
    review_items = clamp_items(comparison.get("suggested_review_structure", []), 4)
    common_h_est = estimate_panel_height("跨方向共同点", common_items, common_w, 21)
    review_h_est = estimate_panel_height("组织视角", review_items, legend_w, 20)
    top_row_h = max(common_h_est, review_h_est)

    map_y = common_y + top_row_h + 34
    map_h = 660
    map_w = width - 2 * MARGIN

    diff_by_id = {str(item.get("direction_id")): item for item in comparison.get("cross_direction_differences", [])}
    positions = {
        "D1": (0.60, 0.16),
        "D2": (0.38, 0.38),
        "D3": (0.74, 0.48),
        "D4": (0.88, 0.78),
        "D5": (0.18, 0.88),
    }

    detail_gap = 26
    detail_card_width = width - 2 * MARGIN
    detail_cards: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for direction in mapping.get("directions", []):
        direction_id = str(direction.get("direction_id"))
        info = diff_by_id.get(direction_id, {})
        title_lines = wrap_text(f"{direction_id} {direction.get('direction_name')}", detail_card_width - 36, 18)
        common_lines = wrap_bullets(clamp_items(info.get("what_is_common_with_others", []), 2), detail_card_width - 30, 16)
        diff_lines = wrap_bullets(clamp_items(info.get("what_is_different", []), 2), detail_card_width - 30, 16)
        role_lines = wrap_text(str(info.get("role_in_review", "")), detail_card_width - 30, 15)[:5]
        height_est = 24
        height_est += estimate_text_block_height(len(title_lines), 18, TITLE_GAP)
        height_est += 18
        height_est += 22 + estimate_text_block_height(len(common_lines), 16, LINE_GAP)
        height_est += 18
        height_est += 22 + estimate_text_block_height(len(diff_lines), 16, LINE_GAP)
        height_est += 18
        height_est += 20 + estimate_text_block_height(len(role_lines), 15, LINE_GAP)
        height_est += 24
        detail_cards.append((direction, info, int(height_est)))

    detail_heights = [card[2] for card in detail_cards]
    footer_y_est = map_y + map_h + 34 + sum(detail_heights) + detail_gap * (len(detail_heights) - 1) + 34
    viz_items = clamp_items(comparison.get("suggested_visualizations", []), 4)
    footer_h_est = estimate_panel_height("可进一步延展的比较维度", viz_items, width - 2 * MARGIN, 19)
    height = int(footer_y_est + footer_h_est + 60)

    canvas = SvgCanvas(width, height)
    draw_header(canvas, title, subtitle, accent, width, "总图")
    draw_panel(canvas, common_x, common_y, common_w, "跨方向共同点", common_items, accent, font_size=21)
    draw_panel(canvas, common_x + common_w + 24, common_y, legend_w, "组织视角", review_items, "#2E86DE", font_size=20)

    canvas.rect(MARGIN, map_y, map_w, map_h, COLOR_PANEL, radius=26)
    canvas.text(MARGIN + 22, map_y + 38, ["五方向定位图"], font_size=28, fill=COLOR_TEXT, weight=800)

    plot_x = MARGIN + 120
    plot_y = map_y + 90
    plot_w = map_w - 180
    plot_h = map_h - 170
    canvas.arrow(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, COLOR_AXIS, 2.5)
    canvas.arrow(plot_x, plot_y + plot_h, plot_x, plot_y, COLOR_AXIS, 2.5)
    canvas.text(plot_x + plot_w / 2, plot_y + plot_h + 48, ["时间尺度：实时控制 → 按圈/按循环自适应 → 离线训练/在线执行"], font_size=20, fill=COLOR_MUTED, anchor="middle")
    canvas.text(plot_x - 64, plot_y + plot_h / 2, ["决策对象层级", "航迹/索引/策略/张力/系统"], font_size=20, fill=COLOR_MUTED, anchor="middle")

    x_labels = [("实时", 0.12), ("在线调参", 0.42), ("跨循环学习", 0.58), ("离线训练", 0.86)]
    for label, ratio in x_labels:
        x = plot_x + plot_w * ratio
        canvas.line(x, plot_y + plot_h, x, plot_y + plot_h + 10, COLOR_AXIS, 1.8)
        canvas.text(x, plot_y + plot_h + 28, [label], font_size=18, fill=COLOR_MUTED, anchor="middle")

    y_labels = [("航迹几何", 0.18), ("索引/参数", 0.40), ("策略学习", 0.58), ("张力管理", 0.78), ("系统集成", 0.92)]
    for label, ratio in y_labels:
        y = plot_y + plot_h * ratio
        canvas.line(plot_x - 10, y, plot_x, y, COLOR_AXIS, 1.8)
        canvas.text(plot_x - 20, y + 6, [label], font_size=18, fill=COLOR_MUTED, anchor="end")

    for direction in mapping.get("directions", []):
        direction_id = str(direction.get("direction_id"))
        info = diff_by_id.get(direction_id, {})
        rx, ry = positions.get(direction_id, (0.5, 0.5))
        map_card_w = 228
        map_card_h = 148
        x = plot_x + plot_w * rx - map_card_w / 2
        y = plot_y + plot_h * ry - map_card_h / 2
        accent_local = DIRECTION_COLORS.get(direction_id, "#2E86DE")
        canvas.rect(x, y, map_card_w, map_card_h, COLOR_PANEL, stroke=accent_local, stroke_width=2.0, radius=20)
        canvas.rect(x, y, map_card_w, 14, accent_local, stroke=accent_local, stroke_width=0, radius=18)
        canvas.text(x + 16, y + 38, [direction_id], font_size=20, fill=accent_local, weight=800)
        name_lines = wrap_text(str(direction.get("direction_name", "")), map_card_w - 32, 15)[:2]
        canvas.text(x + 16, y + 60, name_lines, font_size=15, fill=COLOR_TEXT, weight=700)
        focus_lines = wrap_text(str(info.get("main_focus", "")), map_card_w - 32, 12)[:2]
        focus_y = y + 60 + estimate_text_block_height(len(name_lines), 15, LINE_GAP) + 10
        canvas.text(x + 16, focus_y, focus_lines, font_size=12, fill=COLOR_MUTED)

    diff_y = map_y + map_h + 36
    canvas.text(MARGIN, diff_y - 8, ["五方向差异卡片"], font_size=28, fill=COLOR_TEXT, weight=800)

    def draw_detail_card(x: float, y: float, width_: float, direction: dict[str, Any], info: dict[str, Any], height_: int) -> None:
        direction_id = str(direction.get("direction_id"))
        accent_local = DIRECTION_COLORS.get(direction_id, "#2E86DE")
        canvas.rect(x, y, width_, height_, COLOR_PANEL, stroke=accent_local, stroke_width=2.0, radius=22)
        canvas.rect(x, y, width_, 14, accent_local, stroke=accent_local, stroke_width=0, radius=18)
        title_lines = wrap_text(f"{direction_id} {direction.get('direction_name')}", width_ - 36, 18)
        canvas.text(x + 16, y + 38, title_lines, font_size=18, fill=accent_local, weight=800, line_gap=TITLE_GAP)
        title_h = estimate_text_block_height(len(title_lines), 18, TITLE_GAP)

        common_lines = wrap_bullets(clamp_items(info.get("what_is_common_with_others", []), 2), width_ - 30, 16)
        common_title_y = y + 38 + title_h + 18
        canvas.text(x + 16, common_title_y, ["与其他方向相通"], font_size=17, fill=COLOR_TEXT, weight=700)
        common_body_y = common_title_y + 28
        canvas.text(x + 16, common_body_y, common_lines, font_size=16, fill=COLOR_TEXT)
        common_h = estimate_text_block_height(len(common_lines), 16, LINE_GAP)

        split_y = common_body_y + common_h + 18
        canvas.line(x + 14, split_y, x + width_ - 14, split_y, COLOR_BORDER, 1.4)
        diff_lines = wrap_bullets(clamp_items(info.get("what_is_different", []), 2), width_ - 30, 16)
        diff_title_y = split_y + 28
        canvas.text(x + 16, diff_title_y, ["本方向独特性"], font_size=17, fill=COLOR_TEXT, weight=700)
        diff_body_y = diff_title_y + 28
        canvas.text(x + 16, diff_body_y, diff_lines, font_size=16, fill=COLOR_TEXT)
        diff_h = estimate_text_block_height(len(diff_lines), 16, LINE_GAP)

        role_title_y = diff_body_y + diff_h + 20
        role_lines = wrap_text(str(info.get("role_in_review", "")), width_ - 30, 15)[:5]
        canvas.text(x + 16, role_title_y, ["综述中的角色"], font_size=16, fill=COLOR_MUTED, weight=700)
        canvas.text(x + 16, role_title_y + 24, role_lines, font_size=15, fill=COLOR_MUTED)

    next_y = diff_y + 20
    for direction, info, card_height in detail_cards:
        draw_detail_card(MARGIN, next_y, detail_card_width, direction, info, card_height)
        next_y += card_height + detail_gap

    footer_y = next_y + 8
    draw_panel(canvas, MARGIN, footer_y, width - 2 * MARGIN, "可进一步延展的比较维度", viz_items, "#4C9A5F", font_size=19)
    canvas.text(MARGIN, height - 24, ["数据来源：direction_mapping.json + cross_direction_comparison.json"], font_size=16, fill=COLOR_MUTED)

    path = output_dir / "cross_direction_overview.svg"
    canvas.save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="基于结构化输出结果生成综述 SVG 图")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or (input_dir / "review_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_json(input_dir / "directions" / "direction_mapping.json")
    comparison = load_json(input_dir / "comparisons" / "cross_direction_comparison.json")
    paper_lookup = build_paper_lookup(input_dir, mapping)
    single_papers = build_single_paper_lookup(input_dir)

    records_by_id: dict[str, dict[str, Any]] = {}
    for path in (input_dir / "direction_records").glob("*.json"):
        record = load_json(path)
        records_by_id[str(record.get("direction_id"))] = record

    created: list[Path] = []
    for direction in mapping.get("directions", []):
        direction_id = str(direction.get("direction_id"))
        record = records_by_id.get(direction_id)
        if not record:
            continue
        created.append(build_direction_figure(output_dir, direction, record, paper_lookup, single_papers))

    created.append(build_cross_figure(output_dir, mapping, comparison, paper_lookup))

    print("已生成图文件：")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
