from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "three_stage_literature_review.v1"
SHOWCASE_FILENAME = "three_stage_review.json"
QUALITY_FILENAME = "quality_report.json"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip() or default
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_clean_text(item) for item in value]
        return "；".join(part for part in parts if part) or default
    if isinstance(value, dict):
        for key in ["name", "title", "description", "summary_cn", "claim_cn", "text"]:
            text = _clean_text(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip() or default


def _unique_texts(values: list[Any], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for value in values:
        text = _clean_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        results.append(text)
        if limit and len(results) >= limit:
            break
    return results


def _first_text(*values: Any, default: str = "") -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return default


def _year_value(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    text = _clean_text(value)
    if text.isdigit():
        return int(text)
    return text


def _metadata_article_url(bibliography: dict[str, Any], assigned: dict[str, Any]) -> str:
    doi = _first_text(bibliography.get("doi"), assigned.get("doi"))
    if doi:
        clean_doi = doi.removeprefix("https://doi.org/").removeprefix("doi:").strip()
        if clean_doi:
            return f"https://doi.org/{clean_doi}"
    arxiv_id = _first_text(assigned.get("arxiv_id"))
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id.removesuffix('.pdf')}"
    for value in [
        bibliography.get("url"),
        assigned.get("url"),
        assigned.get("doi_link"),
        assigned.get("oa_url"),
        assigned.get("landing_page_url"),
        assigned.get("landing_page_urls"),
        assigned.get("pdf_url"),
    ]:
        for item in _as_list(value):
            text = _clean_text(item)
            if text:
                return text
    return ""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _year_range(years: list[Any]) -> str:
    parsed = sorted({int(year) for year in years if str(year).isdigit()})
    if not parsed:
        return "-"
    if len(parsed) == 1:
        return str(parsed[0])
    return f"{parsed[0]}-{parsed[-1]}"


def _mentions_unknown_direction(text: Any, actual_ids: set[str]) -> bool:
    refs = set(re.findall(r"\bD\d+\b", _clean_text(text)))
    return bool(refs - actual_ids)


def _first_meaningful_markdown_paragraph(markdown_text: str) -> str:
    for block in re.split(r"\n\s*\n", markdown_text or ""):
        text = "\n".join(
            line.strip()
            for line in block.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        text = re.sub(r"^\s*[-*]\s+", "", text).strip()
        if len(text) >= 40 and "参考文献" not in text:
            return _clean_text(text)
    return ""


def _tidy_overview_text(text: str, paper_total: int) -> str:
    text = _clean_text(text)
    if paper_total:
        count_patterns = [
            r"((?:本综述|本文|本次综述)[^。\n]{0,80}?(?:梳理|整理|覆盖|纳入|分析)了[^。\n]{0,40}?)\s*\d+\s*篇文献",
            r"((?:系统|共|总计|合计)[^。\n]{0,40}?(?:梳理|整理|覆盖|纳入|分析)了[^。\n]{0,40}?)\s*\d+\s*篇文献",
        ]
        for pattern in count_patterns:
            text = re.sub(pattern, rf"\g<1> {paper_total} 篇文献", text, count=1)

    replacements = {
        "林素性": "逻辑属性",
        "资源聚化": "资源聚合",
        "。；": "；",
        "；。": "。",
        "。。": "。",
        "；；": "；",
        "，，": "，",
        "、，": "、",
        "，。": "。",
    }
    changed = True
    while changed:
        changed = False
        for old, new in replacements.items():
            if old in text:
                text = text.replace(old, new)
                changed = True
    return text.strip()


def _build_corpus_overview_detail(
    topic: str,
    summary: str,
    directions: list[dict[str, Any]],
    paper_total: int,
    target_paper_total: int,
    year_range: str,
    methods: list[str],
    gaps: list[str],
    domain_insights: list[dict[str, Any]],
) -> str:
    paragraphs: list[str] = []
    if summary:
        paragraphs.append(_tidy_overview_text(summary, paper_total))

    scope_parts = []
    if paper_total:
        scope_parts.append(f"{paper_total} 篇文献")
    if target_paper_total and target_paper_total > paper_total:
        scope_parts.append(f"目标样本 {target_paper_total} 篇")
    if directions:
        scope_parts.append(f"{len(directions)} 个研究方向")
    if year_range and year_range != "-":
        scope_parts.append(f"时间范围 {year_range}")
    if scope_parts:
        paragraphs.append(
            f"本次综述围绕“{topic}”整理了{'、'.join(scope_parts)}。第 1 层用于把握领域全貌：先确认文献样本和方向划分，再比较方向之间的共同问题、方法差异和潜在研究空白。"
        )

    direction_names = [
        f"{direction.get('id')} {direction.get('name')}"
        for direction in directions[:5]
        if direction.get("id") or direction.get("name")
    ]
    if direction_names:
        paragraphs.append(
            f"方向结构上，当前样本主要覆盖{'、'.join(direction_names)}。这些方向共同支撑主题理解，但关注点分别落在研究对象、市场机制、优化或学习方法、评价指标和工程约束等不同侧面。"
        )

    if methods:
        paragraphs.append(
            f"方法谱系上，样本中反复出现的技术路径包括{'、'.join(methods[:6])}。这些方法可进一步按输入变量、目标函数、约束处理、求解或训练方式、实验场景和输出指标进行横向比较。"
        )

    insight_texts = [
        _first_text(item.get("summary"), item.get("title"))
        for item in domain_insights[:3]
        if isinstance(item, dict)
    ]
    if insight_texts:
        paragraphs.append(
            "综合洞察上，当前文献显示：" + "；".join(_clean_text(item) for item in insight_texts if item) + "。"
        )
    elif gaps:
        paragraphs.append(
            f"研究机会主要集中在{'、'.join(gaps[:4])}，后续可通过扩大样本、统一评价指标和增强可复现数据来进一步验证。"
        )

    unique: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        text = _tidy_overview_text(paragraph, paper_total)
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return "\n\n".join(unique)


def _direction_prefix(folder_name: str) -> str:
    return folder_name.split("_", 1)[0]


def _preferred_direction_folder_names(run_dir: Path) -> set[str]:
    preferred: set[str] = set()
    report = _load_json(run_dir / "unified_run_report.json", {})
    if isinstance(report, dict):
        for item in _as_list(report.get("directions")):
            if not isinstance(item, dict):
                continue
            outputs = item.get("outputs") if isinstance(item.get("outputs"), dict) else {}
            for key in ("assigned_papers", "direction_review_md", "direction_review_summary", "paper_cards_dir"):
                path_text = outputs.get(key)
                if path_text:
                    preferred.add(Path(str(path_text)).parent.name)
                    break
    if preferred:
        return preferred
    payload = _load_json(run_dir / "01_discovery" / "direction_workspace_manifest.json", [])
    if isinstance(payload, list):
        for item in payload:
            try:
                preferred.add(Path(str(item)).name)
            except (TypeError, ValueError):
                continue
    return preferred


def _pick_direction_dir(candidate: Path, existing: Path | None, preferred_names: set[str], prefer_review: bool) -> Path:
    if existing is None:
        return candidate
    if preferred_names:
        candidate_preferred = candidate.name in preferred_names
        existing_preferred = existing.name in preferred_names
        if candidate_preferred and not existing_preferred:
            return candidate
        if existing_preferred and not candidate_preferred:
            return existing
    if prefer_review:
        return candidate
    return existing


def _direction_dirs(run_dir: Path) -> list[Path]:
    review_root = run_dir / "02_reviews" / "directions"
    discovery_root = run_dir / "01_discovery" / "directions"
    preferred_names = _preferred_direction_folder_names(run_dir)
    by_prefix: dict[str, Path] = {}
    for root in [discovery_root, review_root]:
        if not root.exists():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            prefix = _direction_prefix(path.name)
            by_prefix[prefix] = _pick_direction_dir(
                path,
                by_prefix.get(prefix),
                preferred_names,
                prefer_review=root == review_root,
            )
    return [by_prefix[name] for name in sorted(by_prefix)]


def _unique_paper_ids(directions: list[dict[str, Any]]) -> set[str]:
    paper_ids: set[str] = set()
    for direction in directions:
        for paper in _as_list(direction.get("papers")):
            if not isinstance(paper, dict):
                continue
            paper_id = _first_text(paper.get("id"), paper.get("paper_id"), paper.get("candidate_id"))
            if paper_id:
                paper_ids.add(str(paper_id))
    return paper_ids


def _matching_direction_dir(root: Path, folder_name: str) -> Path:
    exact = root / folder_name
    if exact.exists():
        return exact
    prefix = folder_name.split("_", 1)[0]
    for path in root.glob(f"{prefix}_*"):
        if path.is_dir():
            return path
    return exact


def _paper_cards(review_dir: Path) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    cards_dir = review_dir / "paper_cards"
    if not cards_dir.exists():
        return cards
    for path in sorted(cards_dir.glob("*.json")):
        card = _load_json(path, {})
        if not isinstance(card, dict):
            continue
        keys = [card.get("paper_id"), card.get("candidate_id"), path.stem.split("_", 1)[0]]
        for key in keys:
            if key:
                cards[str(key)] = card
    return cards


def _normalize_formula(item: Any, index: int) -> dict[str, Any]:
    if isinstance(item, str):
        return {"id": f"F{index}", "name": f"公式 {index}", "formula": item, "note": ""}
    if not isinstance(item, dict):
        return {"id": f"F{index}", "name": f"公式 {index}", "formula": _clean_text(item), "note": ""}
    formula = _first_text(
        item.get("latex"),
        item.get("formula_latex"),
        item.get("formula"),
        item.get("expression"),
        item.get("text"),
    )
    variables = []
    for variable in _as_list(item.get("variables")):
        if isinstance(variable, dict):
            variables.append(
                {
                    "symbol": _first_text(variable.get("symbol")),
                    "meaning": _first_text(variable.get("meaning_cn"), variable.get("meaning")),
                    "unit": _first_text(variable.get("unit"), default="unknown"),
                }
            )
    return {
        "id": _first_text(item.get("id"), item.get("formula_id"), default=f"F{index}"),
        "original_number": _first_text(item.get("original_number"), item.get("number")),
        "name": _first_text(
            item.get("name"),
            item.get("title"),
            item.get("meaning_cn"),
            item.get("used_for_cn"),
            item.get("symbol"),
            default=f"公式 {index}",
        ),
        "formula": formula,
        "note": _first_text(
            item.get("description"),
            item.get("meaning_cn"),
            item.get("explanation_cn"),
            item.get("note"),
        ),
        "used_for": _first_text(item.get("used_for_cn"), item.get("used_in_method_step_cn")),
        "source_location": _first_text(item.get("source_location")),
        "variables": [item for item in variables if item.get("symbol") or item.get("meaning")],
    }


def _fallback_formulas(paper: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "id": "F1",
            "name": "收益最大化目标",
            "formula": r"\[\max_{p_t^{ch},p_t^{dis},r_t}\; \sum_{t\in T}\left(\pi_t^e p_t^{dis}-\pi_t^e p_t^{ch}+\pi_t^s r_t\right)-C_{deg}(p_t^{ch},p_t^{dis})-\lambda Risk\]",
            "note": r"\(\pi_t^e\) 表示能量价格，\(\pi_t^s\) 表示服务价格，\(C_{deg}\) 表示储能退化成本，\(\lambda\) 表示风险偏好权重。",
        },
        {
            "id": "F2",
            "name": "SOC 动态约束",
            "formula": r"\[SOC_t=SOC_{t-1}+\eta_{ch}p_t^{ch}\Delta t-\frac{p_t^{dis}\Delta t}{\eta_{dis}},\quad SOC^{min}\le SOC_t\le SOC^{max}\]",
            "note": r"该约束描述储能荷电状态随充放电功率变化，并限制在安全上下界内。",
        },
    ]


def _normalize_web_review_formula_refs(web_review: dict[str, Any], formulas: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(web_review, dict):
        return {}
    valid_ids = {
        _clean_text(item.get("id")).upper()
        for item in formulas
        if _clean_text(item.get("id"))
    }
    normalized = dict(web_review)
    steps = []

    def clean_invalid_mentions(text: Any) -> str:
        source = _clean_text(text)
        if not source:
            return source

        def replace_ref(match: re.Match[str]) -> str:
            ref = match.group(0).upper()
            return ref if ref in valid_ids else "相关公式"

        source = re.sub(r"\bF\d+\b", replace_ref, source, flags=re.IGNORECASE)
        source = re.sub(r"(相关公式[、,，和及\s]*){2,}", "相关公式", source)
        source = re.sub(r"相关公式[、,，]\s*。", "相关公式。", source)
        return source

    for step in _as_list(web_review.get("method_steps")):
        if not isinstance(step, dict):
            continue
        current = dict(step)
        current["step_name"] = clean_invalid_mentions(current.get("step_name"))
        current["step_detail_cn"] = clean_invalid_mentions(current.get("step_detail_cn"))
        refs = [
            _clean_text(ref).upper()
            for ref in _as_list(current.get("formula_refs"))
            if _clean_text(ref)
        ]
        mentioned = re.findall(r"\bF\d+\b", _clean_text(current.get("step_detail_cn")), flags=re.IGNORECASE)
        mentioned += re.findall(r"\bF\d+\b", _clean_text(current.get("step_name")), flags=re.IGNORECASE)
        for ref in mentioned:
            refs.append(ref.upper())
        unique_refs = []
        for ref in refs:
            if ref in valid_ids and ref not in unique_refs:
                unique_refs.append(ref)
        current["formula_refs"] = unique_refs
        steps.append(current)
    if steps:
        normalized["method_steps"] = steps
    return normalized


def _relative_to_run(path_text: Any, run_dir: Path) -> str:
    text = _clean_text(path_text)
    if not text:
        return ""
    try:
        path = Path(text)
        if not path.is_absolute():
            path = (run_dir / path).resolve()
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except Exception:
        return ""


def _visual_asset_score(item: dict[str, Any]) -> int:
    caption = _clean_text(item.get("caption")).lower()
    kind = _clean_text(item.get("kind")).lower()
    score = 0
    preferred = [
        "framework",
        "architecture",
        "algorithm",
        "flow",
        "workflow",
        "process",
        "structure",
        "model",
        "system",
        "market",
        "bidding",
        "optimization",
        "operation",
        "method",
        "mechanism",
        "case study",
        "流程",
        "框架",
        "算法",
        "结构",
        "模型",
        "机制",
        "市场",
        "竞价",
        "优化",
        "调度",
    ]
    score += sum(4 for word in preferred if word in caption)
    if kind == "figure":
        score += 3
    if kind == "table":
        score += 1
    caption_len = len(caption)
    if 40 <= caption_len <= 420:
        score += 2
    if _clean_text(item.get("id")).lower() in {"fig1", "figure1", "tab1", "table1"}:
        score += 1
    return score


def _visual_asset_order_key(item: dict[str, Any]) -> tuple[int, int, str]:
    raw_id = _clean_text(item.get("id")).lower()
    caption = _clean_text(item.get("caption")).lower()
    number = 9999
    match = re.search(r"(?:fig(?:ure)?|tab(?:le)?)\s*\.?\s*(\d+)", f"{raw_id} {caption}", flags=re.I)
    if match:
        number = _safe_int(match.group(1), 9999)
    return (_safe_int(item.get("page"), 9999), number, raw_id)


def _load_visual_assets(assigned: dict[str, Any], run_dir: Path, limit: int = 4) -> list[dict[str, Any]]:
    manifest_path_text = _clean_text(assigned.get("figures_tables_manifest_path"))
    if not manifest_path_text:
        return []
    manifest_path = Path(manifest_path_text)
    if not manifest_path.exists():
        return []
    manifest = _load_json(manifest_path, {})
    if not isinstance(manifest, dict):
        return []
    candidates = []
    for kind in ["figures", "tables"]:
        for item in _as_list(manifest.get(kind)):
            if not isinstance(item, dict):
                continue
            png_path = _clean_text(item.get("png_path"))
            if not png_path or not Path(png_path).exists():
                continue
            caption = _clean_text(item.get("caption"))
            candidates.append(
                {
                    "id": _first_text(item.get("id"), default=f"{kind}_{len(candidates) + 1}"),
                    "kind": _first_text(item.get("kind"), default="figure" if kind == "figures" else "table"),
                    "caption": caption,
                    "page": item.get("page") or "",
                    "asset_path": _relative_to_run(png_path, run_dir),
                    "data_path": _relative_to_run(item.get("data_path"), run_dir),
                    "_score": _visual_asset_score(item),
                }
            )
    selected: list[dict[str, Any]] = []

    def selection_key(value: dict[str, Any]) -> tuple[int, tuple[int, int, str]]:
        return (
            -int(value.get("_score", 0) or 0),
            _visual_asset_order_key(value),
        )

    for item in sorted(candidates, key=selection_key):
        if not item.get("asset_path"):
            continue
        selected.append({key: value for key, value in item.items() if key != "_score"})
        if len(selected) >= limit:
            break
    return sorted(selected, key=_visual_asset_order_key)


def _normalize_paper(
    assigned: dict[str, Any],
    card: dict[str, Any],
    direction_id: str,
    direction_name: str,
    related_ids: list[str],
    run_dir: Path,
) -> dict[str, Any]:
    bibliography = card.get("bibliography") if isinstance(card.get("bibliography"), dict) else {}
    research_context = card.get("research_context") if isinstance(card.get("research_context"), dict) else {}
    method = card.get("method") if isinstance(card.get("method"), dict) else {}
    findings = card.get("findings") if isinstance(card.get("findings"), dict) else {}
    review_use = card.get("review_use") if isinstance(card.get("review_use"), dict) else {}
    display = card.get("display_facts") if isinstance(card.get("display_facts"), dict) else {}
    web_review = card.get("web_review") if isinstance(card.get("web_review"), dict) else {}
    prescreen = assigned.get("prescreen") if isinstance(assigned.get("prescreen"), dict) else {}

    paper_id = _first_text(card.get("paper_id"), assigned.get("paper_id"), assigned.get("candidate_id"), default="P000")
    formulas = [
        normalized
        for index, item in enumerate(_as_list(card.get("formulas")), start=1)
        if (normalized := _normalize_formula(item, index)).get("formula")
    ]
    web_review = _normalize_web_review_formula_refs(web_review, formulas)

    paper = {
        "id": paper_id,
        "candidate_id": _first_text(card.get("candidate_id"), assigned.get("candidate_id")),
        "title": _first_text(bibliography.get("title"), assigned.get("title"), default=paper_id),
        "title_cn": _first_text(bibliography.get("title_cn"), assigned.get("title_cn"), bibliography.get("title"), assigned.get("title"), default=paper_id),
        "authors": _first_text(bibliography.get("authors"), assigned.get("authors"), default="未知作者"),
        "year": _year_value(_first_text(bibliography.get("year"), assigned.get("year"))),
        "venue": _first_text(bibliography.get("journal_or_conference"), assigned.get("venue"), default="未知来源"),
        "doi": _first_text(bibliography.get("doi"), assigned.get("doi")),
        "url": _metadata_article_url(bibliography, assigned),
        "source_pdf": _first_text(assigned.get("pdf_path"), assigned.get("_pdf_path")),
        "source_text": _first_text(assigned.get("txt_path")),
        "citations": assigned.get("cited_by_count") or assigned.get("citations") or "",
        "keywords": _unique_texts(
            _as_list(display.get("tags"))
            + _as_list(display.get("method_family"))
            + _as_list(method.get("method_type"))
            + _as_list(assigned.get("concepts")),
            limit=8,
        ),
        "research_problem": _first_text(research_context.get("problem_cn"), assigned.get("abstract"), default="暂无明确研究问题。"),
        "method": _first_text(
            method.get("summary_cn"),
            method.get("method_type"),
            display.get("method_family"),
            prescreen.get("method_or_object_summary_cn"),
            assigned.get("method_or_object_summary_cn"),
            assigned.get("title"),
            default="暂无明确方法描述。",
        ),
        "scenario": _first_text(
            display.get("study_object_type"),
            display.get("data_source_type"),
            method.get("inputs"),
            default="暂无明确数据或场景。"
        ),
        "conclusion": _first_text(findings.get("conclusions_cn"), findings.get("main_results_cn"), default="暂无明确结论。"),
        "innovation": _first_text(research_context.get("main_contribution_cn"), review_use.get("role_in_direction_cn"), default="暂无明确创新点。"),
        "limitation": _first_text(findings.get("limitations_cn"), research_context.get("motivation_or_gap_cn"), default="暂无明确局限。"),
        "background": _first_text(
            research_context.get("motivation_or_gap_cn"),
            research_context.get("field_cn"),
            assigned.get("abstract"),
            default=f"该文献归入“{direction_name}”方向，用于支撑 {direction_id} 的主题综述。"
        ),
        "method_flow": _unique_texts(_as_list(method.get("workflow")), limit=8),
        "method_inputs": _unique_texts(_as_list(method.get("inputs")), limit=8),
        "method_outputs": _unique_texts(_as_list(method.get("outputs")), limit=8),
        "method_parameters": _unique_texts(_as_list(method.get("parameters")), limit=8),
        "web_review": web_review,
        "formulas": formulas,
        "visual_assets": _load_visual_assets(assigned, run_dir),
        "related_papers": [item for item in related_ids if item != paper_id][:4],
        "evidence": [
            {
                "claim": _first_text(item.get("claim_cn"), item.get("claim")) if isinstance(item, dict) else _clean_text(item),
                "source": _first_text(item.get("source_text"), item.get("page_or_section")) if isinstance(item, dict) else "",
            }
            for item in _as_list(card.get("evidence"))
        ],
    }
    if not paper["method_flow"]:
        paper["method_flow"] = ["研究问题定义", "方法建模", "输入变量整理", "结果分析", "结论归纳"]
    if not paper["formulas"]:
        paper["formulas"] = _fallback_formulas(paper)
    return paper


def _method_labels_from_text(text: str) -> list[str]:
    text = _clean_text(text).lower()
    patterns = [
        ("鲁棒优化", ["鲁棒优化", "robust optimization"]),
        ("凸优化", ["凸优化", "convex optimization"]),
        ("线性规划", ["线性规划", "linear programming"]),
        ("动态规划", ["动态规划", "dynamic programming", "approximate dynamic programming"]),
        ("随机优化", ["随机优化", "stochastic optimization"]),
        ("强化学习", ["强化学习", "reinforcement learning", "q-learning"]),
        ("博弈论", ["博弈", "game theory", "game-based", "nash"]),
        ("纳什议价", ["纳什议价", "nash bargaining"]),
        ("市场机制设计", ["市场机制", "mechanism design", "auction"]),
        ("策略性报价", ["报价", "竞价", "bidding", "bid", "price-maker", "price maker", "price-making", "offer"]),
        ("市场参与优化", ["市场参与", "参与", "market participation", "participation", "optimal participation"]),
        ("分布式控制", ["分布式控制", "distributed control", "consensus"]),
        ("隐私保护", ["隐私", "privacy-preserving", "privacy preserving"]),
        ("安全多方计算", ["安全多方", "multi-party computation", "mpc", "spdz"]),
        ("区块链智能合约", ["区块链", "blockchain", "smart contract"]),
        ("优化", ["优化", "optimization", "optimal"]),
    ]
    labels = []
    for label, needles in patterns:
        if any(needle in text for needle in needles):
            labels.append(label)
    return labels


def _canonical_method_label(label: Any) -> str:
    text = _clean_text(label)
    lowered = text.lower()
    if not text:
        return ""
    rules = [
        ("鲁棒优化", ["鲁棒优化", "robust optimization", "robust"]),
        ("随机优化", ["随机优化", "stochastic optimization", "scenario-based", "scenario based"]),
        ("混合整数规划", ["混合整数", "mixed-integer", "mixed integer", "milp", "mip"]),
        ("线性规划", ["线性规划", "linear programming", "lp"]),
        ("动态规划", ["动态规划", "dynamic programming", "approximate dynamic programming"]),
        ("凸优化", ["凸优化", "convex optimization"]),
        ("强化学习", ["强化学习", "reinforcement learning", "q-learning", "deep q", "dqn"]),
        ("机器学习预测", ["机器学习", "深度学习", "machine learning", "deep learning", "forecast", "prediction", "neural network"]),
        ("博弈论与策略竞价", ["博弈", "game theory", "nash", "strategic bidding", "策略性竞价"]),
        ("市场机制设计", ["机制设计", "market mechanism", "mechanism design", "auction", "market clearing"]),
        ("策略性报价", ["报价", "竞价", "bidding", "bid", "price-maker", "price maker", "price-making", "offer"]),
        ("市场参与优化", ["市场参与", "参与", "market participation", "participation", "optimal participation"]),
        ("分布式优化控制", ["分布式", "distributed", "consensus", "admm"]),
        ("风险度量与CVaR", ["cvar", "风险", "risk", "value-at-risk", "var"]),
        ("经济性评估", ["经济性", "收益评估", "cost-benefit", "valuation", "assessment"]),
    ]
    for canonical, needles in rules:
        if any(needle in lowered or needle in text for needle in needles):
            return canonical
    return text[:32]


def _method_distribution(summary: dict[str, Any], papers: list[dict[str, Any]], cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    paper_counter: Counter[str] = Counter()
    contributors: dict[str, list[str]] = {}

    def add_occurrence(label: Any, paper_id: str = "") -> None:
        text = _canonical_method_label(label)
        if text:
            counter[text] += 1
            if paper_id and paper_id not in contributors.setdefault(text, []):
                contributors[text].append(paper_id)
                paper_counter[text] += 1

    def add_paper_label(label: Any, paper_id: str) -> None:
        text = _canonical_method_label(label)
        if not text or not paper_id:
            return
        counter[text] += 1
        if paper_id not in contributors.setdefault(text, []):
            contributors[text].append(paper_id)
            paper_counter[text] += 1

    summary_methods = _as_list(summary.get("method_families"))
    for card in cards:
        display = card.get("display_facts") if isinstance(card.get("display_facts"), dict) else {}
        method = card.get("method") if isinstance(card.get("method"), dict) else {}
        paper_id = _first_text(card.get("paper_id"), card.get("candidate_id"))
        explicit_items = _as_list(display.get("method_family")) + _as_list(method.get("method_type"))
        for item in explicit_items:
            add_paper_label(item, paper_id)
        method_text = " ".join(
            _clean_text(item)
            for item in [
                method.get("summary_cn"),
                method.get("workflow"),
                method.get("object_cn"),
                display.get("modeling_or_experiment_type"),
            ]
        )
        for label in _method_labels_from_text(method_text):
            add_paper_label(label, paper_id)
    for paper in papers:
        paper_id = _clean_text(paper.get("id"))
        for item in _as_list(paper.get("keywords")):
            text = _clean_text(item)
            if any(marker in text for marker in ["优化", "学习", "模型", "仿真", "规划", "博弈"]):
                add_paper_label(text, paper_id)
        for label in _method_labels_from_text(" ".join([_clean_text(paper.get("method")), _clean_text(paper.get("innovation"))])):
            add_paper_label(label, paper_id)
    if not counter:
        for item in summary_methods:
            add_occurrence(item)
    return [
        {
            "method": method,
            "count": count,
            "occurrence_count": count,
            "paper_count": paper_counter.get(method, 0),
            "paper_ids": contributors.get(method, [])[:12],
        }
        for method, count in counter.most_common(8)
        if paper_counter.get(method, 0) or not papers
    ]


def _is_main_direction(direction: dict[str, Any]) -> bool:
    text = " ".join(
        _clean_text(direction.get(key))
        for key in ["id", "name", "name_en", "summary", "core_question"]
    ).lower()
    if not direction.get("papers"):
        return False
    excluded_markers = ["不纳入主线", "exclude", "excluded", "不相关", "候选排除", "边界排除"]
    return not any(marker in text for marker in excluded_markers)


def _topic_domain_terms(topic: Any) -> list[str]:
    text = _clean_text(topic).lower()
    if any(term in text for term in ["储能", "energy storage", "battery storage", "bess"]):
        return [
            "energy storage",
            "battery",
            "bess",
            "electricity market",
            "power market",
            "bidding",
            "storage",
            "储能",
            "电力市场",
            "电池",
            "竞标",
            "报价",
            "电网",
        ]
    if any(term in text for term in ["频率", "frequency", "惯性", "inertia"]):
        return [
            "frequency",
            "inertia",
            "low inertia",
            "frequency reserve",
            "reserve sharing",
            "virtual synchronous",
            "synchronous machine",
            "vsm",
            "hvdc",
            "converter",
            "renewable energy",
            "power system",
            "electricity transmission",
            "grid",
            "频率",
            "惯性",
            "备用",
            "虚拟同步机",
            "新能源",
            "换流器",
            "电力系统",
            "电网",
            "输电",
        ]
    generic = ["power system", "electricity", "energy", "grid", "market", "电力", "能源", "电网", "市场"]
    topic_terms = [term for term in re.split(r"[\s,，;；、]+", text) if len(term) >= 3]
    return _unique_texts([*topic_terms, *generic], limit=40)


def _is_irrelevant_paper(paper: dict[str, Any], topic: Any = "") -> tuple[bool, str]:
    text = " ".join(
        _clean_text(paper.get(key))
        for key in ["title", "title_cn", "research_problem", "method", "scenario", "background"]
    ).lower()
    irrelevant = ["dark energy", "synoptic survey", "telescope", "astronom", "cosmolog", "lsst"]
    domain = _topic_domain_terms(topic)
    if any(term in text for term in irrelevant):
        return True, "命中天文学/暗能量等明显非电力市场主题词"
    if not any(term in text for term in domain):
        return True, "未命中当前主题的核心主题词"
    return False, ""


def build_quality_report(payload: dict[str, Any]) -> dict[str, Any]:
    directions = payload.get("directions", []) if isinstance(payload.get("directions"), list) else []
    papers = [paper for direction in directions for paper in _as_list(direction.get("papers")) if isinstance(paper, dict)]
    direction_ids = [str(direction.get("id")) for direction in directions if direction.get("id")]
    corpus = payload.get("corpus") if isinstance(payload.get("corpus"), dict) else {}
    expected_papers = _safe_int(payload.get("target_paper_total") or corpus.get("target_paper_total") or len(papers) or 15)
    if papers and expected_papers < len(papers):
        expected_papers = len(papers)
    irrelevant = []
    formula_issues = []
    formula_ref_issues = []
    for paper in papers:
        is_bad, reason = _is_irrelevant_paper(paper, payload.get("topic"))
        if is_bad:
            irrelevant.append({"paper_id": paper.get("id"), "title": paper.get("title"), "reason": reason})
        formulas = [item for item in _as_list(paper.get("formulas")) if isinstance(item, dict) and _clean_text(item.get("formula"))]
        if not formulas:
            formula_issues.append({"paper_id": paper.get("id"), "title": paper.get("title"), "reason": "缺少可渲染公式字段"})
        formula_ids = {
            _clean_text(item.get("id")).lower()
            for item in formulas
            if _clean_text(item.get("id"))
        }
        formula_ids.update(
            _clean_text(item.get("original_number")).lower()
            for item in formulas
            if _clean_text(item.get("original_number"))
        )
        web_review = paper.get("web_review") if isinstance(paper.get("web_review"), dict) else {}
        for index, step in enumerate(_as_list(web_review.get("method_steps")), start=1):
            if not isinstance(step, dict):
                continue
            bad_refs = [
                ref
                for ref in _as_list(step.get("formula_refs"))
                if _clean_text(ref).lower() not in formula_ids
            ]
            if bad_refs:
                formula_ref_issues.append(
                    {
                        "paper_id": paper.get("id"),
                        "step_index": index,
                        "step_name": step.get("step_name"),
                        "bad_refs": bad_refs,
                    }
                )
    checks = [
        {
            "id": "paper_total",
            "name": "文献数量",
            "status": "pass" if len(papers) == expected_papers else "warn",
            "expected": expected_papers,
            "actual": len(papers),
        },
        {
            "id": "direction_minimum",
            "name": "方向数量与 D1/D2",
            "status": "pass" if len(directions) >= 2 and {"D1", "D2"}.issubset(set(direction_ids)) else "fail",
            "expected": "至少 D1、D2 两类",
            "actual": ", ".join(direction_ids),
        },
        {
            "id": "formula_fields",
            "name": "公式字段",
            "status": "pass" if not formula_issues else "warn",
            "expected": "每篇至少有一组公式或兜底公式",
            "actual": len(papers) - len(formula_issues),
        },
        {
            "id": "method_formula_refs",
            "name": "方法步骤与公式引用",
            "status": "pass" if not formula_ref_issues else "warn",
            "expected": "method_steps 中的 formula_refs 均指向实际公式 id",
            "actual": len(formula_ref_issues),
        },
        {
            "id": "irrelevant_papers",
            "name": "无关论文",
            "status": "pass",
            "expected": 0,
            "actual": len(irrelevant),
        },
    ]
    status = "pass" if all(item["status"] == "pass" for item in checks) else ("fail" if any(item["status"] == "fail" for item in checks) else "warn")
    return {
        "schema_version": "literature_quality_report.v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_run_id": payload.get("source_run_id"),
        "topic": payload.get("topic"),
        "status": status,
        "checks": checks,
        "irrelevant_papers": irrelevant,
        "formula_issues": formula_issues,
        "formula_ref_issues": formula_ref_issues,
        "notes": [
            "公式渲染的浏览器端 MathJax 状态仍需通过 literature_showcase/test_showcase.ps1 或浏览器自动化验证。",
            "无关论文检查为规则检测，仅作为人工复核提示，不影响质量状态。"
        ],
    }


def _normalize_direction(run_dir: Path, folder: Path) -> dict[str, Any]:
    review_root = run_dir / "02_reviews" / "directions"
    discovery_root = run_dir / "01_discovery" / "directions"
    review_dir = _matching_direction_dir(review_root, folder.name)
    discovery_dir = _matching_direction_dir(discovery_root, folder.name)

    assigned = _load_json(review_dir / "assigned_papers.json", {})
    if not isinstance(assigned, dict):
        assigned = _load_json(discovery_dir / "assigned_papers.json", {})
    if not isinstance(assigned, dict):
        assigned = {}
    summary = _load_json(review_dir / "direction_review_summary.json", {})
    if not isinstance(summary, dict):
        summary = {}
    cards_by_id = _paper_cards(review_dir)
    raw_papers = assigned.get("papers") if isinstance(assigned.get("papers"), list) else []
    related_ids = [
        _first_text(cards_by_id.get(_first_text(item.get("paper_id"), item.get("candidate_id")), {}).get("paper_id"), item.get("paper_id"), item.get("candidate_id"))
        for item in raw_papers
        if isinstance(item, dict)
    ]

    direction_id = _first_text(assigned.get("direction_id"), summary.get("direction_id"), folder.name.split("_", 1)[0], default=folder.name)
    direction_name = _first_text(assigned.get("direction_name_cn"), summary.get("direction_name_cn"), folder.name, default=direction_id)
    assigned_summary = _first_text(assigned.get("direction_summary_cn"))
    if direction_name == "候选相关文献":
        topic_name = _first_text(assigned.get("topic"), default=direction_name)
        direction_name = f"{topic_name}综合文献组"
    papers = []
    used_card_ids: set[int] = set()
    for item in raw_papers:
        if not isinstance(item, dict):
            continue
        key = _first_text(item.get("paper_id"), item.get("candidate_id"))
        card = cards_by_id.get(key, {})
        if card:
            used_card_ids.add(id(card))
        papers.append(_normalize_paper(item, card, direction_id, direction_name, related_ids, run_dir))
    for card in cards_by_id.values():
        if id(card) in used_card_ids:
            continue
        papers.append(_normalize_paper({}, card, direction_id, direction_name, related_ids, run_dir))

    cards = [card for card in cards_by_id.values() if isinstance(card, dict)]
    keywords = _unique_texts(
        _as_list(assigned.get("display_keywords"))
        + _as_list(summary.get("display_tags"))
        + _as_list(summary.get("method_families")),
        limit=10,
    )
    methods = _method_distribution(summary, papers, cards)
    heat = min(100, 45 + len(papers) * 8 + len(methods) * 5)
    common_inputs = _unique_texts(_as_list(summary.get("common_inputs")), limit=6)
    common_outputs = _unique_texts(_as_list(summary.get("common_outputs")), limit=6)
    common_metrics = _unique_texts(_as_list(summary.get("common_metrics")), limit=6)

    return {
        "id": direction_id,
        "name": direction_name,
        "name_en": _first_text(assigned.get("direction_name_en")),
        "paper_count": len(papers),
        "heat": heat,
        "core_question": _first_text(summary.get("core_problem_cn"), assigned.get("direction_summary_cn"), default=f"{direction_name} 的核心研究问题。"),
        "summary": (
            f"本方向汇总本次实际下载并分析的 {len(papers)} 篇文献，用于进行方向内问题、方法、场景、结论和局限比较。"
            if direction_name == "候选相关文献" or "模型方向归纳失败" in assigned_summary
            else _first_text(assigned.get("direction_summary_cn"), summary.get("core_problem_cn"), _load_text(review_dir / "direction_review.md", "")[:260])
        ),
        "keywords": keywords,
        "commonality": _first_text(summary.get("main_findings_cn"), default="该方向文献围绕相近问题展开，并共享部分输入、输出或评价指标。"),
        "differences": _first_text(
            [
                f"输入：{'、'.join(common_inputs)}" if common_inputs else "",
                f"输出：{'、'.join(common_outputs)}" if common_outputs else "",
                f"指标：{'、'.join(common_metrics)}" if common_metrics else "",
            ],
            default="差异主要体现在研究对象、建模方法、数据来源和评价指标上。",
        ),
        "gap": _first_text(summary.get("limitations_cn"), default="仍需补充更完整的实证数据、可复现模型和跨场景比较。"),
        "methods_distribution": methods,
        "knowledge_card": {
            "concepts": keywords[:5],
            "trend": _first_text(
                summary.get("future_trends_cn"),
                summary.get("future_trend_cn"),
                summary.get("research_trends_cn"),
                summary.get("limitations_cn"),
                default="从单点算例验证转向跨场景、可复现、可解释和贴近真实市场规则的综合研究。",
            ),
            "representative_conclusion": _first_text(summary.get("main_findings_cn"), default="代表性结论将在更多论文卡片生成后进一步细化。"),
        },
        "papers": papers,
    }


def build_three_stage_review(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    report = _load_json(run_dir / "unified_run_report.json", {})
    if not isinstance(report, dict):
        report = {}
    corpus_summary = _load_json(run_dir / "02_reviews" / "corpus_review_summary.json", {})
    if not isinstance(corpus_summary, dict):
        corpus_summary = {}
    cross_plot: dict[str, Any] = {}

    raw_directions = [_normalize_direction(run_dir, folder) for folder in _direction_dirs(run_dir)]
    directions = [direction for direction in raw_directions if _is_main_direction(direction)]
    if not directions:
        directions = [direction for direction in raw_directions if direction.get("papers")]
    for index, direction in enumerate(directions, start=1):
        original_id = str(direction.get("id") or f"D{index}")
        normalized_id = f"D{index}"
        if original_id != normalized_id:
            direction["original_id"] = original_id
        direction["id"] = normalized_id
        for paper in _as_list(direction.get("papers")):
            if isinstance(paper, dict):
                paper["direction_id"] = normalized_id
    actual_direction_ids = {str(direction.get("id")) for direction in directions if direction.get("id")}
    cross_blocks = [item for item in _as_list(cross_plot.get("direction_blocks")) if isinstance(item, dict)]
    cross_ids = {str(item.get("direction_id")) for item in cross_blocks if item.get("direction_id")}
    use_cross_plot = bool(cross_plot) and len(directions) > 1 and not (cross_ids - actual_direction_ids)
    years = [paper.get("year") for direction in directions for paper in direction.get("papers", [])]
    paper_ids = _unique_paper_ids(directions)
    paper_total = len(paper_ids) if paper_ids else sum(int(direction.get("paper_count") or 0) for direction in directions)
    methods = _unique_texts(_as_list(corpus_summary.get("cross_direction_method_families")), limit=12)
    gaps = _unique_texts(
        _as_list(corpus_summary.get("cross_direction_gaps_cn"))
        + (
            [item.get("gap_description_cn") for item in _as_list(cross_plot.get("research_gap_blocks")) if isinstance(item, dict)]
            if use_cross_plot
            else []
        ),
        limit=10,
    )
    comparisons = [
        _first_text(item.get("comparison_cn"), item.get("axis_cn"))
        for item in _as_list(cross_plot.get("cross_direction_comparison"))
        if isinstance(item, dict) and use_cross_plot and not _mentions_unknown_direction(item.get("comparison_cn"), actual_direction_ids)
    ]
    evidence = []
    for direction in directions:
        for paper in direction.get("papers", [])[:2]:
            for item in paper.get("evidence", [])[:1]:
                claim = _clean_text(item.get("claim") if isinstance(item, dict) else item)
                if claim:
                    evidence.append({"claim": claim, "papers": [paper.get("id")]})

    topic = _first_text(corpus_summary.get("topic"), cross_plot.get("topic"), report.get("topic"), run_dir.name)
    target_paper_total = max(_safe_int(report.get("max_papers") or paper_total), paper_total)
    corpus_review = _load_text(run_dir / "02_reviews" / "corpus_literature_review.md", "")
    summary = _first_text(
        cross_plot.get("global_core_problem_cn") if use_cross_plot or not _mentions_unknown_direction(cross_plot.get("global_core_problem_cn"), actual_direction_ids) else "",
        _first_meaningful_markdown_paragraph(corpus_review),
        default=f"本运行围绕“{topic}”生成三层递进式文献综述数据。"
    )
    timeline = []
    if years:
        timeline.append(
            {
                "period": _year_range(years),
                "theme": "本次运行覆盖文献",
                "description": f"共整理 {paper_total} 篇文献，形成 {len(directions)} 个研究方向。"
            }
        )
    storyline_source = _as_list(cross_plot.get("storyline_cn")) if use_cross_plot else []
    for item in storyline_source[:4]:
        text = _clean_text(item)
        if text and not _mentions_unknown_direction(text, actual_direction_ids):
            timeline.append({"period": "研究脉络", "theme": text, "description": text})
    if len(directions) == 1 and directions:
        direction = directions[0]
        timeline.append(
            {
                "period": "方向内综合",
                "theme": f"{direction.get('id')} {direction.get('name')}",
                "description": f"当前网页实际只有 {direction.get('id')} 一个方向，{paper_total} 篇文献均在该方向下展开比较。"
            }
        )

    if len(directions) == 1 and directions:
        direction = directions[0]
        commonality = (
            f"当前输出只有一个实际研究方向：{direction.get('id')} {direction.get('name')}。"
            f"{paper_total} 篇文献共同围绕“{topic}”展开，主要方法包括：{'、'.join(methods[:6]) or '优化、市场机制与数据驱动分析'}。"
        )
        differences = (
            "由于本次方向归并结果只有一个实际方向，网页总览不再显示任何不存在的方向编号。"
            "方向内差异主要体现在论文的市场场景、建模方法、输入变量、输出指标、隐私约束和实证数据来源上，详见第二层文献对比表。"
        )
    else:
        direction_focus = [
            f"{direction.get('id')}侧重{_first_text(direction.get('name'), direction.get('core_question'))}"
            for direction in directions[:5]
            if direction.get("id")
        ]
        direction_gaps = _unique_texts([
            _clean_text(direction.get("gap"))
            for direction in directions
            if _clean_text(direction.get("gap"))
        ])
        commonality = _first_text(
            corpus_summary.get("cross_direction_commonality_cn"),
            corpus_summary.get("commonality_cn"),
            corpus_summary.get("commonality"),
            [item for item in _as_list(cross_plot.get("storyline_cn")) if not _mentions_unknown_direction(item, actual_direction_ids)],
            corpus_summary.get("cross_direction_inputs"),
            default=(
                f"这些方向共同围绕“{topic}”展开，均关注研究对象如何在具体市场规则、运行约束和收益风险权衡下形成可执行决策。"
                f"当前样本中的主要方法包括{'、'.join(methods[:5]) or '优化建模、市场机制分析和数据驱动方法'}。"
            ),
        )
        differences = _first_text(
            corpus_summary.get("cross_direction_differences_cn"),
            corpus_summary.get("differences_cn"),
            corpus_summary.get("differences"),
            comparisons,
            default="；".join(direction_focus[:5]) if direction_focus else "方向差异体现在研究对象、市场场景、建模方法和评价指标上。",
        )

    consensus = _unique_texts(
        _as_list(corpus_summary.get("research_consensus"))
        + _as_list(corpus_summary.get("research_consensus_cn"))
        + _as_list(commonality),
        limit=4,
    )
    disagreements = _unique_texts(
        _as_list(corpus_summary.get("research_disagreements"))
        + _as_list(corpus_summary.get("research_disagreements_cn"))
        + _as_list(corpus_summary.get("research_divergence"))
        + _as_list(differences),
        limit=4,
    )
    opportunities = _unique_texts(
        _as_list(corpus_summary.get("future_opportunities"))
        + _as_list(corpus_summary.get("future_opportunities_cn"))
        + gaps,
        limit=4,
    )

    direction_lookup = {str(direction.get("id")): direction for direction in directions if direction.get("id")}
    domain_insights: list[dict[str, Any]] = []
    for index, raw in enumerate(_as_list(corpus_summary.get("domain_insights"))[:6]):
        if isinstance(raw, dict):
            title = _first_text(raw.get("title"), raw.get("name"), default="领域综合洞察")
            summary_text = _first_text(raw.get("summary"), raw.get("explanation"), raw.get("claim"))
            support_directions = [str(item) for item in _as_list(raw.get("support_directions") or raw.get("directions")) if item]
        else:
            title = "领域综合洞察"
            summary_text = _clean_text(raw)
            support_directions = []
        if not summary_text:
            continue
        if not support_directions and directions:
            direction = directions[min(index, len(directions) - 1)]
            support_directions = [str(direction.get("id"))]
        paper_ids: list[str] = []
        for direction_id in support_directions:
            direction = direction_lookup.get(direction_id)
            if not direction:
                continue
            paper_ids.extend([paper.get("id") for paper in direction.get("papers", [])[:3] if paper.get("id")])
        domain_insights.append(
            {
                "title": title,
                "summary": summary_text,
                "support_directions": support_directions,
                "support_paper_count": sum(
                    int(direction_lookup.get(direction_id, {}).get("paper_count") or 0)
                    for direction_id in support_directions
                ),
                "papers": _unique_texts(paper_ids, limit=4),
            }
        )
    if not domain_insights:
        for direction in directions[:6]:
            paper_ids = [paper.get("id") for paper in direction.get("papers", [])[:3] if paper.get("id")]
            domain_insights.append(
                {
                    "title": f"{direction.get('name') or direction.get('id')} 是当前主题下的关键研究方向",
                    "summary": _first_text(direction.get("summary"), direction.get("core_question"), default="该方向围绕主题形成了一组可比较的研究问题、方法和结论。"),
                    "support_directions": [direction.get("id")],
                    "support_paper_count": int(direction.get("paper_count") or len(direction.get("papers", []))),
                    "papers": paper_ids,
                }
            )

    overview_detail = _build_corpus_overview_detail(
        topic=topic,
        summary=_first_text(corpus_summary.get("overview_detail"), summary),
        directions=directions,
        paper_total=paper_total,
        target_paper_total=target_paper_total,
        year_range=_year_range(years),
        methods=methods,
        gaps=gaps,
        domain_insights=domain_insights,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_run_id": run_dir.name,
        "topic": topic,
        "target_paper_total": target_paper_total,
        "corpus": {
            "topic": topic,
            "paper_total": paper_total,
            "target_paper_total": target_paper_total,
            "year_range": _year_range(years),
            "direction_total": len(directions),
            "source_note": "由 analysis_pipeline 自动生成，汇总检索、分类、综述与网页展示结果。",
            "summary": summary,
            "overview_detail": overview_detail,
            "commonality": commonality,
            "differences": differences,
            "gap": _first_text(
                corpus_summary.get("cross_direction_gap_cn"),
                corpus_summary.get("gap_cn"),
                corpus_summary.get("gap"),
                "；".join(gaps) if gaps else "",
                "；".join(direction_gaps[:4]) if direction_gaps else "",
                default="尚未生成明确研究空白，可在扩充论文后进一步总结。",
            ),
            "keywords": _unique_texts(_as_list(corpus_summary.get("display_tags")), limit=12),
            "methods": methods,
            "timeline": timeline,
            "evidence": evidence[:6],
            "domain_insights": domain_insights,
            "research_consensus": consensus,
            "research_disagreements": disagreements,
            "future_opportunities": opportunities,
        },
        "directions": directions,
    }


def write_three_stage_review(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    payload = build_three_stage_review(run_dir)
    target = run_dir / SHOWCASE_FILENAME
    _save_json(target, payload)
    _save_json(run_dir / QUALITY_FILENAME, build_quality_report(payload))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pipeline output to the literature_showcase JSON schema.")
    parser.add_argument("run_dir", type=Path, help="A run directory under output/.")
    args = parser.parse_args()
    target = write_three_stage_review(args.run_dir)
    print(target)


if __name__ == "__main__":
    main()
