import argparse
import json
import re
from pathlib import Path
from typing import Any

from generate_review_figures import (
    build_paper_lookup,
    build_single_paper_lookup,
    extract_record_meta,
    normalize_direction_paper,
)
from multi_paper_structured_pipeline_v2 import (
    DEFAULT_MODEL,
    build_client,
    call_api_json,
    compact_single_structure,
    ensure_dir,
    save_json,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "output_awe_20260426_2212"
DEFAULT_OUTPUT_DIRNAME = "plot_ready_structures"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def shorten_text(text: Any, limit: int = 320) -> str:
    value = str(text).strip()
    value = re.sub(r"\s+", " ", value)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def trim_for_prompt(value: Any, max_items: int = 6, max_text: int = 260) -> Any:
    if isinstance(value, dict):
        return {key: trim_for_prompt(val, max_items=max_items, max_text=max_text) for key, val in value.items()}
    if isinstance(value, list):
        return [trim_for_prompt(item, max_items=max_items, max_text=max_text) for item in value[:max_items]]
    if isinstance(value, str):
        return shorten_text(value, max_text)
    return value


def detect_text_tokens_for_repair(payload: dict[str, Any]) -> list[str]:
    found: list[str] = []
    pattern = re.compile(r"[A-Za-zΔΘΦΨγψϕθκβ_]+(?:\([A-Za-z]+\))?")

    def add_matches(text: str) -> None:
        for token in pattern.findall(text):
            normalized = token.strip()
            if normalized and normalized not in found:
                found.append(normalized)

    add_matches(str(payload.get("direction_name", "")))
    add_matches(str(payload.get("core_problem_cn", "")))

    baseline = payload.get("baseline_paper", {})
    for key in ["input_box_cn", "method_box_cn", "output_box_cn"]:
        for item in baseline.get(key, []):
            add_matches(str(item))

    for row in payload.get("comparison_rows", []):
        for key in ["input_diff_cn", "method_diff_cn", "output_diff_cn"]:
            for item in row.get(key, []):
                add_matches(str(item))

    return found


def validate_plot_ready_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_top = ["direction_id", "direction_name", "core_problem_cn", "baseline_paper", "comparison_rows", "symbol_glossary_cn"]
    for key in required_top:
        if key not in payload:
            errors.append(f"缺少顶层字段：{key}")

    baseline = payload.get("baseline_paper", {})
    for key in ["paper_id", "citation_cn", "input_box_cn", "method_box_cn", "output_box_cn"]:
        if key not in baseline:
            errors.append(f"baseline_paper 缺少字段：{key}")

    for key in ["input_box_cn", "method_box_cn", "output_box_cn"]:
        value = baseline.get(key)
        if not isinstance(value, list):
            errors.append(f"baseline_paper.{key} 必须是数组")

    rows = payload.get("comparison_rows", [])
    if not isinstance(rows, list):
        errors.append("comparison_rows 必须是数组")
    else:
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                errors.append(f"comparison_rows[{idx}] 必须是对象")
                continue
            for key in ["paper_id", "citation_cn", "input_diff_cn", "method_diff_cn", "output_diff_cn"]:
                if key not in row:
                    errors.append(f"comparison_rows[{idx}] 缺少字段：{key}")

    glossary = payload.get("symbol_glossary_cn", [])
    if not isinstance(glossary, list):
        errors.append("symbol_glossary_cn 必须是数组")

    return errors


def build_direction_records_map(input_dir: Path) -> dict[str, dict[str, Any]]:
    records_by_id: dict[str, dict[str, Any]] = {}
    for path in (input_dir / "direction_records").glob("*.json"):
        record = load_json(path)
        direction_id = str(record.get("direction_id", "")).strip()
        if direction_id:
            records_by_id[direction_id] = record
            continue
        first = record.get("records", [None])[0]
        if isinstance(first, dict):
            if "paper" in first and isinstance(first["paper"], dict):
                direction_id = str(first.get("direction_id", "") or record.get("direction_id", "")).strip()
            elif "meta" in first and isinstance(first["meta"], dict):
                direction_id = str(first["meta"].get("direction_id", "") or record.get("direction_id", "")).strip()
        if direction_id:
            records_by_id[direction_id] = record
    return records_by_id


def find_direction_record(input_dir: Path, direction_id: str) -> dict[str, Any]:
    for path in (input_dir / "direction_records").glob(f"{direction_id}_*.json"):
        return load_json(path)
    raise FileNotFoundError(f"未找到方向记录：{direction_id}")


def build_assignment_lookup(mapping: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("paper_id", "")).strip(): item for item in mapping.get("paper_assignments", [])}


def build_direction_api_context(
    direction: dict[str, Any],
    direction_record: dict[str, Any],
    paper_lookup: dict[str, dict[str, str]],
    single_papers: dict[str, dict[str, Any]],
    assignment_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    record_by_id: dict[str, dict[str, Any]] = {}
    for item in direction_record.get("records", []):
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
    papers: list[dict[str, Any]] = []
    for paper_id in ordered_ids:
        normalized = normalize_direction_paper(
            record_by_id[paper_id],
            direction.get("direction_definition", ""),
            paper_lookup,
            single_papers,
        )
        raw_single = single_papers.get(paper_id, {})
        assignment = assignment_lookup.get(paper_id, {})
        papers.append(
            {
                "paper_id": paper_id,
                "title": normalized["title"],
                "year": normalized["year"],
                "citation_cn": normalized["citation"],
                "role_hint": assignment.get("role_in_direction", ""),
                "assignment_reason": shorten_text(assignment.get("assignment_reason", ""), 220),
                "normalized_slots": {
                    "core_problem_cn": normalized["problem"],
                    "input_box_cn": normalized["inputs"],
                    "method_box_cn": normalized["methods"],
                    "output_box_cn": normalized["outputs"],
                },
                "single_paper_structure_excerpt": trim_for_prompt(compact_single_structure(raw_single)),
                "direction_record_excerpt": trim_for_prompt(record_by_id[paper_id]),
            }
        )

    return {
        "direction_id": direction.get("direction_id"),
        "direction_name": direction.get("direction_name"),
        "direction_definition_cn": shorten_text(direction.get("direction_definition", ""), 320),
        "baseline_hint_paper_id": baseline_id,
        "baseline_hint_citation_cn": paper_lookup.get(baseline_id, {}).get("citation", "") if baseline_id else "",
        "drawing_rule_cn": {
            "baseline_row": "第一行固定为 输入/条件 -> 方法/模型 -> 输出/结果，右侧只放 作者+年份",
            "comparison_rows": "后续每篇文献只写相对主线不同的 输入差异 / 方法差异 / 输出差异；相同处写 同主线",
            "language": "全部用中文；符号或缩写必须在同一条内解释中文含义",
        },
        "papers": papers,
    }


def build_plot_ready_prompt(direction_context: dict[str, Any]) -> str:
    context_json = json.dumps(direction_context, ensure_ascii=False, indent=2)
    return f"""
你是“论文综述作图结构化助手”。

任务：把给定方向的文献信息整理成“专用于画图”的结构化 JSON，供程序直接绘制单方向可视化总结。

硬性要求：
1. 全部用中文表达。
2. 禁止输出未解释的英文术语、英文缩写、变量符号。
3. 如果必须出现符号或缩写，必须在同一条中立即解释中文含义。
   示例：
   - 可以写：8字航迹宽度 W、8字航迹高度 H
   - 可以写：气动力 F_a
   - 可以写：方位偏移 Δaz
   - 不可以只写：W、H、Fa、Δaz、RLS、DQN
4. 每个方框内只保留适合画图的短句：
   - 每列最多 3 条
   - 每条尽量控制在 28 个汉字以内
   - 不要写长段落
5. 必须忠实于提供的材料，不要猜测。
6. 如果该方向只有 1 篇文献，则 comparison_rows 返回空数组。
7. comparison_rows 只写相对主线不同的地方；如果没有明显差异，写 ["同主线"]。
8. 输出必须是 JSON，不要输出解释文字。

输出 JSON 结构：
{{
  "direction_id": "",
  "direction_name": "",
  "core_problem_cn": "一句中文，说明这一方向主要解决什么问题",
  "baseline_paper": {{
    "paper_id": "",
    "citation_cn": "作者等（年份）",
    "input_box_cn": ["短句1", "短句2", "短句3"],
    "method_box_cn": ["短句1", "短句2", "短句3"],
    "output_box_cn": ["短句1", "短句2", "短句3"]
  }},
  "comparison_rows": [
    {{
      "paper_id": "",
      "citation_cn": "作者等（年份）",
      "input_diff_cn": ["短句1", "短句2"],
      "method_diff_cn": ["短句1", "短句2"],
      "output_diff_cn": ["短句1", "短句2"]
    }}
  ],
  "symbol_glossary_cn": [
    {{
      "symbol": "W",
      "meaning_cn": "8字航迹宽度"
    }}
  ],
  "self_check": {{
    "all_text_chinese": true,
    "unexplained_symbols": [],
    "unexplained_english_terms": []
  }}
}}

补充规则：
1. baseline_paper 必须优先使用 baseline_hint_paper_id 指定的主线文献。
2. baseline_paper 的三列，要概括“需要什么条件/输入、用什么方法、得到什么结果/输出”。
3. comparison_rows 的每一列，只写和主线不同之处。
4. symbol_glossary_cn 只收录实际出现在方框文本中的符号。
5. 如果某条已经写成“8字航迹宽度 W”，那么 glossary 里也要有 W 的中文释义。

以下是输入上下文：
{context_json}
""".strip()


def build_repair_prompt(direction_context: dict[str, Any], payload: dict[str, Any], tokens: list[str]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    context_json = json.dumps(direction_context, ensure_ascii=False, indent=2)
    token_text = "、".join(tokens[:30])
    return f"""
你是“论文综述作图结构化助手”。

下面这个 JSON 已经基本正确，但仍残留不符合要求的英文、缩写或裸符号：
{token_text}

请在不改变事实的前提下，把它重写成更适合作图的 JSON。

硬性要求：
1. direction_name、core_problem_cn、所有输入/方法/输出条目、所有差异条目，全部优先改写成中文。
2. citation_cn 保留“作者等（年份）”格式，不必翻译作者姓氏。
3. 除 citation_cn 与 symbol_glossary_cn.symbol 外，其他字段尽量不要出现英文缩写或裸符号。
4. 如果某个符号确实必须保留，也必须在同一条里说明中文含义，例如“8字航迹宽度 W”“系绳张力 T”“性能指标 J”。
5. 不要出现 J帽(b)、W/H、RLS、DQN、EAR 这类未充分中文化的写法；要改成中文优先表达。
6. 保持原 JSON 结构完全一致。
7. 输出必须是 JSON，不要解释。

原始上下文：
{context_json}

待修正 JSON：
{payload_json}
""".strip()


def generate_plot_ready_payload(
    client: Any,
    model: str,
    direction_context: dict[str, Any],
    dry_run: bool,
) -> tuple[dict[str, Any] | None, str]:
    prompt = build_plot_ready_prompt(direction_context)
    if dry_run:
        return None, prompt
    payload = call_api_json(client=client, model=model, prompt=prompt)
    if not isinstance(payload, dict):
        raise ValueError("API 返回结果不是 JSON 对象")
    repair_tokens = detect_text_tokens_for_repair(payload)
    if repair_tokens:
        repair_prompt = build_repair_prompt(direction_context, payload, repair_tokens)
        repaired = call_api_json(client=client, model=model, prompt=repair_prompt)
        if isinstance(repaired, dict):
            payload = repaired
    errors = validate_plot_ready_payload(payload)
    if errors:
        raise ValueError("画图结构化输出格式不合法：" + "；".join(errors))
    return payload, prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="生成专用于画图的结构化输出（调用 DeepSeek / OpenAI 兼容接口）")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--direction-id", action="append", help="只处理指定方向，可重复传入，如 --direction-id D1")
    parser.add_argument("--dry-run", action="store_true", help="只生成上下文和提示词，不调用 API")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的结果文件")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or (input_dir / DEFAULT_OUTPUT_DIRNAME)
    ensure_dir(output_dir)

    mapping = load_json(input_dir / "directions" / "direction_mapping.json")
    paper_lookup = build_paper_lookup(input_dir, mapping)
    single_papers = build_single_paper_lookup(input_dir)
    assignment_lookup = build_assignment_lookup(mapping)
    client = None if args.dry_run else build_client()

    selected_ids = set(args.direction_id or [])
    directions = [
        direction
        for direction in mapping.get("directions", [])
        if not selected_ids or str(direction.get("direction_id")) in selected_ids
    ]
    if not directions:
        raise ValueError("没有匹配到要处理的方向")

    bundle: list[dict[str, Any]] = []
    for direction in directions:
        direction_id = str(direction.get("direction_id"))
        direction_record = find_direction_record(input_dir, direction_id)
        direction_context = build_direction_api_context(
            direction=direction,
            direction_record=direction_record,
            paper_lookup=paper_lookup,
            single_papers=single_papers,
            assignment_lookup=assignment_lookup,
        )

        context_path = output_dir / f"{direction_id}_drawing_context.json"
        prompt_path = output_dir / f"{direction_id}_drawing_prompt.txt"
        result_path = output_dir / f"{direction_id}_plot_ready.json"
        save_json(context_path, direction_context)

        if result_path.exists() and not args.overwrite and not args.dry_run:
            payload = load_json(result_path)
            bundle.append(payload)
            print(f"跳过已存在结果：{result_path}")
            continue

        payload, prompt = generate_plot_ready_payload(
            client=client,
            model=args.model,
            direction_context=direction_context,
            dry_run=args.dry_run,
        )
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        print(f"已生成上下文：{context_path}")
        print(f"已生成提示词：{prompt_path}")

        if payload is None:
            continue

        self_check = payload.get("self_check", {})
        if self_check.get("unexplained_symbols") or self_check.get("unexplained_english_terms"):
            print(f"警告：{direction_id} 的 self_check 仍报告存在未解释项，请人工复查 {result_path.name}")
        save_json(result_path, payload)
        print(f"已生成画图结构化结果：{result_path}")
        bundle.append(payload)

    if bundle:
        save_json(output_dir / "plot_ready_bundle.json", {"directions": bundle})
        print(f"聚合结果：{output_dir / 'plot_ready_bundle.json'}")


if __name__ == "__main__":
    main()
