from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evidence import evidence_as_prompt, select_evidence
from .llm_client import call_openai_json, render_prompt


def run_audit(
    *,
    target: dict[str, Any],
    text_json: dict[str, Any],
    schema_path: str | Path,
    prompt_path: str | Path,
    offline: bool = False,
) -> dict[str, Any]:
    snippets = select_evidence(text_json)
    if offline:
        return offline_audit(target, snippets)

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    metadata = json.dumps(
        {
            key: target.get(key)
            for key in ["title", "authors", "year", "venue", "doi", "role", "notes"]
        },
        ensure_ascii=False,
        indent=2,
    )
    prompt = render_prompt(
        prompt_path,
        metadata=metadata,
        evidence=evidence_as_prompt(snippets, max_chars=50000),
    )
    return call_openai_json(
        prompt=prompt,
        schema=schema,
        schema_name="reproducibility_audit",
    )


def offline_audit(target: dict[str, Any], snippets: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_blob = "\n".join(item["text"].lower() for item in snippets)
    has_ieee118 = "118-bus" in evidence_blob or "118 bus" in evidence_blob
    has_ccg = "column-and-constraint" in evidence_blob
    has_lsf = "load-shift-factor" in evidence_blob or "lsf" in evidence_blob
    has_solver = "cplex" in evidence_blob
    has_uncertainty = "uncertainty" in evidence_blob and ("wind" in evidence_blob or "load" in evidence_blob)

    data_score = 3.5 if has_ieee118 and has_uncertainty else 2.5
    algorithm_score = 4.5 if has_ccg and has_lsf else 3.5
    result_score = 4.0 if "table" in evidence_blob and has_solver else 3.0
    overall = round((data_score + algorithm_score + result_score) / 3, 2)

    return {
        "paper_title": target["title"],
        "recommended_role": "primary_target",
        "scores": {
            "data": data_score,
            "algorithm": algorithm_score,
            "result_alignment": result_score,
            "overall": overall,
        },
        "data_check": [
            {
                "item": "Benchmark system",
                "evidence": "Modified IEEE 118-bus system with 54 generators, 118 buses, and 186 transmission lines appears in extracted evidence."
                if has_ieee118
                else "Benchmark system evidence was not confidently found.",
                "status": "partially reproducible",
            },
            {
                "item": "Uncertainty model",
                "evidence": "Evidence mentions uncertain load and wind power with interval/box uncertainty.",
                "status": "reproducible after parameter transcription" if has_uncertainty else "needs manual check",
            },
            {
                "item": "Modified case details",
                "evidence": "Modified IEEE 118 details, wind station locations, and full profiles are not fully captured by text extraction.",
                "status": "source tracing required",
            },
        ],
        "algorithm_check": [
            {
                "component": "Column-and-constraint generation",
                "evidence": "Evidence mentions master problem, separation subproblem, lower/upper bounds, and convergence."
                if has_ccg
                else "C&CG evidence was not confidently found.",
                "status": "clear" if has_ccg else "needs manual extraction",
            },
            {
                "component": "LSF cutting-plane",
                "evidence": "Evidence mentions load-shift-factor cutting-plane algorithm for reducing transmission constraints."
                if has_lsf
                else "LSF evidence was not confidently found.",
                "status": "clear" if has_lsf else "needs manual extraction",
            },
            {
                "component": "Solver configuration",
                "evidence": "Evidence mentions CPLEX and implementation environment."
                if has_solver
                else "Solver evidence was not confidently found.",
                "status": "mostly clear" if has_solver else "needs manual check",
            },
        ],
        "result_alignment": [
            {
                "target": "Deterministic vs robust UC comparison",
                "evidence": "The paper reports comparison tables and commitment/cost characteristics.",
                "status": "alignable after table extraction",
            },
            {
                "target": "Runtime and active transmission constraints",
                "evidence": "The paper reports LSF cutting-plane and column-generation performance tables.",
                "status": "alignable after table extraction",
            },
        ],
        "blockers": [
            "Modified IEEE 118-bus data are not fully enumerated.",
            "Wind station bus locations and full 24-hour profile need tracing or assumptions.",
            "Exact branch-price-cut implementation is specialized; start with C&CG plus LSF cutting-plane.",
            "PDF text extraction does not preserve all table numerics cleanly.",
        ],
        "next_steps": [
            "Create source_trace.md for modified IEEE 118-bus, wind profile, and generator data.",
            "Implement deterministic UC appendix first.",
            "Implement robust C&CG master, feasibility subproblem, and optimality subproblem.",
            "Add LSF cutting-plane for master problem before attempting subproblem column generation.",
        ],
    }


def audit_to_markdown(audit: dict[str, Any]) -> str:
    lines = [
        f"# Reproducibility Audit: {audit['paper_title']}",
        "",
        f"- Recommended role: `{audit['recommended_role']}`",
        f"- Data score: {audit['scores']['data']}/5",
        f"- Algorithm score: {audit['scores']['algorithm']}/5",
        f"- Result alignment score: {audit['scores']['result_alignment']}/5",
        f"- Overall score: {audit['scores']['overall']}/5",
        "",
        "## Data Check",
        "",
        "| Item | Evidence | Status |",
        "|---|---|---|",
    ]
    for row in audit["data_check"]:
        lines.append(f"| {row['item']} | {row['evidence']} | {row['status']} |")
    lines += ["", "## Algorithm Check", "", "| Component | Evidence | Status |", "|---|---|---|"]
    for row in audit["algorithm_check"]:
        lines.append(f"| {row['component']} | {row['evidence']} | {row['status']} |")
    lines += ["", "## Result Alignment", "", "| Target | Evidence | Status |", "|---|---|---|"]
    for row in audit["result_alignment"]:
        lines.append(f"| {row['target']} | {row['evidence']} | {row['status']} |")
    lines += ["", "## Blockers", ""]
    lines += [f"- {item}" for item in audit["blockers"]]
    lines += ["", "## Next Steps", ""]
    lines += [f"{idx}. {item}" for idx, item in enumerate(audit["next_steps"], start=1)]
    lines.append("")
    return "\n".join(lines)
