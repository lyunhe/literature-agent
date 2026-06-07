from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


DATA_TEMPLATES: dict[str, list[str]] = {
    "buses.csv": ["bus_id", "base_kv", "area", "zone", "pd_fraction", "notes"],
    "lines.csv": [
        "line_id",
        "from_bus",
        "to_bus",
        "x_pu",
        "rate_mw",
        "lsf_row_source",
        "notes",
    ],
    "generators.csv": [
        "gen_id",
        "bus_id",
        "p_min_mw",
        "p_max_mw",
        "startup_cost",
        "shutdown_cost",
        "fixed_cost",
        "ramp_up_mw",
        "ramp_down_mw",
        "min_up_h",
        "min_down_h",
        "initial_status",
        "initial_p_mw",
        "notes",
    ],
    "generator_cost_segments.csv": [
        "gen_id",
        "segment_id",
        "p_max_segment_mw",
        "marginal_cost",
        "notes",
    ],
    "reserves.csv": [
        "hour",
        "spinning_reserve_mw",
        "operating_reserve_mw",
        "notes",
    ],
    "load_profile.csv": [
        "hour",
        "total_load_mw",
        "nodal_allocation_source",
        "notes",
    ],
    "wind_farms.csv": [
        "wind_id",
        "bus_id",
        "p_nom_mw",
        "profile_source",
        "notes",
    ],
    "wind_profile.csv": [
        "hour",
        "wind_id",
        "forecast_mw",
        "lower_bound_mw",
        "upper_bound_mw",
        "notes",
    ],
    "uncertainty_bounds.csv": [
        "hour",
        "component_type",
        "component_id",
        "forecast_mw",
        "lower_bound_mw",
        "upper_bound_mw",
        "relative_bound",
        "notes",
    ],
    "scenario_probabilities.csv": [
        "scenario_id",
        "probability",
        "source",
        "notes",
    ],
    "load_factors.csv": [
        "hour",
        "load_factor",
        "source",
        "notes",
    ],
    "paper_parameters.csv": [
        "parameter",
        "value",
        "unit",
        "source",
        "status",
    ],
    "wind_scenario_statistics.csv": [
        "scenario_id",
        "probability",
        "scenario_total_mwh",
        "scenario_average_mw",
        "scenario_capacity_factor",
    ],
}


def scaffold_reproduction_package(target: dict[str, Any]) -> list[Path]:
    run_dir = Path(target["run_dir"])
    data_dir = run_dir / "data"
    configs_dir = run_dir / "configs"
    src_dir = run_dir / "src"
    reports_dir = run_dir / "reports"
    for directory in [data_dir, configs_dir, src_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for filename, columns in DATA_TEMPLATES.items():
        path = data_dir / filename
        if not path.exists():
            write_csv_header(path, columns)
            created.append(path)

    for path, content in [
        (configs_dir / "experiment_matrix.json", default_experiment_matrix(target)),
        (configs_dir / "solver_config.json", default_solver_config()),
        (configs_dir / "reproduction_assumptions.json", default_assumptions(target)),
    ]:
        if not path.exists():
            path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
            created.append(path)

    for path, text in [
        (src_dir / "README.md", source_readme(target)),
        (src_dir / "model_interfaces.py", model_interfaces_py()),
        (src_dir / "run_reproduction.py", run_reproduction_py()),
        (reports_dir / "reproduction_checklist.md", reproduction_checklist_md(target)),
    ]:
        if not path.exists():
            path.write_text(text, encoding="utf-8")
            created.append(path)
    return created


def write_csv_header(path: Path, columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)


def default_experiment_matrix(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper": target.get("id"),
        "case": "modified_ieee_118_placeholder",
        "horizon_hours": 24,
        "experiments": [
            {
                "id": "deterministic_uc_baseline",
                "model": "deterministic_uc",
                "uncertainty": "forecast_only",
                "expected_outputs": [
                    "objective",
                    "commitment_matrix",
                    "dispatch",
                    "line_flows",
                ],
            },
            {
                "id": "robust_uc_ccg_full_tlc",
                "model": "two_stage_robust_uc",
                "algorithm": "ccg_full_transmission_constraints",
                "expected_outputs": [
                    "objective",
                    "lower_bound_trace",
                    "upper_bound_trace",
                    "worst_case_scenarios",
                    "runtime",
                ],
            },
            {
                "id": "robust_uc_ccg_lsf_cutting_plane",
                "model": "two_stage_robust_uc",
                "algorithm": "ccg_lsf_cutting_plane_master",
                "expected_outputs": [
                    "objective",
                    "active_line_count",
                    "iteration_count",
                    "runtime",
                ],
            },
        ],
    }


def default_solver_config() -> dict[str, Any]:
    return {
        "preferred_solvers": ["gurobi", "cplex", "highs"],
        "mip_gap": 0.001,
        "time_limit_sec": 7200,
        "threads": None,
        "feasibility_tolerance": None,
        "notes": [
            "Original paper used AMPL/CPLEX and C++/Coin-OR Bcp for specialized subproblem variants.",
            "Record all deviations from the paper solver stack.",
        ],
    }


def default_assumptions(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper": target.get("title"),
        "assumptions": [
            {
                "id": "A1",
                "status": "open",
                "text": "Start from a public IEEE 118-bus base case until the exact modified case is traced.",
            },
            {
                "id": "A2",
                "status": "open",
                "text": "Use documented wind farm locations if author/cited data cannot be found.",
            },
            {
                "id": "A3",
                "status": "open",
                "text": "Use +/-20% wind and +/-3% load uncertainty once the forecast profiles are reconstructed.",
            },
        ],
    }


def source_readme(target: dict[str, Any]) -> str:
    return f"""# Reproduction Source Skeleton

Target paper: {target.get("title")}

This directory intentionally contains interfaces and runners before any solver-specific implementation. Fill the data templates first, run validation, then implement deterministic UC before robust C&CG.
"""


def model_interfaces_py() -> str:
    return '''"""Interfaces for the reproduction implementation.

These functions are deliberately stubs. They define the contract that the
reproduction runner and future Pyomo/JuMP implementation should satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SolveResult:
    status: str
    objective: float | None
    runtime_sec: float | None
    metadata: dict[str, Any]


def load_case_data(data_dir: Path) -> dict[str, Any]:
    raise NotImplementedError("Load CSV templates and construct model data.")


def solve_deterministic_uc(data: dict[str, Any], solver_config: dict[str, Any]) -> SolveResult:
    raise NotImplementedError("Implement deterministic UC first.")


def solve_robust_uc_ccg(data: dict[str, Any], solver_config: dict[str, Any]) -> SolveResult:
    raise NotImplementedError("Implement C&CG robust UC after deterministic UC is validated.")
'''


def run_reproduction_py() -> str:
    return '''from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_interfaces import load_case_data, solve_deterministic_uc, solve_robust_uc_ccg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--solver-config", default="../configs/solver_config.json")
    parser.add_argument("--experiment", choices=["deterministic", "robust_ccg"], default="deterministic")
    args = parser.parse_args()

    data = load_case_data(Path(args.data_dir))
    solver_config = json.loads(Path(args.solver_config).read_text(encoding="utf-8"))
    if args.experiment == "deterministic":
        result = solve_deterministic_uc(data, solver_config)
    else:
        result = solve_robust_uc_ccg(data, solver_config)
    print(result)


if __name__ == "__main__":
    main()
'''


def reproduction_checklist_md(target: dict[str, Any]) -> str:
    return f"""# Reproduction Checklist: {target.get("title")}

## Data

- [ ] Public base case selected and version recorded.
- [ ] Modified IEEE 118-bus changes traced.
- [ ] Generator UC parameters completed.
- [ ] 24-hour load profile completed.
- [ ] Wind farm buses and forecast profiles completed.
- [ ] Line limits and LSF/PTDF construction documented.
- [ ] Unit conventions and base MVA documented.

## Model

- [ ] Deterministic UC implemented.
- [ ] Deterministic UC unit tests pass on toy data.
- [ ] DC/LSF transmission constraints validated.
- [ ] Robust master problem implemented.
- [ ] Feasibility subproblem implemented.
- [ ] Optimality subproblem implemented.
- [ ] C&CG bound trace logged.
- [ ] LSF cutting-plane active-line selection implemented.

## Results

- [ ] Objective values exported.
- [ ] Commitment matrix exported.
- [ ] Worst-case scenarios exported.
- [ ] Runtime and iteration counts exported.
- [ ] Alignment report generated.
"""


def validate_data_templates(target: dict[str, Any]) -> dict[str, Any]:
    data_dir = Path(target["run_dir"]) / "data"
    optional_files = set(target.get("optional_data_files", []) or [])
    checks: list[dict[str, Any]] = []
    for filename, columns in DATA_TEMPLATES.items():
        path = data_dir / filename
        if not path.exists():
            status = "optional_missing" if filename in optional_files else "missing"
            checks.append({"file": filename, "status": status, "rows": 0, "message": "Template missing"})
            continue
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        header = rows[0] if rows else []
        missing_cols = [col for col in columns if col not in header]
        data_rows = max(0, len(rows) - 1)
        status = "ok" if not missing_cols and data_rows > 0 else "empty" if not missing_cols else "bad_header"
        if status == "empty" and filename in optional_files:
            status = "optional_empty"
        checks.append(
            {
                "file": filename,
                "status": status,
                "rows": data_rows,
                "missing_columns": missing_cols,
            }
        )
    summary = {
        "target": target.get("id"),
        "complete_files": sum(1 for item in checks if item["status"] == "ok"),
        "empty_files": sum(1 for item in checks if item["status"] == "empty"),
        "missing_files": sum(1 for item in checks if item["status"] == "missing"),
        "bad_header_files": sum(1 for item in checks if item["status"] == "bad_header"),
        "optional_empty_files": sum(1 for item in checks if item["status"] == "optional_empty"),
        "optional_missing_files": sum(1 for item in checks if item["status"] == "optional_missing"),
        "checks": checks,
    }
    report_dir = Path(target["run_dir"]) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "data_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "data_validation.md").write_text(data_validation_md(summary), encoding="utf-8")
    return summary


def data_validation_md(summary: dict[str, Any]) -> str:
    lines = [
        f"# Data Validation: {summary['target']}",
        "",
        f"- Complete files: {summary['complete_files']}",
        f"- Empty files: {summary['empty_files']}",
        f"- Missing files: {summary['missing_files']}",
        f"- Bad headers: {summary['bad_header_files']}",
        f"- Optional empty files: {summary.get('optional_empty_files', 0)}",
        f"- Optional missing files: {summary.get('optional_missing_files', 0)}",
        "",
        "| File | Status | Rows | Missing Columns |",
        "|---|---|---:|---|",
    ]
    for item in summary["checks"]:
        lines.append(
            f"| {item['file']} | {item['status']} | {item.get('rows', 0)} | {', '.join(item.get('missing_columns', []))} |"
        )
    lines.append("")
    return "\n".join(lines)


def extract_reproduction_manifest(text_json: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_figures: list[dict[str, Any]] = []
    equations: list[dict[str, Any]] = []
    table_re = re.compile(r"\b(TABLE|Table|Fig\.|Figure)\s+([IVXLC]+|\d+)", re.IGNORECASE)
    eq_re = re.compile(r"\((\d+[a-z]?)\)")
    for page in text_json.get("pages", []):
        text = page.get("text", "")
        for match in table_re.finditer(text):
            start = max(0, match.start() - 180)
            end = min(len(text), match.end() + 400)
            table_figures.append(
                {
                    "page": page["page"],
                    "label": match.group(0),
                    "context": text[start:end].replace("\n", " ").strip(),
                }
            )
        for match in eq_re.finditer(text):
            start = max(0, match.start() - 180)
            end = min(len(text), match.end() + 240)
            equations.append(
                {
                    "page": page["page"],
                    "equation": match.group(1),
                    "context": text[start:end].replace("\n", " ").strip(),
                }
            )
    fig_path = out_dir / "figures_tables_manifest.json"
    eq_path = out_dir / "equations_manifest.json"
    fig_path.write_text(json.dumps(table_figures, ensure_ascii=False, indent=2), encoding="utf-8")
    eq_path.write_text(json.dumps(equations, ensure_ascii=False, indent=2), encoding="utf-8")
    return fig_path, eq_path
