from __future__ import annotations

from pathlib import Path
from typing import Any


def write_algorithm_trace(target: dict[str, Any], out_path: str | Path) -> Path:
    lines = [
        f"# Algorithm Trace: {target['title']}",
        "",
        "## Recommended Implementation Path",
        "",
        "1. Build deterministic UC model from the appendix.",
        "2. Implement reduced C&CG master problem with an initial scenario set.",
        "3. Solve robust feasibility subproblem for the current commitment.",
        "4. If infeasible, add the returned worst-case scenario to the master.",
        "5. If feasible, solve optimality subproblem for worst-case dispatch cost.",
        "6. Update lower/upper bounds and stop when the relative gap meets tolerance.",
        "7. Add LSF cutting-plane to the master to include only violated transmission constraints.",
        "8. Treat subproblem column-generation heuristics as a later acceleration.",
        "",
        "## Paper-to-Code Map",
        "",
        "| Paper object | Implementation function | Inputs | Outputs | Notes |",
        "|---|---|---|---|---|",
        "| Deterministic UC appendix | `build_deterministic_uc()` | generator data, load, wind, LSF, reserves | MILP model | First model to validate. |",
        "| Reduced robust master | `solve_master(scenarios, active_lines)` | scenario pool, active TLC set | commitment, dispatch recourse, lower bound | C&CG master. |",
        "| Feasibility subproblem | `solve_feasibility_subproblem(commitment)` | commitment, uncertainty set | violating scenario or feasible flag | Add scenario if dispatch impossible. |",
        "| Optimality subproblem | `solve_optimality_subproblem(commitment)` | commitment, uncertainty set | worst-case scenario, dispatch cost, upper bound | MILP via extreme-point/big-M reformulation. |",
        "| LSF cutting-plane | `update_active_lines(solution)` | dispatch flows, LSF matrix, line limits | violated/critical lines | Use before subproblem column generation. |",
        "| Experiment runner | `run_ccg(config)` | case data, solver config, tolerance | logs, solution, alignment metrics | Persist every iteration. |",
        "",
        "## Minimum Iteration Log",
        "",
        "| Field | Description |",
        "|---|---|",
        "| iteration | C&CG iteration index |",
        "| lower_bound | Master objective |",
        "| upper_bound | Commitment cost plus worst-case dispatch cost |",
        "| relative_gap | Stopping metric |",
        "| feasibility_status | Robustly feasible or violating scenario found |",
        "| scenario_id | Worst-case scenario added or evaluated |",
        "| active_line_count | Number of TLCs retained by LSF cutting-plane |",
        "| master_runtime_sec | Master solve time |",
        "| subproblem_runtime_sec | Subproblem solve time |",
        "",
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_source_trace(target: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "source_trace.md"
    csv_path = out_dir / "dataset_registry.csv"

    md_path.write_text(
        "\n".join(
            [
                f"# Source Trace: {target['title']}",
                "",
                "## Data Dependencies",
                "",
                "| Item | Current Status | Reproduction Action |",
                "|---|---|---|",
                "| Modified IEEE 118-bus network | Partially specified in paper | Start from MATPOWER/PGLib IEEE 118 and trace generator/line modifications. |",
                "| 54-generator UC data | Paper reports count, not full machine-readable table | Trace cited UC data source or reconstruct from standard IEEE 118 UC datasets. |",
                "| Three wind farms | Three identical wind stations are specified, but bus locations need tracing | Search cited references or define documented reproduction assumptions. |",
                "| 24-hour load profile | Peak value is specified; full profile needs transcription/source tracing | Extract table/figure or reuse cited benchmark profile with clear note. |",
                "| Uncertainty set | +/-20% wind and +/-3% load are specified | Directly implement after profile reconstruction. |",
                "| Solver stack | AMPL/CPLEX and C++/Coin-OR Bcp are specified | Reproduce first in Pyomo/JuMP + Gurobi/CPLEX/HiGHS, record solver differences. |",
                "",
                "## Source-Tracing Questions",
                "",
                "- Which IEEE 118-bus data version is modified?",
                "- Which buses host the three wind stations?",
                "- What is the exact 24-hour load shape?",
                "- Are generator startup, shutdown, ramp, reserve and piecewise cost parameters taken from a cited UC benchmark?",
                "- Are line limits original, modified, or scaled?",
                "",
            ]
        ),
        encoding="utf-8",
    )

    csv_path.write_text(
        "\n".join(
            [
                "item,type,source_hint,availability,reproduction_status,notes",
                "Modified IEEE 118-bus network,network,MATPOWER/PGLib IEEE 118,public base case,needs tracing,Paper uses modified case",
                "UC generator parameters,generator data,cited UC benchmark or paper appendix,partially available,needs tracing,54 generators",
                "Wind farms,renewable data,paper/cited references,partially available,needs assumption,Three identical wind stations",
                "Load profile,time series,paper/cited references,partially available,needs extraction,24h peak 3733 MWh",
                "Uncertainty set,model parameter,paper,available,rebuildable,+/-20% wind and +/-3% load",
            ]
        ),
        encoding="utf-8",
    )
    return md_path, csv_path

