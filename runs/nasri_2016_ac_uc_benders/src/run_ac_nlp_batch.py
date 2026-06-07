from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ac_subproblem import solve_ac_nlp_subproblem
from case_data import load_case_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small batch of AC NLP subproblems selected from screening diagnostics.")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--solver-config", default="../configs/solver_config.json")
    parser.add_argument("--case-id", default="case_b_ac_uc_benders")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--selection", choices=["worst-reactive", "worst-line", "first"], default="worst-reactive")
    parser.add_argument("--ac-nlp-solver", choices=["scipy_slsqp", "cyipopt", "cyipopt_constrained"])
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    data_dir = resolve_relative(src_dir, Path(args.data_dir))
    solver_config = json.loads(resolve_relative(src_dir, Path(args.solver_config)).read_text(encoding="utf-8"))
    if args.ac_nlp_solver:
        solver_config["ac_nlp_solver"] = args.ac_nlp_solver
    case = load_case_data(data_dir)
    run_dir = data_dir.parent
    master_results_dir = run_dir / "results" / "case_a_dc_uc"
    screening_path = run_dir / "results" / "ac_subproblem" / f"{args.case_id}_subproblem_summary.csv"
    out_dir = run_dir / "results" / "ac_nlp_subproblem"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = select_subproblems(screening_path, args.selection, args.limit)
    rows = []
    for item in selected.itertuples(index=False):
        result = solve_ac_nlp_subproblem(
            case,
            case_id=args.case_id,
            master_results_dir=master_results_dir,
            out_dir=out_dir,
            scenario_id=int(item.scenario_id),
            hour=int(item.hour),
            solver_config=solver_config,
        )
        rows.append(
            {
                "case_id": args.case_id,
                "scenario_id": int(item.scenario_id),
                "hour": int(item.hour),
                "screening_reactive_violation_mvar": float(item.reactive_violation_mvar),
                "screening_max_line_loading_percent": float(item.max_ac_line_loading_percent),
                "nlp_status": result.status,
                "nlp_success": result.metadata.get("success"),
                "nlp_objective": result.objective,
                "nlp_iterations": result.metadata.get("iterations"),
                "nlp_max_p_residual_mw": result.metadata.get("max_p_residual_mw"),
                "nlp_max_q_residual_mvar": result.metadata.get("max_q_residual_mvar"),
                "nlp_max_line_loading_percent": result.metadata.get("max_ac_line_loading_percent"),
                "summary": result.metadata.get("outputs", {}).get("summary"),
            }
        )
    batch = pd.DataFrame(rows)
    solver_label = solver_config.get("ac_nlp_solver", "scipy_slsqp")
    out_path = out_dir / f"{args.case_id}_{solver_label}_nlp_batch_{args.selection}_{args.limit}.csv"
    batch.to_csv(out_path, index=False)
    report_path = out_dir / f"{args.case_id}_{solver_label}_nlp_batch_{args.selection}_{args.limit}.md"
    report_path.write_text(render_report(batch, args.selection, solver_label), encoding="utf-8")
    print(out_path)
    print(report_path)


def select_subproblems(path: Path, selection: str, limit: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if selection == "worst-reactive":
        return df.sort_values("reactive_violation_mvar", ascending=False).head(limit)
    if selection == "worst-line":
        return df.sort_values("max_ac_line_loading_percent", ascending=False).head(limit)
    return df.sort_values(["scenario_id", "hour"]).head(limit)


def render_report(batch: pd.DataFrame, selection: str, solver_label: str) -> str:
    lines = [
        "# AC NLP Batch Report",
        "",
        f"- Selection: `{selection}`",
        f"- Solver: `{solver_label}`",
        f"- Solved subproblems: {len(batch)}",
        f"- Successful NLP solves: {int(batch['nlp_success'].fillna(False).sum())}",
        f"- Max post-NLP P residual: {batch['nlp_max_p_residual_mw'].max():.6f} MW",
        f"- Max post-NLP Q residual: {batch['nlp_max_q_residual_mvar'].max():.6f} Mvar",
        f"- Max post-NLP line loading: {batch['nlp_max_line_loading_percent'].max():.6f}%",
        "",
        "| Scenario | Hour | Screening Q Violation | NLP P Residual | NLP Q Residual | NLP Line Loading |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in batch.itertuples(index=False):
        lines.append(
            f"| {row.scenario_id} | {row.hour} | {row.screening_reactive_violation_mvar:.3f} | "
            f"{row.nlp_max_p_residual_mw:.6f} | {row.nlp_max_q_residual_mvar:.6f} | {row.nlp_max_line_loading_percent:.3f}% |"
        )
    lines += [
        "",
        "The constrained Ipopt backend writes explicit constraint values and multipliers for future Benders cut construction.",
        "",
    ]
    return "\n".join(lines)


def resolve_relative(base: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


if __name__ == "__main__":
    main()
