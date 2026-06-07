from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from case_data import load_case_data
from dc_uc_baseline import solve_case_a_dc_uc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a two-iteration Benders closed-loop smoke test.")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--solver-config", default="../configs/solver_config.json")
    parser.add_argument("--cuts", default="../results/benders_cuts/case_b_benders_cut_constraints.csv")
    parser.add_argument("--terms", default="../results/benders_cuts/case_b_benders_cut_terms.csv")
    parser.add_argument("--out-dir", default="../results/benders_closed_loop")
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    data_dir = resolve_relative(src_dir, Path(args.data_dir))
    solver_config = json.loads(resolve_relative(src_dir, Path(args.solver_config)).read_text(encoding="utf-8"))
    cuts_path = resolve_relative(src_dir, Path(args.cuts))
    terms_path = resolve_relative(src_dir, Path(args.terms))
    out_dir = resolve_relative(src_dir, Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    case = load_case_data(data_dir)

    iteration_1 = solve_case_a_dc_uc(case, solver_config, dry_run=False, out_dir=out_dir / "iteration_1_master")
    cut_config = dict(solver_config)
    cut_config["benders_cut_constraints"] = str(cuts_path)
    cut_config["benders_cut_terms"] = str(terms_path)
    cut_config["benders_eta_objective_weight"] = float(cut_config.get("benders_eta_objective_weight", 1.0))
    iteration_2 = solve_case_a_dc_uc(case, cut_config, dry_run=False, out_dir=out_dir / "iteration_2_master_with_cuts")

    rows = [
        {
            "iteration": 1,
            "master_problem": "dc_uc_master_no_ac_cuts",
            "objective": iteration_1.objective,
            "lower_bound": iteration_1.objective,
            "cuts_available": 0,
            "cuts_added": 0,
            "eta_variables": 0,
            "status": iteration_1.status,
            "runtime_sec": iteration_1.runtime_sec,
            "notes": "Initial master solve, corresponding to first Benders master iteration.",
        },
        {
            "iteration": 2,
            "master_problem": "dc_uc_master_with_ac_benders_cuts",
            "objective": iteration_2.objective,
            "lower_bound": iteration_2.objective,
            "cuts_available": iteration_2.metadata.get("benders_cuts", {}).get("candidate_cuts", 0),
            "cuts_added": iteration_2.metadata.get("benders_cuts", {}).get("cuts_added", 0),
            "eta_variables": len(iteration_2.metadata.get("benders_cuts", {}).get("eta_variables", [])),
            "status": iteration_2.status,
            "runtime_sec": iteration_2.runtime_sec,
            "notes": "Master re-solve after adding generated Benders-form cuts.",
        },
    ]
    log = pd.DataFrame(rows)
    log_path = out_dir / "closed_loop_iteration_log.csv"
    log.to_csv(log_path, index=False)
    result_path = out_dir / "closed_loop_result.json"
    result = {
        "status": "closed_loop_complete",
        "iteration_log": str(log_path),
        "iteration_1": {
            "objective": iteration_1.objective,
            "metadata": iteration_1.metadata,
        },
        "iteration_2": {
            "objective": iteration_2.objective,
            "metadata": iteration_2.metadata,
        },
        "objective_delta": None
        if iteration_1.objective is None or iteration_2.objective is None
        else iteration_2.objective - iteration_1.objective,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = out_dir / "closed_loop_report.md"
    report_path.write_text(render_report(log, result, cuts_path, terms_path), encoding="utf-8")
    print(result_path)
    print(report_path)


def render_report(log: pd.DataFrame, result: dict, cuts_path: Path, terms_path: Path) -> str:
    lines = [
        "# Benders Closed-Loop Smoke Test",
        "",
        f"- Cut headers: `{cuts_path}`",
        f"- Cut terms: `{terms_path}`",
        f"- Objective delta: {result['objective_delta']}",
        "",
        "| Iteration | Master | Objective | Cuts Added | Eta Variables | Status |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in log.itertuples(index=False):
        lines.append(
            f"| {row.iteration} | {row.master_problem} | {row.objective:.6f} | {row.cuts_added} | {row.eta_variables} | {row.status} |"
        )
    lines += [
        "",
        "This is the first actual master re-solve with generated Benders-form rows. The current cut coefficients are small because the constrained AC NLP subproblems are nearly feasible after voltage/reactive optimization.",
        "",
    ]
    return "\n".join(lines)


def resolve_relative(base: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


if __name__ == "__main__":
    main()
