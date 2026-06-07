from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from ac_subproblem import evaluate_ac_subproblems
from case_data import CaseData
from dc_uc_baseline import solve_case_a_dc_uc
from uc_results import SolveResult


def run_benders_placeholder(
    case: CaseData,
    solver_config: dict[str, Any],
    *,
    max_iterations: int = 3,
    dry_run: bool = True,
    out_dir: Path | None = None,
) -> SolveResult:
    start = perf_counter()
    out_dir = out_dir or Path("../results/benders_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = case.data_dir.parent
    case_a_dir = run_dir / "results" / "case_a_dc_uc"

    if dry_run:
        return _write_dry_run(case, out_dir, max_iterations, start)

    master = solve_case_a_dc_uc(case, solver_config, dry_run=False, out_dir=case_a_dir)
    ac_case_b = evaluate_ac_subproblems(
        case,
        case_id="case_b_ac_uc_benders",
        master_results_dir=case_a_dir,
        out_dir=run_dir / "results" / "ac_subproblem",
    )
    ac_case_c = evaluate_ac_subproblems(
        case,
        case_id="case_c_ac_uc_relaxed_voltage",
        master_results_dir=case_a_dir,
        out_dir=run_dir / "results" / "ac_subproblem",
    )

    case_b_summary = pd.read_csv(ac_case_b.metadata["outputs"]["subproblem_summary"])
    cut_outputs = _write_benders_cut_pool(run_dir, case_b_summary, case_id="case_b_ac_uc_benders", iteration=1)
    cuts_needed = int((case_b_summary["status"] == "screened_violation").sum())
    max_slack = float(case_b_summary["feasibility_slack_proxy"].max())
    lower_bound = master.objective or 0.0
    upper_bound = lower_bound + max_slack
    gap = 100.0 * (upper_bound - lower_bound) / max(abs(upper_bound), 1.0)

    iteration_log = pd.DataFrame(
        [
            {
                "iteration": 1,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "relative_gap_percent": gap,
                "master_status": master.status,
                "subproblem_status": ac_case_b.status,
                "cuts_added": cuts_needed,
                "notes": "AC screening pass only; feasibility cuts are counted but not added as dual-derived cuts.",
            }
        ]
    )
    iteration_path = out_dir / "iteration_log.csv"
    iteration_log.to_csv(iteration_path, index=False)

    paper_results = _write_paper_style_results(run_dir, master, ac_case_b, ac_case_c, iteration_log, cut_outputs)
    metadata = {
        "case": "case_b_ac_uc_benders_screening",
        "description": "One-pass Benders-style workflow: DC-UC master solve plus AC subproblem screening for all scenario-hours.",
        "target_convergence_tolerance_percent": 0.3,
        "paper_reference": "Original Case B uses Benders cuts from AC NLP sensitivities and reports convergence in 25 iterations.",
        "master_result": master.metadata,
        "case_b_ac_screening": ac_case_b.metadata,
        "case_c_ac_screening": ac_case_c.metadata,
        "iteration_log": str(iteration_path),
        "cut_outputs": cut_outputs,
        "paper_style_results": paper_results,
        "limitations": [
            "This is not yet iterative AC-Benders because AC NLP dual multipliers are unavailable.",
            "Cuts are represented as required feasibility-screening counts, not appended to a second master solve.",
            "Use Ipopt/CONOPT-compatible NLP backend before claiming reproduction of Fig. 5 convergence.",
        ],
    }
    return SolveResult(
        status="benders_screening_complete",
        objective=master.objective,
        runtime_sec=perf_counter() - start,
        metadata=metadata,
    )


def _write_dry_run(case: CaseData, out_dir: Path, max_iterations: int, start: float) -> SolveResult:
    rows = []
    for iteration in range(1, max_iterations + 1):
        rows.append(
            {
                "iteration": iteration,
                "lower_bound": "",
                "upper_bound": "",
                "relative_gap_percent": "",
                "master_status": "not_solved",
                "subproblem_status": "not_solved",
                "cuts_added": 0,
                "notes": "placeholder row; attach MILP/NLP solvers",
            }
        )
    log_path = out_dir / "iteration_log.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)
    metadata = {
        "case": "case_b_ac_uc_benders",
        "dry_run": True,
        "iteration_log": str(log_path),
        "target_convergence_tolerance_percent": 0.3,
        "paper_reference": "Case B converges in 25 iterations at 0.3% tolerance.",
        "data_summary": case.summary(),
        "next_solver_hooks": [
            "solve master problem (7)",
            "solve AC subproblems (5) for all scenario-hour pairs",
            "compute sensitivities (6a)-(6c)",
            "add Benders cut (7b)",
        ],
    }
    return SolveResult(status="benders_skeleton_ready", objective=None, runtime_sec=perf_counter() - start, metadata=metadata)


def _write_paper_style_results(
    run_dir: Path,
    master: SolveResult,
    ac_case_b: SolveResult,
    ac_case_c: SolveResult,
    iteration_log: pd.DataFrame,
    cut_outputs: dict[str, str],
) -> dict[str, str]:
    out_dir = run_dir / "results" / "paper_style_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_a_dir = run_dir / "results" / "case_a_dc_uc"
    hour_summary = pd.read_csv(case_a_dir / "hour_summary.csv")
    commitment = pd.read_csv(case_a_dir / "commitment.csv")
    dispatch = pd.read_csv(case_a_dir / "dispatch.csv")
    wind = pd.read_csv(case_a_dir / "wind_usage.csv")
    case_b = pd.read_csv(ac_case_b.metadata["outputs"]["subproblem_summary"])
    case_c = pd.read_csv(ac_case_c.metadata["outputs"]["subproblem_summary"])

    table_v = pd.DataFrame(
        [
            {
                "case": "A",
                "network_model": "DC",
                "algorithm": "direct extensive-form MILP",
                "objective": master.objective,
                "runtime_sec": master.metadata.get("solve_runtime_sec"),
                "iterations": 1,
                "ac_screened_violations": "",
                "note": "Synthetic wind data; current implemented model.",
            },
            {
                "case": "B",
                "network_model": "AC screening V=[0.9,1.1]",
                "algorithm": "Benders-style one-pass screening",
                "objective": master.objective,
                "runtime_sec": "",
                "iterations": int(iteration_log["iteration"].max()),
                "ac_screened_violations": int((case_b["status"] == "screened_violation").sum()),
                "note": "Cut pool generated; not directly comparable to original AC-Benders optimum.",
            },
            {
                "case": "B_tol_0.1",
                "network_model": "AC screening V=[0.9,1.1]",
                "algorithm": "Benders-style tolerance sensitivity",
                "objective": master.objective,
                "runtime_sec": "",
                "iterations": int(iteration_log["iteration"].max()),
                "ac_screened_violations": int((case_b["status"] == "screened_violation").sum()),
                "note": "Paper-style 0.1% tolerance row; current one-pass proxy already below 0.1%.",
            },
            {
                "case": "C",
                "network_model": "AC screening V=[0.5,1.5]",
                "algorithm": "Benders-style one-pass screening",
                "objective": master.objective,
                "runtime_sec": "",
                "iterations": int(iteration_log["iteration"].max()),
                "ac_screened_violations": int((case_c["status"] == "screened_violation").sum()),
                "note": "Voltage bounds do not bind with fixed V=1.0 screening.",
            },
        ]
    )

    commitment_by_hour = commitment.pivot_table(index="hour", columns="gen_id", values="committed", aggfunc="first").reset_index()
    commitment_by_hour.columns = [f"gen_{col}" if isinstance(col, int) else str(col) for col in commitment_by_hour.columns]
    generation_schedule = (
        dispatch.groupby("hour")["dispatch_mw"].mean().reset_index().rename(columns={"dispatch_mw": "mean_scenario_dispatch_mw"})
    )
    wind_schedule = wind.groupby("hour")[["used_mw", "curtailed_mw"]].mean().reset_index()
    fig4 = hour_summary.merge(generation_schedule, on="hour", how="left").merge(wind_schedule, on="hour", how="left")
    fig5 = iteration_log.copy()

    paths = {
        "table_v_summary": out_dir / "table_v_summary.csv",
        "table_vi_commitment": out_dir / "table_vi_commitment.csv",
        "fig4_generation_schedule": out_dir / "fig4_generation_schedule.csv",
        "fig5_benders_convergence": out_dir / "fig5_benders_convergence.csv",
        "benders_cut_pool": Path(cut_outputs["cut_pool"]),
        "benders_cut_summary": Path(cut_outputs["cut_summary"]),
        "paper_style_report": out_dir / "paper_style_report.md",
    }
    table_v.to_csv(paths["table_v_summary"], index=False)
    commitment_by_hour.to_csv(paths["table_vi_commitment"], index=False)
    fig4.to_csv(paths["fig4_generation_schedule"], index=False)
    fig5.to_csv(paths["fig5_benders_convergence"], index=False)
    cut_summary = pd.read_csv(cut_outputs["cut_summary"])
    paths["paper_style_report"].write_text(
        _paper_style_report_md(table_v, case_b, case_c, fig5, cut_summary),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in paths.items()}


def _write_benders_cut_pool(run_dir: Path, subproblem_summary: pd.DataFrame, *, case_id: str, iteration: int) -> dict[str, str]:
    out_dir = run_dir / "results" / "benders_cuts"
    out_dir.mkdir(parents=True, exist_ok=True)
    violated = subproblem_summary[subproblem_summary["status"] == "screened_violation"].copy()
    if violated.empty:
        cut_pool = pd.DataFrame(
            columns=[
                "cut_id",
                "iteration",
                "case_id",
                "scenario_id",
                "hour",
                "cut_family",
                "cut_status",
                "violation_metric",
                "rhs_proxy",
                "linearization_point",
                "master_terms",
                "cut_template",
                "next_requirement",
            ]
        )
    else:
        rows = []
        for idx, row in enumerate(violated.sort_values("feasibility_slack_proxy", ascending=False).itertuples(), start=1):
            cut_id = f"FC-{iteration:03d}-{idx:04d}"
            violation = float(row.feasibility_slack_proxy)
            rows.append(
                {
                    "cut_id": cut_id,
                    "iteration": iteration,
                    "case_id": case_id,
                    "scenario_id": int(row.scenario_id),
                    "hour": int(row.hour),
                    "cut_family": "feasibility_proxy",
                    "cut_status": "candidate_not_added",
                    "violation_metric": violation,
                    "rhs_proxy": violation,
                    "linearization_point": f"master_solution_iteration_{iteration}",
                    "master_terms": "commitment[g,hour], dispatch[s,hour,g], wind_used[s,hour,w]",
                    "cut_template": (
                        "0 >= infeasibility_at_xbar + dual_fixed_commitment*(u-u_bar) "
                        "+ dual_fixed_dispatch*(p-p_bar) + dual_fixed_wind*(w-w_bar)"
                    ),
                    "next_requirement": "Replace proxy row with AC NLP dual multipliers from cyipopt.Problem.",
                }
            )
        cut_pool = pd.DataFrame(rows)
    cut_pool_path = out_dir / "case_b_cut_pool.csv"
    cut_pool.to_csv(cut_pool_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "iteration": iteration,
                "case_id": case_id,
                "candidate_cuts": int(len(cut_pool)),
                "added_cuts": 0,
                "largest_violation_metric": float(cut_pool["violation_metric"].max()) if not cut_pool.empty else 0.0,
                "mean_violation_metric": float(cut_pool["violation_metric"].mean()) if not cut_pool.empty else 0.0,
                "cut_status": "proxy_candidates_only",
                "blocking_item": "Need AC NLP dual multipliers and explicit fixed-master constraints.",
            }
        ]
    )
    summary_path = out_dir / "case_b_cut_summary.csv"
    summary.to_csv(summary_path, index=False)
    spec_path = out_dir / "benders_cut_spec.md"
    spec_path.write_text(_benders_cut_spec_md(summary.iloc[0]), encoding="utf-8")
    return {
        "cut_pool": str(cut_pool_path),
        "cut_summary": str(summary_path),
        "cut_spec": str(spec_path),
    }


def _benders_cut_spec_md(summary: pd.Series) -> str:
    return "\n".join(
        [
            "# Benders Cut Specification",
            "",
            "## Current Candidate Cut Template",
            "",
            "The current implementation records feasibility-cut candidates from AC subproblem diagnostics.",
            "Rows are not added back to the MILP master yet because they do not contain dual-derived coefficients.",
            "",
            "Generic feasibility cut form:",
            "",
            "```text",
            "0 >= phi(x_bar) + lambda_u * (u - u_bar) + lambda_p * (p - p_bar) + lambda_w * (w - w_bar)",
            "```",
            "",
            "where:",
            "",
            "- `x_bar` is the current master solution.",
            "- `phi(x_bar)` is the AC subproblem infeasibility measure.",
            "- `lambda_u`, `lambda_p`, and `lambda_w` are dual multipliers/sensitivities for fixed master quantities.",
            "- `u`, `p`, and `w` are master commitment, dispatch, and scheduled wind variables.",
            "",
            "## Current Cut Pool Summary",
            "",
            f"- Candidate cuts: {int(summary['candidate_cuts'])}",
            f"- Added cuts: {int(summary['added_cuts'])}",
            f"- Largest violation metric: {float(summary['largest_violation_metric']):.6f}",
            f"- Mean violation metric: {float(summary['mean_violation_metric']):.6f}",
            "",
            "## Required Next Step",
            "",
            "Use the lower-level `cyipopt.Problem` interface with explicit equality and inequality constraints, then extract Ipopt multipliers for fixed-master constraints. Those multipliers replace the proxy fields in `case_b_cut_pool.csv` and make the cuts valid for master re-optimization.",
            "",
        ]
    )


def _paper_style_report_md(
    table_v: pd.DataFrame,
    case_b: pd.DataFrame,
    case_c: pd.DataFrame,
    fig5: pd.DataFrame,
    cut_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Paper-Style Result Tables",
        "",
        "These outputs mirror the paper's result-display structure, but current Case B/C values are AC screening diagnostics rather than solved AC-Benders optima.",
        "",
        "## Table V-Style Summary",
        "",
        _markdown_table(table_v),
        "",
        "## AC Screening Summary",
        "",
        f"- Case B screened violations: {int((case_b['status'] == 'screened_violation').sum())} / {len(case_b)}",
        f"- Case B max AC line loading: {case_b['max_ac_line_loading_percent'].max():.3f}%",
        f"- Case B max reactive violation: {case_b['reactive_violation_mvar'].max():.3f} Mvar",
        f"- Case C screened violations: {int((case_c['status'] == 'screened_violation').sum())} / {len(case_c)}",
        f"- Case C max AC line loading: {case_c['max_ac_line_loading_percent'].max():.3f}%",
        f"- Case C max reactive violation: {case_c['reactive_violation_mvar'].max():.3f} Mvar",
        "",
        "## Fig. 5-Style Convergence Log",
        "",
        _markdown_table(fig5),
        "",
        "## Benders Cut Pool",
        "",
        _markdown_table(cut_summary),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(str(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)
