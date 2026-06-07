from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from ac_subproblem import evaluate_ac_subproblems, solve_ac_nlp_subproblem
from case_data import load_case_data
from dc_uc_baseline import solve_case_a_dc_uc
from generate_benders_cut_constraints import build_cut_constraints


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an automatic Benders-style master/subproblem/cut loop.")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--solver-config", default="../configs/solver_config.json")
    parser.add_argument("--out-dir", default="../results/benders_auto_loop")
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--min-iterations", type=int, default=2)
    parser.add_argument("--cuts-per-iteration", type=int, default=3)
    parser.add_argument("--tolerance-percent", type=float, default=0.3)
    parser.add_argument("--case-id", default="case_b_ac_uc_benders")
    args = parser.parse_args()

    start = perf_counter()
    src_dir = Path(__file__).resolve().parent
    data_dir = resolve_relative(src_dir, Path(args.data_dir))
    solver_config = json.loads(resolve_relative(src_dir, Path(args.solver_config)).read_text(encoding="utf-8"))
    solver_config["ac_nlp_solver"] = "cyipopt_constrained"
    out_dir = resolve_relative(src_dir, Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    case = load_case_data(data_dir)

    cumulative_headers = pd.DataFrame()
    cumulative_terms = pd.DataFrame()
    cut_subproblem_keys: set[tuple[int, int]] = set()
    failed_subproblem_keys: set[tuple[int, int]] = set()
    log_rows: list[dict[str, Any]] = []
    stop_reason = "max_iterations_reached"

    for iteration in range(1, args.max_iterations + 1):
        iteration_dir = out_dir / f"iteration_{iteration:02d}"
        master_dir = iteration_dir / "master"
        cut_config = dict(solver_config)
        if not cumulative_headers.empty:
            cuts_dir = out_dir / "cumulative_cuts"
            cuts_dir.mkdir(parents=True, exist_ok=True)
            headers_path = cuts_dir / "benders_cut_constraints.csv"
            terms_path = cuts_dir / "benders_cut_terms.csv"
            cumulative_headers.to_csv(headers_path, index=False)
            cumulative_terms.to_csv(terms_path, index=False)
            cut_config["benders_cut_constraints"] = str(headers_path)
            cut_config["benders_cut_terms"] = str(terms_path)
            cut_config["benders_eta_objective_weight"] = float(cut_config.get("benders_eta_objective_weight", 1.0))

        master = solve_case_a_dc_uc(case, cut_config, dry_run=False, out_dir=master_dir)
        if master.status != "solved":
            log_rows.append(
                {
                    "iteration": iteration,
                    "lower_bound": float(master.objective or 0.0),
                    "upper_bound_proxy": float(master.objective or 0.0),
                    "evaluated_upper_bound": float(master.objective or 0.0),
                    "relative_gap_percent": 0.0,
                    "master_status": master.status,
                    "master_objective": master.objective,
                    "first_stage_proxy_cost": float(master.objective or 0.0),
                    "expected_master_eta_cost": 0.0,
                    "cuts_active_in_master": 0,
                    "eta_variables": 0,
                    "new_cuts_generated": 0,
                    "cumulative_cuts": len(cumulative_headers),
                    "selected_subproblems": 0,
                    "successful_nlp_subproblems": 0,
                    "failed_nlp_subproblems": 0,
                    "max_successful_nlp_objective": 0.0,
                    "expected_successful_nlp_objective": 0.0,
                    "max_failed_nlp_objective": 0.0,
                    "max_screening_reactive_violation_mvar": 0.0,
                    "previously_cut_candidates_skipped": 0,
                    "previously_failed_candidates_skipped": 0,
                    "notes": f"Stopped because master solve failed with status {master.status}.",
                }
            )
            stop_reason = f"master_failed_{master.status}"
            break
        screening = evaluate_ac_subproblems(
            case,
            case_id=args.case_id,
            master_results_dir=master_dir,
            out_dir=iteration_dir / "ac_screening",
        )
        screening_summary = pd.read_csv(screening.metadata["outputs"]["subproblem_summary"])
        selected, selection_stats = select_subproblems(
            screening_summary,
            args.cuts_per_iteration,
            cut_subproblem_keys=cut_subproblem_keys,
            failed_subproblem_keys=failed_subproblem_keys,
        )

        nlp_rows = []
        for item in selected.itertuples(index=False):
            nlp = solve_ac_nlp_subproblem(
                case,
                case_id=args.case_id,
                master_results_dir=master_dir,
                out_dir=iteration_dir / "ac_nlp",
                scenario_id=int(item.scenario_id),
                hour=int(item.hour),
                solver_config=solver_config,
            )
            nlp_rows.append(
                {
                    "case_id": args.case_id,
                    "scenario_id": int(item.scenario_id),
                    "hour": int(item.hour),
                    "screening_reactive_violation_mvar": float(item.reactive_violation_mvar),
                    "screening_max_line_loading_percent": float(item.max_ac_line_loading_percent),
                    "nlp_status": nlp.status,
                    "nlp_success": bool(nlp.metadata.get("success")),
                    "nlp_objective": nlp.objective,
                    "nlp_max_p_residual_mw": nlp.metadata.get("max_p_residual_mw"),
                    "nlp_max_q_residual_mvar": nlp.metadata.get("max_q_residual_mvar"),
                    "nlp_max_line_loading_percent": nlp.metadata.get("max_ac_line_loading_percent"),
                    "nlp_sum_load_shed_mw": nlp.metadata.get("sum_load_shed_mw"),
                    "nlp_sum_wind_spillage_mw": nlp.metadata.get("sum_wind_spillage_mw"),
                    "nlp_sum_reserve_up_mw": nlp.metadata.get("sum_reserve_up_mw"),
                    "nlp_sum_reserve_down_mw": nlp.metadata.get("sum_reserve_down_mw"),
                    "summary": nlp.metadata.get("outputs", {}).get("summary"),
                    "multipliers": nlp.metadata.get("outputs", {}).get("multipliers"),
                }
            )
        nlp_batch = pd.DataFrame(nlp_rows)
        nlp_batch_path = iteration_dir / "ac_nlp_batch.csv"
        nlp_batch.to_csv(nlp_batch_path, index=False)

        successful_batch = successful_nlp_rows(nlp_batch)
        for item in successful_batch.itertuples(index=False):
            cut_subproblem_keys.add((int(item.scenario_id), int(item.hour)))
        failed_batch = failed_nlp_rows(nlp_batch)
        for item in failed_batch.itertuples(index=False):
            failed_subproblem_keys.add((int(item.scenario_id), int(item.hour)))

        coeffs = build_dual_coefficients(case.data_dir, successful_batch)
        coeffs_path = iteration_dir / "dual_cut_coefficients.csv"
        coeffs.to_csv(coeffs_path, index=False)
        dispatch = pd.read_csv(master_dir / "dispatch.csv")
        wind = pd.read_csv(master_dir / "wind_usage.csv")
        summaries = load_summary_map(successful_batch)
        if coeffs.empty:
            headers = empty_cut_headers()
            terms = empty_cut_terms()
        else:
            probabilities = load_probabilities(case.data_dir)
            headers, terms = build_cut_constraints(coeffs, dispatch, wind, summaries, cut_type="optimality_cut", probabilities=probabilities)
            headers = relabel_cuts(headers, iteration)
            terms = relabel_terms(terms, headers)
        headers_path = iteration_dir / "benders_cut_constraints.csv"
        terms_path = iteration_dir / "benders_cut_terms.csv"
        headers.to_csv(headers_path, index=False)
        terms.to_csv(terms_path, index=False)

        cumulative_headers = pd.concat([cumulative_headers, headers], ignore_index=True)
        cumulative_terms = pd.concat([cumulative_terms, terms], ignore_index=True)

        probabilities = load_probabilities(case.data_dir)
        expected_success_phi = expected_phi(successful_batch, probabilities)
        max_success_phi = finite_max(successful_batch, "nlp_objective")
        max_failed_phi = finite_max(failed_batch, "nlp_objective")
        lower_bound = float(master.objective or 0.0)
        objective_breakdown = master.metadata.get("objective_breakdown", {})
        first_stage_proxy = (
            float(objective_breakdown.get("startup_cost", 0.0))
            + float(objective_breakdown.get("expected_dispatch_cost", 0.0))
            + float(objective_breakdown.get("expected_load_shed_cost", 0.0))
        )
        expected_master_eta_cost = float(objective_breakdown.get("expected_eta_cost", 0.0))
        evaluated_upper_bound = first_stage_proxy + expected_success_phi if not successful_batch.empty else None
        upper_bound_proxy = evaluated_upper_bound if evaluated_upper_bound is not None else lower_bound
        gap = (
            100.0 * max(0.0, float(evaluated_upper_bound) - lower_bound) / max(abs(float(evaluated_upper_bound)), 1.0)
            if evaluated_upper_bound is not None
            else float("nan")
        )
        benders_cuts = master.metadata.get("benders_cuts", {})
        log_rows.append(
            {
                "iteration": iteration,
                "lower_bound": lower_bound,
                "upper_bound_proxy": upper_bound_proxy,
                "evaluated_upper_bound": evaluated_upper_bound,
                "relative_gap_percent": gap,
                "master_status": master.status,
                "master_objective": master.objective,
                "first_stage_proxy_cost": first_stage_proxy,
                "expected_master_eta_cost": expected_master_eta_cost,
                "cuts_active_in_master": benders_cuts.get("cuts_added", 0),
                "eta_variables": len(benders_cuts.get("eta_variables", [])),
                "new_cuts_generated": len(headers),
                "cumulative_cuts": len(cumulative_headers),
                "selected_subproblems": len(selected),
                "successful_nlp_subproblems": int(nlp_batch["nlp_success"].fillna(False).sum()) if not nlp_batch.empty else 0,
                "failed_nlp_subproblems": int((~nlp_batch["nlp_success"].fillna(False)).sum()) if not nlp_batch.empty else 0,
                "max_successful_nlp_objective": max_success_phi,
                "expected_successful_nlp_objective": expected_success_phi,
                "max_failed_nlp_objective": max_failed_phi,
                "max_screening_reactive_violation_mvar": float(selected["reactive_violation_mvar"].max()) if not selected.empty else 0.0,
                "previously_cut_candidates_skipped": selection_stats["previously_cut_candidates_skipped"],
                "previously_failed_candidates_skipped": selection_stats["previously_failed_candidates_skipped"],
                "notes": "Paper-style loop: solve master, solve selected AC NLP subproblems, generate cuts. Lower bound is the master objective. Evaluated upper bound is reconstructed from first-stage proxy cost plus expected AC recourse on successful NLP subproblems.",
            }
        )
        write_iteration_report(iteration_dir / "iteration_report.md", log_rows[-1], selected, nlp_batch)
        if (
            iteration >= args.min_iterations
            and evaluated_upper_bound is not None
            and pd.notna(gap)
            and gap <= args.tolerance_percent
            and log_rows[-1]["failed_nlp_subproblems"] == 0
        ):
            stop_reason = "tolerance_reached"
            break

    cumulative_dir = out_dir / "cumulative_cuts"
    cumulative_dir.mkdir(parents=True, exist_ok=True)
    cumulative_headers_path = cumulative_dir / "benders_cut_constraints.csv"
    cumulative_terms_path = cumulative_dir / "benders_cut_terms.csv"
    cumulative_headers.to_csv(cumulative_headers_path, index=False)
    cumulative_terms.to_csv(cumulative_terms_path, index=False)
    log = pd.DataFrame(log_rows)
    log_path = out_dir / "auto_loop_iteration_log.csv"
    log.to_csv(log_path, index=False)
    result = {
        "status": "auto_loop_complete",
        "stop_reason": stop_reason,
        "runtime_sec": perf_counter() - start,
        "iterations": len(log_rows),
        "tolerance_percent": args.tolerance_percent,
        "cuts_per_iteration": args.cuts_per_iteration,
        "cumulative_cuts": int(len(cumulative_headers)),
        "unique_cut_subproblems": int(len(cut_subproblem_keys)),
        "failed_subproblems_seen": int(len(failed_subproblem_keys)),
        "iteration_log": str(log_path),
        "cumulative_cut_constraints": str(cumulative_headers_path),
        "cumulative_cut_terms": str(cumulative_terms_path),
    }
    result_path = out_dir / "auto_loop_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = out_dir / "auto_loop_report.md"
    report_path.write_text(render_report(result, log), encoding="utf-8")
    print(result_path)
    print(report_path)


def select_subproblems(
    summary: pd.DataFrame,
    limit: int,
    *,
    cut_subproblem_keys: set[tuple[int, int]],
    failed_subproblem_keys: set[tuple[int, int]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    ranked = summary.sort_values(
        ["feasibility_slack_proxy", "reactive_violation_mvar", "max_ac_line_loading_percent"],
        ascending=False,
    ).copy()
    ranked["_key"] = list(zip(ranked["scenario_id"].astype(int), ranked["hour"].astype(int)))
    previously_cut = ranked["_key"].isin(cut_subproblem_keys)
    previously_failed = ranked["_key"].isin(failed_subproblem_keys)
    fresh = ranked[~previously_cut & ~previously_failed].head(limit)
    if len(fresh) < limit:
        backfill = ranked[~ranked["_key"].isin(set(fresh["_key"])) & ~previously_cut].head(limit - len(fresh))
        selected = pd.concat([fresh, backfill], ignore_index=True)
    else:
        selected = fresh
    selected = selected.drop(columns=["_key"])
    stats = {
        "previously_cut_candidates_skipped": int(previously_cut.sum()),
        "previously_failed_candidates_skipped": int(previously_failed.sum()),
    }
    return selected, stats


def successful_nlp_rows(batch: pd.DataFrame) -> pd.DataFrame:
    if batch.empty or "nlp_success" not in batch.columns:
        return batch.iloc[0:0].copy()
    mask = batch["nlp_success"].fillna(False).astype(bool)
    if "multipliers" in batch.columns:
        mask &= batch["multipliers"].fillna("").astype(str).map(lambda value: value != "" and Path(value).exists())
    if "summary" in batch.columns:
        mask &= batch["summary"].fillna("").astype(str).map(lambda value: value != "" and Path(value).exists())
    if "nlp_objective" in batch.columns:
        mask &= pd.to_numeric(batch["nlp_objective"], errors="coerce").map(pd.notna)
    return batch[mask].copy()


def failed_nlp_rows(batch: pd.DataFrame) -> pd.DataFrame:
    if batch.empty or "nlp_success" not in batch.columns:
        return batch.iloc[0:0].copy()
    return batch[~batch["nlp_success"].fillna(False).astype(bool)].copy()


def finite_max(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.max())


def expected_phi(batch: pd.DataFrame, probabilities: dict[int, float]) -> float:
    if batch.empty or "nlp_objective" not in batch.columns:
        return 0.0
    total = 0.0
    for row in batch.itertuples(index=False):
        probability = float(probabilities.get(int(row.scenario_id), 0.0))
        objective = pd.to_numeric(row.nlp_objective, errors="coerce")
        if pd.isna(objective):
            continue
        total += probability * float(objective)
    return total


def load_probabilities(data_dir: Path) -> dict[int, float]:
    path = data_dir / "scenario_probabilities.csv"
    if not path.exists():
        return {}
    probabilities = pd.read_csv(path)
    return probabilities.set_index("scenario_id")["probability"].astype(float).to_dict()


def empty_cut_headers() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cut_id",
            "case_id",
            "scenario_id",
            "hour",
            "cut_type",
            "lhs",
            "sense",
            "rhs_constant",
            "eta_variable",
            "scenario_probability",
            "eta_objective_weight",
            "subproblem_objective_phi",
            "xbar_dot_beta",
            "term_count",
            "status",
            "algebra",
        ]
    )


def empty_cut_terms() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cut_id",
            "case_id",
            "scenario_id",
            "hour",
            "variable_name",
            "component_type",
            "component_id",
            "coefficient",
            "xbar",
            "source_constraint",
            "notes",
        ]
    )


def build_dual_coefficients(data_dir: Path, batch: pd.DataFrame) -> pd.DataFrame:
    generators = pd.read_csv(data_dir / "generators.csv")
    wind_farms = pd.read_csv(data_dir / "wind_farms.csv")
    gen_bus = generators.set_index("gen_id")["bus_id"].astype(int).to_dict()
    wind_bus = wind_farms.set_index("wind_id")["bus_id"].astype(int).to_dict()
    rows = []
    for case in batch.itertuples(index=False):
        multipliers_path = Path(case.multipliers)
        multipliers = pd.read_csv(multipliers_path)
        if {"fixed_master_dispatch_eq", "fixed_master_wind_eq"}.intersection(set(multipliers["constraint_type"].astype(str))):
            for item in multipliers[multipliers["constraint_type"] == "fixed_master_dispatch_eq"].itertuples(index=False):
                gen_id = int(item.component_id)
                rows.append(
                    {
                        "case_id": case.case_id,
                        "scenario_id": int(case.scenario_id),
                        "hour": int(case.hour),
                        "coefficient_family": "fixed_dispatch_active_power",
                        "component_type": "generator",
                        "component_id": gen_id,
                        "bus_id": gen_bus.get(gen_id, -1),
                        "coefficient": float(item.benders_coefficient),
                        "source_constraint": "fixed_master_dispatch_eq",
                        "notes": "Direct coefficient from explicit master-coupling equality Pg_ac - Pg_master_bar = 0.",
                    }
                )
            for item in multipliers[multipliers["constraint_type"] == "fixed_master_wind_eq"].itertuples(index=False):
                wind_id = str(item.component_id)
                rows.append(
                    {
                        "case_id": case.case_id,
                        "scenario_id": int(case.scenario_id),
                        "hour": int(case.hour),
                        "coefficient_family": "fixed_wind_active_power",
                        "component_type": "wind_farm",
                        "component_id": wind_id,
                        "bus_id": wind_bus.get(wind_id, -1),
                        "coefficient": float(item.benders_coefficient),
                        "source_constraint": "fixed_master_wind_eq",
                        "notes": "Direct coefficient from explicit master-coupling equality Wind_ac - Wind_master_bar = 0.",
                    }
                )
            continue
        p_lambda = multipliers[multipliers["constraint_type"] == "p_balance_eq"].set_index("component_id")["multiplier"].astype(float).to_dict()
        q_lambda = multipliers[multipliers["constraint_type"] == "q_balance_eq"].set_index("component_id")["multiplier"].astype(float).to_dict()
        for gen_id, bus_id in gen_bus.items():
            rows.append(
                {
                    "case_id": case.case_id,
                    "scenario_id": int(case.scenario_id),
                    "hour": int(case.hour),
                    "coefficient_family": "fixed_dispatch_active_power",
                    "component_type": "generator",
                    "component_id": gen_id,
                    "bus_id": bus_id,
                    "coefficient": p_lambda.get(bus_id, 0.0),
                    "source_constraint": "p_balance_eq",
                    "notes": "Coefficient from constrained Ipopt P-balance multiplier.",
                }
            )
        for wind_id, bus_id in wind_bus.items():
            rows.append(
                {
                    "case_id": case.case_id,
                    "scenario_id": int(case.scenario_id),
                    "hour": int(case.hour),
                    "coefficient_family": "fixed_wind_active_power",
                    "component_type": "wind_farm",
                    "component_id": wind_id,
                    "bus_id": bus_id,
                    "coefficient": p_lambda.get(bus_id, 0.0),
                    "source_constraint": "p_balance_eq",
                    "notes": "Coefficient from constrained Ipopt P-balance multiplier.",
                }
            )
        for bus_id, value in q_lambda.items():
            rows.append(
                {
                    "case_id": case.case_id,
                    "scenario_id": int(case.scenario_id),
                    "hour": int(case.hour),
                    "coefficient_family": "reactive_balance_sensitivity",
                    "component_type": "bus",
                    "component_id": bus_id,
                    "bus_id": bus_id,
                    "coefficient": value,
                    "source_constraint": "q_balance_eq",
                    "notes": "Diagnostic coefficient; no reactive master variable yet.",
                }
            )
    return pd.DataFrame(rows)


def load_summary_map(batch: pd.DataFrame) -> dict[tuple[str, int, int], dict]:
    out = {}
    for row in batch.itertuples(index=False):
        out[(row.case_id, int(row.scenario_id), int(row.hour))] = json.loads(Path(row.summary).read_text(encoding="utf-8"))
    return out


def relabel_cuts(headers: pd.DataFrame, iteration: int) -> pd.DataFrame:
    headers = headers.copy()
    mapping = {}
    for idx, old_id in enumerate(headers["cut_id"], start=1):
        mapping[old_id] = f"BC-{iteration:03d}-{idx:04d}"
    headers["cut_id"] = headers["cut_id"].map(mapping)
    return headers


def relabel_terms(terms: pd.DataFrame, headers: pd.DataFrame) -> pd.DataFrame:
    # build_cut_constraints starts each iteration at BC-0001. Match by order.
    terms = terms.copy()
    old_ids = list(dict.fromkeys(terms["cut_id"].tolist()))
    new_ids = headers["cut_id"].tolist()
    mapping = {old: new for old, new in zip(old_ids, new_ids)}
    terms["cut_id"] = terms["cut_id"].map(mapping)
    return terms


def write_iteration_report(path: Path, log_row: dict[str, Any], selected: pd.DataFrame, nlp_batch: pd.DataFrame) -> None:
    lines = [
        f"# Benders Auto Loop Iteration {log_row['iteration']}",
        "",
        f"- Master objective: {log_row['master_objective']}",
        f"- First-stage proxy cost: {log_row.get('first_stage_proxy_cost')}",
        f"- Master eta cost: {log_row.get('expected_master_eta_cost')}",
        f"- Evaluated upper bound: {log_row.get('evaluated_upper_bound')}",
        f"- Active cuts in master: {log_row['cuts_active_in_master']}",
        f"- New cuts generated: {log_row['new_cuts_generated']}",
        f"- Relative gap proxy: {log_row['relative_gap_percent']:.8f}%",
        f"- Successful NLP subproblems: {log_row['successful_nlp_subproblems']} / {log_row['selected_subproblems']}",
        f"- Failed NLP subproblems: {log_row['failed_nlp_subproblems']}",
        f"- Previously cut candidates skipped: {log_row['previously_cut_candidates_skipped']}",
        f"- Previously failed candidates skipped: {log_row['previously_failed_candidates_skipped']}",
        "",
        "## Selected Subproblems",
        "",
        "| Scenario | Hour | Screening Reactive Violation | NLP Objective | NLP Success | NLP Status |",
        "|---:|---:|---:|---:|---|---|",
    ]
    nlp_lookup = {(int(row.scenario_id), int(row.hour)): row for row in nlp_batch.itertuples(index=False)}
    for row in selected.itertuples(index=False):
        nlp = nlp_lookup.get((int(row.scenario_id), int(row.hour)))
        lines.append(
            f"| {int(row.scenario_id)} | {int(row.hour)} | {float(row.reactive_violation_mvar):.6f} | "
            f"{float(nlp.nlp_objective) if nlp is not None else 0.0:.6e} | {bool(nlp.nlp_success) if nlp is not None else False} | "
            f"{str(nlp.nlp_status) if nlp is not None else 'not_run'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def render_report(result: dict[str, Any], log: pd.DataFrame) -> str:
    lines = [
        "# Benders Automatic Loop Report",
        "",
        f"- Status: `{result['status']}`",
        f"- Stop reason: `{result['stop_reason']}`",
        f"- Iterations: {result['iterations']}",
        f"- Tolerance: {result['tolerance_percent']}%",
        f"- Cumulative cuts: {result['cumulative_cuts']}",
        f"- Unique cut subproblems: {result['unique_cut_subproblems']}",
        f"- Failed subproblems seen: {result['failed_subproblems_seen']}",
        f"- Runtime: {result['runtime_sec']:.3f} seconds",
        "",
        "| Iteration | Lower Bound | Evaluated Upper Bound | Gap % | Cuts Active | New Cuts | NLP Solves | Failed NLP | Skipped Cut/Failed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in log.itertuples(index=False):
        lines.append(
            f"| {row.iteration} | {row.lower_bound:.6f} | {float(row.evaluated_upper_bound) if pd.notna(row.evaluated_upper_bound) else float('nan'):.6f} | "
            f"{row.relative_gap_percent:.8f} | {row.cuts_active_in_master} | {row.new_cuts_generated} | "
            f"{row.successful_nlp_subproblems}/{row.selected_subproblems} | {row.failed_nlp_subproblems} | "
            f"{row.previously_cut_candidates_skipped}/{row.previously_failed_candidates_skipped} |"
        )
    lines += [
        "",
        "This loop follows the paper's sequence at workflow level: master solve, AC subproblem solves, cut generation, and master re-solve. Failed AC NLP rows are retained as diagnostics but are excluded from cut generation. The reported evaluated upper bound is a reconstructed paper-style proxy built from first-stage cost plus expected AC recourse over the successful NLP evaluations.",
        "",
    ]
    return "\n".join(lines)


def resolve_relative(base: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


if __name__ == "__main__":
    main()
