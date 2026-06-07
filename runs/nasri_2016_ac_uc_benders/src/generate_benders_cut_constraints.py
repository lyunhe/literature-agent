from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Benders-form cut constraints from dual coefficient drafts.")
    parser.add_argument("--coefficients", default="../results/benders_cuts/case_b_dual_cut_coefficients.csv")
    parser.add_argument("--case-a-dir", default="../results/case_a_dc_uc")
    parser.add_argument("--scenario-probabilities", default="../data/scenario_probabilities.csv")
    parser.add_argument("--out-dir", default="../results/benders_cuts")
    parser.add_argument("--cut-type", choices=["optimality_cut", "optimality_proxy", "feasibility_proxy"], default="optimality_cut")
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    coeff_path = resolve_relative(src_dir, Path(args.coefficients))
    case_a_dir = resolve_relative(src_dir, Path(args.case_a_dir))
    probabilities_path = resolve_relative(src_dir, Path(args.scenario_probabilities))
    out_dir = resolve_relative(src_dir, Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    coefficients = pd.read_csv(coeff_path)
    dispatch = pd.read_csv(case_a_dir / "dispatch.csv")
    wind = pd.read_csv(case_a_dir / "wind_usage.csv")
    probabilities = load_probabilities(probabilities_path)
    nlp_summaries = load_nlp_summaries(coefficients)
    headers, terms = build_cut_constraints(coefficients, dispatch, wind, nlp_summaries, cut_type=args.cut_type, probabilities=probabilities)

    headers_path = out_dir / "case_b_benders_cut_constraints.csv"
    terms_path = out_dir / "case_b_benders_cut_terms.csv"
    lp_path = out_dir / "case_b_benders_cuts.lp.txt"
    report_path = out_dir / "case_b_benders_cut_constraints.md"
    headers.to_csv(headers_path, index=False)
    terms.to_csv(terms_path, index=False)
    lp_path.write_text(render_lp(headers, terms), encoding="utf-8")
    report_path.write_text(render_report(headers, terms, lp_path), encoding="utf-8")
    print(headers_path)
    print(terms_path)
    print(lp_path)
    print(report_path)


def load_nlp_summaries(coefficients: pd.DataFrame) -> dict[tuple[str, int, int], dict]:
    summaries: dict[tuple[str, int, int], dict] = {}
    # Coefficient rows are generated from the constrained batch; infer summary paths
    # from the standard output naming convention.
    base = Path(__file__).resolve().parents[1] / "results" / "ac_nlp_subproblem"
    for case_id, scenario_id, hour in coefficients[["case_id", "scenario_id", "hour"]].drop_duplicates().itertuples(index=False):
        path = base / f"{case_id}_cyipopt_constrained_scenario_{int(scenario_id)}_hour_{int(hour)}_nlp_summary.json"
        if path.exists():
            summaries[(case_id, int(scenario_id), int(hour))] = json.loads(path.read_text(encoding="utf-8"))
    return summaries


def build_cut_constraints(
    coefficients: pd.DataFrame,
    dispatch: pd.DataFrame,
    wind: pd.DataFrame,
    nlp_summaries: dict[tuple[str, int, int], dict],
    *,
    cut_type: str,
    probabilities: dict[int, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    headers = []
    terms = []
    for idx, (case_id, scenario_id, hour) in enumerate(
        coefficients[["case_id", "scenario_id", "hour"]].drop_duplicates().itertuples(index=False),
        start=1,
    ):
        scenario_id = int(scenario_id)
        hour = int(hour)
        cut_id = f"BC-{idx:04d}"
        scenario_probability = float((probabilities or {}).get(scenario_id, 1.0))
        summary = nlp_summaries.get((case_id, scenario_id, hour), {})
        phi = float(summary.get("objective", summary.get("metadata", {}).get("objective", 0.0)))
        selected = coefficients[
            (coefficients["case_id"] == case_id)
            & (coefficients["scenario_id"].astype(int) == scenario_id)
            & (coefficients["hour"].astype(int) == hour)
            & (coefficients["coefficient_family"].isin(["fixed_dispatch_active_power", "fixed_wind_active_power"]))
        ].copy()
        xbar_dot_beta = 0.0
        for row in selected.itertuples(index=False):
            variable_name, xbar = master_variable_and_value(row, dispatch, wind)
            coefficient = float(row.coefficient)
            xbar_dot_beta += coefficient * xbar
            terms.append(
                {
                    "cut_id": cut_id,
                    "case_id": case_id,
                    "scenario_id": scenario_id,
                    "hour": hour,
                    "variable_name": variable_name,
                    "component_type": row.component_type,
                    "component_id": row.component_id,
                    "coefficient": coefficient,
                    "xbar": xbar,
                    "source_constraint": row.source_constraint,
                    "notes": row.notes,
                }
            )
        eta_name = f"eta_ac_s{scenario_id}_t{hour}"
        alpha = phi - xbar_dot_beta
        if cut_type in {"optimality_cut", "optimality_proxy"}:
            sense = ">="
            lhs = eta_name
            rhs_constant = alpha
            algebra = f"{eta_name} >= {alpha:.12g} + sum(beta_i * x_i)"
        else:
            sense = "<="
            lhs = "0"
            rhs_constant = alpha
            algebra = f"0 >= {alpha:.12g} + sum(beta_i * x_i)"
        headers.append(
            {
                "cut_id": cut_id,
                "case_id": case_id,
                "scenario_id": scenario_id,
                "hour": hour,
                "cut_type": cut_type,
                "scenario_probability": scenario_probability,
                "eta_objective_weight": scenario_probability,
                "lhs": lhs,
                "sense": sense,
                "rhs_constant": rhs_constant,
                "eta_variable": eta_name if cut_type in {"optimality_cut", "optimality_proxy"} else "",
                "subproblem_objective_phi": phi,
                "xbar_dot_beta": xbar_dot_beta,
                "term_count": int(len(selected)),
                "status": "generated_not_added_to_master",
                "algebra": algebra,
            }
        )
    return pd.DataFrame(headers), pd.DataFrame(terms)


def load_probabilities(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    data = pd.read_csv(path)
    if "scenario_id" not in data.columns or "probability" not in data.columns:
        return {}
    return data.set_index("scenario_id")["probability"].astype(float).to_dict()


def master_variable_and_value(row: object, dispatch: pd.DataFrame, wind: pd.DataFrame) -> tuple[str, float]:
    scenario_id = int(row.scenario_id)
    hour = int(row.hour)
    if row.coefficient_family == "fixed_dispatch_active_power":
        gen_id = int(row.component_id)
        values = dispatch[
            (dispatch["scenario_id"].astype(int) == scenario_id)
            & (dispatch["hour"].astype(int) == hour)
            & (dispatch["gen_id"].astype(int) == gen_id)
        ]["dispatch_mw"]
        return f"p_s{scenario_id}_t{hour}_g{gen_id}", float(values.iloc[0]) if not values.empty else 0.0
    wind_id = str(row.component_id)
    values = wind[
        (wind["scenario_id"].astype(int) == scenario_id)
        & (wind["hour"].astype(int) == hour)
        & (wind["wind_id"].astype(str) == wind_id)
    ]["used_mw"]
    return f"wind_s{scenario_id}_t{hour}_{wind_id}", float(values.iloc[0]) if not values.empty else 0.0


def render_lp(headers: pd.DataFrame, terms: pd.DataFrame) -> str:
    lines = ["\\ Benders cut constraints generated from constrained Ipopt multipliers", "Subject To"]
    for header in headers.itertuples(index=False):
        cut_terms = terms[terms["cut_id"] == header.cut_id]
        # eta - sum(beta*x) >= alpha
        pieces = []
        if header.eta_variable:
            pieces.append(header.eta_variable)
        for term in cut_terms.itertuples(index=False):
            coefficient = -float(term.coefficient) if header.cut_type in {"optimality_cut", "optimality_proxy"} else -float(term.coefficient)
            pieces.append(format_lp_term(coefficient, term.variable_name))
        lhs = " ".join(pieces) if pieces else "0"
        lines.append(f" {header.cut_id}: {lhs} >= {float(header.rhs_constant):.12g}")
    lines.append("End")
    lines.append("")
    return "\n".join(lines)


def format_lp_term(coefficient: float, variable_name: str) -> str:
    sign = "+" if coefficient >= 0 else "-"
    return f"{sign} {abs(coefficient):.12g} {variable_name}"


def render_report(headers: pd.DataFrame, terms: pd.DataFrame, lp_path: Path) -> str:
    lines = [
        "# Benders Cut Constraints",
        "",
        f"- Cuts generated: {len(headers)}",
        f"- Term rows: {len(terms)}",
        f"- LP text: `{lp_path}`",
        "",
        "Generated cut form:",
        "",
        "```text",
        "eta_ac_s_t >= phi_s_t(x_bar) + sum_i beta_i * (x_i - xbar_i)",
        "```",
        "",
        "Equivalent LP row:",
        "",
        "```text",
        "eta_ac_s_t - sum_i beta_i * x_i >= phi_s_t(x_bar) - sum_i beta_i * xbar_i",
        "```",
        "",
        "The expected second-stage objective is represented in the master by assigning each eta variable an objective weight equal to its scenario probability.",
        "",
        "| Cut | Scenario | Hour | Probability | Phi | RHS Constant | Terms | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in headers.itertuples(index=False):
        lines.append(
            f"| {row.cut_id} | {row.scenario_id} | {row.hour} | {row.scenario_probability:.6f} | {row.subproblem_objective_phi:.6e} | "
            f"{row.rhs_constant:.6e} | {row.term_count} | {row.status} |"
        )
    lines += [
        "",
        "These rows are now in Benders algebraic form and use master variable names from the HiGHS model. They still need eta variables and cut insertion in the master solve loop.",
        "",
    ]
    return "\n".join(lines)


def resolve_relative(base: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


if __name__ == "__main__":
    main()
