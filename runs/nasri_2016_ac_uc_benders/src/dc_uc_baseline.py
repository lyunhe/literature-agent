from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from case_data import CaseData
from uc_results import SolveResult


BASE_MVA = 100.0
LOAD_SHED_VALUE_USD_PER_PU = 10000.0


def solve_case_a_dc_uc(
    case: CaseData,
    solver_config: dict[str, Any],
    *,
    dry_run: bool = True,
    out_dir: Path | None = None,
) -> SolveResult:
    """Solve a synthetic-data Case A DC-UC extensive-form MILP with HiGHS."""
    start = perf_counter()
    missing_or_empty = case.missing_or_empty()
    if missing_or_empty:
        return SolveResult(
            status="blocked_missing_data",
            objective=None,
            runtime_sec=perf_counter() - start,
            metadata={"missing_or_empty_tables": missing_or_empty, "data_summary": case.summary()},
        )
    if dry_run:
        return _dry_run_result(case, solver_config, start)

    try:
        import highspy
    except ImportError as exc:
        return SolveResult(
            status="blocked_missing_solver",
            objective=None,
            runtime_sec=perf_counter() - start,
            metadata={"error": str(exc), "required_package": "highspy"},
        )

    out_dir = out_dir or Path("../results/case_a_dc_uc")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _prepare_model_data(case)
    model = highspy.Highs()
    model.setOptionValue("output_flag", bool(solver_config.get("output_flag", False)))
    if solver_config.get("time_limit_sec"):
        model.setOptionValue("time_limit", float(solver_config["time_limit_sec"]))
    if solver_config.get("mip_gap") is not None:
        model.setOptionValue("mip_rel_gap", float(solver_config["mip_gap"]))

    u: dict[tuple[int, int], Any] = {}
    startup: dict[tuple[int, int], Any] = {}
    p: dict[tuple[int, int, int], Any] = {}
    theta: dict[tuple[int, int, int], Any] = {}
    wind_used: dict[tuple[int, int, str], Any] = {}
    load_shed: dict[tuple[int, int, int], Any] = {}
    variable_map: dict[str, Any] = {}

    for gen_id, gen in data["gens"].items():
        for hour in data["hours"]:
            u[gen_id, hour] = model.addBinary(name=f"u_g{gen_id}_t{hour}")
            variable_map[f"u_g{gen_id}_t{hour}"] = u[gen_id, hour]
            startup[gen_id, hour] = model.addBinary(obj=float(gen["startup_cost"]), name=f"start_g{gen_id}_t{hour}")
            variable_map[f"start_g{gen_id}_t{hour}"] = startup[gen_id, hour]

    for scenario_id, probability in data["probabilities"].items():
        for hour in data["hours"]:
            for gen_id, gen in data["gens"].items():
                variable_cost = probability * float(gen["cost_usd_per_pu"]) / BASE_MVA
                p[scenario_id, hour, gen_id] = model.addVariable(
                    lb=0.0,
                    ub=float(gen["p_max_mw"]),
                    obj=variable_cost,
                    name=f"p_s{scenario_id}_t{hour}_g{gen_id}",
                )
                variable_map[f"p_s{scenario_id}_t{hour}_g{gen_id}"] = p[scenario_id, hour, gen_id]
            for bus_id in data["bus_ids"]:
                theta[scenario_id, hour, bus_id] = model.addVariable(
                    lb=-10.0,
                    ub=10.0,
                    obj=0.0,
                    name=f"theta_s{scenario_id}_t{hour}_b{bus_id}",
                )
                variable_map[f"theta_s{scenario_id}_t{hour}_b{bus_id}"] = theta[scenario_id, hour, bus_id]
                load_mw = data["loads"][hour][bus_id]
                load_shed[scenario_id, hour, bus_id] = model.addVariable(
                    lb=0.0,
                    ub=load_mw,
                    obj=probability * LOAD_SHED_VALUE_USD_PER_PU / BASE_MVA,
                    name=f"shed_s{scenario_id}_t{hour}_b{bus_id}",
                )
                variable_map[f"shed_s{scenario_id}_t{hour}_b{bus_id}"] = load_shed[scenario_id, hour, bus_id]
            for wind_id, available_mw in data["wind_available"][scenario_id][hour].items():
                wind_used[scenario_id, hour, wind_id] = model.addVariable(
                    lb=0.0,
                    ub=available_mw,
                    obj=0.0,
                    name=f"wind_s{scenario_id}_t{hour}_{wind_id}",
                )
                variable_map[f"wind_s{scenario_id}_t{hour}_{wind_id}"] = wind_used[scenario_id, hour, wind_id]

    for gen_id, gen in data["gens"].items():
        initial_status = float(gen["initial_status"])
        for hour in data["hours"]:
            prev_u = initial_status if hour == data["hours"][0] else u[gen_id, hour - 1]
            model.addConstr(startup[gen_id, hour] >= u[gen_id, hour] - prev_u, name=f"startup_logic_g{gen_id}_t{hour}")
            for scenario_id in data["scenario_ids"]:
                model.addConstr(
                    p[scenario_id, hour, gen_id] <= float(gen["p_max_mw"]) * u[gen_id, hour],
                    name=f"pmax_s{scenario_id}_t{hour}_g{gen_id}",
                )
                model.addConstr(
                    p[scenario_id, hour, gen_id] >= float(gen["p_min_mw"]) * u[gen_id, hour],
                    name=f"pmin_s{scenario_id}_t{hour}_g{gen_id}",
                )

    gen_by_bus: dict[int, list[int]] = defaultdict(list)
    for gen_id, gen in data["gens"].items():
        gen_by_bus[int(gen["bus_id"])].append(gen_id)
    wind_bus = data["wind_bus"]

    for scenario_id in data["scenario_ids"]:
        for hour in data["hours"]:
            model.addConstr(theta[scenario_id, hour, data["slack_bus"]] == 0.0, name=f"slack_s{scenario_id}_t{hour}")
            for bus_id in data["bus_ids"]:
                expr = 0.0
                for gen_id in gen_by_bus.get(bus_id, []):
                    expr += p[scenario_id, hour, gen_id]
                for wind_id, bus in wind_bus.items():
                    if bus == bus_id:
                        expr += wind_used[scenario_id, hour, wind_id]
                expr += load_shed[scenario_id, hour, bus_id]
                expr -= data["loads"][hour][bus_id]
                for line in data["incidence"][bus_id]:
                    sign = 1.0 if line["from_bus"] == bus_id else -1.0
                    expr -= sign * (theta[scenario_id, hour, line["from_bus"]] - theta[scenario_id, hour, line["to_bus"]]) * (
                        BASE_MVA / line["x_pu"]
                    )
                model.addConstr(expr == 0.0, name=f"balance_s{scenario_id}_t{hour}_b{bus_id}")

            for line in data["lines"]:
                flow = (theta[scenario_id, hour, line["from_bus"]] - theta[scenario_id, hour, line["to_bus"]]) * (
                    BASE_MVA / line["x_pu"]
                )
                model.addConstr(flow <= line["rate_mw"], name=f"line_pos_s{scenario_id}_t{hour}_l{line['line_id']}")
                model.addConstr(flow >= -line["rate_mw"], name=f"line_neg_s{scenario_id}_t{hour}_l{line['line_id']}")

    benders_cut_info = _add_benders_cut_constraints(model, variable_map, solver_config)

    solve_start = perf_counter()
    model.run()
    solve_runtime = perf_counter() - solve_start
    status = str(model.getModelStatus()).replace("HighsModelStatus.", "")
    objective = float(model.getObjectiveValue()) if status == "kOptimal" else None
    solution = model.getSolution()
    col_values = solution.col_value

    outputs = _write_solution_outputs(
        out_dir=out_dir,
        data=data,
        objective=objective,
        status=status,
        solve_runtime=solve_runtime,
        u=u,
        startup=startup,
        p=p,
        theta=theta,
        wind_used=wind_used,
        load_shed=load_shed,
        col_values=col_values,
    )
    startup_cost_total = 0.0
    for (gen_id, hour), var in startup.items():
        startup_cost_total += float(data["gens"][gen_id]["startup_cost"]) * float(col_values[var.index])
    expected_dispatch_cost_total = 0.0
    for (scenario_id, hour, gen_id), var in p.items():
        expected_dispatch_cost_total += (
            float(data["probabilities"][scenario_id])
            * float(data["gens"][gen_id]["cost_usd_per_pu"])
            / BASE_MVA
            * float(col_values[var.index])
        )
    expected_load_shed_cost_total = 0.0
    for (scenario_id, hour, bus_id), var in load_shed.items():
        expected_load_shed_cost_total += (
            float(data["probabilities"][scenario_id]) * LOAD_SHED_VALUE_USD_PER_PU / BASE_MVA * float(col_values[var.index])
        )

    metadata = {
        "case": "case_a_dc_uc_synthetic_extensive_form",
        "description": "Solver-backed DC-UC MILP over 40 synthetic wind scenarios and 24 hours.",
        "solver": "HiGHS via highspy",
        "model_status": status,
        "solve_runtime_sec": solve_runtime,
        "variables": int(model.getNumCol()),
        "constraints": int(model.getNumRow()),
        "scenarios": len(data["scenario_ids"]),
        "hours": len(data["hours"]),
        "generators": len(data["gens"]),
        "buses": len(data["bus_ids"]),
        "lines": len(data["lines"]),
        "objective_breakdown": {
            "startup_cost": startup_cost_total,
            "expected_dispatch_cost": expected_dispatch_cost_total,
            "expected_load_shed_cost": expected_load_shed_cost_total,
            "expected_eta_cost": max(
                0.0,
                float(objective or 0.0) - startup_cost_total - expected_dispatch_cost_total - expected_load_shed_cost_total,
            ),
        },
        "outputs": outputs,
        "benders_cuts": benders_cut_info,
        "limitations": [
            "Synthetic calibrated wind scenarios are used instead of original Fig. 3 trajectories.",
            "This stage includes unit commitment, startup, generator limits, DC nodal balance, line limits, wind curtailment, and load shedding.",
            "Minimum up/down time, ramping chronology, reserve constraints, and AC feasibility cuts are not yet implemented.",
        ],
    }
    return SolveResult(status="solved" if status == "kOptimal" else status, objective=objective, runtime_sec=perf_counter() - start, metadata=metadata)


def _dry_run_result(case: CaseData, solver_config: dict[str, Any], start: float) -> SolveResult:
    metadata = {
        "case": "case_a_dc_uc",
        "description": "DC-UC baseline; both stages use DC network representation.",
        "data_summary": case.summary(),
        "missing_or_empty_tables": [],
        "solver_config": solver_config,
        "dry_run": True,
        "next_solver_hook": "build and solve MILP master/DC extensive form",
    }
    return SolveResult(status="ready_for_solver", objective=None, runtime_sec=perf_counter() - start, metadata=metadata)


def _add_benders_cut_constraints(model: Any, variable_map: dict[str, Any], solver_config: dict[str, Any]) -> dict[str, Any]:
    headers_path = solver_config.get("benders_cut_constraints")
    terms_path = solver_config.get("benders_cut_terms")
    if not headers_path or not terms_path:
        return {"enabled": False, "cuts_added": 0}
    headers_path = Path(headers_path)
    terms_path = Path(terms_path)
    if not headers_path.exists() or not terms_path.exists():
        return {
            "enabled": True,
            "cuts_added": 0,
            "status": "missing_cut_files",
            "headers_path": str(headers_path),
            "terms_path": str(terms_path),
        }
    headers = pd.read_csv(headers_path)
    terms = pd.read_csv(terms_path)
    cuts_added = 0
    eta_variables = []
    missing_terms: list[dict[str, Any]] = []
    default_eta_weight = float(solver_config.get("benders_eta_objective_weight", 1.0))
    for header in headers.itertuples(index=False):
        eta_name = str(header.eta_variable) if not pd.isna(header.eta_variable) and str(header.eta_variable) else f"eta_{header.cut_id}"
        eta_var = variable_map.get(eta_name)
        if eta_var is None:
            eta_weight = _header_eta_weight(header, default_eta_weight)
            eta_var = model.addVariable(lb=0.0, ub=float(solver_config.get("benders_eta_upper_bound", 1.0e6)), obj=eta_weight, name=eta_name)
            variable_map[eta_name] = eta_var
            eta_variables.append(eta_name)
        expr = eta_var
        cut_terms = terms[terms["cut_id"] == header.cut_id]
        term_missing = False
        for term in cut_terms.itertuples(index=False):
            var = variable_map.get(str(term.variable_name))
            if var is None:
                missing_terms.append({"cut_id": header.cut_id, "variable_name": term.variable_name})
                term_missing = True
                continue
            expr -= float(term.coefficient) * var
        if term_missing and bool(solver_config.get("benders_skip_incomplete_cuts", True)):
            continue
        model.addConstr(expr >= float(header.rhs_constant), name=f"benders_{header.cut_id}")
        cuts_added += 1
    return {
        "enabled": True,
        "cuts_added": cuts_added,
        "candidate_cuts": int(len(headers)),
        "eta_variables": eta_variables,
        "missing_terms": missing_terms,
        "headers_path": str(headers_path),
        "terms_path": str(terms_path),
        "eta_objective_weight": default_eta_weight,
    }


def _header_eta_weight(header: Any, default: float) -> float:
    value = getattr(header, "eta_objective_weight", default)
    if pd.isna(value):
        return default
    return float(value)


def _prepare_model_data(case: CaseData) -> dict[str, Any]:
    buses = case.table("buses").copy()
    lines_df = case.table("lines").copy()
    gens_df = case.table("generators").copy()
    load_profile = case.table("load_profile").copy()
    wind_profile = case.table("wind_profile").copy()
    wind_farms = case.table("wind_farms").copy()
    probabilities = case.table("scenario_probabilities").copy()

    bus_ids = sorted(buses["bus_id"].astype(int).tolist())
    hours = sorted(load_profile["hour"].astype(int).tolist())
    scenario_ids = sorted(probabilities["scenario_id"].astype(int).tolist())
    slack_bus = 13 if 13 in bus_ids else bus_ids[0]

    pd_fraction = buses.set_index("bus_id")["pd_fraction"].astype(float).to_dict()
    loads: dict[int, dict[int, float]] = {}
    for row in load_profile.itertuples():
        total_load = float(row.total_load_mw)
        loads[int(row.hour)] = {bus_id: total_load * float(pd_fraction.get(bus_id, 0.0)) for bus_id in bus_ids}

    gens: dict[int, dict[str, float]] = {}
    for row in gens_df.itertuples():
        gen_id = int(row.gen_id)
        gens[gen_id] = {
            "bus_id": int(row.bus_id),
            "p_min_mw": float(row.p_min_mw),
            "p_max_mw": float(row.p_max_mw),
            "startup_cost": _clean_float(getattr(row, "startup_cost", 0.0)),
            "initial_status": _clean_float(getattr(row, "initial_status", 0.0)),
            "cost_usd_per_pu": _clean_float(getattr(row, "paper_cost_usd_per_pu", 0.0)),
        }

    probability_map = probabilities.set_index("scenario_id")["probability"].astype(float).to_dict()
    wind_bus = {str(row.wind_id): int(row.bus_id) for row in wind_farms.itertuples()}
    wind_available: dict[int, dict[int, dict[str, float]]] = {
        scenario_id: {hour: {wind_id: 0.0 for wind_id in wind_bus} for hour in hours} for scenario_id in scenario_ids
    }
    for row in wind_profile.itertuples():
        wind_available[int(row.scenario_id)][int(row.hour)][str(row.wind_id)] = float(row.production_mw)

    lines: list[dict[str, float]] = []
    incidence: dict[int, list[dict[str, float]]] = {bus_id: [] for bus_id in bus_ids}
    for row in lines_df.itertuples():
        line = {
            "line_id": int(row.line_id),
            "from_bus": int(row.from_bus),
            "to_bus": int(row.to_bus),
            "x_pu": float(row.x_pu),
            "rate_mw": float(row.rate_mw),
        }
        lines.append(line)
        incidence[line["from_bus"]].append(line)
        incidence[line["to_bus"]].append(line)

    return {
        "bus_ids": bus_ids,
        "hours": hours,
        "scenario_ids": scenario_ids,
        "slack_bus": slack_bus,
        "loads": loads,
        "gens": gens,
        "probabilities": probability_map,
        "wind_bus": wind_bus,
        "wind_available": wind_available,
        "lines": lines,
        "incidence": incidence,
    }


def _write_solution_outputs(
    *,
    out_dir: Path,
    data: dict[str, Any],
    objective: float | None,
    status: str,
    solve_runtime: float,
    u: dict[tuple[int, int], Any],
    startup: dict[tuple[int, int], Any],
    p: dict[tuple[int, int, int], Any],
    theta: dict[tuple[int, int, int], Any],
    wind_used: dict[tuple[int, int, str], Any],
    load_shed: dict[tuple[int, int, int], Any],
    col_values: list[float],
) -> dict[str, str]:
    commitment_rows = []
    for (gen_id, hour), var in u.items():
        commitment_rows.append(
            {
                "gen_id": gen_id,
                "hour": hour,
                "committed": round(col_values[var.index]),
                "startup": round(col_values[startup[gen_id, hour].index]),
            }
        )

    dispatch_rows = []
    for (scenario_id, hour, gen_id), var in p.items():
        dispatch_rows.append(
            {
                "scenario_id": scenario_id,
                "hour": hour,
                "gen_id": gen_id,
                "dispatch_mw": col_values[var.index],
            }
        )

    wind_rows = []
    for (scenario_id, hour, wind_id), var in wind_used.items():
        available = data["wind_available"][scenario_id][hour][wind_id]
        used = col_values[var.index]
        wind_rows.append(
            {
                "scenario_id": scenario_id,
                "hour": hour,
                "wind_id": wind_id,
                "available_mw": available,
                "used_mw": used,
                "curtailed_mw": max(0.0, available - used),
            }
        )

    shed_rows = []
    for (scenario_id, hour, bus_id), var in load_shed.items():
        value = col_values[var.index]
        if value > 1.0e-5:
            shed_rows.append(
                {
                    "scenario_id": scenario_id,
                    "hour": hour,
                    "bus_id": bus_id,
                    "load_shed_mw": value,
                }
            )

    flow_rows = []
    angle_rows = []
    for scenario_id in data["scenario_ids"]:
        for hour in data["hours"]:
            for bus_id in data["bus_ids"]:
                angle_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "hour": hour,
                        "bus_id": bus_id,
                        "theta_rad": col_values[theta[scenario_id, hour, bus_id].index],
                    }
                )
            for line in data["lines"]:
                angle_from = col_values[theta[scenario_id, hour, line["from_bus"]].index]
                angle_to = col_values[theta[scenario_id, hour, line["to_bus"]].index]
                flow_mw = (angle_from - angle_to) * BASE_MVA / line["x_pu"]
                flow_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "hour": hour,
                        "line_id": line["line_id"],
                        "from_bus": line["from_bus"],
                        "to_bus": line["to_bus"],
                        "flow_mw": flow_mw,
                        "rate_mw": line["rate_mw"],
                        "loading_percent": 100.0 * flow_mw / line["rate_mw"] if line["rate_mw"] else np.nan,
                    }
                )

    commitment = pd.DataFrame(commitment_rows)
    dispatch = pd.DataFrame(dispatch_rows)
    wind = pd.DataFrame(wind_rows)
    shedding = pd.DataFrame(shed_rows, columns=["scenario_id", "hour", "bus_id", "load_shed_mw"])
    flows = pd.DataFrame(flow_rows)
    angles = pd.DataFrame(angle_rows)
    global _PROBABILITY_CACHE
    _PROBABILITY_CACHE = dict(data["probabilities"])
    scenario_summary = _scenario_summary(data, dispatch, wind, shedding, flows)
    hour_summary = _hour_summary(commitment, dispatch, wind, shedding, flows)

    paths = {
        "commitment": out_dir / "commitment.csv",
        "dispatch": out_dir / "dispatch.csv",
        "wind": out_dir / "wind_usage.csv",
        "load_shedding": out_dir / "load_shedding.csv",
        "line_flows": out_dir / "line_flows.csv",
        "bus_angles": out_dir / "bus_angles.csv",
        "scenario_summary": out_dir / "scenario_summary.csv",
        "hour_summary": out_dir / "hour_summary.csv",
        "experiment_summary": out_dir / "experiment_summary.md",
    }
    commitment.to_csv(paths["commitment"], index=False)
    dispatch.to_csv(paths["dispatch"], index=False)
    wind.to_csv(paths["wind"], index=False)
    shedding.to_csv(paths["load_shedding"], index=False)
    flows.to_csv(paths["line_flows"], index=False)
    angles.to_csv(paths["bus_angles"], index=False)
    scenario_summary.to_csv(paths["scenario_summary"], index=False)
    hour_summary.to_csv(paths["hour_summary"], index=False)
    paths["experiment_summary"].write_text(
        _experiment_summary_md(
            objective=objective,
            status=status,
            solve_runtime=solve_runtime,
            commitment=commitment,
            dispatch=dispatch,
            wind=wind,
            shedding=shedding,
            flows=flows,
            scenario_summary=scenario_summary,
        ),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in paths.items()}


def _scenario_summary(
    data: dict[str, Any],
    dispatch: pd.DataFrame,
    wind: pd.DataFrame,
    shedding: pd.DataFrame,
    flows: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    shed_group = shedding.groupby("scenario_id")["load_shed_mw"].sum() if not shedding.empty else {}
    for scenario_id in data["scenario_ids"]:
        scenario_dispatch = dispatch[dispatch["scenario_id"] == scenario_id]
        scenario_wind = wind[wind["scenario_id"] == scenario_id]
        scenario_flows = flows[flows["scenario_id"] == scenario_id]
        rows.append(
            {
                "scenario_id": scenario_id,
                "probability": data["probabilities"][scenario_id],
                "thermal_mwh": float(scenario_dispatch["dispatch_mw"].sum()),
                "wind_used_mwh": float(scenario_wind["used_mw"].sum()),
                "wind_curtailed_mwh": float(scenario_wind["curtailed_mw"].sum()),
                "load_shed_mwh": float(shed_group.get(scenario_id, 0.0)) if hasattr(shed_group, "get") else 0.0,
                "max_abs_line_loading_percent": float(scenario_flows["loading_percent"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _hour_summary(
    commitment: pd.DataFrame,
    dispatch: pd.DataFrame,
    wind: pd.DataFrame,
    shedding: pd.DataFrame,
    flows: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for hour in sorted(dispatch["hour"].unique()):
        hour_commitment = commitment[commitment["hour"] == hour]
        hour_dispatch = dispatch[dispatch["hour"] == hour]
        hour_wind = wind[wind["hour"] == hour]
        hour_shedding = shedding[shedding["hour"] == hour] if not shedding.empty else shedding
        hour_flows = flows[flows["hour"] == hour]
        rows.append(
            {
                "hour": int(hour),
                "committed_units": int(hour_commitment["committed"].sum()),
                "startups": int(hour_commitment["startup"].sum()),
                "expected_thermal_mw": _expected_by_hour(hour_dispatch, "dispatch_mw"),
                "expected_wind_used_mw": _expected_by_hour(hour_wind, "used_mw"),
                "expected_wind_curtailed_mw": _expected_by_hour(hour_wind, "curtailed_mw"),
                "total_load_shed_mw": float(hour_shedding["load_shed_mw"].sum()) if not hour_shedding.empty else 0.0,
                "max_abs_line_loading_percent": float(hour_flows["loading_percent"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _expected_by_hour(df: pd.DataFrame, value_col: str) -> float:
    if df.empty:
        return 0.0
    probabilities = df[["scenario_id"]].drop_duplicates().copy()
    probability_map = _PROBABILITY_CACHE
    grouped = df.groupby("scenario_id")[value_col].sum()
    return float(sum(grouped.loc[scenario_id] * probability_map.get(int(scenario_id), 0.0) for scenario_id in grouped.index))


_PROBABILITY_CACHE: dict[int, float] = {}


def _experiment_summary_md(
    *,
    objective: float | None,
    status: str,
    solve_runtime: float,
    commitment: pd.DataFrame,
    dispatch: pd.DataFrame,
    wind: pd.DataFrame,
    shedding: pd.DataFrame,
    flows: pd.DataFrame,
    scenario_summary: pd.DataFrame,
) -> str:
    expected_thermal = float((scenario_summary["thermal_mwh"] * scenario_summary["probability"]).sum())
    expected_wind_used = float((scenario_summary["wind_used_mwh"] * scenario_summary["probability"]).sum())
    expected_wind_curtailed = float((scenario_summary["wind_curtailed_mwh"] * scenario_summary["probability"]).sum())
    expected_shed = float((scenario_summary["load_shed_mwh"] * scenario_summary["probability"]).sum())
    max_loading = float(flows["loading_percent"].abs().max())
    return "\n".join(
        [
            "# Case A DC-UC Synthetic Experiment Summary",
            "",
            f"- Solver status: `{status}`",
            f"- Objective: {objective if objective is not None else ''}",
            f"- Solver runtime: {solve_runtime:.3f} seconds",
            f"- Committed unit-hours: {int(commitment['committed'].sum())}",
            f"- Startup count: {int(commitment['startup'].sum())}",
            f"- Expected thermal energy: {expected_thermal:.3f} MWh",
            f"- Expected wind used: {expected_wind_used:.3f} MWh",
            f"- Expected wind curtailed: {expected_wind_curtailed:.3f} MWh",
            f"- Expected load shedding: {expected_shed:.6f} MWh",
            f"- Max absolute line loading: {max_loading:.3f}%",
            "",
            "This is a synthetic-data DC-UC stage result. It is not the full AC-Benders reproduction.",
            "",
        ]
    )


def _clean_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_case_a_plan(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "case_a_dc_uc_plan.md"
    path.write_text(
        """# Case A DC-UC Implementation Plan

1. Build unit commitment binary variables for each generator and hour.
2. Add active generation, scheduled wind, voltage angle, startup cost variables.
3. Add nodal active-power balance using DC line flow.
4. Add generator min/max output, ramping, line capacity, reference angle, and startup constraints.
5. Solve direct MILP and export objective, commitment matrix, dispatch, and line flows.
6. Align with Table VI and Fig. 4 in the paper.
""",
        encoding="utf-8",
    )
    return path
