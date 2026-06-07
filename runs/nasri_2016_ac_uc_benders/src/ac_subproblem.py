from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ac_power_flow import apparent_flow_sq_pu, voltage_bounds_for_case
from case_data import CaseData
from uc_results import SolveResult


BASE_MVA = 100.0


def solve_ac_subproblem_placeholder(
    case: CaseData,
    case_id: str,
    scenario_id: int,
    hour: int,
    solver_config: dict[str, Any],
    *,
    dry_run: bool = True,
) -> SolveResult:
    start = perf_counter()
    run_dir = case.data_dir.parent
    case_a_dir = run_dir / "results" / "case_a_dc_uc"
    if not (case_a_dir / "bus_angles.csv").exists():
        metadata = {
            "case": case_id,
            "scenario_id": scenario_id,
            "hour": hour,
            "required_input": str(case_a_dir / "bus_angles.csv"),
            "next_step": "Run `python run_reproduction.py --experiment case-a --solve` first.",
        }
        return SolveResult(status="blocked_missing_master_solution", objective=None, runtime_sec=perf_counter() - start, metadata=metadata)

    if not dry_run:
        return solve_ac_nlp_subproblem(
            case,
            case_id=case_id,
            master_results_dir=case_a_dir,
            out_dir=run_dir / "results" / "ac_nlp_subproblem",
            scenario_id=scenario_id,
            hour=hour,
            solver_config=solver_config,
        )

    result = evaluate_ac_subproblems(
        case,
        case_id=case_id,
        master_results_dir=case_a_dir,
        out_dir=run_dir / "results" / "ac_subproblem",
        scenario_filter=scenario_id,
        hour_filter=hour,
    )
    result.metadata["solver_config"] = solver_config
    result.metadata["dry_run"] = dry_run
    return result


def evaluate_ac_subproblems(
    case: CaseData,
    *,
    case_id: str,
    master_results_dir: Path,
    out_dir: Path,
    scenario_filter: int | None = None,
    hour_filter: int | None = None,
) -> SolveResult:
    start = perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    buses = case.table("buses").copy()
    lines = case.table("lines").copy()
    generators = case.table("generators").copy()
    load_profile = case.table("load_profile").copy()
    commitment = pd.read_csv(master_results_dir / "commitment.csv")
    dispatch = pd.read_csv(master_results_dir / "dispatch.csv")
    wind = pd.read_csv(master_results_dir / "wind_usage.csv")
    shedding = pd.read_csv(master_results_dir / "load_shedding.csv")
    angles = pd.read_csv(master_results_dir / "bus_angles.csv")

    if scenario_filter is not None:
        dispatch = dispatch[dispatch["scenario_id"] == scenario_filter]
        wind = wind[wind["scenario_id"] == scenario_filter]
        shedding = shedding[shedding["scenario_id"] == scenario_filter]
        angles = angles[angles["scenario_id"] == scenario_filter]
    if hour_filter is not None:
        dispatch = dispatch[dispatch["hour"] == hour_filter]
        wind = wind[wind["hour"] == hour_filter]
        shedding = shedding[shedding["hour"] == hour_filter]
        angles = angles[angles["hour"] == hour_filter]

    qd_fraction = buses.set_index("bus_id")["qd_fraction"].astype(float).to_dict()
    q_bounds = _q_bounds(generators)
    gen_bus = generators.set_index("gen_id")["bus_id"].astype(int).to_dict()
    wind_bus = case.table("wind_farms").set_index("wind_id")["bus_id"].astype(int).to_dict()
    bus_ids = sorted(buses["bus_id"].astype(int).tolist())
    voltage_min, voltage_max = voltage_bounds_for_case(case_id) or (1.0, 1.0)
    voltage_pu = 1.0

    subproblem_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    reactive_rows: list[dict[str, Any]] = []

    scenario_hours = angles[["scenario_id", "hour"]].drop_duplicates().sort_values(["scenario_id", "hour"])
    for item in scenario_hours.itertuples(index=False):
        scenario_id = int(item.scenario_id)
        hour = int(item.hour)
        theta = {
            int(row.bus_id): float(row.theta_rad)
            for row in angles[(angles["scenario_id"] == scenario_id) & (angles["hour"] == hour)].itertuples()
        }
        total_q_load = float(load_profile[load_profile["hour"].astype(int) == hour]["total_reactive_load_mvar"].iloc[0])
        p_gen_by_bus = _sum_by_bus(dispatch, scenario_id, hour, "gen_id", "dispatch_mw", gen_bus)
        wind_by_bus = _sum_by_bus(wind, scenario_id, hour, "wind_id", "used_mw", wind_bus)
        shed_by_bus = (
            shedding[(shedding["scenario_id"] == scenario_id) & (shedding["hour"] == hour)]
            .groupby("bus_id")["load_shed_mw"]
            .sum()
            .to_dict()
            if not shedding.empty
            else {}
        )
        committed = commitment[commitment["hour"] == hour].set_index("gen_id")["committed"].to_dict()

        max_apparent_loading = 0.0
        overloaded_count = 0
        reactive_violation_mvar = 0.0
        max_active_residual_mw = 0.0
        q_flow_injection = {bus_id: 0.0 for bus_id in bus_ids}
        p_flow_injection = {bus_id: 0.0 for bus_id in bus_ids}

        for line in lines.itertuples():
            from_bus = int(line.from_bus)
            to_bus = int(line.to_bus)
            x_pu = float(line.x_pu)
            delta_ft = theta[from_bus] - theta[to_bus]
            delta_tf = theta[to_bus] - theta[from_bus]
            p_ft = (voltage_pu * voltage_pu / x_pu) * math.sin(delta_ft) * BASE_MVA
            q_ft = ((voltage_pu * voltage_pu) - voltage_pu * voltage_pu * math.cos(delta_ft)) / x_pu * BASE_MVA
            p_tf = (voltage_pu * voltage_pu / x_pu) * math.sin(delta_tf) * BASE_MVA
            q_tf = ((voltage_pu * voltage_pu) - voltage_pu * voltage_pu * math.cos(delta_tf)) / x_pu * BASE_MVA
            apparent = math.sqrt(apparent_flow_sq_pu(p_ft / BASE_MVA, q_ft / BASE_MVA)) * BASE_MVA
            loading = 100.0 * apparent / float(line.rate_mw)
            max_apparent_loading = max(max_apparent_loading, abs(loading))
            if apparent > float(line.rate_mw) + 1.0e-6:
                overloaded_count += 1
            p_flow_injection[from_bus] += p_ft
            p_flow_injection[to_bus] += p_tf
            q_flow_injection[from_bus] += q_ft
            q_flow_injection[to_bus] += q_tf
            line_rows.append(
                {
                    "case_id": case_id,
                    "scenario_id": scenario_id,
                    "hour": hour,
                    "line_id": int(line.line_id),
                    "from_bus": from_bus,
                    "to_bus": to_bus,
                    "p_from_to_mw": p_ft,
                    "q_from_to_mvar": q_ft,
                    "apparent_from_to_mva": apparent,
                    "rate_mva": float(line.rate_mw),
                    "loading_percent": loading,
                    "violated": apparent > float(line.rate_mw) + 1.0e-6,
                }
            )

        for bus_id in bus_ids:
            total_load = float(load_profile[load_profile["hour"].astype(int) == hour]["total_load_mw"].iloc[0])
            p_load = total_load * float(buses.set_index("bus_id").loc[bus_id, "pd_fraction"])
            q_load = total_q_load * float(qd_fraction.get(bus_id, 0.0))
            p_balance = p_gen_by_bus.get(bus_id, 0.0) + wind_by_bus.get(bus_id, 0.0) + shed_by_bus.get(bus_id, 0.0) - p_load - p_flow_injection[bus_id]
            max_active_residual_mw = max(max_active_residual_mw, abs(p_balance))
            q_needed = q_load + q_flow_injection[bus_id]
            q_min, q_max = _bus_q_limits(generators, q_bounds, committed, bus_id)
            q_violation = max(0.0, q_needed - q_max, q_min - q_needed)
            reactive_violation_mvar += q_violation
            reactive_rows.append(
                {
                    "case_id": case_id,
                    "scenario_id": scenario_id,
                    "hour": hour,
                    "bus_id": bus_id,
                    "q_needed_mvar": q_needed,
                    "q_min_available_mvar": q_min,
                    "q_max_available_mvar": q_max,
                    "q_violation_mvar": q_violation,
                    "p_balance_residual_mw": p_balance,
                    "voltage_pu": voltage_pu,
                }
            )

        voltage_violation = max(0.0, voltage_min - voltage_pu, voltage_pu - voltage_max)
        subproblem_rows.append(
            {
                "case_id": case_id,
                "scenario_id": scenario_id,
                "hour": hour,
                "status": "screened_violation" if overloaded_count or reactive_violation_mvar or voltage_violation else "screened_feasible",
                "max_active_balance_residual_mw": max_active_residual_mw,
                "reactive_violation_mvar": reactive_violation_mvar,
                "voltage_violation_pu": voltage_violation,
                "ac_overloaded_line_count": overloaded_count,
                "max_ac_line_loading_percent": max_apparent_loading,
                "feasibility_slack_proxy": reactive_violation_mvar + voltage_violation * BASE_MVA + overloaded_count,
            }
        )

    subproblems = pd.DataFrame(subproblem_rows)
    line_eval = pd.DataFrame(line_rows)
    reactive_eval = pd.DataFrame(reactive_rows)
    sub_path = out_dir / f"{case_id}_subproblem_summary.csv"
    line_path = out_dir / f"{case_id}_line_eval.csv"
    reactive_path = out_dir / f"{case_id}_reactive_eval.csv"
    subproblems.to_csv(sub_path, index=False)
    line_eval.to_csv(line_path, index=False)
    reactive_eval.to_csv(reactive_path, index=False)

    metadata = {
        "case": case_id,
        "description": "AC subproblem screening using fixed DC-UC dispatch and lossless AC flow equations; not an NLP optimum.",
        "scenario_filter": scenario_filter,
        "hour_filter": hour_filter,
        "subproblems_evaluated": int(len(subproblems)),
        "screened_feasible_count": int((subproblems["status"] == "screened_feasible").sum()) if not subproblems.empty else 0,
        "screened_violation_count": int((subproblems["status"] == "screened_violation").sum()) if not subproblems.empty else 0,
        "max_ac_line_loading_percent": float(subproblems["max_ac_line_loading_percent"].max()) if not subproblems.empty else None,
        "max_reactive_violation_mvar": float(subproblems["reactive_violation_mvar"].max()) if not subproblems.empty else None,
        "outputs": {
            "subproblem_summary": str(sub_path),
            "line_eval": str(line_path),
            "reactive_eval": str(reactive_path),
        },
        "limitations": [
            "No AC NLP optimization, no dual multipliers, and no exact Benders cuts yet.",
            "Voltage magnitudes are fixed at 1.0 p.u. for screening.",
            "Branch resistance, charging, transformer taps, and shunts are not yet reconstructed in the AC evaluator.",
        ],
    }
    return SolveResult(status="ac_screening_complete", objective=None, runtime_sec=perf_counter() - start, metadata=metadata)


def solve_ac_nlp_subproblem(
    case: CaseData,
    *,
    case_id: str,
    master_results_dir: Path,
    out_dir: Path,
    scenario_id: int,
    hour: int,
    solver_config: dict[str, Any],
) -> SolveResult:
    """Solve one AC feasibility NLP prototype.

    The model fixes the DC-UC active dispatch and wind schedule, then searches
    voltage magnitudes, voltage angles, and reactive generation within bounds to
    reduce active/reactive balance residuals and apparent-flow overloads.
    """
    start = perf_counter()
    backend = str(solver_config.get("ac_nlp_solver", "scipy_slsqp"))
    if backend == "cyipopt_constrained":
        return solve_ac_constrained_ipopt_subproblem(
            case,
            case_id=case_id,
            master_results_dir=master_results_dir,
            out_dir=out_dir,
            scenario_id=scenario_id,
            hour=hour,
            solver_config=solver_config,
        )
    try:
        minimize, solver_label = _load_nlp_backend(backend)
    except ImportError as exc:
        return SolveResult(
            status="blocked_missing_nlp_solver",
            objective=None,
            runtime_sec=perf_counter() - start,
            metadata={"backend": backend, "error": str(exc)},
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    buses = case.table("buses").copy()
    lines = case.table("lines").copy()
    generators = case.table("generators").copy()
    load_profile = case.table("load_profile").copy()
    wind_farms = case.table("wind_farms").copy()
    commitment = pd.read_csv(master_results_dir / "commitment.csv")
    dispatch = pd.read_csv(master_results_dir / "dispatch.csv")
    wind = pd.read_csv(master_results_dir / "wind_usage.csv")
    angles = pd.read_csv(master_results_dir / "bus_angles.csv")

    bus_ids = sorted(buses["bus_id"].astype(int).tolist())
    slack_bus = 13 if 13 in bus_ids else bus_ids[0]
    theta_seed = (
        angles[(angles["scenario_id"] == scenario_id) & (angles["hour"] == hour)]
        .set_index("bus_id")["theta_rad"]
        .astype(float)
        .to_dict()
    )
    if not theta_seed:
        return SolveResult(
            status="blocked_missing_scenario_hour",
            objective=None,
            runtime_sec=perf_counter() - start,
            metadata={"scenario_id": scenario_id, "hour": hour, "source": str(master_results_dir / "bus_angles.csv")},
        )

    voltage_min, voltage_max = voltage_bounds_for_case(case_id) or (1.0, 1.0)
    non_slack_buses = [bus_id for bus_id in bus_ids if bus_id != slack_bus]
    committed = commitment[commitment["hour"] == hour].set_index("gen_id")["committed"].to_dict()
    active_gens = [
        int(row.gen_id)
        for row in generators.itertuples()
        if round(float(committed.get(int(row.gen_id), 0.0))) > 0
    ]
    q_bounds = _q_bounds(generators)
    gen_bus = generators.set_index("gen_id")["bus_id"].astype(int).to_dict()
    wind_bus = wind_farms.set_index("wind_id")["bus_id"].astype(int).to_dict()
    p_bounds = {
        int(row.gen_id): (float(row.p_min_mw), float(row.p_max_mw))
        for row in generators.itertuples()
    }
    wind_bounds = wind_farms.set_index("wind_id")["p_nom_mw"].astype(float).to_dict()
    dispatch_hour = dispatch[(dispatch["scenario_id"].astype(int) == scenario_id) & (dispatch["hour"].astype(int) == hour)].copy()
    wind_hour = wind[(wind["scenario_id"].astype(int) == scenario_id) & (wind["hour"].astype(int) == hour)].copy()
    master_pg = dispatch_hour.set_index("gen_id")["dispatch_mw"].astype(float).to_dict()
    master_pg = {int(gen_id): _clean_near_zero(float(value)) for gen_id, value in master_pg.items()}
    master_wind = wind_hour.set_index("wind_id")["used_mw"].astype(float).to_dict()
    master_wind = {str(wind_id): _clean_near_zero(float(value)) for wind_id, value in master_wind.items()}
    wind_available = wind_hour.set_index("wind_id")["available_mw"].astype(float).to_dict()
    wind_available = {str(wind_id): float(value) for wind_id, value in wind_available.items()}

    total_load = float(load_profile[load_profile["hour"].astype(int) == hour]["total_load_mw"].iloc[0])
    total_q_load = float(load_profile[load_profile["hour"].astype(int) == hour]["total_reactive_load_mvar"].iloc[0])
    p_load = buses.set_index("bus_id")["pd_fraction"].astype(float).mul(total_load).to_dict()
    q_load = buses.set_index("bus_id")["qd_fraction"].astype(float).mul(total_q_load).to_dict()
    p_gen_by_bus = _sum_by_bus(dispatch, scenario_id, hour, "gen_id", "dispatch_mw", gen_bus)
    wind_by_bus = _sum_by_bus(wind, scenario_id, hour, "wind_id", "used_mw", wind_bus)

    n_theta = len(non_slack_buses)
    n_v = len(bus_ids)
    n_q = len(active_gens)
    theta0 = np.array([theta_seed[bus_id] for bus_id in non_slack_buses], dtype=float)
    v0 = np.ones(n_v, dtype=float)
    q0 = _initial_q_dispatch(active_gens, generators, q_bounds, gen_bus, q_load)
    x0 = np.concatenate([theta0, v0, q0])
    bounds = [(-0.75, 0.75)] * n_theta
    bounds += [(voltage_min, voltage_max)] * n_v
    bounds += [q_bounds.get(gen_id, (0.0, 0.0)) for gen_id in active_gens]

    def unpack(x: np.ndarray) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
        theta = {slack_bus: 0.0}
        theta.update({bus_id: float(x[idx]) for idx, bus_id in enumerate(non_slack_buses)})
        offset = n_theta
        voltage = {bus_id: float(x[offset + idx]) for idx, bus_id in enumerate(bus_ids)}
        offset += n_v
        q_gen = {gen_id: float(x[offset + idx]) for idx, gen_id in enumerate(active_gens)}
        return theta, voltage, q_gen

    def evaluate(x: np.ndarray) -> dict[str, Any]:
        theta, voltage, q_gen = unpack(x)
        p_flow = {bus_id: 0.0 for bus_id in bus_ids}
        q_flow = {bus_id: 0.0 for bus_id in bus_ids}
        max_loading = 0.0
        overload_sq = 0.0
        for line in lines.itertuples():
            from_bus = int(line.from_bus)
            to_bus = int(line.to_bus)
            x_pu = float(line.x_pu)
            p_ft, q_ft = _lossless_ac_flow_mw(voltage[from_bus], theta[from_bus], voltage[to_bus], theta[to_bus], x_pu)
            p_tf, q_tf = _lossless_ac_flow_mw(voltage[to_bus], theta[to_bus], voltage[from_bus], theta[from_bus], x_pu)
            apparent = math.sqrt(apparent_flow_sq_pu(p_ft / BASE_MVA, q_ft / BASE_MVA)) * BASE_MVA
            loading = 100.0 * apparent / float(line.rate_mw)
            max_loading = max(max_loading, abs(loading))
            overload_sq += max(0.0, apparent - float(line.rate_mw)) ** 2
            p_flow[from_bus] += p_ft
            p_flow[to_bus] += p_tf
            q_flow[from_bus] += q_ft
            q_flow[to_bus] += q_tf
        q_gen_by_bus: dict[int, float] = {}
        for gen_id, q_value in q_gen.items():
            bus_id = gen_bus[gen_id]
            q_gen_by_bus[bus_id] = q_gen_by_bus.get(bus_id, 0.0) + q_value
        p_res = []
        q_res = []
        for bus_id in bus_ids:
            p_res.append(p_gen_by_bus.get(bus_id, 0.0) + wind_by_bus.get(bus_id, 0.0) - p_load.get(bus_id, 0.0) - p_flow[bus_id])
            q_res.append(q_gen_by_bus.get(bus_id, 0.0) - q_load.get(bus_id, 0.0) - q_flow[bus_id])
        p_arr = np.array(p_res, dtype=float)
        q_arr = np.array(q_res, dtype=float)
        objective = float(np.sum((p_arr / 100.0) ** 2) + np.sum((q_arr / 100.0) ** 2) + 10.0 * overload_sq / (100.0**2))
        return {
            "objective": objective,
            "max_p_residual_mw": float(np.max(np.abs(p_arr))),
            "max_q_residual_mvar": float(np.max(np.abs(q_arr))),
            "sum_abs_p_residual_mw": float(np.sum(np.abs(p_arr))),
            "sum_abs_q_residual_mvar": float(np.sum(np.abs(q_arr))),
            "max_ac_line_loading_percent": max_loading,
            "line_overload_penalty": float(overload_sq),
            "theta": theta,
            "voltage": voltage,
            "q_gen": q_gen,
        }

    maxiter = int(solver_config.get("ac_nlp_maxiter", 400) or 400)
    solve_start = perf_counter()
    result = _run_nlp_backend(minimize, backend, lambda x: evaluate(x)["objective"], x0, bounds, maxiter)
    solve_runtime = perf_counter() - solve_start
    metrics = evaluate(result.x)
    solution_path = out_dir / f"{case_id}_{backend}_scenario_{scenario_id}_hour_{hour}_nlp_solution.csv"
    _write_nlp_solution(solution_path, metrics, active_gens)
    summary_path = out_dir / f"{case_id}_{backend}_scenario_{scenario_id}_hour_{hour}_nlp_summary.json"
    metadata = {
        "case": case_id,
        "scenario_id": scenario_id,
        "hour": hour,
        "solver": solver_label,
        "backend": backend,
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", getattr(result, "niter", -1))),
        "solve_runtime_sec": solve_runtime,
        "objective": metrics["objective"],
        "max_p_residual_mw": metrics["max_p_residual_mw"],
        "max_q_residual_mvar": metrics["max_q_residual_mvar"],
        "sum_abs_p_residual_mw": metrics["sum_abs_p_residual_mw"],
        "sum_abs_q_residual_mvar": metrics["sum_abs_q_residual_mvar"],
        "max_ac_line_loading_percent": metrics["max_ac_line_loading_percent"],
        "outputs": {"solution": str(solution_path), "summary": str(summary_path)},
        "limitations": [
            "Prototype feasibility NLP; active generation and wind usage are fixed from DC-UC master.",
            "This prototype does not yet expose the dual multipliers needed for exact Benders cuts.",
            "Lossless branch model is used until full MATPOWER AC parameters are wired in.",
        ],
    }
    solve_result = SolveResult(
        status="ac_nlp_solved" if result.success else "ac_nlp_failed",
        objective=metrics["objective"],
        runtime_sec=perf_counter() - start,
        metadata=metadata,
    )
    solve_result.write_json(summary_path)
    return solve_result


def solve_ac_constrained_ipopt_subproblem(
    case: CaseData,
    *,
    case_id: str,
    master_results_dir: Path,
    out_dir: Path,
    scenario_id: int,
    hour: int,
    solver_config: dict[str, Any],
) -> SolveResult:
    start = perf_counter()
    try:
        import cyipopt
    except ImportError as exc:
        return SolveResult(
            status="blocked_missing_nlp_solver",
            objective=None,
            runtime_sec=perf_counter() - start,
            metadata={"backend": "cyipopt_constrained", "error": str(exc)},
        )

    data = _build_ac_nlp_data(case, master_results_dir, scenario_id, hour, case_id)
    if data.get("blocked"):
        return SolveResult(
            status="blocked_missing_scenario_hour",
            objective=None,
            runtime_sec=perf_counter() - start,
            metadata=data,
        )

    problem = ExplicitACSubproblem(data)
    multistart_mode = str(solver_config.get("ac_nlp_multistart_mode", "paper")).strip().lower()
    if multistart_mode == "off":
        start_points = [("dc_seed", problem.initial_point("dc_seed"))]
    elif multistart_mode == "basic":
        start_points = [
            ("dc_seed", problem.initial_point("dc_seed")),
            ("flat_start", problem.initial_point("flat_start")),
        ]
    else:
        start_points = [
            ("dc_seed", problem.initial_point("dc_seed")),
            ("flat_start", problem.initial_point("flat_start")),
            ("flat_start_high_v", problem.initial_point("flat_start_high_v")),
        ]

    best_x: np.ndarray | None = None
    best_info: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_objective: float | None = None
    best_label = ""
    attempt_rows: list[dict[str, Any]] = []
    solve_runtime = 0.0

    for label, x0 in start_points:
        nlp = cyipopt.Problem(
            n=problem.n_variables,
            m=problem.n_constraints,
            problem_obj=problem,
            lb=problem.variable_lower_bounds(),
            ub=problem.variable_upper_bounds(),
            cl=problem.constraint_lower_bounds(),
            cu=problem.constraint_upper_bounds(),
        )
        nlp.add_option("print_level", int(solver_config.get("ipopt_print_level", 0) or 0))
        nlp.add_option("max_iter", int(solver_config.get("ac_nlp_maxiter", 400) or 400))
        nlp.add_option("tol", float(solver_config.get("ac_nlp_tol", 1.0e-7) or 1.0e-7))
        nlp.add_option("hessian_approximation", "limited-memory")
        nlp.add_option("mu_strategy", "adaptive")

        attempt_start = perf_counter()
        x_try, info_try = nlp.solve(x0)
        attempt_runtime = perf_counter() - attempt_start
        solve_runtime += attempt_runtime
        metrics_try = problem.evaluate_metrics(x_try)
        objective_try = problem.objective(x_try)
        success_try = int(info_try.get("status", -999)) in {0, 1}
        infeas_measure = (
            abs(metrics_try["max_p_residual_mw"])
            + abs(metrics_try["max_q_residual_mvar"])
            + abs(metrics_try["max_line_slack_mva"])
            + abs(metrics_try["sum_p_slack_mw"])
            + abs(metrics_try["sum_q_slack_mvar"])
        )
        attempt_rows.append(
            {
                "initialization": label,
                "success": bool(success_try),
                "status_code": int(info_try.get("status", -999)),
                "objective": float(objective_try),
                "runtime_sec": attempt_runtime,
                "infeasibility_measure": float(infeas_measure),
            }
        )
        if best_x is None:
            best_x, best_info, best_metrics, best_objective, best_label = x_try, info_try, metrics_try, objective_try, label
            continue
        best_success = int((best_info or {}).get("status", -999)) in {0, 1}
        better = False
        if success_try and not best_success:
            better = True
        elif success_try == best_success:
            current_infeas = (
                abs((best_metrics or {}).get("max_p_residual_mw", 0.0))
                + abs((best_metrics or {}).get("max_q_residual_mvar", 0.0))
                + abs((best_metrics or {}).get("max_line_slack_mva", 0.0))
                + abs((best_metrics or {}).get("sum_p_slack_mw", 0.0))
                + abs((best_metrics or {}).get("sum_q_slack_mvar", 0.0))
            )
            if infeas_measure + 1.0e-8 < current_infeas:
                better = True
            elif abs(infeas_measure - current_infeas) <= 1.0e-8 and float(objective_try) < float(best_objective or 0.0):
                better = True
        if better:
            best_x, best_info, best_metrics, best_objective, best_label = x_try, info_try, metrics_try, objective_try, label

    assert best_x is not None and best_info is not None and best_metrics is not None and best_objective is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{case_id}_cyipopt_constrained_scenario_{scenario_id}_hour_{hour}"
    solution_path = out_dir / f"{prefix}_nlp_solution.csv"
    constraints_path = out_dir / f"{prefix}_nlp_constraints.csv"
    multipliers_path = out_dir / f"{prefix}_nlp_multipliers.csv"
    summary_path = out_dir / f"{prefix}_nlp_summary.json"
    attempts_path = out_dir / f"{prefix}_nlp_attempts.csv"
    problem.write_solution(solution_path, best_x)
    problem.write_constraints(constraints_path, best_x)
    problem.write_multipliers(multipliers_path, best_info)
    pd.DataFrame(attempt_rows).to_csv(attempts_path, index=False)

    success = int(best_info.get("status", -999)) in {0, 1}
    metadata = {
        "case": case_id,
        "scenario_id": scenario_id,
        "hour": hour,
        "solver": "cyipopt.Problem",
        "backend": "cyipopt_constrained",
        "multistart_mode": multistart_mode,
        "selected_initialization": best_label,
        "attempt_count": len(attempt_rows),
        "success": bool(success),
        "status_code": int(best_info.get("status", -999)),
        "message": _decode_ipopt_message(best_info.get("status_msg", "")),
        "solve_runtime_sec": solve_runtime,
        "objective": best_objective,
        "max_p_residual_mw": best_metrics["max_p_residual_mw"],
        "max_q_residual_mvar": best_metrics["max_q_residual_mvar"],
        "sum_p_slack_mw": best_metrics["sum_p_slack_mw"],
        "sum_q_slack_mvar": best_metrics["sum_q_slack_mvar"],
        "max_line_slack_mva": best_metrics["max_line_slack_mva"],
        "max_ac_line_loading_percent": best_metrics["max_ac_line_loading_percent"],
        "sum_load_shed_mw": best_metrics.get("sum_load_shed_mw"),
        "sum_wind_spillage_mw": best_metrics.get("sum_wind_spillage_mw"),
        "sum_reserve_up_mw": best_metrics.get("sum_reserve_up_mw"),
        "sum_reserve_down_mw": best_metrics.get("sum_reserve_down_mw"),
        "n_variables": problem.n_variables,
        "n_constraints": problem.n_constraints,
        "outputs": {
            "solution": str(solution_path),
            "constraints": str(constraints_path),
            "multipliers": str(multipliers_path),
            "summary": str(summary_path),
            "attempts": str(attempts_path),
        },
        "cut_multiplier_note": "Rows with constraint_type fixed_master_* are explicit master-coupling equalities. Their benders_coefficient column is the current direct source for cut terms.",
    }
    solve_result = SolveResult(
        status="ac_constrained_nlp_solved" if success else "ac_constrained_nlp_failed",
        objective=metadata["objective"],
        runtime_sec=perf_counter() - start,
        metadata=metadata,
    )
    solve_result.write_json(summary_path)
    return solve_result


class ExplicitACSubproblem:
    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.bus_ids = data["bus_ids"]
        self.non_slack_buses = data["non_slack_buses"]
        self.active_gens = data["active_gens"]
        self.p_gen_ids = data["p_gen_ids"]
        self.wind_ids = data["wind_ids"]
        self.lines = data["lines"]
        self.n_theta = len(self.non_slack_buses)
        self.n_v = len(self.bus_ids)
        self.n_q = len(self.active_gens)
        self.n_pg = len(self.p_gen_ids)
        self.n_wind = len(self.wind_ids)
        self.n_reserve_up = len(self.p_gen_ids)
        self.n_reserve_down = len(self.p_gen_ids)
        self.n_load_shed = len(self.bus_ids)
        self.n_wind_spill = len(self.wind_ids)
        self.n_bus = len(self.bus_ids)
        self.n_line = len(self.lines)
        offset = 0
        self.idx_theta = slice(offset, offset + self.n_theta)
        offset += self.n_theta
        self.idx_v = slice(offset, offset + self.n_v)
        offset += self.n_v
        self.idx_q = slice(offset, offset + self.n_q)
        offset += self.n_q
        self.idx_pg = slice(offset, offset + self.n_pg)
        offset += self.n_pg
        self.idx_wind = slice(offset, offset + self.n_wind)
        offset += self.n_wind
        self.idx_reserve_up = slice(offset, offset + self.n_reserve_up)
        offset += self.n_reserve_up
        self.idx_reserve_down = slice(offset, offset + self.n_reserve_down)
        offset += self.n_reserve_down
        self.idx_load_shed = slice(offset, offset + self.n_load_shed)
        offset += self.n_load_shed
        self.idx_wind_spill = slice(offset, offset + self.n_wind_spill)
        offset += self.n_wind_spill
        self.idx_p_pos = slice(offset, offset + self.n_bus)
        offset += self.n_bus
        self.idx_p_neg = slice(offset, offset + self.n_bus)
        offset += self.n_bus
        self.idx_q_pos = slice(offset, offset + self.n_bus)
        offset += self.n_bus
        self.idx_q_neg = slice(offset, offset + self.n_bus)
        offset += self.n_bus
        self.idx_line_slack = slice(offset, offset + self.n_line)
        offset += self.n_line
        self.n_variables = offset
        self.n_constraints = 2 * self.n_bus + self.n_line + self.n_pg + self.n_wind
        rows, cols = np.indices((self.n_constraints, self.n_variables))
        self._jac_rows = rows.ravel().astype(np.int32)
        self._jac_cols = cols.ravel().astype(np.int32)

    def initial_point(self, mode: str = "dc_seed") -> np.ndarray:
        theta_seed = np.array([self.data["theta_seed"][bus_id] for bus_id in self.non_slack_buses], dtype=float)
        if mode == "flat_start":
            theta0 = np.zeros(self.n_theta, dtype=float)
            v0 = np.ones(self.n_v, dtype=float)
        elif mode == "flat_start_high_v":
            theta0 = np.zeros(self.n_theta, dtype=float)
            v0 = np.full(self.n_v, min(self.data["voltage_bounds"][1], 1.05), dtype=float)
        else:
            theta0 = theta_seed
            v0 = np.ones(self.n_v, dtype=float)
        q0 = _initial_q_dispatch(
            self.active_gens,
            self.data["generators"],
            self.data["q_bounds"],
            self.data["gen_bus"],
            self.data["q_load"],
        )
        base = np.concatenate(
            [
                theta0,
                v0,
                q0,
                np.array([self.data["master_pg"].get(gen_id, 0.0) for gen_id in self.p_gen_ids], dtype=float),
                np.array([self.data["master_wind"].get(wind_id, 0.0) for wind_id in self.wind_ids], dtype=float),
                np.zeros(self.n_reserve_up),
                np.zeros(self.n_reserve_down),
                np.zeros(self.n_load_shed),
                np.zeros(self.n_wind_spill),
                np.zeros(self.n_bus),
                np.zeros(self.n_bus),
                np.zeros(self.n_bus),
                np.zeros(self.n_bus),
                np.zeros(self.n_line),
            ]
        )
        raw = self._raw_residuals(base)
        p_res = raw["p_residual"]
        q_res = raw["q_residual"]
        line_expr = raw["line_expr"]
        base[self.idx_p_pos] = np.maximum(-p_res, 0.0)
        base[self.idx_p_neg] = np.maximum(p_res, 0.0)
        base[self.idx_q_pos] = np.maximum(-q_res, 0.0)
        base[self.idx_q_neg] = np.maximum(q_res, 0.0)
        base[self.idx_line_slack] = np.maximum(line_expr, 0.0)
        return base

    def variable_lower_bounds(self) -> np.ndarray:
        voltage_min, _ = self.data["voltage_bounds"]
        bounds = [-0.75] * self.n_theta
        bounds += [voltage_min] * self.n_v
        bounds += [self.data["q_bounds"].get(gen_id, (0.0, 0.0))[0] for gen_id in self.active_gens]
        bounds += [0.0] * self.n_pg
        bounds += [0.0] * self.n_wind
        bounds += [0.0] * self.n_reserve_up
        bounds += [0.0] * self.n_reserve_down
        bounds += [0.0] * self.n_load_shed
        bounds += [0.0] * self.n_wind_spill
        bounds += [0.0] * (4 * self.n_bus + self.n_line)
        return np.array(bounds, dtype=float)

    def variable_upper_bounds(self) -> np.ndarray:
        _, voltage_max = self.data["voltage_bounds"]
        bounds = [0.75] * self.n_theta
        bounds += [voltage_max] * self.n_v
        bounds += [self.data["q_bounds"].get(gen_id, (0.0, 0.0))[1] for gen_id in self.active_gens]
        bounds += [self.data["p_bounds"].get(gen_id, (0.0, 0.0))[1] for gen_id in self.p_gen_ids]
        bounds += [self.data["master_wind"].get(wind_id, self.data["wind_available"].get(wind_id, self.data["wind_bounds"].get(wind_id, 0.0))) for wind_id in self.wind_ids]
        bounds += [self.data["reserve_up_bounds"].get(gen_id, 0.0) for gen_id in self.p_gen_ids]
        bounds += [self.data["reserve_down_bounds"].get(gen_id, 0.0) for gen_id in self.p_gen_ids]
        bounds += [self.data["p_load"].get(bus_id, 0.0) for bus_id in self.bus_ids]
        bounds += [self.data["master_wind"].get(wind_id, self.data["wind_available"].get(wind_id, 0.0)) for wind_id in self.wind_ids]
        bounds += [1.0e4] * (4 * self.n_bus)
        bounds += [1.0e4] * self.n_line
        return np.array(bounds, dtype=float)

    def constraint_lower_bounds(self) -> np.ndarray:
        return np.array([0.0] * (2 * self.n_bus) + [-1.0e19] * self.n_line + [0.0] * (self.n_pg + self.n_wind), dtype=float)

    def constraint_upper_bounds(self) -> np.ndarray:
        return np.array([0.0] * (2 * self.n_bus) + [0.0] * self.n_line + [0.0] * (self.n_pg + self.n_wind), dtype=float)

    def objective(self, x: np.ndarray) -> float:
        p_slack = x[self.idx_p_pos] + x[self.idx_p_neg]
        q_slack = x[self.idx_q_pos] + x[self.idx_q_neg]
        line_slack = x[self.idx_line_slack]
        reserve_up = x[self.idx_reserve_up]
        reserve_down = x[self.idx_reserve_down]
        load_shed = x[self.idx_load_shed]
        return float(
            1000.0 * np.sum((p_slack / 100.0) ** 2)
            + 1000.0 * np.sum((q_slack / 100.0) ** 2)
            + 1000.0 * np.sum(line_slack**2)
            + np.sum(np.array([self.data["reserve_up_cost"].get(gen_id, 0.0) for gen_id in self.p_gen_ids], dtype=float) * reserve_up)
            + np.sum(np.array([self.data["reserve_down_cost"].get(gen_id, 0.0) for gen_id in self.p_gen_ids], dtype=float) * reserve_down)
            + self.data["load_shed_penalty_per_mw"] * np.sum(load_shed)
            + self.data["wind_spill_penalty_per_mw"] * np.sum(x[self.idx_wind_spill])
        )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros(self.n_variables, dtype=float)
        grad[self.idx_p_pos] = 2000.0 * (x[self.idx_p_pos] + x[self.idx_p_neg]) / (100.0**2)
        grad[self.idx_p_neg] = 2000.0 * (x[self.idx_p_pos] + x[self.idx_p_neg]) / (100.0**2)
        grad[self.idx_q_pos] = 2000.0 * (x[self.idx_q_pos] + x[self.idx_q_neg]) / (100.0**2)
        grad[self.idx_q_neg] = 2000.0 * (x[self.idx_q_pos] + x[self.idx_q_neg]) / (100.0**2)
        grad[self.idx_line_slack] = 2000.0 * x[self.idx_line_slack]
        grad[self.idx_reserve_up] = np.array([self.data["reserve_up_cost"].get(gen_id, 0.0) for gen_id in self.p_gen_ids], dtype=float)
        grad[self.idx_reserve_down] = np.array([self.data["reserve_down_cost"].get(gen_id, 0.0) for gen_id in self.p_gen_ids], dtype=float)
        grad[self.idx_load_shed] = self.data["load_shed_penalty_per_mw"]
        grad[self.idx_wind_spill] = self.data["wind_spill_penalty_per_mw"]
        return grad

    def constraints(self, x: np.ndarray) -> np.ndarray:
        raw = self._raw_residuals(x)
        return np.concatenate(
            [
                raw["p_residual"] + x[self.idx_p_pos] - x[self.idx_p_neg],
                raw["q_residual"] + x[self.idx_q_pos] - x[self.idx_q_neg],
                raw["line_expr"] - x[self.idx_line_slack],
                x[self.idx_pg]
                - np.array([self.data["master_pg"].get(gen_id, 0.0) for gen_id in self.p_gen_ids], dtype=float)
                - x[self.idx_reserve_up]
                + x[self.idx_reserve_down],
                x[self.idx_wind] + x[self.idx_wind_spill] - np.array([self.data["master_wind"].get(wind_id, 0.0) for wind_id in self.wind_ids], dtype=float),
            ]
        )

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        return self._jac_rows, self._jac_cols

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return _finite_difference_jacobian(lambda value: self.constraints(value), x).ravel()

    def _unpack(self, x: np.ndarray) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
        theta = {self.data["slack_bus"]: 0.0}
        theta.update({bus_id: float(value) for bus_id, value in zip(self.non_slack_buses, x[self.idx_theta])})
        voltage = {bus_id: float(value) for bus_id, value in zip(self.bus_ids, x[self.idx_v])}
        q_gen = {gen_id: float(value) for gen_id, value in zip(self.active_gens, x[self.idx_q])}
        return theta, voltage, q_gen

    def _unpack_master_coupling(self, x: np.ndarray) -> tuple[dict[int, float], dict[str, float]]:
        p_gen = {gen_id: float(value) for gen_id, value in zip(self.p_gen_ids, x[self.idx_pg])}
        wind = {wind_id: float(value) for wind_id, value in zip(self.wind_ids, x[self.idx_wind])}
        return p_gen, wind

    def _unpack_recourse_adjustments(self, x: np.ndarray) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[str, float]]:
        reserve_up = {gen_id: _clean_nonnegative(float(value)) for gen_id, value in zip(self.p_gen_ids, x[self.idx_reserve_up])}
        reserve_down = {gen_id: _clean_nonnegative(float(value)) for gen_id, value in zip(self.p_gen_ids, x[self.idx_reserve_down])}
        load_shed = {bus_id: _clean_nonnegative(float(value)) for bus_id, value in zip(self.bus_ids, x[self.idx_load_shed])}
        wind_spill = {wind_id: _clean_nonnegative(float(value)) for wind_id, value in zip(self.wind_ids, x[self.idx_wind_spill])}
        return reserve_up, reserve_down, load_shed, wind_spill

    def _raw_residuals(self, x: np.ndarray) -> dict[str, np.ndarray]:
        theta, voltage, q_gen = self._unpack(x)
        p_flow = {bus_id: 0.0 for bus_id in self.bus_ids}
        q_flow = {bus_id: 0.0 for bus_id in self.bus_ids}
        line_expr = []
        line_loading = []
        for line in self.lines:
            from_bus = int(line["from_bus"])
            to_bus = int(line["to_bus"])
            x_pu = float(line["x_pu"])
            p_ft, q_ft = _lossless_ac_flow_mw(voltage[from_bus], theta[from_bus], voltage[to_bus], theta[to_bus], x_pu)
            p_tf, q_tf = _lossless_ac_flow_mw(voltage[to_bus], theta[to_bus], voltage[from_bus], theta[from_bus], x_pu)
            apparent_sq_pu = apparent_flow_sq_pu(p_ft / BASE_MVA, q_ft / BASE_MVA)
            rate_pu = float(line["rate_mw"]) / BASE_MVA
            line_expr.append(apparent_sq_pu - rate_pu * rate_pu)
            line_loading.append(100.0 * math.sqrt(apparent_sq_pu) / rate_pu if rate_pu else 0.0)
            p_flow[from_bus] += p_ft
            p_flow[to_bus] += p_tf
            q_flow[from_bus] += q_ft
            q_flow[to_bus] += q_tf
        q_gen_by_bus: dict[int, float] = {}
        for gen_id, q_value in q_gen.items():
            bus_id = self.data["gen_bus"][gen_id]
            q_gen_by_bus[bus_id] = q_gen_by_bus.get(bus_id, 0.0) + q_value
        p_gen, wind = self._unpack_master_coupling(x)
        _, _, load_shed, _ = self._unpack_recourse_adjustments(x)
        p_gen_by_bus: dict[int, float] = {}
        for gen_id, p_value in p_gen.items():
            bus_id = self.data["gen_bus"][gen_id]
            p_gen_by_bus[bus_id] = p_gen_by_bus.get(bus_id, 0.0) + p_value
        wind_by_bus: dict[int, float] = {}
        for wind_id, wind_value in wind.items():
            bus_id = self.data["wind_bus"][wind_id]
            wind_by_bus[bus_id] = wind_by_bus.get(bus_id, 0.0) + wind_value
        p_res = []
        q_res = []
        for bus_id in self.bus_ids:
            p_res.append(
                p_gen_by_bus.get(bus_id, 0.0)
                + wind_by_bus.get(bus_id, 0.0)
                + load_shed.get(bus_id, 0.0)
                - self.data["p_load"].get(bus_id, 0.0)
                - p_flow[bus_id]
            )
            q_res.append(q_gen_by_bus.get(bus_id, 0.0) - self.data["q_load"].get(bus_id, 0.0) - q_flow[bus_id])
        return {
            "p_residual": np.array(p_res, dtype=float),
            "q_residual": np.array(q_res, dtype=float),
            "line_expr": np.array(line_expr, dtype=float),
            "line_loading": np.array(line_loading, dtype=float),
        }

    def evaluate_metrics(self, x: np.ndarray) -> dict[str, float]:
        raw = self._raw_residuals(x)
        p_slack = x[self.idx_p_pos] + x[self.idx_p_neg]
        q_slack = x[self.idx_q_pos] + x[self.idx_q_neg]
        line_slack = x[self.idx_line_slack]
        reserve_up, reserve_down, load_shed, wind_spill = self._unpack_recourse_adjustments(x)
        return {
            "max_p_residual_mw": float(np.max(np.abs(raw["p_residual"]))),
            "max_q_residual_mvar": float(np.max(np.abs(raw["q_residual"]))),
            "sum_p_slack_mw": float(np.sum(p_slack)),
            "sum_q_slack_mvar": float(np.sum(q_slack)),
            "max_line_slack_mva": float(np.sqrt(max(float(np.max(line_slack)), 0.0)) * BASE_MVA if len(line_slack) else 0.0),
            "max_ac_line_loading_percent": float(np.max(raw["line_loading"])) if len(raw["line_loading"]) else 0.0,
            "sum_load_shed_mw": float(sum(load_shed.values())),
            "sum_wind_spillage_mw": float(sum(wind_spill.values())),
            "sum_reserve_up_mw": float(sum(reserve_up.values())),
            "sum_reserve_down_mw": float(sum(reserve_down.values())),
        }

    def write_solution(self, path: Path, x: np.ndarray) -> None:
        theta, voltage, q_gen = self._unpack(x)
        rows = []
        for bus_id, value in voltage.items():
            rows.append({"component_type": "bus_voltage", "component_id": bus_id, "value": value, "unit": "p.u."})
        for bus_id, value in theta.items():
            rows.append({"component_type": "bus_angle", "component_id": bus_id, "value": value, "unit": "rad"})
        for gen_id, value in q_gen.items():
            rows.append({"component_type": "reactive_generation", "component_id": gen_id, "value": value, "unit": "Mvar"})
        p_gen, wind = self._unpack_master_coupling(x)
        reserve_up, reserve_down, load_shed, wind_spill = self._unpack_recourse_adjustments(x)
        for gen_id, value in p_gen.items():
            rows.append({"component_type": "real_time_active_generation", "component_id": gen_id, "value": value, "unit": "MW"})
        for wind_id, value in wind.items():
            rows.append({"component_type": "real_time_wind_injection", "component_id": wind_id, "value": value, "unit": "MW"})
        for gen_id, value in reserve_up.items():
            rows.append({"component_type": "reserve_up_deployment", "component_id": gen_id, "value": value, "unit": "MW"})
        for gen_id, value in reserve_down.items():
            rows.append({"component_type": "reserve_down_deployment", "component_id": gen_id, "value": value, "unit": "MW"})
        for bus_id, value in load_shed.items():
            rows.append({"component_type": "load_shedding", "component_id": bus_id, "value": value, "unit": "MW"})
        for wind_id, value in wind_spill.items():
            rows.append({"component_type": "wind_spillage", "component_id": wind_id, "value": value, "unit": "MW"})
        for bus_id, pos, neg in zip(self.bus_ids, x[self.idx_p_pos], x[self.idx_p_neg]):
            rows.append({"component_type": "p_balance_slack", "component_id": bus_id, "value": pos + neg, "unit": "MW"})
        for bus_id, pos, neg in zip(self.bus_ids, x[self.idx_q_pos], x[self.idx_q_neg]):
            rows.append({"component_type": "q_balance_slack", "component_id": bus_id, "value": pos + neg, "unit": "Mvar"})
        for line, value in zip(self.lines, x[self.idx_line_slack]):
            rows.append({"component_type": "line_mva_slack_sq_pu", "component_id": line["line_id"], "value": value, "unit": "p.u.^2"})
        pd.DataFrame(rows).to_csv(path, index=False)

    def write_constraints(self, path: Path, x: np.ndarray) -> None:
        values = self.constraints(x)
        rows = []
        offset = 0
        for bus_id, value in zip(self.bus_ids, values[offset : offset + self.n_bus]):
            rows.append({"constraint_type": "p_balance_eq", "component_id": bus_id, "value": value, "lower": 0.0, "upper": 0.0})
        offset += self.n_bus
        for bus_id, value in zip(self.bus_ids, values[offset : offset + self.n_bus]):
            rows.append({"constraint_type": "q_balance_eq", "component_id": bus_id, "value": value, "lower": 0.0, "upper": 0.0})
        offset += self.n_bus
        for line, value in zip(self.lines, values[offset : offset + self.n_line]):
            rows.append({"constraint_type": "line_mva_ineq", "component_id": line["line_id"], "value": value, "lower": -1.0e19, "upper": 0.0})
        offset += self.n_line
        for gen_id, value in zip(self.p_gen_ids, values[offset : offset + self.n_pg]):
            rows.append(
                {
                    "constraint_type": "fixed_master_dispatch_eq",
                    "component_id": gen_id,
                    "variable_name": f"p_s{self.data['scenario_id']}_t{self.data['hour']}_g{gen_id}",
                    "master_value": self.data["master_pg"].get(gen_id, 0.0),
                    "value": value,
                    "lower": 0.0,
                    "upper": 0.0,
                }
            )
        offset += self.n_pg
        for wind_id, value in zip(self.wind_ids, values[offset : offset + self.n_wind]):
            rows.append(
                {
                    "constraint_type": "fixed_master_wind_eq",
                    "component_id": wind_id,
                    "variable_name": f"wind_s{self.data['scenario_id']}_t{self.data['hour']}_{wind_id}",
                    "master_value": self.data["master_wind"].get(wind_id, 0.0),
                    "value": value,
                    "lower": 0.0,
                    "upper": 0.0,
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)

    def write_multipliers(self, path: Path, info: dict[str, Any]) -> None:
        mult_g = np.array(info.get("mult_g", np.zeros(self.n_constraints)), dtype=float)
        rows = []
        offset = 0
        for bus_id, value in zip(self.bus_ids, mult_g[offset : offset + self.n_bus]):
            rows.append({"constraint_type": "p_balance_eq", "component_id": bus_id, "multiplier": value})
        offset += self.n_bus
        for bus_id, value in zip(self.bus_ids, mult_g[offset : offset + self.n_bus]):
            rows.append({"constraint_type": "q_balance_eq", "component_id": bus_id, "multiplier": value})
        offset += self.n_bus
        for line, value in zip(self.lines, mult_g[offset : offset + self.n_line]):
            rows.append({"constraint_type": "line_mva_ineq", "component_id": line["line_id"], "multiplier": value, "benders_coefficient": ""})
        offset += self.n_line
        for gen_id, value in zip(self.p_gen_ids, mult_g[offset : offset + self.n_pg]):
            rows.append(
                {
                    "constraint_type": "fixed_master_dispatch_eq",
                    "component_id": gen_id,
                    "multiplier": value,
                    "benders_coefficient": -float(value),
                    "variable_name": f"p_s{self.data['scenario_id']}_t{self.data['hour']}_g{gen_id}",
                    "master_value": self.data["master_pg"].get(gen_id, 0.0),
                    "notes": "Coefficient is -multiplier for constraint p_ac - p_master_bar = 0.",
                }
            )
        offset += self.n_pg
        for wind_id, value in zip(self.wind_ids, mult_g[offset : offset + self.n_wind]):
            rows.append(
                {
                    "constraint_type": "fixed_master_wind_eq",
                    "component_id": wind_id,
                    "multiplier": value,
                    "benders_coefficient": -float(value),
                    "variable_name": f"wind_s{self.data['scenario_id']}_t{self.data['hour']}_{wind_id}",
                    "master_value": self.data["master_wind"].get(wind_id, 0.0),
                    "notes": "Coefficient is -multiplier for constraint wind_ac - wind_master_bar = 0.",
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)


def _build_ac_nlp_data(
    case: CaseData,
    master_results_dir: Path,
    scenario_id: int,
    hour: int,
    case_id: str,
) -> dict[str, Any]:
    buses = case.table("buses").copy()
    lines_df = case.table("lines").copy()
    generators = case.table("generators").copy()
    load_profile = case.table("load_profile").copy()
    wind_farms = case.table("wind_farms").copy()
    commitment = pd.read_csv(master_results_dir / "commitment.csv")
    dispatch = pd.read_csv(master_results_dir / "dispatch.csv")
    wind = pd.read_csv(master_results_dir / "wind_usage.csv")
    angles = pd.read_csv(master_results_dir / "bus_angles.csv")
    parameters = case.table("paper_parameters").copy() if "paper_parameters" in case.tables else pd.DataFrame()
    bus_ids = sorted(buses["bus_id"].astype(int).tolist())
    slack_bus = 13 if 13 in bus_ids else bus_ids[0]
    theta_seed = (
        angles[(angles["scenario_id"] == scenario_id) & (angles["hour"] == hour)]
        .set_index("bus_id")["theta_rad"]
        .astype(float)
        .to_dict()
    )
    if not theta_seed:
        return {"blocked": True, "scenario_id": scenario_id, "hour": hour}
    committed = commitment[commitment["hour"] == hour].set_index("gen_id")["committed"].to_dict()
    active_gens = [
        int(row.gen_id)
        for row in generators.itertuples()
        if round(float(committed.get(int(row.gen_id), 0.0))) > 0
    ]
    gen_bus = generators.set_index("gen_id")["bus_id"].astype(int).to_dict()
    wind_bus = wind_farms.set_index("wind_id")["bus_id"].astype(int).to_dict()
    p_bounds = {
        int(row.gen_id): (float(row.p_min_mw), float(row.p_max_mw))
        for row in generators.itertuples()
    }
    wind_bounds = wind_farms.set_index("wind_id")["p_nom_mw"].astype(float).to_dict()
    dispatch_hour = dispatch[(dispatch["scenario_id"].astype(int) == scenario_id) & (dispatch["hour"].astype(int) == hour)].copy()
    wind_hour = wind[(wind["scenario_id"].astype(int) == scenario_id) & (wind["hour"].astype(int) == hour)].copy()
    master_pg = dispatch_hour.set_index("gen_id")["dispatch_mw"].astype(float).to_dict()
    master_pg = {int(gen_id): _clean_near_zero(float(value)) for gen_id, value in master_pg.items()}
    master_wind = wind_hour.set_index("wind_id")["used_mw"].astype(float).to_dict()
    master_wind = {str(wind_id): _clean_near_zero(float(value)) for wind_id, value in master_wind.items()}
    wind_available = wind_hour.set_index("wind_id")["available_mw"].astype(float).to_dict()
    wind_available = {str(wind_id): float(value) for wind_id, value in wind_available.items()}
    reserve_up_bounds = {
        gen_id: max(0.0, min(float(generators[generators["gen_id"] == gen_id]["reserve_up_mw"].iloc[0]), max(p_bounds.get(gen_id, (0.0, 0.0))[1] - master_pg.get(gen_id, 0.0), 0.0)))
        for gen_id in master_pg
    }
    reserve_down_bounds = {
        gen_id: max(0.0, min(float(generators[generators["gen_id"] == gen_id]["reserve_down_mw"].iloc[0]), max(master_pg.get(gen_id, 0.0) - p_bounds.get(gen_id, (0.0, 0.0))[0], 0.0)))
        for gen_id in master_pg
    }
    reserve_up_cost = {gen_id: float(generators[generators["gen_id"] == gen_id]["paper_cost_usd_per_pu"].iloc[0]) / BASE_MVA for gen_id in master_pg}
    reserve_down_cost = {gen_id: 0.0 for gen_id in master_pg}
    load_shed_penalty_per_mw = 100.0
    if not parameters.empty:
        row = parameters[parameters["parameter"] == "load_shed_value"]
        if not row.empty:
            load_shed_penalty_per_mw = float(row.iloc[0]["value"])
    wind_spill_penalty_per_mw = 0.0
    total_load = float(load_profile[load_profile["hour"].astype(int) == hour]["total_load_mw"].iloc[0])
    total_q_load = float(load_profile[load_profile["hour"].astype(int) == hour]["total_reactive_load_mvar"].iloc[0])
    lines = [
        {
            "line_id": int(row.line_id),
            "from_bus": int(row.from_bus),
            "to_bus": int(row.to_bus),
            "x_pu": float(row.x_pu),
            "rate_mw": float(row.rate_mw),
        }
        for row in lines_df.itertuples()
    ]
    return {
        "blocked": False,
        "bus_ids": bus_ids,
        "non_slack_buses": [bus_id for bus_id in bus_ids if bus_id != slack_bus],
        "slack_bus": slack_bus,
        "theta_seed": theta_seed,
        "voltage_bounds": voltage_bounds_for_case(case_id) or (1.0, 1.0),
        "generators": generators,
        "active_gens": active_gens,
        "p_gen_ids": sorted(gen_id for gen_id, value in master_pg.items() if gen_id in active_gens or abs(value) > 1.0e-7),
        "wind_ids": sorted(master_wind),
        "q_bounds": _q_bounds(generators),
        "p_bounds": p_bounds,
        "wind_bounds": wind_bounds,
        "gen_bus": gen_bus,
        "wind_bus": wind_bus,
        "master_pg": master_pg,
        "master_wind": master_wind,
        "wind_available": wind_available,
        "reserve_up_bounds": reserve_up_bounds,
        "reserve_down_bounds": reserve_down_bounds,
        "reserve_up_cost": reserve_up_cost,
        "reserve_down_cost": reserve_down_cost,
        "load_shed_penalty_per_mw": load_shed_penalty_per_mw,
        "wind_spill_penalty_per_mw": wind_spill_penalty_per_mw,
        "p_load": buses.set_index("bus_id")["pd_fraction"].astype(float).mul(total_load).to_dict(),
        "q_load": buses.set_index("bus_id")["qd_fraction"].astype(float).mul(total_q_load).to_dict(),
        "scenario_id": scenario_id,
        "hour": hour,
        "lines": lines,
    }


def _load_nlp_backend(backend: str):
    if backend == "scipy_slsqp":
        from scipy.optimize import minimize

        return minimize, "scipy.optimize.minimize(method='SLSQP')"
    if backend == "cyipopt":
        from cyipopt import minimize_ipopt

        return minimize_ipopt, "cyipopt.minimize_ipopt"
    raise ImportError(f"Unsupported AC NLP backend: {backend}")


def _run_nlp_backend(minimize_fn: Any, backend: str, objective: Any, x0: np.ndarray, bounds: list[tuple[float, float]], maxiter: int):
    if backend == "scipy_slsqp":
        return minimize_fn(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1.0e-7, "disp": False},
        )
    if backend == "cyipopt":
        from scipy.optimize import minimize as scipy_minimize

        warm = scipy_minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": min(maxiter, 200), "ftol": 1.0e-6, "disp": False},
        )
        start_x = warm.x if getattr(warm, "success", False) else x0
        return minimize_fn(
            objective,
            x0=start_x,
            jac=lambda x: _finite_difference_gradient(objective, x),
            bounds=bounds,
            options={"max_iter": maxiter, "tol": 1.0e-7, "print_level": 0},
        )
    raise ValueError(f"Unsupported AC NLP backend: {backend}")


def _finite_difference_gradient(objective: Any, x: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(x, dtype=float)
    f0 = objective(x)
    for idx in range(len(x)):
        step = 1.0e-6 * max(1.0, abs(float(x[idx])))
        xp = x.copy()
        xp[idx] += step
        grad[idx] = (objective(xp) - f0) / step
    return grad


def _finite_difference_jacobian(function: Any, x: np.ndarray) -> np.ndarray:
    f0 = np.array(function(x), dtype=float)
    jac = np.zeros((len(f0), len(x)), dtype=float)
    for idx in range(len(x)):
        step = 1.0e-6 * max(1.0, abs(float(x[idx])))
        xp = x.copy()
        xp[idx] += step
        jac[:, idx] = (np.array(function(xp), dtype=float) - f0) / step
    return jac


def _decode_ipopt_message(message: Any) -> str:
    if isinstance(message, bytes):
        return message.decode("utf-8", errors="replace")
    return str(message)


def _clean_near_zero(value: float, tolerance: float = 1.0e-9) -> float:
    return 0.0 if abs(value) <= tolerance else value


def _clean_nonnegative(value: float, tolerance: float = 1.0e-7) -> float:
    if value < 0.0 and abs(value) <= tolerance:
        return 0.0
    return value


def _q_bounds(generators: pd.DataFrame) -> dict[int, tuple[float, float]]:
    return {
        int(row.gen_id): (float(row.q_min_mvar), float(row.q_max_mvar))
        for row in generators.itertuples()
        if hasattr(row, "q_min_mvar") and hasattr(row, "q_max_mvar")
    }


def _initial_q_dispatch(
    active_gens: list[int],
    generators: pd.DataFrame,
    q_bounds: dict[int, tuple[float, float]],
    gen_bus: dict[int, int],
    q_load: dict[int, float],
) -> np.ndarray:
    values = []
    for gen_id in active_gens:
        q_min, q_max = q_bounds.get(gen_id, (0.0, 0.0))
        bus_id = gen_bus[gen_id]
        online_at_bus = [
            int(row.gen_id)
            for row in generators[generators["bus_id"].astype(int) == bus_id].itertuples()
            if int(row.gen_id) in active_gens
        ]
        share = q_load.get(bus_id, 0.0) / max(len(online_at_bus), 1)
        values.append(min(max(share, q_min), q_max))
    return np.array(values, dtype=float)


def _lossless_ac_flow_mw(v_i: float, theta_i: float, v_j: float, theta_j: float, x_pu: float) -> tuple[float, float]:
    delta = theta_i - theta_j
    p_mw = (v_i * v_j / x_pu) * math.sin(delta) * BASE_MVA
    q_mvar = ((v_i * v_i) - v_i * v_j * math.cos(delta)) / x_pu * BASE_MVA
    return p_mw, q_mvar


def _write_nlp_solution(path: Path, metrics: dict[str, Any], active_gens: list[int]) -> None:
    rows = []
    for bus_id, voltage in metrics["voltage"].items():
        rows.append({"component_type": "bus_voltage", "component_id": bus_id, "value": voltage, "unit": "p.u."})
    for bus_id, theta in metrics["theta"].items():
        rows.append({"component_type": "bus_angle", "component_id": bus_id, "value": theta, "unit": "rad"})
    for gen_id in active_gens:
        rows.append(
            {
                "component_type": "reactive_generation",
                "component_id": gen_id,
                "value": metrics["q_gen"].get(gen_id, 0.0),
                "unit": "Mvar",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _bus_q_limits(
    generators: pd.DataFrame,
    q_bounds: dict[int, tuple[float, float]],
    committed: dict[int, float],
    bus_id: int,
) -> tuple[float, float]:
    q_min = 0.0
    q_max = 0.0
    for row in generators[generators["bus_id"].astype(int) == bus_id].itertuples():
        gen_id = int(row.gen_id)
        if round(float(committed.get(gen_id, 0.0))) <= 0:
            continue
        bounds = q_bounds.get(gen_id, (0.0, 0.0))
        q_min += bounds[0]
        q_max += bounds[1]
    return q_min, q_max


def _sum_by_bus(
    df: pd.DataFrame,
    scenario_id: int,
    hour: int,
    component_col: str,
    value_col: str,
    bus_map: dict[Any, int],
) -> dict[int, float]:
    rows = df[(df["scenario_id"] == scenario_id) & (df["hour"] == hour)]
    out: dict[int, float] = {}
    for row in rows.itertuples():
        component = getattr(row, component_col)
        bus_id = int(bus_map[component])
        out[bus_id] = out.get(bus_id, 0.0) + float(getattr(row, value_col))
    return out


def write_ac_subproblem_plan(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ac_subproblem_plan.md"
    path.write_text(
        """# AC Subproblem Implementation Plan

1. Implement Appendix A AC active/reactive/apparent flow equations.
2. For a fixed commitment and scheduled dispatch, solve one NLP for each scenario-hour pair.
3. Include slack variables for reactive generation and voltage magnitude infeasibility as in subproblem (5).
4. Export objective, slack values, voltages, line flows, and dual multipliers for fixed first-stage constraints.
5. Use these dual multipliers to build Benders cuts for the MILP master.
6. Validate first on one scenario and one hour before enabling all 40 x 24 subproblems.
""",
        encoding="utf-8",
    )
    return path
