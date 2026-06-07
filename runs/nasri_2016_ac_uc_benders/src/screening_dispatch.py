from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from case_data import CaseData
from uc_results import SolveResult


@dataclass
class DispatchScreeningConfig:
    scenario_id: int | None = None
    overload_tolerance: float = 1.0e-6


def run_dispatch_screening(
    case: CaseData,
    solver_config: dict[str, Any],
    out_dir: Path,
    *,
    config: DispatchScreeningConfig | None = None,
) -> SolveResult:
    """Run a no-solver screening pass over scenario-hour data.

    This is not a UC solve. It uses a merit-order dispatch and linear DC flow
    reconstruction to test whether the reconstructed data are internally usable
    before a MILP/NLP backend is attached.
    """
    start = perf_counter()
    config = config or DispatchScreeningConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    buses = case.table("buses").copy()
    lines = case.table("lines").copy()
    gens = case.table("generators").copy()
    load_profile = case.table("load_profile").copy()
    wind_profile = case.table("wind_profile").copy()
    wind_farms = case.table("wind_farms").copy()

    if config.scenario_id is not None:
        wind_profile = wind_profile[wind_profile["scenario_id"] == config.scenario_id].copy()

    bus_ids = sorted(buses["bus_id"].astype(int).tolist())
    bus_index = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
    slack_bus = 13 if 13 in bus_index else bus_ids[0]
    b_matrix = build_dc_b_matrix(lines, bus_index, case.base_mva)

    gen_cost = generator_cost(gens)
    rows: list[dict[str, Any]] = []
    dispatch_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []

    for scenario_id in sorted(wind_profile["scenario_id"].astype(int).unique()):
        scenario_wind = wind_profile[wind_profile["scenario_id"] == scenario_id]
        for hour in sorted(load_profile["hour"].astype(int).unique()):
            load_row = load_profile[load_profile["hour"].astype(int) == hour].iloc[0]
            wind_rows = scenario_wind[scenario_wind["hour"].astype(int) == hour]
            total_load = float(load_row["total_load_mw"])
            dispatch = merit_order_dispatch(total_load - float(wind_rows["production_mw"].sum()), gens, gen_cost)
            injections = nodal_injections(
                bus_ids=bus_ids,
                buses=buses,
                gens=gens,
                dispatch=dispatch,
                wind_rows=wind_rows,
                wind_farms=wind_farms,
                total_load_mw=total_load,
            )
            theta, balanced_injections = solve_dc_angles(b_matrix, injections, bus_index[slack_bus])
            flows = compute_line_flows(lines, theta, bus_index, case.base_mva)
            overloaded = flows[np.abs(flows["flow_mw"]) > flows["rate_mw"] + config.overload_tolerance]

            for gen_id, p_mw in dispatch["dispatch_mw"].items():
                dispatch_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "hour": hour,
                        "gen_id": gen_id,
                        "dispatch_mw": p_mw,
                    }
                )
            for _, flow in flows.iterrows():
                flow_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "hour": hour,
                        "line_id": int(flow["line_id"]),
                        "from_bus": int(flow["from_bus"]),
                        "to_bus": int(flow["to_bus"]),
                        "flow_mw": float(flow["flow_mw"]),
                        "rate_mw": float(flow["rate_mw"]),
                        "loading_percent": float(flow["loading_percent"]),
                    }
                )

            rows.append(
                {
                    "scenario_id": scenario_id,
                    "hour": hour,
                    "total_load_mw": total_load,
                    "wind_mw": float(wind_rows["production_mw"].sum()),
                    "thermal_dispatch_mw": float(sum(dispatch["dispatch_mw"].values())),
                    "shortage_mw": float(dispatch["shortage_mw"]),
                    "curtailment_mw": float(dispatch["curtailment_mw"]),
                    "slack_bus": slack_bus,
                    "slack_adjustment_mw": float(balanced_injections[bus_index[slack_bus]] - injections[bus_index[slack_bus]]),
                    "max_abs_line_loading_percent": float(flows["loading_percent"].abs().max()),
                    "overloaded_line_count": int(len(overloaded)),
                }
            )

    summary = pd.DataFrame(rows)
    dispatch_df = pd.DataFrame(dispatch_rows)
    flows_df = pd.DataFrame(flow_rows)
    summary_path = out_dir / "summary.csv"
    dispatch_path = out_dir / "dispatch.csv"
    flows_path = out_dir / "line_flows.csv"
    summary.to_csv(summary_path, index=False)
    dispatch_df.to_csv(dispatch_path, index=False)
    flows_df.to_csv(flows_path, index=False)

    metadata = {
        "case": "screening_dispatch",
        "description": "No-solver merit-order dispatch plus DC flow screening; not a paper-level UC solution.",
        "scenario_filter": config.scenario_id,
        "hours_evaluated": int(summary["hour"].nunique()) if not summary.empty else 0,
        "scenarios_evaluated": int(summary["scenario_id"].nunique()) if not summary.empty else 0,
        "max_shortage_mw": float(summary["shortage_mw"].max()) if not summary.empty else None,
        "max_curtailment_mw": float(summary["curtailment_mw"].max()) if not summary.empty else None,
        "max_abs_line_loading_percent": float(summary["max_abs_line_loading_percent"].max()) if not summary.empty else None,
        "hours_with_overloads": int((summary["overloaded_line_count"] > 0).sum()) if not summary.empty else 0,
        "outputs": {
            "summary": str(summary_path),
            "dispatch": str(dispatch_path),
            "line_flows": str(flows_path),
        },
        "solver_config": solver_config,
        "limitations": [
            "No commitment binaries, startup/shutdown costs, ramp chronology, reserve scheduling, or Benders cuts.",
            "Uses synthetic calibrated wind profiles, so results are synthetic-data reproduction rather than exact paper-result reproduction.",
        ],
    }
    return SolveResult(
        status="screening_complete",
        objective=None,
        runtime_sec=perf_counter() - start,
        metadata=metadata,
    )


def generator_cost(gens: pd.DataFrame) -> dict[int, float]:
    if "paper_cost_usd_per_pu" in gens.columns:
        cost_col = "paper_cost_usd_per_pu"
    else:
        cost_col = "p_max_mw"
    return {int(row.gen_id): float(getattr(row, cost_col)) for row in gens.itertuples()}


def merit_order_dispatch(net_load_mw: float, gens: pd.DataFrame, gen_cost: dict[int, float]) -> dict[str, Any]:
    remaining = max(0.0, net_load_mw)
    dispatch: dict[int, float] = {}
    for row in sorted(gens.itertuples(), key=lambda item: (gen_cost[int(item.gen_id)], int(item.gen_id))):
        gen_id = int(row.gen_id)
        p = min(float(row.p_max_mw), remaining)
        dispatch[gen_id] = p
        remaining -= p
        if remaining <= 1.0e-9:
            remaining = 0.0
    for row in gens.itertuples():
        dispatch.setdefault(int(row.gen_id), 0.0)
    total_dispatch = sum(dispatch.values())
    return {
        "dispatch_mw": dispatch,
        "shortage_mw": max(0.0, net_load_mw - total_dispatch),
        "curtailment_mw": max(0.0, -net_load_mw),
    }


def nodal_injections(
    *,
    bus_ids: list[int],
    buses: pd.DataFrame,
    gens: pd.DataFrame,
    dispatch: dict[str, Any],
    wind_rows: pd.DataFrame,
    wind_farms: pd.DataFrame,
    total_load_mw: float,
) -> np.ndarray:
    injections = np.zeros(len(bus_ids), dtype=float)
    bus_index = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
    for row in buses.itertuples():
        injections[bus_index[int(row.bus_id)]] -= total_load_mw * float(row.pd_fraction)
    for row in gens.itertuples():
        injections[bus_index[int(row.bus_id)]] += float(dispatch["dispatch_mw"][int(row.gen_id)])
    farm_bus = {str(row.wind_id): int(row.bus_id) for row in wind_farms.itertuples()}
    for row in wind_rows.itertuples():
        injections[bus_index[farm_bus[str(row.wind_id)]]] += float(row.production_mw)
    return injections


def build_dc_b_matrix(lines: pd.DataFrame, bus_index: dict[int, int], base_mva: float) -> np.ndarray:
    n_bus = len(bus_index)
    b = np.zeros((n_bus, n_bus), dtype=float)
    for row in lines.itertuples():
        i = bus_index[int(row.from_bus)]
        j = bus_index[int(row.to_bus)]
        susceptance = base_mva / float(row.x_pu)
        b[i, i] += susceptance
        b[j, j] += susceptance
        b[i, j] -= susceptance
        b[j, i] -= susceptance
    return b


def solve_dc_angles(b_matrix: np.ndarray, injections: np.ndarray, slack_index: int) -> tuple[np.ndarray, np.ndarray]:
    balanced = injections.copy()
    balanced[slack_index] -= balanced.sum()
    keep = [idx for idx in range(len(injections)) if idx != slack_index]
    theta = np.zeros(len(injections), dtype=float)
    theta[keep] = np.linalg.solve(b_matrix[np.ix_(keep, keep)], balanced[keep])
    return theta, balanced


def compute_line_flows(lines: pd.DataFrame, theta: np.ndarray, bus_index: dict[int, int], base_mva: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in lines.itertuples():
        i = bus_index[int(row.from_bus)]
        j = bus_index[int(row.to_bus)]
        flow_mw = (theta[i] - theta[j]) / float(row.x_pu) * base_mva
        rate_mw = float(row.rate_mw)
        rows.append(
            {
                "line_id": int(row.line_id),
                "from_bus": int(row.from_bus),
                "to_bus": int(row.to_bus),
                "flow_mw": flow_mw,
                "rate_mw": rate_mw,
                "loading_percent": 100.0 * flow_mw / rate_mw if rate_mw else np.nan,
            }
        )
    return pd.DataFrame(rows)
