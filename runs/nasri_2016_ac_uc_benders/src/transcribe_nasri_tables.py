from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


BASE_MVA = 100.0

TABLE_I_LINE_LIMITS_PU = {
    (1, 3): 2.3,
    (3, 9): 1.6,
    (6, 10): 2.5,
    (10, 11): 3.0,
    (11, 14): 2.3,
    (14, 16): 2.1,
}

TABLE_II_GROUPS = [
    ([1, 2], 1, 0.100, 0.200, 0.00, 0.10, 0.000, 0.100, 1109, 300, 1, 0.20, 0.00),
    ([3, 4], 1, 0.152, 0.760, -0.25, 0.30, 0.100, 0.500, 1246, 400, 1, 0.76, 0.00),
    ([5, 6], 2, 0.100, 0.200, 0.00, 0.10, 0.000, 0.100, 1109, 300, 1, 0.20, 0.00),
    ([7, 8], 2, 0.152, 0.760, -0.25, 0.30, 0.100, 0.500, 1246, 400, 1, 0.76, 0.00),
    ([9], 7, 0.800, 3.500, -0.25, 1.50, 1.000, 2.500, 1720, 100, 0, 0.00, 0.00),
    ([10, 11], 7, 0.150, 1.000, 0.00, 0.60, 0.550, 0.850, 1660, 275, 1, 0.55, 0.45),
    ([12, 13, 14], 13, 0.620, 1.970, 0.00, 0.80, 0.450, 1.150, 1408, 300, 1, 1.97, 0.00),
    ([15, 16, 17, 18, 19], 15, 0.024, 0.120, 0.00, 0.06, 0.096, 0.096, 2141, 400, 0, 0.00, 0.00),
    ([20], 15, 0.500, 1.550, -0.50, 0.80, 0.450, 1.000, 1592, 200, 1, 1.10, 0.45),
    ([21], 16, 0.500, 1.550, -0.50, 0.80, 0.450, 1.000, 1592, 200, 1, 1.10, 0.45),
    ([22], 18, 1.000, 4.000, -0.50, 2.00, 1.500, 2.800, 1917, 250, 0, 0.00, 0.00),
    ([23], 21, 1.000, 4.000, -0.50, 2.00, 1.500, 2.800, 1917, 250, 0, 0.00, 0.00),
    ([24, 25, 26, 27, 28, 29], 22, 0.000, 0.500, -0.10, 0.16, 0.150, 0.500, 0, 100, 1, 0.50, 0.00),
    ([30, 31], 23, 0.500, 1.550, -0.50, 0.80, 0.450, 1.000, 1592, 200, 1, 1.10, 0.45),
    ([32], 23, 0.800, 3.500, -0.25, 1.50, 1.000, 2.500, 1720, 100, 0, 0.00, 0.00),
]

TABLE_III_LOAD_FACTORS = [
    0.75, 0.70, 0.65, 0.60, 0.62, 0.63, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78,
    0.80, 0.85, 0.85, 0.90, 0.92, 0.95, 0.98, 1.00, 0.97, 0.93, 0.91, 0.92,
]

TABLE_IV_SCENARIO_PROBABILITIES = {
    1: 0.01, 2: 0.01, 3: 0.01, 4: 0.02, 5: 0.02,
    6: 0.03, 7: 0.03, 8: 0.03, 9: 0.04, 10: 0.04,
    11: 0.05, 12: 0.05, 13: 0.01, 14: 0.01, 15: 0.01,
    16: 0.01, 17: 0.01, 18: 0.02, 19: 0.02, 20: 0.03,
    21: 0.03, 22: 0.04, 23: 0.04, 24: 0.05, 25: 0.05,
    26: 0.01, 27: 0.01, 28: 0.01, 29: 0.01, 30: 0.01,
    31: 0.02, 32: 0.02, 33: 0.02, 34: 0.02, 35: 0.02,
    36: 0.03, 37: 0.03, 38: 0.04, 39: 0.04, 40: 0.04,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe visible Nasri 2016 Tables I-IV.")
    parser.add_argument("--data-dir", default="../data")
    args = parser.parse_args()
    data_dir = resolve_relative(Path(args.data_dir))
    transcribe_tables(data_dir)


def transcribe_tables(data_dir: Path) -> None:
    apply_table_i_line_limits(data_dir)
    write_table_ii_generators(data_dir)
    write_load_factors(data_dir)
    write_scenario_probabilities(data_dir)
    write_load_profile_from_matpower(data_dir)
    write_report(data_dir)


def apply_table_i_line_limits(data_dir: Path) -> None:
    path = data_dir / "lines.csv"
    lines = pd.read_csv(path)
    lines["table_i_sbar_pu"] = ""
    lines["table_i_source"] = ""
    for idx, row in lines.iterrows():
        key = (int(row["from_bus"]), int(row["to_bus"]))
        rev = (key[1], key[0])
        if key in TABLE_I_LINE_LIMITS_PU or rev in TABLE_I_LINE_LIMITS_PU:
            value = TABLE_I_LINE_LIMITS_PU.get(key, TABLE_I_LINE_LIMITS_PU.get(rev))
            lines.loc[idx, "table_i_sbar_pu"] = value
            lines.loc[idx, "rate_mw"] = value * BASE_MVA
            lines.loc[idx, "table_i_source"] = "Nasri 2016 Table I"
            lines.loc[idx, "notes"] = "rate overridden from Nasri 2016 Table I"
    lines.to_csv(path, index=False)


def write_table_ii_generators(data_dir: Path) -> None:
    rows = []
    cost_rows = []
    for units, node, p_min, p_max, q_min, q_max, ramp_ud, reserve_ud, cost, startup, u_ini, p_ini, r_ini in TABLE_II_GROUPS:
        for unit in units:
            rows.append(
                {
                    "gen_id": unit,
                    "bus_id": node,
                    "p_min_mw": p_min * BASE_MVA,
                    "p_max_mw": p_max * BASE_MVA,
                    "startup_cost": startup,
                    "shutdown_cost": "",
                    "fixed_cost": "",
                    "ramp_up_mw": ramp_ud * BASE_MVA,
                    "ramp_down_mw": ramp_ud * BASE_MVA,
                    "min_up_h": "",
                    "min_down_h": "",
                    "initial_status": u_ini,
                    "initial_p_mw": p_ini * BASE_MVA,
                    "q_min_mvar": q_min * BASE_MVA,
                    "q_max_mvar": q_max * BASE_MVA,
                    "reserve_up_mw": reserve_ud * BASE_MVA,
                    "reserve_down_mw": reserve_ud * BASE_MVA,
                    "initial_reserve_mw": r_ini * BASE_MVA,
                    "paper_cost_usd_per_pu": cost,
                    "notes": "transcribed from Nasri 2016 Table II",
                }
            )
            cost_rows.append(
                {
                    "gen_id": unit,
                    "segment_id": 1,
                    "p_max_segment_mw": p_max * BASE_MVA,
                    "marginal_cost": cost,
                    "notes": "C_i from Nasri 2016 Table II; units USD per p.u.",
                }
            )
    pd.DataFrame(rows).to_csv(data_dir / "generators.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(data_dir / "generator_cost_segments.csv", index=False)


def write_load_factors(data_dir: Path) -> None:
    rows = [
        {
            "hour": hour,
            "load_factor": factor,
            "source": "Nasri 2016 Table III",
            "notes": "",
        }
        for hour, factor in enumerate(TABLE_III_LOAD_FACTORS, start=1)
    ]
    pd.DataFrame(rows).to_csv(data_dir / "load_factors.csv", index=False)


def write_scenario_probabilities(data_dir: Path) -> None:
    rows = [
        {
            "scenario_id": scenario,
            "probability": probability,
            "source": "Nasri 2016 Table IV",
            "notes": "",
        }
        for scenario, probability in sorted(TABLE_IV_SCENARIO_PROBABILITIES.items())
    ]
    pd.DataFrame(rows).to_csv(data_dir / "scenario_probabilities.csv", index=False)


def write_load_profile_from_matpower(data_dir: Path) -> None:
    bus_path = data_dir.parent / "sources" / "case24_ieee_rts.m"
    if not bus_path.exists():
        return
    bus_rows = parse_matpower_bus_rows(bus_path)
    total_p = sum(row[2] for row in bus_rows)
    total_q = sum(row[3] for row in bus_rows)
    rows = []
    for hour, factor in enumerate(TABLE_III_LOAD_FACTORS, start=1):
        rows.append(
            {
                "hour": hour,
                "total_load_mw": total_p * factor,
                "total_reactive_load_mvar": total_q * factor,
                "nodal_allocation_source": "MATPOWER case24_ieee_rts bus Pd/Qd scaled by Nasri Table III",
                "notes": f"base total Pd={total_p}; base total Qd={total_q}",
            }
        )
    pd.DataFrame(rows).to_csv(data_dir / "load_profile.csv", index=False)


def parse_matpower_bus_rows(case_path: Path) -> list[list[float]]:
    text = case_path.read_text(encoding="utf-8", errors="replace")
    body = text.split("mpc.bus = [", 1)[1].split("];", 1)[0]
    rows = []
    for raw in body.split(";"):
        line = raw.split("%", 1)[0].strip()
        if not line:
            continue
        rows.append([float(item) for item in line.split()])
    return rows


def write_report(data_dir: Path) -> None:
    report = data_dir.parent / "reports" / "transcribed_tables_i_iv.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    prob_sum = sum(TABLE_IV_SCENARIO_PROBABILITIES.values())
    report.write_text(
        "\n".join(
            [
                "# Transcribed Nasri 2016 Tables I-IV",
                "",
                "- Table I line capacity overrides applied to `data/lines.csv`.",
                "- Table II generator data written to `data/generators.csv` and `data/generator_cost_segments.csv`.",
                "- Table III load factors written to `data/load_factors.csv`.",
                "- Table IV scenario probabilities written to `data/scenario_probabilities.csv`.",
                f"- Scenario probability sum: {prob_sum:.6f}.",
                "- `data/load_profile.csv` generated from MATPOWER RTS base Pd/Qd scaled by Table III.",
                "",
                "Remaining gap:",
                "",
                "- Fig. 3 wind scenario trajectories are still not digitized; `data/wind_profile.csv` remains incomplete.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(report)


def resolve_relative(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


if __name__ == "__main__":
    main()

