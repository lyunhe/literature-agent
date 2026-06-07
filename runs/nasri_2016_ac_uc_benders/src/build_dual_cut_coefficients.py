from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build first-pass Benders cut coefficients from constrained Ipopt multipliers.")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--batch", default="../results/ac_nlp_subproblem/case_b_ac_uc_benders_cyipopt_constrained_nlp_batch_worst-reactive_3.csv")
    parser.add_argument("--out", default="../results/benders_cuts/case_b_dual_cut_coefficients.csv")
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    data_dir = resolve_relative(src_dir, Path(args.data_dir))
    batch_path = resolve_relative(src_dir, Path(args.batch))
    out_path = resolve_relative(src_dir, Path(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generators = pd.read_csv(data_dir / "generators.csv")
    wind_farms = pd.read_csv(data_dir / "wind_farms.csv")
    gen_bus = generators.set_index("gen_id")["bus_id"].astype(int).to_dict()
    wind_bus = wind_farms.set_index("wind_id")["bus_id"].astype(int).to_dict()
    batch = pd.read_csv(batch_path)

    rows = []
    for case in batch.itertuples(index=False):
        summary = json.loads(Path(case.summary).read_text(encoding="utf-8"))
        multipliers_path = Path(summary["metadata"]["outputs"]["multipliers"])
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
        p_lambda = (
            multipliers[multipliers["constraint_type"] == "p_balance_eq"]
            .set_index("component_id")["multiplier"]
            .astype(float)
            .to_dict()
        )
        q_lambda = (
            multipliers[multipliers["constraint_type"] == "q_balance_eq"]
            .set_index("component_id")["multiplier"]
            .astype(float)
            .to_dict()
        )
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
                    "notes": "First-pass coefficient from Ipopt P-balance multiplier; sign convention follows p_gen injection term.",
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
                    "notes": "First-pass coefficient from Ipopt P-balance multiplier; sign convention follows wind injection term.",
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
                    "notes": "Diagnostic coefficient; not yet linked to a master reactive decision variable.",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    report = out_path.with_suffix(".md")
    report.write_text(render_report(out, batch_path), encoding="utf-8")
    print(out_path)
    print(report)


def render_report(coefficients: pd.DataFrame, batch_path: Path) -> str:
    grouped = coefficients.groupby("coefficient_family")["coefficient"].agg(["count", "min", "max", "mean"]).reset_index()
    lines = [
        "# Dual-Based Cut Coefficient Draft",
        "",
        f"- Source batch: `{batch_path}`",
        f"- Rows: {len(coefficients)}",
        "",
        "These coefficients are extracted from constrained Ipopt multipliers. When available, fixed_master_* equality multipliers are used as the direct Benders coupling source; older files fall back to P-balance multipliers.",
        "",
        "| Family | Count | Min | Max | Mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in grouped.itertuples(index=False):
        lines.append(f"| {row.coefficient_family} | {int(row.count)} | {row.min:.6e} | {row.max:.6e} | {row.mean:.6e} |")
    lines.append("")
    return "\n".join(lines)


def resolve_relative(base: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


if __name__ == "__main__":
    main()
