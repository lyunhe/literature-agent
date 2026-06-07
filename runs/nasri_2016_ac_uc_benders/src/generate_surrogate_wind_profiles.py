from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_MVA = 100.0
TARGET_CAPACITY_FACTOR = 0.2960
SEED = 2016
SOURCE_LABEL = "SYNTHETIC_CALIBRATED_TO_NASRI_2016"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate surrogate wind profiles for solver development.")
    parser.add_argument("--data-dir", default="../data")
    args = parser.parse_args()
    data_dir = resolve_relative(Path(args.data_dir))
    generate_surrogate(data_dir)


def generate_surrogate(data_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    wind_farms = pd.read_csv(data_dir / "wind_farms.csv")
    probabilities = pd.read_csv(data_dir / "scenario_probabilities.csv")
    hours = np.arange(1, 25)

    # Smooth daily shape with higher production overnight/morning, then normalized.
    shape = 0.85 + 0.25 * np.sin((hours - 4) / 24 * 2 * np.pi) + 0.10 * np.cos((hours - 12) / 24 * 4 * np.pi)
    shape = np.clip(shape, 0.25, None)
    shape = shape / shape.mean()

    rows = []
    for _, farm in wind_farms.iterrows():
        p_nom_mw = float(farm["p_nom_mw"])
        for _, prob_row in probabilities.iterrows():
            scenario = int(prob_row["scenario_id"])
            probability = float(prob_row["probability"])
            scenario_scale = np.clip(rng.normal(loc=1.0, scale=0.28), 0.05, 1.65)
            hourly_noise = np.clip(rng.normal(loc=1.0, scale=0.10, size=24), 0.60, 1.35)
            production = p_nom_mw * TARGET_CAPACITY_FACTOR * shape * scenario_scale * hourly_noise
            production = np.clip(production, 0, p_nom_mw)
            for hour, value in zip(hours, production):
                rows.append(
                    {
                        "hour": int(hour),
                        "scenario_id": scenario,
                        "probability": probability,
                        "wind_id": farm["wind_id"],
                        "forecast_mw": "",
                        "production_mw": float(value),
                        "production_pu": float(value / BASE_MVA),
                        "lower_bound_mw": "",
                        "upper_bound_mw": "",
                        "source": SOURCE_LABEL,
                        "notes": (
                            "Synthetic wind scenario generated for reproduction workflow; "
                            "calibrated to Nasri 2016 reported 29.60% expected wind production."
                        ),
                    }
                )

    out = pd.DataFrame(rows)
    weighted_average_mw = (
        out.groupby(["scenario_id", "probability", "hour"])["production_mw"].sum().reset_index()
    )
    expected_mean_mw = float((weighted_average_mw["production_mw"] * weighted_average_mw["probability"]).sum() / 24)
    target_mean_mw = float(wind_farms["p_nom_mw"].sum() * TARGET_CAPACITY_FACTOR)
    scale = target_mean_mw / expected_mean_mw if expected_mean_mw else 1.0
    out["production_mw"] = out["production_mw"] * scale
    capacity = wind_farms.set_index("wind_id")["p_nom_mw"].to_dict()
    out["production_mw"] = [
        min(value, float(capacity[wind_id]))
        for value, wind_id in zip(out["production_mw"], out["wind_id"])
    ]
    out["production_pu"] = out["production_mw"] / BASE_MVA
    out.to_csv(data_dir / "wind_profile.csv", index=False)

    scenario_totals = (
        out.groupby(["scenario_id", "probability"], as_index=False)["production_mw"]
        .sum()
        .rename(columns={"production_mw": "scenario_total_mwh"})
    )
    scenario_totals["scenario_average_mw"] = scenario_totals["scenario_total_mwh"] / 24.0
    scenario_totals["scenario_capacity_factor"] = scenario_totals["scenario_total_mwh"] / (
        float(wind_farms["p_nom_mw"].sum()) * 24.0
    )
    scenario_totals.to_csv(data_dir / "wind_scenario_statistics.csv", index=False)

    expected_total_mwh = float((scenario_totals["scenario_total_mwh"] * scenario_totals["probability"]).sum())
    expected_average_mw = expected_total_mwh / 24.0
    expected_average_pu = expected_average_mw / BASE_MVA
    expected_total_pu_hours = expected_total_mwh / BASE_MVA
    achieved_capacity_factor = expected_total_mwh / (float(wind_farms["p_nom_mw"].sum()) * 24.0)

    report = data_dir.parent / "reports" / "surrogate_wind_profile_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Surrogate Wind Profile Report",
                "",
                "Generated `data/wind_profile.csv` as a synthetic wind scenario dataset for the reproduction workflow.",
                "",
                f"- Seed: {SEED}",
                f"- Source label: `{SOURCE_LABEL}`",
                f"- Target capacity factor: {TARGET_CAPACITY_FACTOR}",
                f"- Achieved capacity factor: {achieved_capacity_factor:.6f}",
                f"- Expected average wind production: {expected_average_mw:.6f} MW",
                f"- Expected average wind production on 100 MVA base: {expected_average_pu:.6f} p.u.",
                f"- Expected total wind production over 24 hours: {expected_total_pu_hours:.6f} p.u.-h",
                f"- Wind farms: {len(wind_farms)}",
                f"- Scenarios: {len(probabilities)}",
                f"- Rows: {len(out)}",
                "",
                "This dataset is not digitized from Nasri 2016 Fig. 3. It is a documented synthetic substitute accepted for continuing the implementation.",
                "Results produced with this file should be described as synthetic-data reproduction, not exact paper-result reproduction.",
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
