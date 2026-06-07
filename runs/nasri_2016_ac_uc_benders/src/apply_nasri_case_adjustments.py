from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_MVA = 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply paper-specific Nasri 2016 case adjustments.")
    parser.add_argument("--data-dir", default="../data")
    args = parser.parse_args()
    data_dir = resolve_relative(Path(args.data_dir))
    apply_adjustments(data_dir)


def apply_adjustments(data_dir: Path) -> None:
    generators_path = data_dir / "generators.csv"
    generators = pd.read_csv(generators_path)
    before = len(generators)
    mask_condenser = (
        (generators["bus_id"].astype(str) == "14")
        & (generators["p_max_mw"].fillna(0).astype(float) == 0.0)
    )
    generators = generators.loc[~mask_condenser].copy()
    generators["notes"] = generators["notes"].fillna("")
    generators.to_csv(generators_path, index=False)

    wind = pd.DataFrame(
        [
            {
                "wind_id": "W3",
                "bus_id": 3,
                "p_nom_mw": 2.85 * BASE_MVA,
                "profile_source": "Nasri 2016 Section IV-A / Fig. 3",
                "notes": "Installed capacity 2.85 p.u.; exact 40 scenario traces still need digitization.",
            },
            {
                "wind_id": "W14",
                "bus_id": 14,
                "p_nom_mw": 2.96 * BASE_MVA,
                "profile_source": "Nasri 2016 Section IV-A / Fig. 3",
                "notes": "Installed capacity 2.96 p.u.; exact 40 scenario traces still need digitization.",
            },
        ]
    )
    wind.to_csv(data_dir / "wind_farms.csv", index=False)

    assumptions = pd.DataFrame(
        [
            {
                "parameter": "base_mva",
                "value": BASE_MVA,
                "unit": "MW per p.u.",
                "source": "Nasri 2016 Section IV-A",
                "status": "available",
            },
            {
                "parameter": "load_shed_value",
                "value": 10000,
                "unit": "USD per p.u.",
                "source": "Nasri 2016 Section IV-A",
                "status": "available",
            },
            {
                "parameter": "case_b_voltage_min",
                "value": 0.9,
                "unit": "p.u.",
                "source": "Nasri 2016 Section IV-B",
                "status": "available",
            },
            {
                "parameter": "case_b_voltage_max",
                "value": 1.1,
                "unit": "p.u.",
                "source": "Nasri 2016 Section IV-B",
                "status": "available",
            },
            {
                "parameter": "case_c_voltage_min",
                "value": 0.5,
                "unit": "p.u.",
                "source": "Nasri 2016 Section IV-B",
                "status": "available",
            },
            {
                "parameter": "case_c_voltage_max",
                "value": 1.5,
                "unit": "p.u.",
                "source": "Nasri 2016 Section IV-B",
                "status": "available",
            },
            {
                "parameter": "benders_tolerance",
                "value": 0.3,
                "unit": "percent of objective",
                "source": "Nasri 2016 Section IV-A",
                "status": "available",
            },
        ]
    )
    assumptions.to_csv(data_dir / "paper_parameters.csv", index=False)

    report = data_dir.parent / "reports" / "nasri_case_adjustments.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    removed = before - len(generators)
    report.write_text(
        "\n".join(
            [
                "# Nasri 2016 Case Adjustments",
                "",
                f"- Removed synchronous condenser rows at node 14: {removed}",
                "- Added wind farm W3 at node 3 with 2.85 p.u. = 285 MW.",
                "- Added wind farm W14 at node 14 with 2.96 p.u. = 296 MW.",
                "- Recorded paper constants in `data/paper_parameters.csv`.",
                "",
                "Remaining data gaps:",
                "",
                "- Table I network modifications need manual transcription.",
                "- Table II generator UC/reserve/cost data need manual transcription.",
                "- Table III hourly load factors need manual transcription.",
                "- Table IV scenario probabilities need manual transcription.",
                "- Fig. 3 wind scenario traces need digitization or original data.",
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

