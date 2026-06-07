from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_TABLES = {
    "buses": "buses.csv",
    "lines": "lines.csv",
    "generators": "generators.csv",
    "generator_cost_segments": "generator_cost_segments.csv",
    "load_profile": "load_profile.csv",
    "wind_farms": "wind_farms.csv",
    "wind_profile": "wind_profile.csv",
    "scenario_probabilities": "scenario_probabilities.csv",
    "load_factors": "load_factors.csv",
}

OPTIONAL_TABLES = {
    "paper_parameters": "paper_parameters.csv",
    "reserves": "reserves.csv",
    "uncertainty_bounds": "uncertainty_bounds.csv",
}


@dataclass
class CaseData:
    data_dir: Path
    tables: dict[str, pd.DataFrame]
    base_mva: float = 100.0

    def table(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"Missing table: {name}")
        return self.tables[name]

    def summary(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "base_mva": self.base_mva,
            "tables": {
                name: {
                    "rows": int(len(df)),
                    "columns": list(df.columns),
                }
                for name, df in self.tables.items()
            },
        }

    def missing_or_empty(self) -> list[str]:
        return [name for name in REQUIRED_TABLES if self.tables.get(name, pd.DataFrame()).empty]


def load_case_data(data_dir: Path, *, strict: bool = False) -> CaseData:
    data_dir = Path(data_dir)
    tables: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for name, filename in REQUIRED_TABLES.items():
        path = data_dir / filename
        if not path.exists():
            missing.append(filename)
            tables[name] = pd.DataFrame()
            continue
        tables[name] = pd.read_csv(path)
    for name, filename in OPTIONAL_TABLES.items():
        path = data_dir / filename
        tables[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if strict:
        empty = [name for name, df in tables.items() if df.empty]
        if missing or empty:
            raise ValueError(f"Case data incomplete. Missing={missing}; empty={empty}")
    return CaseData(data_dir=data_dir, tables=tables)


def write_summary(case: CaseData, out_path: Path) -> None:
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(case.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
