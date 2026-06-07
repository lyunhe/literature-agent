from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


MATRIX_RE = re.compile(r"mpc\.bus\s*=\s*\[(.*?)\];", re.DOTALL)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "sources" / "case24_ieee_rts.m"
    buses_path = root / "data" / "buses.csv"
    buses = pd.read_csv(buses_path)
    matpower_bus = parse_bus_matrix(source)

    base_pd = sum(row["pd_mw"] for row in matpower_bus.values())
    base_qd = sum(row["qd_mvar"] for row in matpower_bus.values())
    if base_pd <= 0:
        raise ValueError(f"MATPOWER source has non-positive total Pd: {source}")

    buses["base_pd_mw"] = buses["bus_id"].astype(int).map(lambda bus: matpower_bus[bus]["pd_mw"])
    buses["base_qd_mvar"] = buses["bus_id"].astype(int).map(lambda bus: matpower_bus[bus]["qd_mvar"])
    buses["pd_fraction"] = buses["base_pd_mw"] / base_pd
    buses["qd_fraction"] = buses["base_qd_mvar"] / base_qd if base_qd > 0 else 0.0
    buses["notes"] = buses["notes"].fillna("").apply(
        lambda text: text
        if "load fractions filled from MATPOWER Pd/Qd" in text
        else f"{text}; load fractions filled from MATPOWER Pd/Qd".strip("; ")
    )
    buses.to_csv(buses_path, index=False)

    report = root / "reports" / "bus_load_fraction_report.md"
    report.write_text(
        "\n".join(
            [
                "# Bus Load Fraction Reconstruction",
                "",
                f"- Source: `{source}`",
                f"- Output: `{buses_path}`",
                f"- Base total Pd: {base_pd:.6f} MW",
                f"- Base total Qd: {base_qd:.6f} Mvar",
                f"- Sum pd_fraction: {buses['pd_fraction'].sum():.12f}",
                f"- Sum qd_fraction: {buses['qd_fraction'].sum():.12f}",
                "",
                "These fractions allocate Nasri Table III hourly total load over the RTS-24 buses.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(report)


def parse_bus_matrix(path: Path) -> dict[int, dict[str, float]]:
    text = strip_matlab_comments(path.read_text(encoding="utf-8", errors="replace"))
    match = MATRIX_RE.search(text)
    if not match:
        raise ValueError(f"Could not find mpc.bus matrix in {path}")
    rows: dict[int, dict[str, float]] = {}
    for raw in match.group(1).replace(";", "\n").splitlines():
        line = raw.strip()
        if not line:
            continue
        values = [float(item) for item in line.split()]
        bus_id = int(values[0])
        rows[bus_id] = {
            "pd_mw": values[2],
            "qd_mvar": values[3],
        }
    return rows


def strip_matlab_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "%" in line:
            line = line.split("%", 1)[0]
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
