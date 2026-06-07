from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


MATRIX_RE = re.compile(r"mpc\.(bus|branch|gen|gencost)\s*=\s*\[(.*?)\];", re.DOTALL)


def import_matpower_case(case_path: str | Path, target: dict[str, Any]) -> dict[str, Any]:
    case_path = Path(case_path)
    text = strip_matlab_comments(case_path.read_text(encoding="utf-8", errors="replace"))
    matrices = {name: parse_matrix(body) for name, body in MATRIX_RE.findall(text)}
    data_dir = Path(target["run_dir"]) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    if "bus" in matrices:
        write_buses(data_dir / "buses.csv", matrices["bus"])
        written.append("buses.csv")
    if "branch" in matrices:
        write_lines(data_dir / "lines.csv", matrices["branch"])
        written.append("lines.csv")
    if "gen" in matrices:
        write_generators(data_dir / "generators.csv", matrices["gen"])
        written.append("generators.csv")
    if "gencost" in matrices:
        write_gencost(data_dir / "generator_cost_segments.csv", matrices["gencost"])
        written.append("generator_cost_segments.csv")

    manifest = {
        "source": str(case_path),
        "matrices": {name: len(rows) for name, rows in matrices.items()},
        "written": written,
        "notes": [
            "MATPOWER import fills electrical base fields only.",
            "UC-specific startup/shutdown/ramp/min-up/min-down fields may still need external data.",
        ],
    }
    out = Path(target["run_dir"]) / "artifacts" / "matpower_import_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def strip_matlab_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "%" in line:
            line = line.split("%", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def parse_matrix(body: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw in body.replace(";", "\n").splitlines():
        line = raw.strip()
        if not line:
            continue
        values = []
        for item in line.split():
            try:
                values.append(float(item))
            except ValueError:
                pass
        if values:
            rows.append(values)
    return rows


def write_rows(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def write_buses(path: Path, rows: list[list[float]]) -> None:
    out = []
    for row in rows:
        bus_id = int(row[0])
        base_kv = row[9] if len(row) > 9 else ""
        area = int(row[6]) if len(row) > 6 else ""
        zone = int(row[10]) if len(row) > 10 else ""
        out.append([bus_id, base_kv, area, zone, "", "imported from MATPOWER bus matrix"])
    write_rows(path, ["bus_id", "base_kv", "area", "zone", "pd_fraction", "notes"], out)


def write_lines(path: Path, rows: list[list[float]]) -> None:
    out = []
    for idx, row in enumerate(rows, start=1):
        from_bus = int(row[0])
        to_bus = int(row[1])
        x_pu = row[3] if len(row) > 3 else ""
        rate_mw = row[5] if len(row) > 5 else ""
        out.append([idx, from_bus, to_bus, x_pu, rate_mw, "compute_lsf_from_case", "imported from MATPOWER branch matrix"])
    write_rows(
        path,
        ["line_id", "from_bus", "to_bus", "x_pu", "rate_mw", "lsf_row_source", "notes"],
        out,
    )


def write_generators(path: Path, rows: list[list[float]]) -> None:
    out = []
    for idx, row in enumerate(rows, start=1):
        bus_id = int(row[0])
        p_max = row[8] if len(row) > 8 else ""
        p_min = row[9] if len(row) > 9 else ""
        ramp = row[16] if len(row) > 16 else ""
        out.append(
            [
                idx,
                bus_id,
                p_min,
                p_max,
                "",
                "",
                "",
                ramp,
                ramp,
                "",
                "",
                "",
                row[1] if len(row) > 1 else "",
                "imported from MATPOWER gen matrix; UC fields incomplete",
            ]
        )
    write_rows(
        path,
        [
            "gen_id",
            "bus_id",
            "p_min_mw",
            "p_max_mw",
            "startup_cost",
            "shutdown_cost",
            "fixed_cost",
            "ramp_up_mw",
            "ramp_down_mw",
            "min_up_h",
            "min_down_h",
            "initial_status",
            "initial_p_mw",
            "notes",
        ],
        out,
    )


def write_gencost(path: Path, rows: list[list[float]]) -> None:
    out = []
    for gen_id, row in enumerate(rows, start=1):
        startup = row[1] if len(row) > 1 else ""
        shutdown = row[2] if len(row) > 2 else ""
        ncost = int(row[3]) if len(row) > 3 else 0
        coeffs = row[4 : 4 + ncost]
        if coeffs:
            out.append([gen_id, 1, "", " ".join(str(v) for v in coeffs), f"startup={startup}; shutdown={shutdown}; MATPOWER gencost"])
        else:
            out.append([gen_id, 1, "", "", f"startup={startup}; shutdown={shutdown}; MATPOWER gencost"])
    write_rows(path, ["gen_id", "segment_id", "p_max_segment_mw", "marginal_cost", "notes"], out)

