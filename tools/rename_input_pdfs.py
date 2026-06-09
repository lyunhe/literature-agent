"""Rename input PDFs to article titles; extract chapter from Springer book."""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "input_pdfs" / "高空风能强化学习"

# Printed book pages 706-714 -> PDF pages 719-727 (1-based)
BOOK_CHAPTER_PDF_PAGES = range(719, 728)
BOOK_CHAPTER_TITLE = (
    "Energy Optimization of Airborne Wind Energy System via Deep Reinforcement Learning"
)

# filename -> canonical English title (explicit overrides)
TITLE_OVERRIDES: dict[str, str] = {
    "978-981-19-6203-5.pdf": BOOK_CHAPTER_TITLE,
    "Airborne_Wind_Energy_Resource_Analysis.pdf": "Airborne Wind Energy Resource Analysis",
    "Autonomous_Take-Off_and_Flight_of_a_Tethered_Aircraft_for_Airborne_Wind_Energy.pdf": (
        "Autonomous Take-Off and Flight of a Tethered Aircraft for Airborne Wind Energy"
    ),
    "Basile_2025_EPL_152_43001.pdf": (
        "Harvesting energy from turbulent winds with reinforcement learning (EPL)"
    ),
    "Control_of_a_Rigid_Wing_Pumping_Airborne_Wind_Energy_System_in_all_Operational_Phases.pdf": (
        "Control of a Rigid Wing Pumping Airborne Wind Energy System in all Operational Phases"
    ),
    "Design_of_a_small-scale_prototype_for_research_in_airborne_wind_energy.pdf": (
        "Design of a small-scale prototype for research in airborne wind energy"
    ),
    "Flight_control_of_tethered_kites_in_autonomous_pumping_cycles_for_airborne_wind_energy.pdf": (
        "Flight control of tethered kites in autonomous pumping cycles for airborne wind energy"
    ),
    "Harvesting_energy_from_turbulent_winds_with_Reinforcement_Learning.pdf": (
        "Harvesting energy from turbulent winds with reinforcement learning (preprint)"
    ),
    "Optimizing_Airborne_Wind_Energy_with_Reinforcement_Learning.pdf": (
        "Optimizing Airborne Wind Energy with Reinforcement Learning (preprint)"
    ),
    "Vertical_Airborne_Wind_Energy_Farms_with_High_Power_Density_per_Ground_Area_based_on_Multi.pdf": (
        "Vertical Airborne Wind Energy Farms with High Power Density per Ground Area based on Multi-Aircraft Systems"
    ),
    "Waypoint_Optimization_Using_Bayesian_Optimization_A_Case_Study_in_Airborne_Wind_Energy_Sys.pdf": (
        "Waypoint Optimization Using Bayesian Optimization: A Case Study in Airborne Wind Energy Systems"
    ),
    "s10189-022-00259-2.pdf": (
        "Optimizing Airborne Wind Energy with Reinforcement Learning (EPJ)"
    ),
    "selje-et-al-2024-waypoint-optimization-using-reinforcement-learning-for-maximizing-energy-harvesting-for-high-altitude.pdf": (
        "Waypoint Optimization Using Reinforcement Learning for Maximizing Energy Harvesting for High Altitude Airborne Wind Energy Systems"
    ),
    "selje-et-al-2026-tension-control-using-reinforcement-learning-for-airborne-wind-energy-systems.pdf": (
        "Tension Control using Reinforcement Learning for Airborne Wind Energy Systems"
    ),
}


def sanitize_filename(title: str, max_len: int = 180) -> str:
    title = unicodedata.normalize("NFKC", title)
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r'[<>:"/\\|?*]', "", title)
    title = title.rstrip(". ")
    if len(title) > max_len:
        title = title[: max_len - 3].rstrip() + "..."
    return f"{title}.pdf"


def extract_chapter(book_path: Path, out_path: Path) -> None:
    reader = PdfReader(str(book_path))
    writer = PdfWriter()
    for page_num in BOOK_CHAPTER_PDF_PAGES:
        writer.add_page(reader.pages[page_num - 1])
    with out_path.open("wb") as f:
        writer.write(f)


def main() -> None:
    if not FOLDER.is_dir():
        raise SystemExit(f"Folder not found: {FOLDER}")

    book_name = "978-981-19-6203-5.pdf"
    book_path = FOLDER / book_name
    chapter_filename = sanitize_filename(BOOK_CHAPTER_TITLE)
    chapter_path = FOLDER / chapter_filename

    if book_path.exists():
        extract_chapter(book_path, chapter_path)
        book_path.unlink()
        print(f"EXTRACTED+DELETED: {book_name} -> {chapter_filename}")
    elif chapter_path.exists():
        print(f"SKIP extract (book gone, chapter exists): {chapter_filename}")
    else:
        print(f"WARN: missing {book_name} and chapter PDF")

    # Build rename plan (skip book if still present)
    plans: list[tuple[Path, Path, str]] = []
    used_names: set[str] = set()

    for src_name, title in sorted(TITLE_OVERRIDES.items()):
        if src_name == book_name:
            src = chapter_path
            if not src.exists():
                continue
        else:
            src = FOLDER / src_name
            if not src.exists():
                print(f"SKIP missing: {src_name}")
                continue

        dst_name = sanitize_filename(title)
        base, ext = os.path.splitext(dst_name)
        candidate = dst_name
        n = 2
        while candidate in used_names or (FOLDER / candidate).exists() and (FOLDER / candidate) != src:
            candidate = f"{base} ({n}){ext}"
            n += 1
        used_names.add(candidate)
        dst = FOLDER / candidate
        if src.resolve() == dst.resolve():
            print(f"OK (already named): {candidate}")
            plans.append((src, dst, title))
            continue
        plans.append((src, dst, title))

    # Two-phase rename to avoid collisions
    temp_moves: list[tuple[Path, Path, str, Path]] = []
    for i, (src, dst, title) in enumerate(plans):
        if src.resolve() == dst.resolve():
            continue
        tmp = FOLDER / f"__renaming_{i}.pdf"
        src.rename(tmp)
        temp_moves.append((tmp, dst, title, src))

    for tmp, dst, title, _orig in temp_moves:
        if dst.exists():
            dst.unlink()
        tmp.rename(dst)
        print(f"RENAMED: {dst.name}")

    print("\n=== FINAL TITLES ===")
    for _src, dst, title in plans:
        if dst.exists():
            print(title)


if __name__ == "__main__":
    main()
