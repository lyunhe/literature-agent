from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("pdf_formula_regions_output_v2")
DEFAULT_OUTPUT_DIR = Path("formula_ocr_pix2tex_output")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("_") or "paper"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_manifests(input_dir: Path, manifest_args: list[Path] | None) -> list[Path]:
    if manifest_args:
        return [path for path in manifest_args if path.exists()]
    return sorted(input_dir.glob("*/manifest.json"))


def text_hint_to_latex(text: str, formula_number: str) -> str:
    text = (text or "").strip()
    if formula_number:
        text = re.sub(rf"[\(（]\s*{re.escape(formula_number)}\s*[\)）]\s*$", "", text).strip()
    text = text.replace("−", "-").replace("×", r"\times ")
    return text or r"\text{OCR unavailable; see formula crop.}"


def try_pix2tex() -> Any | None:
    try:
        from pix2tex.cli import LatexOCR

        return LatexOCR()
    except Exception:
        return None


def recognize_formula(model: Any | None, png_path: Path, text_hint: str, formula_number: str) -> tuple[str, str]:
    if model is not None and png_path.exists():
        try:
            from PIL import Image

            latex = str(model(Image.open(png_path))).strip()
            if latex:
                return latex, "pix2tex"
        except Exception:
            pass
    return text_hint_to_latex(text_hint, formula_number), "text_hint_fallback"


def process_manifest(manifest_path: Path, output_root: Path, model: Any | None, overwrite: bool = False) -> Path:
    manifest = load_json(manifest_path)
    paper_name = safe_stem(manifest_path.parent.name)
    paper_dir = ensure_dir(output_root / paper_name)
    json_path = paper_dir / "formula_ocr.json"
    md_path = paper_dir / "formulas.md"
    if json_path.exists() and md_path.exists() and not overwrite:
        return json_path

    rows = []
    md_lines = [f"# {paper_name}", ""]
    for item in manifest.get("formulas", []):
        png_path = Path(item.get("png_path", ""))
        formula_number = str(item.get("formula_number", "")).strip()
        latex, engine = recognize_formula(model, png_path, str(item.get("text_hint", "")), formula_number)
        row = {
            "id": item.get("id"),
            "page": item.get("page"),
            "formula_number": formula_number,
            "png_path": str(png_path),
            "latex": latex,
            "ocr_engine": engine,
        }
        rows.append(row)
        tag = f" \\tag{{{formula_number}}}" if formula_number else ""
        md_lines.extend([f"## {row['id']} page {row['page']}", "", f"![{row['id']}]({png_path.as_posix()})", "", f"$$\n{latex}{tag}\n$$", ""])

    payload = {
        "source_manifest": str(manifest_path),
        "formula_count": len(rows),
        "formulas": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate formula LaTeX/Markdown from extracted formula crops.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-pix2tex", action="store_true", help="Use text-layer fallback only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = collect_manifests(args.input_dir, args.manifest)
    if not manifests:
        raise SystemExit("No manifest.json files found.")

    output_root = ensure_dir(args.output_dir)
    model = None if args.no_pix2tex else try_pix2tex()
    summary = []
    for manifest_path in manifests:
        json_path = process_manifest(manifest_path, output_root, model, overwrite=args.overwrite)
        payload = load_json(json_path)
        summary.append(
            {
                "manifest": str(manifest_path),
                "formula_ocr": str(json_path),
                "formula_count": payload.get("formula_count", 0),
            }
        )
        print(f"Done: {manifest_path}")
        print(f"OCR JSON: {json_path}")

    (output_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
