from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import fitz


DEFAULT_OUTPUT_DIR = Path("pdf_formula_regions_output_v2")
FORMULA_NUMBER_RE = re.compile(r"(?P<number>[\(（]\s*(?:\d{1,3}(?:\.\d{1,3})*|[A-Z]\.\d+)[a-zA-Z]?\s*[\)）])\s*$")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("_") or "paper"


def bbox_to_tuple(rect: fitz.Rect) -> tuple[float, float, float, float]:
    return (round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2))


def normalize_formula_number(token: str) -> str:
    token = token.strip().replace("（", "(").replace("）", ")")
    return token.strip("() ")


def looks_like_year(number: str) -> bool:
    return number.isdigit() and 1900 <= int(number) <= 2099


def line_text(line: dict[str, Any]) -> str:
    parts: list[str] = []
    for span in line.get("spans", []):
        parts.append(span.get("text", ""))
    return "".join(parts).strip()


def candidate_rect(page: fitz.Page, block: dict[str, Any], line: dict[str, Any]) -> fitz.Rect:
    page_rect = fitz.Rect(page.rect)
    block_rect = fitz.Rect(block.get("bbox", line.get("bbox")))
    line_rect = fitz.Rect(line.get("bbox", block.get("bbox")))

    x0 = max(page_rect.x0 + 8, min(block_rect.x0, page_rect.x0 + 24))
    x1 = min(page_rect.x1 - 8, max(block_rect.x1, page_rect.x1 - 24))
    y0 = max(page_rect.y0 + 8, line_rect.y0 - 34)
    y1 = min(page_rect.y1 - 8, line_rect.y1 + 18)
    return fitz.Rect(x0, y0, x1, y1)


def render_crop(page: fitz.Page, rect: fitz.Rect, out_path: Path, dpi: int) -> None:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
    pix.save(out_path)


def extract_from_pdf(pdf_path: Path, output_root: Path, dpi: int = 220, overwrite: bool = False) -> Path:
    paper_dir = ensure_dir(output_root / safe_stem(pdf_path.stem))
    crop_dir = ensure_dir(paper_dir / "formula_crops")
    manifest_path = paper_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return manifest_path

    manifest: dict[str, Any] = {
        "pdf_path": str(pdf_path),
        "formula_crops_dir": str(crop_dir),
        "formulas": [],
        "pages": [],
    }
    seen: set[tuple[int, str, tuple[float, float, float, float]]] = set()

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text_dict = page.get_text("dict")
            page_count = 0
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    text = line_text(line)
                    match = FORMULA_NUMBER_RE.search(text)
                    if not match:
                        continue
                    number = normalize_formula_number(match.group("number"))
                    if looks_like_year(number):
                        continue
                    rect = candidate_rect(page, block, line)
                    bbox = bbox_to_tuple(rect)
                    key = (page_index, number, bbox)
                    if key in seen:
                        continue
                    seen.add(key)

                    formula_id = f"eq_{len(manifest['formulas']) + 1:03d}"
                    png_path = crop_dir / f"{formula_id}.png"
                    render_crop(page, rect, png_path, dpi)
                    manifest["formulas"].append(
                        {
                            "id": formula_id,
                            "page": page_index,
                            "formula_number": number,
                            "bbox": bbox,
                            "png_path": str(png_path),
                            "text_hint": text,
                            "detection_method": "text_line_number_heuristic",
                        }
                    )
                    page_count += 1
            manifest["pages"].append({"page": page_index, "formula_count": page_count})

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def collect_pdf_files(single_pdf: Path | None, pdf_dir: Path | None) -> list[Path]:
    files: list[Path] = []
    if single_pdf:
        files.append(single_pdf)
    if pdf_dir:
        files.extend(sorted(pdf_dir.glob("*.pdf")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen and path.exists():
            unique.append(path)
            seen.add(resolved)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract numbered formula regions from PDF files.")
    parser.add_argument("--pdf", type=Path, default=None, help="A single PDF file to process.")
    parser.add_argument("--pdf-dir", type=Path, default=None, help="A directory containing PDF files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_files = collect_pdf_files(args.pdf, args.pdf_dir)
    if not pdf_files:
        raise SystemExit("No PDF files found. Use --pdf or --pdf-dir.")

    output_root = ensure_dir(args.output_dir)
    summary = []
    for pdf_path in pdf_files:
        manifest_path = extract_from_pdf(pdf_path, output_root, dpi=args.dpi, overwrite=args.overwrite)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary.append(
            {
                "pdf": str(pdf_path),
                "manifest": str(manifest_path),
                "formula_count": len(payload.get("formulas", [])),
            }
        )
        print(f"Done: {pdf_path}")
        print(f"Manifest: {manifest_path}")

    (output_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
