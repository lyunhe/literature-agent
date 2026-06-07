from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def extract_pdf(pdf_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required for PDF extraction. Install with `pip install -r requirements.txt`."
        ) from exc

    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(
            {
                "page": page_number,
                "text": normalize_text(text),
                "char_count": len(text),
            }
        )

    document = {
        "pdf_path": str(pdf_path),
        "page_count": len(pages),
        "pages": pages,
        "sections": infer_sections(pages),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return document


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_sections(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    heading_re = re.compile(
        r"^(?:[IVX]+\.\s+|[0-9]+(?:\.[0-9]+)*\s+)([A-Z][A-Za-z0-9 ,:;()/\\-]+)$"
    )
    for page in pages:
        for line in page["text"].splitlines():
            clean = line.strip()
            if len(clean) > 90:
                continue
            if heading_re.match(clean) or clean in {
                "Abstract",
                "References",
                "Appendix",
                "Conclusion",
            }:
                sections.append({"title": clean, "page": page["page"]})
    return sections
