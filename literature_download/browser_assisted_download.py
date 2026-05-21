from __future__ import annotations

import argparse
import json
import shutil
import time
import webbrowser
from pathlib import Path
from typing import Any


def _pdfs_newer_than(directory: Path, started_at: float) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [path for path in directory.glob("*.pdf") if path.stat().st_mtime >= started_at],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_queue(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("items", [])
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Open manual PDF download pages and copy newly downloaded PDFs into the library.")
    parser.add_argument("--queue", type=Path, required=True, help="manual_download_queue.json from the download workflow")
    parser.add_argument("--downloads-dir", type=Path, default=Path.home() / "Downloads", help="Browser download directory")
    parser.add_argument("--library-pdf-dir", type=Path, default=Path("library/pdfs"), help="Where confirmed PDFs should be copied")
    parser.add_argument("--open-limit", type=int, default=20, help="Maximum queued papers to open")
    parser.add_argument("--no-open", action="store_true", help="Only print URLs; do not open the browser")
    args = parser.parse_args()

    queue = load_queue(args.queue)
    args.library_pdf_dir.mkdir(parents=True, exist_ok=True)
    confirmations = []
    for index, item in enumerate(queue[: args.open_limit], start=1):
        title = item.get("title") or item.get("paper_key") or f"paper-{index}"
        candidates = item.get("manual_candidates") or item.get("resolution", {}).get("candidates", [])
        urls = [c.get("url") for c in candidates if c.get("url")]
        if not urls:
            continue
        print(f"\n[{index}] {title}")
        for url_index, url in enumerate(urls, start=1):
            print(f"  {url_index}. {url}")
        started_at = time.time()
        if not args.no_open:
            webbrowser.open(urls[0])
        input("请在浏览器中用高校账号登录并下载 PDF。下载完成后按 Enter；跳过也直接按 Enter。")
        new_pdfs = _pdfs_newer_than(args.downloads_dir, started_at)
        copied = ""
        if new_pdfs:
            source = new_pdfs[0]
            target = args.library_pdf_dir / source.name
            shutil.copy2(source, target)
            copied = str(target.resolve())
            print(f"已复制：{copied}")
        else:
            print("未检测到新的 PDF 下载。")
        confirmations.append({
            "title": title,
            "opened_url": urls[0],
            "copied_pdf": copied,
        })

    out_path = args.queue.with_name("manual_download_confirmations.json")
    out_path.write_text(json.dumps(confirmations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n确认记录：{out_path}")


if __name__ == "__main__":
    main()
