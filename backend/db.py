"""
SQLite-backed local literature library.
Stores paper metadata; PDFs are saved as files under library/pdfs/.
"""
from __future__ import annotations

import sqlite3, os, json
from .paths import (
    DB_PATH,
    LIBRARY_DIR,
    LIBRARY_PDF_DIR,
    display_path,
    ensure_library_dirs,
    normalize_library_path,
    pdf_candidates,
)


def init_db():
    """Create library directory and papers table if they don't exist."""
    ensure_library_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT    UNIQUE,
            authors     TEXT,
            abstract    TEXT,
            arxiv_id    TEXT,
            doi         TEXT,
            ieee_id     TEXT,
            source      TEXT    DEFAULT 'unknown',
            pdf_path    TEXT,
            year        INTEGER,
            date_added  TEXT    DEFAULT (datetime('now', 'localtime'))
        )
    """)
    # Full-text search on title/abstract
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
            title, abstract, authors, content='papers', content_rowid='id'
        )
    """)
    conn.commit()
    conn.close()


def _connect():
    return sqlite3.connect(str(DB_PATH))


def _format_row(row: tuple, cols: list) -> dict:
    item = dict(zip(cols, row))
    if item.get("pdf_path"):
        item["pdf_path"] = normalize_library_path(item["pdf_path"])
        item["pdf_abs_path"] = display_path(item["pdf_path"])
    return item


def add_paper(paper: dict) -> str:
    """
    Insert or update a paper record.
    Returns a summary string.
    """
    conn = _connect()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO papers
                (title, authors, abstract, arxiv_id, doi, ieee_id, source, pdf_path, year)
            VALUES (:title, :authors, :abstract, :arxiv_id, :doi, :ieee_id, :source, :pdf_path, :year)
        """, {
            "title":    paper.get("title", ""),
            "authors":  paper.get("authors", ""),
            "abstract": paper.get("abstract", ""),
            "arxiv_id": paper.get("arxiv_id"),
            "doi":      paper.get("doi"),
            "ieee_id":  paper.get("ieee_id"),
            "source":   paper.get("source", "unknown"),
            "pdf_path": normalize_library_path(paper.get("pdf_path")),
            "year":     paper.get("year"),
        })
        conn.commit()
        return f"Paper saved: {paper.get('title', '')}"
    except sqlite3.Error as e:
        return f"Database error: {e}"
    finally:
        conn.close()


def list_papers(limit: int = 50) -> list[dict]:
    """Return the most recently added papers."""
    conn = _connect()
    cols = ["id","title","authors","abstract","arxiv_id","doi","ieee_id","source","pdf_path","year","date_added"]
    rows = conn.execute(
        f"SELECT {','.join(cols)} FROM papers ORDER BY date_added DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [_format_row(r, cols) for r in rows]


def search_papers(keyword: str) -> list[dict]:
    """
    Full-text search on title, abstract, and authors.
    Falls back to LIKE if FTS is unavailable.
    """
    conn = _connect()
    try:
        # Try FTS5 first
        rows = conn.execute("""
            SELECT p.id,p.title,p.authors,p.abstract,p.arxiv_id,p.doi,
                   p.ieee_id,p.source,p.pdf_path,p.year,p.date_added
            FROM papers p
            JOIN papers_fts f ON p.id = f.rowid
            WHERE papers_fts MATCH ?
            ORDER BY rank
        """, (keyword,)).fetchall()
    except sqlite3.Error:
        # Fallback to LIKE
        like = f"%{keyword}%"
        rows = conn.execute("""
            SELECT id,title,authors,abstract,arxiv_id,doi,ieee_id,source,pdf_path,year,date_added
            FROM papers
            WHERE title LIKE ? OR abstract LIKE ? OR authors LIKE ?
            ORDER BY date_added DESC
        """, (like, like, like)).fetchall()
    conn.close()
    cols = ["id","title","authors","abstract","arxiv_id","doi","ieee_id","source","pdf_path","year","date_added"]
    return [_format_row(r, cols) for r in rows]


def get_paper(identifier: str, source: str = "arxiv") -> dict | None:
    """Retrieve a single paper by arxiv_id, doi, or ieee_id."""
    conn = _connect()
    col = {"arxiv": "arxiv_id", "doi": "doi", "ieee": "ieee_id"}.get(source, "arxiv_id")
    cols = ["id","title","authors","abstract","arxiv_id","doi","ieee_id","source","pdf_path","year","date_added"]
    row = conn.execute(
        f"SELECT {','.join(cols)} FROM papers WHERE {col}=? LIMIT 1",
        (identifier,)
    ).fetchone()
    conn.close()
    return _format_row(row, cols) if row else None


def update_pdf_path(paper_id: int, pdf_path: str) -> str:
    pdf_path = normalize_library_path(pdf_path)
    conn = _connect()
    conn.execute("UPDATE papers SET pdf_path=? WHERE id=?", (pdf_path, paper_id))
    conn.commit()
    conn.close()
    return pdf_path


def normalize_pdf_paths() -> dict:
    """Rewrite stored PDF paths into portable library-relative form."""
    conn = _connect()
    rows = conn.execute("SELECT id, arxiv_id, pdf_path FROM papers").fetchall()
    changed = 0
    filled = 0
    for paper_id, arxiv_id, pdf_path in rows:
        normalized = normalize_library_path(pdf_path)
        if not normalized:
            for candidate in pdf_candidates(arxiv_id=arxiv_id):
                if candidate.exists():
                    normalized = normalize_library_path(candidate)
                    filled += 1
                    break
        if normalized and normalized != pdf_path:
            conn.execute("UPDATE papers SET pdf_path=? WHERE id=?", (normalized, paper_id))
            changed += 1
    conn.commit()
    conn.close()
    return {"checked": len(rows), "changed": changed, "filled_missing": filled}
