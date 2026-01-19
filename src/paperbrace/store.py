from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import typer

from .retriever import PageForIndex, Retriever

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

logger = logging.getLogger("paperbrace")

_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","so",
    "which","what","who","whom","whose","where","when","why","how",
    "is","are","was","were","be","been","being",
    "do","does","did",
    "in","on","at","to","from","by","for","of","with","without",
    "this","that","these","those",
    "paper","papers","article","articles","study","studies",
    "discuss","discusses","discussed","mention","mentions","mentioned",
    "report","reports","reported","show","shows","shown",
}

_NEGATIONS = {"no", "not", "without", "never"}


def nl_to_fts_query(text: str) -> str:
    """
    Convert a natural-language question into a friendlier SQLite FTS5 query.

    Strategy:
      - Extract word tokens (letters+digits+underscore).
      - Drop very short tokens (<=2 chars).
      - Drop common stopwords (but keep negations).
      - Join remaining tokens with OR to favor recall.
      - If nothing remains, fall back to the original text.

    This avoids the FTS default where spaces behave like AND, which is too strict
    for natural-language questions.
    """
    toks = re.findall(r"\w+", (text or "").lower())
    kept: list[str] = []
    for t in toks:
        if len(t) <= 2:
            continue
        if t in _STOPWORDS and t not in _NEGATIONS:
            continue
        kept.append(t)

    if not kept:
        return text  # last resort

    # De-dupe while keeping order
    deduped = list(dict.fromkeys(kept))
    return " OR ".join(deduped)


def iter_pdfs(pdf_dir: Path) -> Iterable[Path]:
    """
    Yield all PDFs under a directory recursively.

    Designed for Zotero-style libraries where attachments can be nested deeply.

    Args:
        pdf_dir: Root directory containing PDFs.

    Yields:
        Path objects for each *.pdf found under pdf_dir.
    """
    yield from pdf_dir.rglob("*.pdf")


def index_pdfs(conn: sqlite3.Connection, pdf_dir: Path, commit_every: int = 200) -> int:
    """
    Scan a PDF directory and upsert file metadata into the `papers` table.

    - Recursively finds *.pdf under pdf_dir.
    - Upserts (path, mtime, size) into `papers`.
    - Does not extract text; it only records file-level metadata.
    - Commits periodically and once at the end.

    Args:
        conn: Open SQLite connection.
        pdf_dir: Directory to scan for PDFs (recursively).
        commit_every: Commit after every N PDFs processed (0 disables intermediate commits).

    Returns:
        Number of PDFs indexed.
    """
    pdf_dir = pdf_dir.expanduser().resolve()
    n = 0
    t0 = time.time()

    for p in iter_pdfs(pdf_dir):
        try:
            st = p.stat()
        except FileNotFoundError:
            logger.warning("File disappeared: %s", p)
            continue

        conn.execute(
            """
            INSERT INTO papers (path, mtime, size)
            VALUES (?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                mtime=excluded.mtime,
                size=excluded.size,
                updated_at=datetime('now')
            """,
            (str(p.resolve()), float(st.st_mtime), int(st.st_size)),
        )
        n += 1

        if commit_every and n % commit_every == 0:
            conn.commit()
            logger.debug("Committed at %d", n)

    conn.commit()
    dt = time.time() - t0
    logger.info("Indexed %d PDFs from %s in %.2fs", n, pdf_dir, dt)
    return n


def list_papers(conn: sqlite3.Connection, limit: int = 20):
    """
    Return known papers (id + path + extracted_mtime), for display.

    Args:
        conn: Open SQLite connection.
        limit: Max number of rows.

    Returns:
        List of rows: (id, path, extracted_mtime)
    """
    return conn.execute(
        "SELECT id, path, extracted_mtime FROM papers ORDER BY id LIMIT ?",
        (limit,),
    ).fetchall()


def extract_one(conn: sqlite3.Connection, paper_id: int, force: bool) -> tuple[bool, int]:
    """
    Extract per-page text for a single paper into the DB (and update FTS index).

    Behavior:
      - Looks up the paper by id in `papers`.
      - If not forced and extracted_mtime == file mtime, skips extraction.
      - Otherwise deletes existing rows for that paper from `pages` and `pages_fts`,
        re-opens the PDF, and inserts one row per page into both tables.
      - Updates `papers.extracted_mtime` to match the current file mtime.

    Important:
      - Does NOT commit. Caller decides commit cadence (single vs batch).

    Args:
        conn: Open SQLite connection.
        paper_id: Row id from the `papers` table.
        force: If True, re-extract even if up-to-date.

    Returns:
        (skipped, page_count)

    Raises:
        typer.BadParameter: if PyMuPDF is unavailable or paper_id not found.
    """
    if fitz is None:
        raise typer.BadParameter("PyMuPDF not available. Did you install pymupdf?")

    row = conn.execute(
        "SELECT path, mtime, extracted_mtime FROM papers WHERE id=?",
        (paper_id,),
    ).fetchone()

    if not row:
        raise typer.BadParameter(f"No paper with id={paper_id}. Run `paperbrace list`.")

    path_str, mtime, extracted_mtime = row
    pdf_path = Path(path_str)

    if (not force) and extracted_mtime is not None and float(extracted_mtime) == float(mtime):
        logger.info("Skip already extracted (up-to-date): %s", pdf_path.name)
        return True, 0

    # Clear existing rows for this paper (keep pages + FTS in sync)
    conn.execute("DELETE FROM pages WHERE paper_id=?", (paper_id,))
    conn.execute("DELETE FROM pages_fts WHERE paper_id=?", (paper_id,))

    doc = fitz.open(str(pdf_path))
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        conn.execute(
            "INSERT INTO pages (paper_id, page_num, text) VALUES (?, ?, ?)",
            (paper_id, i + 1, text),
        )
        conn.execute(
            "INSERT INTO pages_fts (text, paper_id, page_num) VALUES (?, ?, ?)",
            (text, paper_id, i + 1),
        )

    conn.execute(
        "UPDATE papers SET extracted_mtime=?, updated_at=datetime('now') WHERE id=?",
        (float(mtime), paper_id),
    )
    logger.info("Extracted %d pages → paper_id=%s (%s)", doc.page_count, paper_id, pdf_path.name)
    return False, doc.page_count


def extract_all(conn: sqlite3.Connection, force: bool, commit_every: int = 5) -> tuple[int, int]:
    """
    Extract per-page text for all papers in the DB.

    Skips up-to-date papers unless force=True. Commits periodically.

    Args:
        conn: Open SQLite connection.
        force: Re-extract even if up-to-date.
        commit_every: Commit frequency (in number of papers processed; 0 disables intermediate commits).

    Returns:
        (extracted_count, skipped_count)
    """
    rows = conn.execute("SELECT id FROM papers ORDER BY id").fetchall()
    ids = [int(r[0]) for r in rows]
    logger.info("Extracting %d papers (commit every %d)...", len(ids), commit_every)

    extracted = skipped = 0
    for i, pid in enumerate(ids, start=1):
        was_skipped, _ = extract_one(conn, paper_id=pid, force=force)
        skipped += int(was_skipped)
        extracted += int(not was_skipped)

        if commit_every and i % commit_every == 0:
            conn.commit()
            logger.debug("Committed at %d/%d", i, len(ids))

    conn.commit()
    return extracted, skipped


def purge_db(conn: sqlite3.Connection, reset_ids: bool = True, vacuum: bool = False) -> None:
    """
    Delete all data from the DB (papers/pages/pages_fts) and optionally reset ids.

    Note: this does NOT delete any semantic vector store (e.g., Chroma persist dir).
    For a fully clean slate you may also delete the vector store directory.

    Args:
        conn: Open SQLite connection.
        reset_ids: If True, resets AUTOINCREMENT counter for papers.
        vacuum: If True, runs VACUUM to reclaim space (can take time).
    """
    # Keep FTS + pages in sync
    conn.execute("DELETE FROM pages_fts;")
    conn.execute("DELETE FROM pages;")
    conn.execute("DELETE FROM papers;")

    if reset_ids:
        # sqlite_sequence may not exist on a brand new DB; ignore if missing
        try:
            conn.execute("DELETE FROM sqlite_sequence WHERE name='papers';")
        except sqlite3.OperationalError:
            pass

    conn.commit()

    if vacuum:
        conn.execute("VACUUM;")
        conn.commit()


def search_pages(conn: sqlite3.Connection, query: str, limit: int = 10):
    """
    Keyword search over extracted text using SQLite FTS5.

    Args:
        conn: Open SQLite connection.
        query: FTS query string (supports AND/OR/NOT, quotes, prefix *).
        limit: Max results to return.

    Returns:
        List of rows:
          (paper_id, page_num, path, snippet, score)
    """
    return conn.execute(
        """
        SELECT
          pages_fts.paper_id,
          pages_fts.page_num,
          papers.path,
          snippet(pages_fts, 0, '[', ']', '…', 12) AS snip,
          bm25(pages_fts) AS score
        FROM pages_fts
        JOIN papers ON papers.id = pages_fts.paper_id
        WHERE pages_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()


def get_evidence_pages(conn: sqlite3.Connection, query: str, limit: int = 8):
    """
    Retrieve top matching pages for a query, including full page text.

    Uses FTS5 ranking to pick candidate pages, then joins to `pages` to fetch
    the full extracted text.

    Returns rows:
      (paper_id, page_num, path, text, score)
    """
    fts_query = nl_to_fts_query(query)

    return conn.execute(
        """
        SELECT
          f.paper_id,
          f.page_num,
          p.path,
          pg.text,
          f.score
        FROM (
          SELECT
            pages_fts.paper_id AS paper_id,
            pages_fts.page_num AS page_num,
            bm25(pages_fts) AS score
          FROM pages_fts
          WHERE pages_fts MATCH ?
          ORDER BY score
          LIMIT ?
        ) AS f
        JOIN papers p ON p.id = f.paper_id
        JOIN pages  pg ON pg.paper_id = f.paper_id AND pg.page_num = f.page_num
        ORDER BY f.score
        """,
        (fts_query, limit),
    ).fetchall()


def iter_pages_for_paper(conn: sqlite3.Connection, paper_id: int) -> list[PageForIndex]:
    """
    Load all extracted pages for a given paper_id and convert them to PageForIndex.

    Args:
        conn: Open SQLite connection.
        paper_id: Paper id.

    Returns:
        List of PageForIndex (one per page).
    """
    row = conn.execute("SELECT path FROM papers WHERE id=?", (paper_id,)).fetchone()
    if not row:
        raise typer.BadParameter(f"No paper with id={paper_id}. Run `paperbrace list`.")
    (path_str,) = row

    rows = conn.execute(
        "SELECT page_num, text FROM pages WHERE paper_id=? ORDER BY page_num",
        (paper_id,),
    ).fetchall()

    return [
        PageForIndex(
            paper_id=int(paper_id),
            page_num=int(page_num),
            path=str(path_str),
            text=str(text),
        )
        for page_num, text in rows
    ]


def embed_one(
    conn: sqlite3.Connection,
    retriever: Retriever,
    paper_id: int,
    force: bool,
) -> tuple[bool, int]:
    """
    Embed pages for a single paper into the semantic retriever (e.g., Chroma).

    Behavior:
      - Requires extracted pages to exist in `pages`.
      - Skips if up-to-date and not forced:
          embedded_mtime == extracted_mtime
      - Otherwise deletes old vectors for this paper (via retriever.delete_paper),
        upserts all pages (retriever.upsert_pages), and updates papers.embedded_mtime.

    Important:
      - Does NOT commit. Caller controls commit cadence.

    Args:
        conn: Open SQLite connection.
        retriever: A semantic retriever backend (Chroma now; others later).
        paper_id: Paper id.
        force: If True, re-embed even if up-to-date.

    Returns:
        (skipped, page_count)
          skipped: True if embeddings were skipped as up-to-date.
          page_count: number of pages embedded (0 if skipped).

    Raises:
        typer.BadParameter if paper_id not found.
    """
    row = conn.execute(
        "SELECT path, extracted_mtime, embedded_mtime FROM papers WHERE id=?",
        (paper_id,),
    ).fetchone()

    if not row:
        raise typer.BadParameter(f"No paper with id={paper_id}. Run `paperbrace list`.")

    path_str, extracted_mtime, embedded_mtime = row
    pdf_name = Path(str(path_str)).name

    if extracted_mtime is None:
        logger.info("Skip not extracted yet: paper_id=%d (%s)", paper_id, pdf_name)
        return True, 0

    if (not force) and embedded_mtime is not None and float(embedded_mtime) == float(extracted_mtime):
        logger.info("Skip already embedded (up-to-date): paper_id=%d (%s)", paper_id, pdf_name)
        return True, 0

    pages = iter_pages_for_paper(conn, paper_id)
    if not pages:
        logger.info("Skip no extracted pages (run extract?): paper_id=%d (%s)", paper_id, pdf_name)
        return True, 0

    # Re-embed: delete then upsert
    retriever.delete_paper(paper_id)
    retriever.upsert_pages(pages)

    conn.execute(
        "UPDATE papers SET embedded_mtime=?, updated_at=datetime('now') WHERE id=?",
        (float(extracted_mtime), paper_id),
    )

    logger.info("Embedded %d pages → paper_id=%d (%s)", len(pages), paper_id, pdf_name)
    return False, len(pages)


def embed_all(
    conn: sqlite3.Connection,
    retriever: Retriever,
    force: bool,
    commit_every: int = 5,
) -> tuple[int, int]:
    """
    Embed pages for all extracted papers into the semantic retriever.

    Skips up-to-date papers unless force=True. Commits periodically.

    Args:
        conn: Open SQLite connection.
        retriever: A semantic retriever backend (Chroma now; others later).
        force: Re-embed even if up-to-date.
        commit_every: Commit frequency (in number of papers processed; 0 disables intermediate commits).

    Returns:
        (embedded_count, skipped_count)
    """
    rows = conn.execute(
        "SELECT id FROM papers WHERE extracted_mtime IS NOT NULL ORDER BY id"
    ).fetchall()
    ids = [int(r[0]) for r in rows]
    logger.info("Embedding %d papers (commit every %d)...", len(ids), commit_every)

    embedded = skipped = 0
    for i, pid in enumerate(ids, start=1):
        was_skipped, _ = embed_one(conn, retriever=retriever, paper_id=pid, force=force)
        skipped += int(was_skipped)
        embedded += int(not was_skipped)

        if commit_every and i % commit_every == 0:
            conn.commit()
            logger.debug("Committed at %d/%d", i, len(ids))

    conn.commit()
    return embedded, skipped
