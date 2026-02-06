from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

import typer

from paperbrace.retriever import ChunkForIndex, Retriever

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
    Scan a PDF directory and upsert file metadata into the `sources` table.

    - Recursively finds *.pdf under pdf_dir.
    - Upserts (path, mtime, size) into `sources`.
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
            INSERT INTO sources (path, mtime, size)
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


def list_sources(conn: sqlite3.Connection, limit: int = 20):
    """
    Return known sources (id + path + extracted_mtime), for display.

    Args:
        conn: Open SQLite connection.
        limit: Max number of rows.

    Returns:
        List of rows: (id, path, extracted_mtime)
    """
    return conn.execute(
        "SELECT id, path, extracted_mtime FROM sources ORDER BY id LIMIT ?",
        (limit,),
    ).fetchall()


def extract_one(conn: sqlite3.Connection, source_id: int, force: bool) -> tuple[bool, int]:
    """
    Extract per-page text for a single paper into the DB (and update FTS index).

    Behavior:
      - Looks up the paper by id in `sources`.
      - If not forced and extracted_mtime == file mtime, skips extraction.
      - Otherwise deletes existing rows for that paper from `pages` and `pages_fts`,
        re-opens the PDF, and inserts one row per page into both tables.
      - Updates `sources.extracted_mtime` to match the current file mtime.

    Important:
      - Does NOT commit. Caller decides commit cadence (single vs batch).

    Args:
        conn: Open SQLite connection.
        source_id: Row id from the `sources` table.
        force: If True, re-extract even if up-to-date.

    Returns:
        (skipped, page_count)

    Raises:
        typer.BadParameter: if PyMuPDF is unavailable or source_id not found.
    """
    if fitz is None:
        raise typer.BadParameter("PyMuPDF not available. Did you install pymupdf?")

    row = conn.execute(
        "SELECT path, mtime, extracted_mtime FROM sources WHERE id=?",
        (source_id,),
    ).fetchone()

    if not row:
        raise typer.BadParameter(f"No paper with id={source_id}. Run `paperbrace list`.")

    path_str, mtime, extracted_mtime = row
    pdf_path = Path(path_str)

    if (not force) and extracted_mtime is not None and float(extracted_mtime) == float(mtime):
        logger.info("Skip already extracted (up-to-date): %s", pdf_path.name)
        return True, 0

    # Clear existing rows for this paper (keep pages + FTS in sync)
    conn.execute("DELETE FROM pages WHERE source_id=?", (source_id,))
    conn.execute("DELETE FROM pages_fts WHERE source_id=?", (source_id,))

    doc = fitz.open(str(pdf_path))
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        conn.execute(
            "INSERT INTO pages (source_id, page_num, text) VALUES (?, ?, ?)",
            (source_id, i + 1, text),
        )
        conn.execute(
            "INSERT INTO pages_fts (text, source_id, page_num) VALUES (?, ?, ?)",
            (text, source_id, i + 1),
        )

    conn.execute(
        "UPDATE sources SET extracted_mtime=?, updated_at=datetime('now') WHERE id=?",
        (float(mtime), source_id),
    )
    logger.info("Extracted %d pages → source_id=%s (%s)", doc.page_count, source_id, pdf_path.name)
    return False, doc.page_count


def extract_all(conn: sqlite3.Connection, force: bool, commit_every: int = 5) -> tuple[int, int]:
    """
    Extract per-page text for all sources in the DB.

    Skips up-to-date sources unless force=True. Commits periodically.

    Args:
        conn: Open SQLite connection.
        force: Re-extract even if up-to-date.
        commit_every: Commit frequency (in number of sources processed; 0 disables intermediate commits).

    Returns:
        (extracted_count, skipped_count)
    """
    rows = conn.execute("SELECT id FROM sources ORDER BY id").fetchall()
    ids = [int(r[0]) for r in rows]
    logger.info("Extracting %d sources (commit every %d)...", len(ids), commit_every)

    extracted = skipped = 0
    for i, sid in enumerate(ids, start=1):
        was_skipped, _ = extract_one(conn, source_id=sid, force=force)
        skipped += int(was_skipped)
        extracted += int(not was_skipped)

        if commit_every and i % commit_every == 0:
            conn.commit()
            logger.debug("Committed at %d/%d", i, len(ids))

    conn.commit()
    return extracted, skipped


def purge_db(conn: sqlite3.Connection, reset_ids: bool = True, vacuum: bool = False) -> None:
    """
    Delete all data from the DB (sources/pages/pages_fts) and optionally reset ids.

    Note: this does NOT delete any semantic vector store (e.g., Chroma persist dir).
    For a fully clean slate you may also delete the vector store directory.

    Args:
        conn: Open SQLite connection.
        reset_ids: If True, resets AUTOINCREMENT counter for sources.
        vacuum: If True, runs VACUUM to reclaim space (can take time).
    """
    # Keep FTS + pages in sync
    conn.execute("DELETE FROM pages_fts;")
    conn.execute("DELETE FROM pages;")
    conn.execute("DELETE FROM sources;")

    if reset_ids:
        # sqlite_sequence may not exist on a brand new DB; ignore if missing
        try:
            conn.execute("DELETE FROM sqlite_sequence WHERE name='sources';")
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
          (source_id, page_num, path, snippet, distance)
    """
    query = nl_to_fts_query(query)
    return conn.execute(
        """
        SELECT
          pages_fts.source_id,
          pages_fts.page_num,
          sources.path,
          snippet(pages_fts, 0, '[', ']', '…', 12) AS snip,
          bm25(pages_fts) AS distance
        FROM pages_fts
        JOIN sources ON sources.id = pages_fts.source_id
        WHERE pages_fts MATCH ?
        ORDER BY distance
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
      (source_id, page_num, path, text, distance)
    """
    fts_query = nl_to_fts_query(query)

    return conn.execute(
        """
        SELECT
          f.source_id,
          f.page_num,
          p.path,
          pg.text,
          f.distance
        FROM (
          SELECT
            pages_fts.source_id AS source_id,
            pages_fts.page_num AS page_num,
            bm25(pages_fts) AS distance
          FROM pages_fts
          WHERE pages_fts MATCH ?
          ORDER BY distance
          LIMIT ?
        ) AS f
        JOIN sources p ON p.id = f.source_id
        JOIN pages  pg ON pg.source_id = f.source_id AND pg.page_num = f.page_num
        ORDER BY f.distance
        """,
        (fts_query, limit),
    ).fetchall()


def iter_pages_for_paper(conn: sqlite3.Connection, source_id: int) -> list[ChunkForIndex]:
    """
    Load all extracted pages for a given source_id and convert them to ChunkForIndex.

    Args:
        conn: Open SQLite connection.
        source_id: Source id.

    Returns:
        List of ChunkForIndex (one per page).
    """
    row = conn.execute("SELECT path FROM sources WHERE id=?", (source_id,)).fetchone()
    if not row:
        raise typer.BadParameter(f"No paper with id={source_id}. Run `paperbrace list`.")
    (path_str,) = row

    rows = conn.execute(
        "SELECT page_num, text FROM pages WHERE source_id=? ORDER BY page_num",
        (source_id,),
    ).fetchall()

    return [
        ChunkForIndex(
            source_id=int(source_id),
            page_num=int(page_num),
            path=str(path_str),
            text=str(text),
        )
        for page_num, text in rows
    ]


def set_vector_index(
    conn,
    *,
    backend: str,
    distance: str,
    collection_name: str,
    embedding_model: str,
) -> None:
    conn.execute(
        """
        INSERT INTO vector_indexes(backend, collection_name, distance, embedding_model)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(backend, collection_name) DO UPDATE SET
          collection_name=excluded.collection_name,
          distance=excluded.distance,
          embedding_model=excluded.embedding_model,
          updated_at=datetime('now')
        """,
        (backend, collection_name, distance, embedding_model),
    )


def get_vector_index(
    conn,
    *,
    backend: str,
    collection_name: str,
) -> Optional[tuple[str, str, str]]:
    """
    Returns (collection_name, distance, embedding_model) or None.
    """
    row = conn.execute(
        """
        SELECT collection_name, distance, embedding_model
        FROM vector_indexes
        WHERE backend=? AND collection_name=?
        """,
        (backend, collection_name),
    ).fetchone()
    if not row:
        return None
    return (str(row[0]), str(row[1]), str(row[2]))


def embed_one(
    conn: sqlite3.Connection,
    retriever: Retriever,
    source_id: int,
    force: bool,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 200,
) -> tuple[bool, int]:
    """
    Embed chunked page text for a single paper into the semantic retriever.

    This is the chunking-aware version of embedding:

    - Reads extracted pages from SQLite `pages` for `source_id`.
    - Splits each page's text into overlapping character chunks (configurable).
    - Deletes any existing vectors for this paper (retriever.delete_source).
    - Upserts all chunks (retriever.upsert_source).
    - Updates `sources.embedded_mtime` to match `sources.extracted_mtime`.

    Notes / expectations:
      - Changing chunk parameters (chunk_size/overlap/min_chunk_chars) does NOT
        automatically invalidate `embedded_mtime`. If you change chunking, run
        embed with `--force` (or pass force=True).
      - This function assumes your `ChunkForIndex` supports chunk metadata fields:
          - chunk_id: int
          - char_start: int
          - char_end: int
          - fingerprint: str
        If your current `ChunkForIndex` does not have these yet, add them as
        optional fields (retrievers can ignore what they don't use).

    Important:
      - Does NOT commit. Caller controls commit cadence.

    Args:
        conn: Open SQLite connection.
        retriever: A semantic retriever backend (e.g., Chroma or Flat).
        source_id: Source id to embed.
        force: If True, re-embed even if up-to-date.
        chunk_size: Target chunk size in characters (<=0 disables chunking; whole page as one chunk).
        chunk_overlap: Overlap between consecutive chunks in characters.
        min_chunk_chars: Drop chunks shorter than this (after trimming).

    Returns:
        (skipped, item_count)
          skipped: True if embedding was skipped as up-to-date or not possible.
          item_count: number of chunks embedded (0 if skipped).

    Raises:
        typer.BadParameter: if source_id not found.
    """
    import hashlib

    def _sha1(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _chunk_text(text: str) -> list[tuple[int, int, str]]:
        """
        Return list of (char_start, char_end, chunk_text) over `text`.

        We try to end chunks on whitespace/newline when possible for cleaner chunks.
        Offsets are relative to the original `text` (before trimming).
        """
        t = text or ""
        n = len(t)
        if chunk_size <= 0 or n <= chunk_size:
            chunk = t.strip()
            if len(chunk) < min_chunk_chars and n > 0:
                # allow very small pages; otherwise they'd vanish
                return [(0, n, t)]
            return [(0, n, t)]

        out: list[tuple[int, int, str]] = []
        i = 0
        # guard overlap
        ov = max(0, min(int(chunk_overlap), max(0, chunk_size - 1)))
        # how far back we search for a "nice" break near the end
        backtrack = 200

        while i < n:
            j = min(i + chunk_size, n)

            # try to break at whitespace near the end for readability
            if j < n:
                lo = max(i + min_chunk_chars, j - backtrack)
                cut = -1
                for k in range(j - 1, lo - 1, -1):
                    if t[k].isspace():
                        cut = k + 1
                        break
                if cut != -1 and cut > i:
                    j = cut

            raw = t[i:j]
            # preserve offsets into the original text while trimming
            ltrim = len(raw) - len(raw.lstrip())
            rtrim = len(raw) - len(raw.rstrip())
            start = i + ltrim
            end = j - rtrim
            chunk_txt = t[start:end]

            if len(chunk_txt) >= min_chunk_chars or (n <= chunk_size):
                out.append((start, end, chunk_txt))

            # advance with overlap (ensure progress)
            next_i = max(end - ov, i + 1)
            if next_i <= i:
                next_i = j
            i = next_i

        # if everything got filtered out, fall back to whole page
        if not out and n > 0:
            out = [(0, n, t)]
        return out

    row = conn.execute(
        "SELECT path, extracted_mtime, embedded_mtime FROM sources WHERE id=?",
        (source_id,),
    ).fetchone()

    if not row:
        raise typer.BadParameter(f"No paper with id={source_id}. Run `paperbrace list`.")

    path_str, extracted_mtime, embedded_mtime = row
    pdf_name = Path(str(path_str)).name

    if extracted_mtime is None:
        logger.info("Skip not extracted yet: source_id=%d (%s)", source_id, pdf_name)
        return True, 0

    if (not force) and embedded_mtime is not None and float(embedded_mtime) == float(extracted_mtime):
        logger.info("Skip already embedded (up-to-date): source_id=%d (%s)", source_id, pdf_name)
        return True, 0

    pages = list(iter_pages_for_paper(conn, source_id))
    if not pages:
        logger.info("Skip no extracted pages (run extract?): source_id=%d (%s)", source_id, pdf_name)
        return True, 0

    # Build chunk-level items for indexing
    items: list[ChunkForIndex] = []
    total_pages = 0

    for p in pages:
        total_pages += 1
        page_text = p.text or ""
        chunks = _chunk_text(page_text)

        for chunk_id, (cs, ce, chunk_txt) in enumerate(chunks):
            # If you haven’t added these fields to ChunkForIndex yet,
            # add them as Optional[...] in the dataclass.
            items.append(
                ChunkForIndex(
                    source_id=p.source_id,
                    page_num=p.page_num,
                    path=p.path,
                    text=chunk_txt,
                    chunk_id=chunk_id,
                    char_start=cs,
                    char_end=ce,
                    fingerprint=_sha1(chunk_txt),
                )
            )

    if not items:
        logger.info("Skip no chunks produced: source_id=%d (%s)", source_id, pdf_name)
        return True, 0

    # Re-embed: delete then upsert
    retriever.delete_source(source_id)
    retriever.upsert_source(items)

    conn.execute(
        "UPDATE sources SET embedded_mtime=?, updated_at=datetime('now') WHERE id=?",
        (float(extracted_mtime), source_id),
    )

    logger.info(
        "Embedded %d chunks from %d pages → source_id=%d (%s)",
        len(items),
        total_pages,
        source_id,
        pdf_name,
    )
    return False, len(items)



def embed_all(
    conn: sqlite3.Connection,
    retriever: Retriever,
    force: bool,
    commit_every: int = 5,
) -> tuple[int, int]:
    """
    Embed pages for all extracted sources into the semantic retriever.

    Skips up-to-date sources unless force=True. Commits periodically.

    Args:
        conn: Open SQLite connection.
        retriever: A semantic retriever backend (Chroma now; others later).
        force: Re-embed even if up-to-date.
        commit_every: Commit frequency (in number of sources processed; 0 disables intermediate commits).

    Returns:
        (embedded_count, skipped_count)
    """
    rows = conn.execute(
        "SELECT id FROM sources WHERE extracted_mtime IS NOT NULL ORDER BY id"
    ).fetchall()
    ids = [int(r[0]) for r in rows]
    logger.info("Embedding %d sources (commit every %d)...", len(ids), commit_every)

    embedded = skipped = 0
    for i, sid in enumerate(ids, start=1):
        was_skipped, _ = embed_one(conn, retriever=retriever, source_id=sid, force=force)
        skipped += int(was_skipped)
        embedded += int(not was_skipped)

        if commit_every and i % commit_every == 0:
            conn.commit()
            logger.debug("Committed at %d/%d", i, len(ids))

    conn.commit()
    return embedded, skipped


def get_source_id_by_filename(conn, file_name: str) -> int:
    """
    Resolve a source id from a filename (basename), e.g. "imf_inequality_wp1920.pdf".

    Matches by basename against sources.path. Raises if not found or ambiguous.
    Works even if paths contain backslashes (Windows) by normalizing separators.
    """
    name = (file_name or "").strip()
    if not name:
        raise ValueError("file_name is empty")

    def _basename(p: str) -> str:
        return (p or "").replace("\\", "/").rsplit("/", 1)[-1]

    # Cheap prefilter using LIKE, then exact basename match in Python.
    rows = conn.execute(
        "SELECT id, path FROM sources WHERE path LIKE ?",
        (f"%{name}",),
    ).fetchall()

    matches = [(int(sid), str(path)) for sid, path in rows if _basename(str(path)) == name]

    if not matches:
        raise KeyError(f"No source found with filename={name!r}")

    if len(matches) > 1:
        # Safer to force uniqueness than silently pick one.
        ids = [sid for sid, _ in matches]
        raise ValueError(f"Ambiguous filename={name!r}; matches source_ids={ids}")

    return matches[0][0]
