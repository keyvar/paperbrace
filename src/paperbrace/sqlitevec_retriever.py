from __future__ import annotations

import re
from array import array
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# sqlite-vec requires extension loading support; some Python sqlite3 builds omit it.
try:
    import sqlite3 as _sqlite3  # type: ignore

    _has_ext = hasattr(_sqlite3.Connection, "enable_load_extension") and hasattr(
        _sqlite3.Connection, "load_extension"
    )
    if not _has_ext:
        raise ImportError("sqlite3 lacks extension loading support")
    sqlite3 = _sqlite3
except Exception:  # pragma: no cover
    try:
        import pysqlite3 as sqlite3  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "sqlite-vec requires SQLite extension loading, but your sqlite3 build "
            "does not support it. Install pysqlite3-binary:\n\n"
            "  pip install pysqlite3-binary\n"
        ) from e

import sqlite_vec
from sentence_transformers import SentenceTransformer

from paperbrace.retriever import PageForIndex, RetrievedPage


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_ident(name: str) -> str:
    """Allow only safe SQLite identifiers (avoids SQL injection via table names)."""
    if not _IDENT_RE.match(name):
        raise ValueError(f"Invalid identifier: {name!r} (use letters/digits/_ only)")
    return name


def _serialize_f32(vec: Sequence[float]) -> bytes:
    """Serialize a float vector to float32 bytes for sqlite-vec."""
    return array("f", vec).tobytes()


def _page_key(paper_id: int, page_num: int) -> int:
    """
    Deterministic integer key for (paper_id, page_num).

    Assumes page_num < 100_000 (true for PDFs). This lets us upsert deterministically.
    """
    return int(paper_id) * 100_000 + int(page_num)


class SqliteVecPageRetriever:
    """
    sqlite-vec retriever backed by its OWN SQLite database.

    Storage layout (inside `db_path`):
      - <meta_table> : page_id -> (paper_id, page_num, path, text)
      - <vec_table>  : vec0 virtual table mapping page_id -> embedding, plus query distance

    This backend is fully self-contained: it does NOT depend on your main Paperbrace DB.
    """

    def __init__(
        self,
        *,
        db_path: Path,
        table: str = "pages_vec",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.db_path = Path(db_path)
        self.vec_table = _validate_ident(table)
        self.meta_table = _validate_ident(f"{table}_meta")

        self._st = SentenceTransformer(embedding_model)
        self.dim = int(self._st.get_sentence_embedding_dimension())

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")

        # Load sqlite-vec extension
        self.conn.enable_load_extension(True)  # type: ignore[attr-defined]
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)  # type: ignore[attr-defined]

        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """
        Create tables if needed and validate embedding dimension consistency.
        """
        self.conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS vec_meta (
              k TEXT PRIMARY KEY,
              v TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS {self.meta_table} (
              page_id INTEGER PRIMARY KEY,
              paper_id INTEGER NOT NULL,
              page_num INTEGER NOT NULL,
              path TEXT NOT NULL,
              text TEXT NOT NULL
            );
            """
        )

        row = self.conn.execute("SELECT v FROM vec_meta WHERE k='vec_dim'").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO vec_meta(k, v) VALUES('vec_dim', ?)",
                (str(self.dim),),
            )
        else:
            if int(row[0]) != self.dim:
                raise ValueError(
                    f"Vector dim mismatch: DB has {row[0]}, model produces {self.dim}. "
                    "Use a new sqlite-vec DB path or purge/rebuild vectors."
                )

        # vec0 table: page_id + embedding, auto-provides v.distance during MATCH queries
        self.conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.vec_table}
            USING vec0(
              page_id INTEGER PRIMARY KEY,
              embedding FLOAT[{self.dim}]
            );
            """
        )
        self.conn.commit()

    def upsert_pages(self, pages: Sequence[PageForIndex], **kwargs: Any) -> None:
        """
        Embed and upsert pages into sqlite-vec.

        Args:
            pages: Sequence of PageForIndex to index.
            **kwargs:
              - batch_size (int): SentenceTransformer encode batch size (default 64)
              - show_progress_bar (bool): show embedding progress bar (default False)
        """
        if not pages:
            return

        batch_size = int(kwargs.get("batch_size", 64))
        show_pb = bool(kwargs.get("show_progress_bar", False))

        docs = [p.text for p in pages]
        embs = self._st.encode(docs, batch_size=batch_size, show_progress_bar=show_pb)

        meta_stmt = f"""
          INSERT OR REPLACE INTO {self.meta_table}(page_id, paper_id, page_num, path, text)
          VALUES (?, ?, ?, ?, ?)
        """
        vec_stmt = f"""
          INSERT OR REPLACE INTO {self.vec_table}(page_id, embedding)
          VALUES (?, ?)
        """

        with self.conn:
            for p, e in zip(pages, embs):
                pid = _page_key(p.paper_id, p.page_num)
                self.conn.execute(
                    meta_stmt,
                    (pid, int(p.paper_id), int(p.page_num), str(p.path), str(p.text)),
                )
                self.conn.execute(vec_stmt, (pid, _serialize_f32(e)))

    def delete_paper(self, paper_id: int, **kwargs: Any) -> None:
        """
        Delete all vectors/pages for a given paper_id from this sqlite-vec DB.
        """
        paper_id = int(paper_id)
        with self.conn:
            page_ids = [
                int(r[0])
                for r in self.conn.execute(
                    f"SELECT page_id FROM {self.meta_table} WHERE paper_id=?",
                    (paper_id,),
                ).fetchall()
            ]
            if not page_ids:
                return

            self.conn.execute(
                f"DELETE FROM {self.meta_table} WHERE paper_id=?",
                (paper_id,),
            )
            # Delete from vec table by page_id
            # Use executemany for portability
            self.conn.executemany(
                f"DELETE FROM {self.vec_table} WHERE page_id=?",
                [(pid,) for pid in page_ids],
            )

    def query_pages(
        self,
        query: str,
        k: int,
        where: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedPage]:
        """
        Vector-search for top-k pages.

        Args:
            query: Natural-language query text.
            k: Number of results to return.
            where: Optional filters. Supports equality filters on:
                   - paper_id (int)
                   - page_num (int)
                   - path (str)
        Returns:
            List of RetrievedPage. score is distance (lower is better).
        """
        q = self._st.encode([query], show_progress_bar=False)[0]
        q_blob = _serialize_f32(q)

        filters_sql = ""
        params: List[Any] = [q_blob]

        if where:
            for key, val in where.items():
                if key not in {"paper_id", "page_num", "path"}:
                    raise ValueError(f"Unsupported filter key: {key!r}")
                filters_sql += f" AND m.{key}=?"
                params.append(val)

        params.append(int(k))

        rows = self.conn.execute(
            f"""
            SELECT
              m.paper_id,
              m.page_num,
              m.path,
              m.text,
              v.distance
            FROM {self.vec_table} AS v
            JOIN {self.meta_table} AS m ON m.page_id = v.page_id
            WHERE v.embedding MATCH ?
            {filters_sql}
            ORDER BY v.distance
            LIMIT ?
            """,
            params,
        ).fetchall()

        out: List[RetrievedPage] = []
        for paper_id, page_num, path, text, dist in rows:
            out.append(
                RetrievedPage(
                    paper_id=int(paper_id),
                    page_num=int(page_num),
                    path=str(path),
                    text=str(text),
                    score=float(dist),  # lower is better
                )
            )
        return out
