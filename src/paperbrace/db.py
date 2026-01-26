from __future__ import annotations

from pathlib import Path
import sqlite3


def connect(db_path: Path) -> sqlite3.Connection:
    """
    Open (or create) the Paperbrace SQLite database.

    - Ensures the parent directory exists.
    - Enables WAL journal mode for better concurrency / write performance.
    - Enables foreign key enforcement (required for ON DELETE CASCADE).

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        An open sqlite3.Connection. Caller is responsible for closing it.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == col for r in rows)  # r[1] = column name


def _add_column(conn: sqlite3.Connection, table: str, col: str, col_ddl: str) -> None:
    """
    Add a column if missing. Keeps migrations additive/simple.
    """
    if not _column_exists(conn, table, col):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_ddl}")


def init_db(conn: sqlite3.Connection) -> None:
    """
    Ensure the database schema exists and apply lightweight additive migrations.

    Creates:
      - sources: one row per PDF (file path + file metadata + extraction markers)
      - pages: extracted text per page (source_id, page_num) -> text
      - pages_fts: FTS5 full-text index over page text (keyword retrieval)

    Lightweight migrations (additive):
      - Adds missing columns (e.g., embedded_mtime) via ALTER TABLE.
      - Creates missing tables/virtual tables if they don't exist.

    Notes:
      - This is idempotent (safe to call on every command).
      - We intentionally avoid a full schema-version migration system for now.
        If/when the schema stabilizes, we can add a `meta` table with a
        schema_version and run versioned migrations.

    Args:
        conn: Open SQLite connection.

    Returns:
        None
    """
    # Base tables (fresh install)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL,
            extracted_mtime REAL,
            added_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS pages (
            source_id INTEGER NOT NULL,
            page_num INTEGER NOT NULL,
            text TEXT NOT NULL,
            PRIMARY KEY (source_id, page_num),
            FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS vector_indexes (
            backend TEXT NOT NULL,
            collection_name TEXT NOT NULL,  -- e.g. paperbrace_pages
            distance TEXT NOT NULL,              -- cosine | l2 | ip
            embedding_model TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (backend, collection_name)
        );
        """
    )

    # Additive migration: track embedding freshness (semantic index)
    _add_column(conn, "sources", "embedded_mtime", "REAL")

    # Additive migration: keyword index
    if not _table_exists(conn, "pages_fts"):
        conn.executescript(
            """
            CREATE VIRTUAL TABLE pages_fts
            USING fts5(
                text,
                source_id UNINDEXED,
                page_num UNINDEXED
            );
            """
        )

    conn.commit()
