# src/paperbrace/paths.py
from __future__ import annotations
from pathlib import Path

STATE_DIR = Path(".paperbrace")

DEFAULT_DB = STATE_DIR / "db" / "paperbrace.db"
DEFAULT_CHROMA_DIR = STATE_DIR / "chroma"

# sqlite-vec usually lives INSIDE the same SQLite DB as vectors in a vec0 table.
# If you still want a separate db file for vectors, use this:
DEFAULT_SQLITEVEC_DB = STATE_DIR / "sqlitevec" / "paperbrace_vec.db"
