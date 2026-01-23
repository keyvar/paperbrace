# src/paperbrace/paths.py
from __future__ import annotations
from pathlib import Path

STATE_DIR = Path(".paperbrace")

DEFAULT_DB = STATE_DIR / "db" / "paperbrace.db"
DEFAULT_CHROMA_DIR = STATE_DIR / "chroma"
DEFAULT_FLAT_DIR = STATE_DIR / "flat"
