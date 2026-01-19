import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from paperbrace.cli import app

runner = CliRunner()


def _make_pdf(pdf_path: Path, pages: list[str]) -> None:
    try:
        import fitz  # PyMuPDF
    except Exception:
        pytest.skip("PyMuPDF (fitz) not available")

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)  # simple, searchable text
    doc.save(str(pdf_path))
    doc.close()


def _db_count(db_path: Path, sql: str, params=()) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(sql, params).fetchone()[0]
    finally:
        conn.close()


def _first_paper_id(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT id FROM papers ORDER BY id LIMIT 1").fetchone()
        assert row is not None
        return int(row[0])
    finally:
        conn.close()


def test_index_inserts_papers(tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    db = tmp_path / "paperbrace.db"

    _make_pdf(pdf_dir / "a.pdf", ["hello a p1"])
    _make_pdf(pdf_dir / "b.pdf", ["hello b p1"])

    r = runner.invoke(app, ["index", "--pdf-dir", str(pdf_dir), "--db", str(db)])
    assert r.exit_code == 0

    assert _db_count(db, "SELECT COUNT(*) FROM papers") == 2


def test_extract_populates_pages(tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    db = tmp_path / "paperbrace.db"

    _make_pdf(pdf_dir / "a.pdf", ["needle page1", "page2 text"])
    runner.invoke(app, ["index", "--pdf-dir", str(pdf_dir), "--db", str(db)])

    pid = _first_paper_id(db)
    r = runner.invoke(app, ["extract", "--paper-id", str(pid), "--db", str(db)])
    assert r.exit_code == 0

    assert _db_count(db, "SELECT COUNT(*) FROM pages WHERE paper_id=?", (pid,)) == 2
    assert _db_count(db, "SELECT COUNT(*) FROM pages WHERE paper_id=? AND text LIKE '%needle%'", (pid,)) == 1


def test_extract_skips_when_uptodate(tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    db = tmp_path / "paperbrace.db"

    _make_pdf(pdf_dir / "a.pdf", ["hello"])
    runner.invoke(app, ["index", "--pdf-dir", str(pdf_dir), "--db", str(db)])

    pid = _first_paper_id(db)
    runner.invoke(app, ["extract", "--paper-id", str(pid), "--db", str(db)])

    before = _db_count(db, "SELECT COUNT(*) FROM pages WHERE paper_id=?", (pid,))
    r2 = runner.invoke(app, ["extract", "--paper-id", str(pid), "--db", str(db)])
    after = _db_count(db, "SELECT COUNT(*) FROM pages WHERE paper_id=?", (pid,))

    assert r2.exit_code == 0
    assert before == after
    
    mtime1 = sqlite3.connect(str(db)).execute(
    "SELECT extracted_mtime FROM papers WHERE id=?", (pid,)
    ).fetchone()[0]

    runner.invoke(app, ["extract", "--paper-id", str(pid), "--db", str(db)])

    mtime2 = sqlite3.connect(str(db)).execute(
        "SELECT extracted_mtime FROM papers WHERE id=?", (pid,)
    ).fetchone()[0]

    assert mtime1 == mtime2