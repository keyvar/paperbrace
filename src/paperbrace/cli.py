# src/paperbrace/cli.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from paperbrace.config import load_config
from paperbrace.db import connect, init_db
from paperbrace.llm_client import LLMConfig, generate as llm_generate
from paperbrace.render import Evidence, to_markdown
from paperbrace import store
from paperbrace.paths import DEFAULT_DB, DEFAULT_CHROMA_DIR, DEFAULT_FLAT_DIR
from paperbrace.retriever import make_retriever

logger = logging.getLogger("paperbrace")
console = Console()

app = typer.Typer(add_completion=False)


def setup_logging(verbose: bool) -> None:
    """
    Configure process-wide logging for CLI runs.

    Args:
        verbose: If True, sets DEBUG level; otherwise INFO.

    Returns:
        None
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _cap_filename(name: str, n: int = 30) -> str:
    """
    Truncate a filename for compact CLI display.

    Keeps the beginning and end of the filename and inserts an ellipsis in the middle.

    Args:
        name: Filename (no directory).
        n: Maximum output length.

    Returns:
        A shortened filename if needed.
    """
    if len(name) <= n:
        return name
    left = (n - 1) // 2
    right = n - 1 - left
    return name[:left] + "…" + name[-right:]


@app.command()
def index(
    pdf_dir: Path = typer.Option(
        ...,
        "--pdf-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to scan for PDFs (recursively).",
    ),
    db_path: Path = typer.Option(
        DEFAULT_DB,
        "--db-path",
        help="SQLite database path (papers/pages/FTS).",
    ),
    commit_every: int = typer.Option(
        200,
        "--commit-every",
        min=1,
        max=5000,
        help="Commit frequency while indexing (performance vs safety).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Scan PDFs and upsert file metadata into SQLite.

    Records file-level metadata (path, mtime, size) in the `papers` table.
    Does not extract page text. Use `paperbrace extract` for extraction.

    Args:
        pdf_dir: Folder containing PDFs.
        db_path: SQLite DB file path.
        commit_every: Commit after every N PDFs processed.
        verbose: Enable debug logging.

    Returns:
        None
    """
    setup_logging(verbose)
    conn = connect(db_path)
    init_db(conn)

    pdf_dir = pdf_dir.expanduser().resolve()
    logger.info("Scanning %s", pdf_dir)

    n = store.index_pdfs(conn, pdf_dir=pdf_dir, commit_every=commit_every)
    logger.info("Indexed %d PDFs → %s", n, db_path)


@app.command()
def list(
    limit: int = typer.Option(20, "--limit", "-n", min=1, max=200, help="Max rows to display."),
    db_path: Path = typer.Option(DEFAULT_DB, "--db-path", help="SQLite database path."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    List known papers (id + extraction status + path).

    Args:
        limit: Maximum number of papers to display.
        db_path: SQLite DB file path.
        verbose: Enable debug logging.

    Returns:
        None
    """
    setup_logging(verbose)
    conn = connect(db_path)
    init_db(conn)

    rows = store.list_papers(conn, limit=limit)

    table = Table(title=f"Papers (showing {len(rows)})")
    table.add_column("id", justify="right")
    table.add_column("extracted?", justify="center")
    table.add_column("path")

    for pid, path, extracted_mtime in rows:
        table.add_row(str(pid), "yes" if extracted_mtime else "no", path)

    console.print(table)


@app.command()
def extract(
    paper_id: Optional[int] = typer.Option(None, "--paper-id", min=1, help="Extract a single paper by id."),
    all_: bool = typer.Option(False, "--all", help="Extract all papers in the DB."),
    db_path: Path = typer.Option(DEFAULT_DB, "--db-path", help="SQLite database path."),
    force: bool = typer.Option(False, "--force", help="Re-extract even if up-to-date."),
    commit_every: int = typer.Option(5, "--commit-every", min=1, max=200, help="Commit frequency during --all."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Extract per-page text for one paper or all papers.

    Stores page text in `pages` and updates `pages_fts` for keyword search.

    Args:
        paper_id: Extract a single paper by id (mutually exclusive with --all).
        all_: Extract all papers (mutually exclusive with --paper-id).
        db_path: SQLite DB file path.
        force: Re-extract even if PDF mtime is unchanged.
        commit_every: Commit frequency in batch mode.
        verbose: Enable debug logging.

    Returns:
        None
    """
    setup_logging(verbose)
    conn = connect(db_path)
    init_db(conn)

    if all_ == (paper_id is not None):
        raise typer.BadParameter("Use either --paper-id ID or --all (not both).")

    if all_:
        extracted, skipped = store.extract_all(conn, force=force, commit_every=commit_every)
        logger.info("Done. Extracted=%d Skipped=%d", extracted, skipped)
        return

    assert paper_id is not None
    store.extract_one(conn, paper_id=paper_id, force=force)
    conn.commit()


@app.command()
def purge(
    db_path: Path = typer.Option(DEFAULT_DB, "--db-path", help="SQLite database path."),
    yes: bool = typer.Option(False, "--yes", help="Confirm destructive purge."),
    vacuum: bool = typer.Option(False, "--vacuum", help="Reclaim space after purge (slower)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Delete ALL rows from the SQLite DB (papers/pages/pages_fts).

    Args:
        db_path: SQLite DB file path.
        yes: Must be set to confirm.
        vacuum: If True, VACUUM after purge.
        verbose: Enable debug logging.

    Returns:
        None
    """
    setup_logging(verbose)
    if not yes:
        raise typer.BadParameter("Refusing to purge without --yes")

    conn = connect(db_path)
    init_db(conn)
    store.purge_db(conn, reset_ids=True, vacuum=vacuum)
    logger.info("Purged DB: %s", db_path)


@app.command()
def search(
    query: str = typer.Argument(..., help="FTS query (supports AND/OR/NOT, quotes, prefix *)."),
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=50, help="Max results to display."),
    db_path: Path = typer.Option(DEFAULT_DB, "--db-path", help="SQLite database path."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Keyword search over extracted text using SQLite FTS5.

    Args:
        query: FTS query string.
        limit: Max results to show.
        db_path: SQLite DB file path.
        verbose: Enable debug logging.

    Returns:
        None
    """
    setup_logging(verbose)
    conn = connect(db_path)
    init_db(conn)

    rows = store.search_pages(conn, query=query, limit=limit)
    table = Table(title=f"Search: {query!r} (showing {len(rows)})")
    table.add_column("paper_id", justify="right")
    table.add_column("page", justify="right")
    table.add_column("file")
    table.add_column("snippet")

    for paper_id, page_num, path, snip, score in rows:
        fname = _cap_filename(Path(path).name, 30)
        table.add_row(str(paper_id), str(page_num), fname, snip)

    console.print(table)


@app.command()
def embed(
    paper_id: int = typer.Option(0, "--paper-id", min=0, help="Embed one paper id (0 means use --all)."),
    all_: bool = typer.Option(False, "--all", help="Embed all extracted papers."),
    db_path: Path = typer.Option(DEFAULT_DB, "--db-path", help="SQLite database path (source pages)."),
    backend: str = typer.Option(
        "chroma",
        "--backend",
        help="Vector backend: chroma | flat",
    ),
    force: bool = typer.Option(False, "--force", help="Re-embed even if up-to-date."),
    commit_every: int = typer.Option(5, "--commit-every", min=1, max=200),
    chroma_db_path: Path = typer.Option(DEFAULT_CHROMA_DIR, "--chroma-db-path", help="Chroma persistence directory."),
    flat_index_dir: Path = typer.Option(DEFAULT_FLAT_DIR, "--flat-index-dir", help="Flat index directory."),
    collection: str = typer.Option("paperbrace_pages", "--collection", help="Collection/index name (backend-specific)."),
    embedding_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--embedding-model"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Build/update the semantic vector index from extracted pages.

    Requires that PDFs are already extracted into `pages` (run `paperbrace extract` first).

    Args:
        paper_id: Single paper id to embed (0 means use --all).
        all_: Embed all extracted papers.
        db_path: SQLite DB file path (papers/pages).
        backend: Vector backend: chroma or flat.
        force: Re-embed even if unchanged.
        commit_every: Commit frequency for embed bookkeeping.
        chroma_db_path: Where Chroma persists its index.
        flat_index_dir: Where the flat index persists (embeddings + metadata).
        collection: Collection/index name for the backend.
        embedding_model: Embedding model id for SentenceTransformers.
        verbose: Enable debug logging.

    Returns:
        None
    """
    setup_logging(verbose)
    conn = connect(db_path)
    init_db(conn)

    retr = make_retriever(
        backend=backend,
        chroma_db_path=chroma_db_path.expanduser().resolve(),
        flat_index_dir=flat_index_dir.expanduser().resolve(),
        collection=collection,
        embedding_model=embedding_model,
        db_path=db_path,
    )

    if all_ == (paper_id != 0):
        raise typer.BadParameter("Use either --paper-id ID or --all (not both).")

    if all_:
        embedded, skipped = store.embed_all(conn, retriever=retr, force=force, commit_every=commit_every)
        logger.info("Done. Embedded=%d Skipped=%d", embedded, skipped)
    else:
        was_skipped, n_pages = store.embed_one(conn, retriever=retr, paper_id=paper_id, force=force)
        if was_skipped:
            logger.info("Skipped embedding paper_id=%d", paper_id)
        else:
            conn.commit()
            logger.info("Embedded paper_id=%d (%d pages)", paper_id, n_pages)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from your library."),
    k: int = typer.Option(8, "--k", min=1, max=30, help="Number of evidence pages."),
    retriever: str = typer.Option(
        "keyword",
        "--retriever",
        help="Retrieval mode: keyword (FTS5), semantic (embeddings), hybrid (both).",
    ),
    model: Optional[str] = typer.Option(None, "--model", help="HF model id (overrides config)."),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to paperbrace.toml (default: ./paperbrace.toml)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write answer + evidence to Markdown."),
    db_path: Path = typer.Option(DEFAULT_DB, "--db-path", help="SQLite database path (papers/pages)."),
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="Vector backend for semantic/hybrid: auto | chroma | flat",
    ),
    chroma_db_path: Path = typer.Option(DEFAULT_CHROMA_DIR, "--chroma-db-path", help="Chroma persistence directory."),
    flat_index_dir: Path = typer.Option(Path(".paperbrace/flat"), "--flat-index-dir", help="Flat index directory."),
    collection: str = typer.Option("paperbrace_pages", "--collection", help="Collection name (backend-specific)."),
    embedding_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--embedding-model"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Answer a question using retrieved evidence + a local Hugging Face LLM.

    For pure keyword lookup without an LLM, use `paperbrace search`.

    Retrieval modes:
      - keyword: SQLite FTS5 over extracted pages (pages_fts), ranked by bm25().
      - semantic: vector search over embedded pages (requires `paperbrace embed`).
      - hybrid: semantic results first, then keyword fill; de-duped by (paper_id, page_num).

    Vector backends (for semantic/hybrid):
      - chroma: persistent Chroma DB (requires `paperbrace[chroma]`)
      - flat: local NumPy index (requires `paperbrace[flat]`)
      - auto: prefer chroma if installed, else flat if installed
    """
    setup_logging(verbose)
    conn = connect(db_path)
    init_db(conn)

    # Resolve model (LLM-only)
    cfg_file = load_config(config)
    resolved_model = model or cfg_file.llm_model
    if not resolved_model:
        logger.error(
            "No LLM model configured. Set [llm].model in paperbrace.toml or pass --model. "
            'For keyword search, use: paperbrace search "..."'
        )
        raise typer.Exit(code=2)

    mode = (retriever or "").strip().lower()
    if mode not in {"keyword", "semantic", "hybrid"}:
        raise typer.BadParameter("Invalid --retriever. Use: keyword | semantic | hybrid")

    rows: list[tuple[int, int, str, str, float]] = []

    if mode == "keyword":
        rows = store.get_evidence_pages(conn, query=question, limit=k)

    else:
        # IMPORTANT: semantic/hybrid goes through make_retriever (no chromadb-only logic here)
        from paperbrace.retriever import make_retriever

        try:
            vec = make_retriever(
                backend=backend,
                chroma_db_path=chroma_db_path.expanduser().resolve(),
                flat_index_dir=flat_index_dir.expanduser().resolve(),
                collection=collection,
                embedding_model=embedding_model,
                db_path=db_path,
            )
        except Exception as e:
            logger.error("%s", e)
            raise typer.Exit(code=2)

        sem_hits = vec.query_pages(question, k=k, where=None)
        sem_rows = [(h.paper_id, h.page_num, h.path, h.text, float(h.score)) for h in sem_hits]

        if mode == "semantic":
            rows = sem_rows
        else:
            kw_rows = store.get_evidence_pages(conn, query=question, limit=max(k * 2, k))
            seen: set[tuple[int, int]] = set()
            merged: list[tuple[int, int, str, str, float]] = []

            for pid, pn, path, text, score in sem_rows + kw_rows:
                key = (int(pid), int(pn))
                if key in seen:
                    continue
                seen.add(key)
                merged.append((int(pid), int(pn), str(path), str(text), float(score)))
                if len(merged) >= k:
                    break

            rows = merged

    items = [Evidence(int(pid), int(pn), str(path), str(text), float(score)) for pid, pn, path, text, score in rows]

    table = Table(title=f"Evidence ({mode}) for: {question!r} (showing {len(items)})")
    table.add_column("#", justify="right")
    table.add_column("paper_id", justify="right")
    table.add_column("page", justify="right")
    table.add_column("file")
    for i, e in enumerate(items, start=1):
        table.add_row(str(i), str(e.paper_id), str(e.page_num), _cap_filename(Path(e.path).name, 30))
    console.print(table)

    if not items:
        answer = "Not found in the provided papers."
        print(answer)
        if out:
            out.write_text(to_markdown(question, items, answer=answer), encoding="utf-8")
            logger.info("Wrote → %s", out)
        return

    system = (
        "You are Paperbrace. Answer ONLY using the provided evidence. "
        "Cite claims with bracketed numbers like [1] or [2]. "
        'If the evidence is insufficient, say: "Not found in the provided papers."'
    )

    blocks: list[str] = []
    for i, e in enumerate(items, start=1):
        excerpt = (e.text or "").strip()
        if len(excerpt) > 1200:
            excerpt = excerpt[:1200] + "…"
        blocks.append(f"[{i}] paper_id={e.paper_id} page={e.page_num} file={Path(e.path).name}\n{excerpt}")

    user = (
        f"Question: {question}\n\n"
        "Evidence:\n\n"
        + "\n\n---\n\n".join(blocks)
        + "\n\nWrite a concise answer with citations like [1], [2]."
    )

    cfg = LLMConfig(
        model=resolved_model,
        max_new_tokens=cfg_file.llm_max_new_tokens,
        temperature=cfg_file.llm_temperature,
    )
    answer = llm_generate(system=system, user=user, cfg=cfg)

    print(answer.strip())

    if out:
        out.write_text(to_markdown(question, items, answer=answer), encoding="utf-8")
        logger.info("Wrote → %s", out)



if __name__ == "__main__":
    app()
