from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from paperbrace.db import connect, init_db
from paperbrace import store
from paperbrace.render import Evidence, to_markdown
from paperbrace.llm_client import LLMConfig, generate as llm_generate
from paperbrace.config import load_config

logger = logging.getLogger("paperbrace")
console = Console()

app = typer.Typer(add_completion=False)
DEFAULT_DB = "paperbrace.db"


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

    Keeps the beginning and end of the filename and inserts an ellipsis in the
    middle (useful for tables).

    Args:
        name: Filename (no directory).
        n: Maximum output length (characters).

    Returns:
        A possibly shortened filename with a middle ellipsis.
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
    db: Path = typer.Option(
        Path(DEFAULT_DB),
        "--db",
        help="SQLite database path.",
    ),
    commit_every: int = typer.Option(
        200,
        "--commit-every",
        min=1,
        max=5000,
        help="Commit frequency while indexing (performance vs safety).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """
    Scan PDFs and upsert file metadata into SQLite.

    This command records PDF file metadata (path, mtime, size) in the `papers`
    table. It does not extract text. Extraction is done via `paperbrace extract`.

    Args:
        pdf_dir: Folder containing PDFs to index.
        db: SQLite DB file path.
        commit_every: Commit after every N PDFs processed.
        verbose: Enable debug logging.

    Returns:
        None
    """
    setup_logging(verbose)
    conn = connect(db)
    init_db(conn)

    pdf_dir = pdf_dir.expanduser().resolve()
    logger.info("Scanning %s", pdf_dir)

    n = store.index_pdfs(conn, pdf_dir=pdf_dir, commit_every=commit_every)
    logger.info("Indexed %d PDFs → %s", n, db)


@app.command()
def list(
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        min=1,
        max=200,
        help="Max rows to display.",
    ),
    db: Path = typer.Option(
        Path(DEFAULT_DB),
        "--db",
        help="SQLite database path.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """
    List known papers (id + extraction status + path).

    Useful for discovering paper ids (for single-paper extraction/debugging).

    Args:
        limit: Maximum number of papers to display.
        db: SQLite DB file path.
        verbose: Enable debug logging.

    Returns:
        None
    """
    setup_logging(verbose)
    conn = connect(db)
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
    paper_id: Optional[int] = typer.Option(
        None,
        "--paper-id",
        min=1,
        help="Extract a single paper by id.",
    ),
    all_: bool = typer.Option(
        False,
        "--all",
        help="Extract all papers in the DB.",
    ),
    db: Path = typer.Option(
        Path(DEFAULT_DB),
        "--db",
        help="SQLite database path.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-extract even if up-to-date.",
    ),
    commit_every: int = typer.Option(
        5,
        "--commit-every",
        min=1,
        max=200,
        help="Commit frequency during --all extraction.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """
    Extract per-page text for one paper or all papers.

    Extraction stores page text in `pages` and builds the keyword index in
    `pages_fts` for fast searching. Up-to-date PDFs are skipped unless --force.

    Args:
        paper_id: Extract a single paper by id (mutually exclusive with --all).
        all_: Extract all papers (mutually exclusive with --paper-id).
        db: SQLite DB file path.
        force: Re-extract even if PDF mtime is unchanged.
        commit_every: Commit frequency in batch mode.
        verbose: Enable debug logging.

    Returns:
        None

    Raises:
        typer.BadParameter: If both/neither of --paper-id and --all are provided.
    """
    setup_logging(verbose)
    conn = connect(db)
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
    db: Path = typer.Option(Path(DEFAULT_DB), "--db"),
    yes: bool = typer.Option(False, "--yes", help="Confirm destructive purge."),
    vacuum: bool = typer.Option(False, "--vacuum", help="Reclaim space after purge (slower)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Delete ALL rows from the DB (papers/pages/pages_fts)."""
    setup_logging(verbose)
    if not yes:
        raise typer.BadParameter("Refusing to purge without --yes")

    conn = connect(db)
    init_db(conn)
    store.purge_db(conn, reset_ids=True, vacuum=vacuum)
    logger.info("Purged DB: %s", db)


@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help="FTS query (supports FTS syntax: AND/OR/NOT, quotes, prefix *).",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        min=1,
        max=50,
        help="Max results to display.",
    ),
    db: Path = typer.Option(
        Path(DEFAULT_DB),
        "--db",
        help="SQLite database path.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """
    Keyword search over extracted text using SQLite FTS5.

    Displays ranked results with highlighted snippets and filenames. Results are
    page-level: (paper_id, page_num).

    Args:
        query: FTS query string.
        limit: Maximum number of results to display.
        db: SQLite DB file path.
        verbose: Enable debug logging.

    Returns:
        None
    """
    setup_logging(verbose)
    conn = connect(db)
    init_db(conn)

    rows = store.search_pages(conn, query=query, limit=limit)
    # Expected row shape: (paper_id, page_num, path, snippet, score)

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
    db: Path = typer.Option(Path(DEFAULT_DB), "--db"),
    force: bool = typer.Option(False, "--force"),
    commit_every: int = typer.Option(5, "--commit-every", min=1, max=200),
    persist_dir: Path = typer.Option(Path("data/chroma"), "--persist-dir"),
    collection: str = typer.Option("paperbrace_pages", "--collection"),
    embedding_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--embedding-model"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Build/update the semantic vector index (Chroma) from extracted pages.
    """
    setup_logging(verbose)

    conn = connect(db)
    init_db(conn)

    from .retriever import ChromaPageRetriever  # local import to keep deps optional
    retr = ChromaPageRetriever(
        persist_dir=persist_dir.expanduser().resolve(),
        collection=collection,
        embedding_model=embedding_model,
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
    model: Optional[str] = typer.Option(
        None, "--model", help="HF model id (overrides config)."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", help="Path to paperbrace.toml (default: ./paperbrace.toml)."
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Write answer + evidence to Markdown."),
    db: Path = typer.Option(Path(DEFAULT_DB), "--db"),
    # Semantic retriever knobs (used when --retriever semantic|hybrid)
    persist_dir: Path = typer.Option(Path("data/chroma"), "--persist-dir", help="Chroma persistence directory."),
    collection: str = typer.Option("paperbrace_pages", "--collection", help="Chroma collection name."),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2", "--embedding-model", help="SentenceTransformers model for embeddings."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Answer a question using retrieved evidence + a local Hugging Face LLM.

    This command is LLM-only: if no model is configured/resolved, it exits with
    an error message. For pure keyword lookup without an LLM, use `paperbrace search`.

    Retrieval modes (via --retriever):
      - keyword: SQLite FTS5 over extracted pages (pages_fts), ranked by bm25().
                Natural-language questions are converted to a friendlier OR-based FTS query
                via `store.nl_to_fts_query`.
      - semantic: Chroma vector search over embedded pages. Requires running:
                `paperbrace embed --all ...` beforehand.
      - hybrid: merge keyword + semantic results, de-duplicating by (paper_id, page_num).

    Model resolution precedence:
      1) --model overrides everything
      2) config file [llm].model from paperbrace.toml (via --config or default ./paperbrace.toml)
      3) if still missing, the command exits (code=2)

    Args:
        question: Natural-language question to answer.
        k: Number of evidence pages (top-k) to retrieve and cite.
        retriever: Retrieval mode ("keyword", "semantic", "hybrid").
        model: Hugging Face model id to use for local inference (override).
        config: Optional path to a TOML config file.
        out: Optional path to write Markdown output (Answer + Evidence).
        db: SQLite database path.
        persist_dir: Chroma persistence directory (semantic/hybrid).
        collection: Chroma collection name (semantic/hybrid).
        embedding_model: SentenceTransformers embedding model (semantic/hybrid).
        verbose: Enable debug logging.

    Returns:
        None

    Raises:
        typer.Exit: if no model is configured/resolved.
        typer.BadParameter: if retriever mode is invalid.
    """
    setup_logging(verbose)
    conn = connect(db)
    init_db(conn)

    # Resolve model (LLM-only)
    cfg_file = load_config(config)
    resolved_model = model or cfg_file.llm_model
    if not resolved_model:
        logger.error(
            "No LLM model configured. Set [llm].model in paperbrace.toml or pass --model. "
            "For keyword search, use: paperbrace search \"...\""
        )
        raise typer.Exit(code=2)

    mode = (retriever or "").strip().lower()
    if mode not in {"keyword", "semantic", "hybrid"}:
        raise typer.BadParameter("Invalid --retriever. Use: keyword | semantic | hybrid")

    # Retrieve evidence
    rows: list[tuple[int, int, str, str, float]] = []

    if mode == "keyword":
        rows = store.get_evidence_pages(conn, query=question, limit=k)

    else:
        # Semantic retrieval (Chroma)
        from .retriever import ChromaPageRetriever  # keep optional deps lazy
        chroma = ChromaPageRetriever(
            persist_dir=persist_dir.expanduser().resolve(),
            collection=collection,
            embedding_model=embedding_model,
        )

        sem_hits = chroma.query_pages(question, k=k, where=None)
        sem_rows = [
            (h.paper_id, h.page_num, h.path, h.text, float(h.score))
            for h in sem_hits
        ]

        if mode == "semantic":
            rows = sem_rows

        else:  # hybrid
            # Keyword side: fetch a bit more to help overlap, then merge/dedupe down to k
            kw_rows = store.get_evidence_pages(conn, query=question, limit=max(k * 2, k))
            seen: set[tuple[int, int]] = set()
            merged: list[tuple[int, int, str, str, float]] = []

            # Simple merge strategy: semantic first (often higher precision), then keyword fill.
            for pid, pn, path, text, score in sem_rows + kw_rows:
                key = (int(pid), int(pn))
                if key in seen:
                    continue
                seen.add(key)
                merged.append((int(pid), int(pn), str(path), str(text), float(score)))
                if len(merged) >= k:
                    break

            rows = merged

        if not rows:
            logger.warning(
                "No semantic hits. Did you run: paperbrace embed --all --db %s ?", db
            )

    items = [
        Evidence(int(pid), int(pn), str(path), str(text), float(score))
        for pid, pn, path, text, score in rows
    ]

    # Always show compact evidence table
    table = Table(title=f"Evidence ({mode}) for: {question!r} (showing {len(items)})")
    table.add_column("#", justify="right")
    table.add_column("paper_id", justify="right")
    table.add_column("page", justify="right")
    table.add_column("file")
    for i, e in enumerate(items, start=1):
        table.add_row(
            str(i),
            str(e.paper_id),
            str(e.page_num),
            _cap_filename(Path(e.path).name, 30),
        )
    console.print(table)

    # If there's no evidence, don't hallucinate
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
        "If the evidence is insufficient, say: \"Not found in the provided papers.\""
    )

    blocks: list[str] = []
    for i, e in enumerate(items, start=1):
        excerpt = (e.text or "").strip()
        if len(excerpt) > 1200:
            excerpt = excerpt[:1200] + "…"
        blocks.append(
            f"[{i}] paper_id={e.paper_id} page={e.page_num} file={Path(e.path).name}\n{excerpt}"
        )

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
