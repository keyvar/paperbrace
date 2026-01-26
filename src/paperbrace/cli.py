# src/paperbrace/cli.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, TypeAlias, Any

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
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    # Silence noisy third-party libs (keep your own logs)
    noisy_level = logging.WARNING if verbose else logging.ERROR
    for name in (
        "sentence_transformers",
        "transformers",
        "torch",
        "chromadb",
        "huggingface_hub",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(noisy_level)

    # Hugging Face specific verbosity control (if installed)
    try:
        from transformers.utils import logging as hf_logging  # type: ignore

        hf_logging.set_verbosity_error()
        hf_logging.disable_default_handler()
        hf_logging.enable_propagation(False)
    except Exception:
        pass



def _opt_int(x: Any) -> int | None:
    """Gives you a proper int | None return type."""
    return None if x is None else int(x)


def _fmt_null(x: object | None, na: str = "—") -> str:
    return na if x is None else str(x)


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


def _cosine_equiv_distance(d: float, metric: str) -> float:
    """
    Convert semantic distance to a cosine-equivalent scale (assuming embeddings+query are L2-normalized).
    - cosine: already 1 - dot
    - ip (your convention): already 1 - dot
    - l2 (Chroma 'l2' is squared L2): l2_sq = 2 * (1 - dot)  =>  (1 - dot) = l2_sq / 2
    """
    m = (metric or "cosine").strip().lower()
    return (d / 2.0) if m == "l2" else d


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
        help="SQLite database path (sources/pages/FTS).",
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

    Records file-level metadata (path, mtime, size) in the `sources` table.
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
    List known sources (id + extraction status + path).

    Args:
        limit: Maximum number of sources to display.
        db_path: SQLite DB file path.
        verbose: Enable debug logging.

    Returns:
        None
    """
    setup_logging(verbose)
    conn = connect(db_path)
    init_db(conn)

    rows = store.list_sources(conn, limit=limit)

    table = Table(title=f"Sources (showing {len(rows)})")
    table.add_column("id", justify="right")
    table.add_column("extracted?", justify="center")
    table.add_column("path")

    for sid, path, extracted_mtime in rows:
        table.add_row(str(sid), "yes" if extracted_mtime else "no", path)

    console.print(table)


@app.command()
def extract(
    source_id: Optional[int] = typer.Option(None, "--paper-id", min=1, help="Extract a single paper by id."),
    all_: bool = typer.Option(False, "--all", help="Extract all sources in the DB."),
    db_path: Path = typer.Option(DEFAULT_DB, "--db-path", help="SQLite database path."),
    force: bool = typer.Option(False, "--force", help="Re-extract even if up-to-date."),
    commit_every: int = typer.Option(5, "--commit-every", min=1, max=200, help="Commit frequency during --all."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Extract per-page text for one paper or all sources.

    Stores page text in `pages` and updates `pages_fts` for keyword search.

    Args:
        source_id: Extract a single paper by id (mutually exclusive with --all).
        all_: Extract all sources (mutually exclusive with --paper-id).
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

    if all_ == (source_id is not None):
        raise typer.BadParameter("Use either --paper-id ID or --all (not both).")

    if all_:
        extracted, skipped = store.extract_all(conn, force=force, commit_every=commit_every)
        logger.info("Done. Extracted=%d Skipped=%d", extracted, skipped)
        return

    assert source_id is not None
    store.extract_one(conn, source_id=source_id, force=force)
    conn.commit()


@app.command()
def purge(
    db_path: Path = typer.Option(DEFAULT_DB, "--db-path", help="SQLite database path."),
    yes: bool = typer.Option(False, "--yes", help="Confirm destructive purge."),
    vacuum: bool = typer.Option(False, "--vacuum", help="Reclaim space after purge (slower)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Delete ALL rows from the SQLite DB (sources/pages/pages_fts).

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
    table.add_column("source_id", justify="right")
    table.add_column("page", justify="right")
    table.add_column("file")
    table.add_column("snippet")

    for source_id, page_num, path, snip, distance in rows:
        fname = _cap_filename(Path(path).name, 30)
        table.add_row(str(source_id), str(page_num), fname, snip)

    console.print(table)


@app.command()
def embed(
    source_id: int = typer.Option(0, "--paper-id", min=0, help="Embed one paper id (0 means use --all)."),
    all_: bool = typer.Option(False, "--all", help="Embed all extracted sources."),
    db_path: Path = typer.Option(DEFAULT_DB, "--db-path", help="SQLite database path (source pages)."),
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="Vector backend: auto | chroma | flat",
    ),
    force: bool = typer.Option(False, "--force", help="Re-embed even if up-to-date."),
    commit_every: int = typer.Option(5, "--commit-every", min=1, max=200),
    chroma_db_path: Path = typer.Option(DEFAULT_CHROMA_DIR, "--chroma-db-path", help="Chroma persistence directory."),
    flat_index_dir: Path = typer.Option(DEFAULT_FLAT_DIR, "--flat-index-dir", help="Flat index directory."),
    collection: str = typer.Option("paperbrace_pages", "--collection", help="Collection/index name (backend-specific)."),
    embedding_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--embedding-model"),
    distance: str = typer.Option(
        "cosine",
        "--distance",
        help="Embedding-time distance metric for the vector index: cosine | l2 | ip",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Build/update the semantic vector index from extracted pages.

    Requires that PDFs are already extracted into `pages` (run `paperbrace extract` first).

    Args:
        source_id: Single paper id to embed (0 means use --all).
        all_: Embed all extracted sources.
        db_path: SQLite DB file path (sources/pages).
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

    b = (backend or "").strip().lower()
    if b not in {"chroma", "flat"}:
        raise typer.BadParameter("Invalid --backend. Use: chroma | flat")

    dist = (distance or "").strip().lower()
    if dist not in {"cosine", "l2", "ip"}:
        raise typer.BadParameter("Invalid --distance. Use: cosine | l2 | ip")

    retr = make_retriever(
        backend=backend,
        chroma_db_path=chroma_db_path.expanduser().resolve(),
        flat_index_dir=flat_index_dir.expanduser().resolve(),
        collection=collection,
        embedding_model=embedding_model,
        db_path=db_path,
        distance=distance,
    )

    if all_ == (source_id != 0):
        raise typer.BadParameter("Use either --paper-id ID or --all (not both).")

    if all_:
        embedded, skipped = store.embed_all(conn, retriever=retr, force=force, commit_every=commit_every)
        logger.info("Done. Embedded=%d Skipped=%d", embedded, skipped)
    else:
        was_skipped, n_chunks = store.embed_one(conn, retriever=retr, source_id=source_id, force=force)
        if was_skipped:
            logger.info("Skipped embedding source_id=%d", source_id)
        else:
            logger.info("Embedded source_id=%d (%d chunks)", source_id, n_chunks)

    # record mapping so ask can find it
    store.set_vector_index(
        conn,
        backend=b,
        distance=dist,
        collection_name=collection,
        embedding_model=embedding_model,
    )

    conn.commit()


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
    db_path: Path = typer.Option(DEFAULT_DB, "--db-path", help="SQLite database path (sources/pages)."),
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="Vector backend for semantic/hybrid: auto | chroma | flat",
    ),
    chroma_db_path: Path = typer.Option(DEFAULT_CHROMA_DIR, "--chroma-db-path", help="Chroma persistence directory."),
    flat_index_dir: Path = typer.Option(Path(".paperbrace/flat"), "--flat-index-dir", help="Flat index directory."),
    collection: str = typer.Option("paperbrace_pages", "--collection", help="Collection name (backend-specific)."),
    embedding_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--embedding-model"),
    distance_cutoff: Optional[float] = typer.Option(
        None,
        "--distance-cutoff",
        help="Discard semantic hits with cosine-equivalent distance > this value (lower is better)." \
        "Default: None (no cuttoff); balanced: 0.60; Strict (high precision): 0.45–0.50; Loose (high recall): 0.70–0.80",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
        debug: bool = typer.Option(
        False,
        "--debug/--no-debug",
        help="Show extra debugging details (e.g., raw bm25, raw semantic distances, backend/metric info).",
    ),
) -> None:
    """
    Answer a question using retrieved evidence + a local Hugging Face LLM.

    For pure keyword lookup without an LLM, use `paperbrace search`.

    Retrieval modes:
      - keyword: SQLite FTS5 over extracted pages (pages_fts), ranked by bm25().
      - semantic: vector search over embedded pages (requires `paperbrace embed`).
      - hybrid: semantic results first, then keyword fill; de-duped by (source_id, page_num).

    Vector backends (for semantic/hybrid):
      - chroma: persistent Chroma DB (requires `paperbrace[chroma]`)
      - flat: local NumPy index (requires `paperbrace[flat]`)
      - auto: prefer chroma if installed, else flat if installed
    """
    setup_logging(verbose)
    conn = connect(db_path)
    init_db(conn)

    b = (backend or "").strip().lower()
    if b not in {"auto", "chroma", "flat"}:
        raise typer.BadParameter("Invalid --backend. Use: auto | chroma | flat")

    # If user asked auto, make_retriever will pick the backend,
    # but we still need to know which "embedded mapping" to use.
    # Strategy: when auto is selected, try chroma mapping first, then flat mapping.
    def _backend_info(preferred_backend: str) -> tuple[str, str, str]:
        info = store.get_vector_index(conn, backend=preferred_backend, collection_name=collection)
        if info:
            col, dist, *_ = info
            return preferred_backend, col, dist
        # fallback: assume cosine on base collection if never embedded
        return preferred_backend, collection, "cosine"

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

    # Row shape used by ask(): (source_id, page_num, chunk_id, char_start, chat_end, fingerprint, path, text, distance, retrieval)
    Row: TypeAlias = Tuple[int, int, Optional[int], Optional[int], Optional[int], Optional[str], str, str, float, str]
    
    def _chunk_key(r: Row) -> tuple[int, int, Optional[int]]:
        sid, pn, cid, *_ = r
        return (sid, pn, cid)

    def _page_key(r: Row) -> tuple[int, int]:
        sid, pn, *_ = r
        return (sid, pn)

    if mode == "keyword":
        kw_rows_raw = store.get_evidence_pages(conn, query=question, limit=k)
        # attach chunk_id=None for keyword/page-level evidence
        rows: list[Row] = [
            (int(sid), int(pn), None, None, None, "", str(path), str(text), float(distance), "keyword",)
            for sid, pn, path, text, distance in kw_rows_raw
        ]

    else:
        try:
            if b == "auto":
                # Try chroma mapping; if none, try flat
                if store.get_vector_index(conn, backend="chroma", collection_name=collection):
                    chosen_backend, chosen_collection, dist = _backend_info("chroma")
                else:
                    chosen_backend, chosen_collection, dist = _backend_info("flat")
            else: # It’s for when b is explicitly "chroma" or "flat"
                chosen_backend, chosen_collection, dist = _backend_info(b)

            vec = make_retriever(
                backend=chosen_backend,
                chroma_db_path=chroma_db_path.expanduser().resolve(),
                flat_index_dir=flat_index_dir.expanduser().resolve(),
                collection=chosen_collection,           # <-- uses embedded metric collection
                embedding_model=embedding_model,     # you can also choose to enforce the stored one
                db_path=db_path,
                distance=dist,                       # <-- pass for flat validation / chroma creation
            )    
        except Exception as e:
            logger.error("%s", e)
            raise typer.Exit(code=2)

        sem_hits = vec.query(question, k=k, where=None)

        if distance_cutoff is not None:
            sem_hits = [
                h for h in sem_hits
                if _cosine_equiv_distance(float(h.distance), dist) <= distance_cutoff
            ]

        # semantic rows have chunk_id (and maybe offsets)
        sem_rows: list[Row] = [
            (
                int(h.source_id),
                int(h.page_num),
                # Following exist only in semntic retriever. Avoid crashing for keyword retriever
                _opt_int(getattr(h, "chunk_id", None)),   
                _opt_int(getattr(h, "char_start", None)),
                _opt_int(getattr(h, "char_end", None)),
                getattr(h, "fingerprint", None),
                str(h.path),
                str(h.text),
                float(h.distance),
                "semantic",
            )
            for h in sem_hits
        ]

        if mode == "semantic":
            rows = sem_rows

        else:
            # Hybrid: Append keyword fill to AFTER Semantic result. Keyword search is page-level (chunk_id=None).
            kw_rows_raw = store.get_evidence_pages(conn, query=question, limit=max(k * 2, k))
            kw_rows: list[Row] = [
                (int(sid), int(pn), None, None, None, "", str(path), str(text), float(distance), "keyword",)
                for sid, pn, path, text, distance in kw_rows_raw
            ]

            merged: list[Row] = []
            seen_chunks: set[tuple[int, int, Optional[int]]] = set()
            pages_with_semantic: set[tuple[int, int]] = set()

            # 1) semantic first: dedupe by chunk key, also remember pages covered
            for r in sem_rows:
                ck = _chunk_key(r)
                if ck in seen_chunks:
                    continue
                seen_chunks.add(ck)
                pages_with_semantic.add(_page_key(r))
                merged.append(r)
                if len(merged) >= k:
                    break

            # 2) keyword fill: only add pages not already covered by semantic
            if len(merged) < k:
                for r in kw_rows:
                    pk = _page_key(r)
                    if pk in pages_with_semantic:
                        continue
                    # page-level evidence uses chunk_id=None; still dedupe if repeated
                    ck = _chunk_key(r)
                    if ck in seen_chunks:
                        continue
                    seen_chunks.add(ck)
                    merged.append(r)
                    if len(merged) >= k:
                        break

            rows = merged

    # Final dedupe (defensive): keep all semantic chunks distinct; keyword pages distinct.
    seen_chunks: set[tuple[int, int, Optional[int]]] = set()
    deduped: list[Row] = []
    for r in rows:
        ck = _chunk_key(r)
        if ck in seen_chunks:
            continue
        seen_chunks.add(ck)
        deduped.append(r)
    rows = deduped[:k]

    items = [
        Evidence(
            source_id=int(sid),
            page_num=int(pn),
            path=str(path),
            text=str(text),
            distance=float(dist),
            chunk_id=_opt_int(cid),
            char_start=_opt_int(start) ,
            char_end=_opt_int(end),
            fingerprint=fing,
            retrieval=ret
        )
        for sid, pn, cid, start, end, fing, path, text, dist, ret in rows
    ]

    table = Table(title=f"Evidence ({mode}) for: {question!r} (showing {len(items)})")
    table.add_column("E", justify="right")
    table.add_column("source_id", justify="right")
    table.add_column("page", justify="right")
    table.add_column("chunk", justify="right")
    table.add_column("char start", justify="right")
    table.add_column("char end", justify="right")
    table.add_column("distance", justify="right")
    table.add_column("retrieval", justify="right")
    table.add_column("file")
    for i, e in enumerate(items, start=1):
        table.add_row(
            f"E{i}", 
            str(e.source_id), 
            str(e.page_num), 
            str(e.chunk_id + 1) if e.chunk_id is not None else _fmt_null(e.chunk_id), # make chunk_id 1 based for user
            _fmt_null(e.char_start), 
            _fmt_null(e.char_end), 
            f"{e.distance:.4f}" if (debug or e.retrieval != "keyword") else "-",
            e.retrieval,
            _cap_filename(Path(e.path).name, 30),
        )
    console.print(table)

    if not items:
        answer = "Not found in the provided sources."
        console.print(answer)
        if out:
            out.write_text(to_markdown(question, items, answer=answer), encoding="utf-8")
            logger.info("Wrote → %s", out)
        return

    system = (
        "You are Paperbrace. Answer ONLY using the provided evidence. "
        "Cite claims using evidence labels like E1, E2, ... only. "
        "Do NOT cite bracketed references like [10] that appear inside the PDF text. "
        "If evidence is insufficient, say: \"Not found in the provided sources.\""
    )

    blocks: list[str] = []
    for i, e in enumerate(items, start=1):
        label = f"E{i}"
        excerpt = (e.text or "").strip()
        if len(excerpt) > 1200:
            excerpt = excerpt[:1200] + "…"
        blocks.append(
            f"{label} source_id={e.source_id} page={e.page_num} file={Path(e.path).name}\n{excerpt}"
    )

    user = (
        f"Question: {question}\n\n"
        "Evidence:\n\n"
        + "\n\n---\n\n".join(blocks)
        + "\n\nWrite a concise answer with citations like E1, E2 (only these evidence labels).."
    )

    cfg = LLMConfig(
        model=resolved_model,
        max_new_tokens=cfg_file.llm_max_new_tokens,
        temperature=cfg_file.llm_temperature,
    )
    answer = llm_generate(system=system, user=user, cfg=cfg)

    console.print(answer.strip())

    if out:
        out.write_text(to_markdown(question, items, answer=answer), encoding="utf-8")
        logger.info("Wrote → %s", out)



if __name__ == "__main__":
    app()
