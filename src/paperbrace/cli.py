# src/paperbrace/cli.py
from __future__ import annotations

from paperbrace import __version__

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

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
        "httpx",
        "httpcore",
        "hpack",
        "h2",
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


def _set_hf_offline(offline: bool) -> None:
    """
    Enforce zero-network for HF Hub / Transformers.
    Must be called before instantiating SentenceTransformer / Transformers models.
    """
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


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


# Row shape used by ask() / eval(): (source_id, page_num, chunk_id, char_start, char_end, fingerprint, path, text, distance, retrieval)
Row: TypeAlias = Tuple[int, int, Optional[int], Optional[int], Optional[int], Optional[str], str, str, float, str]


def _chunk_key(r: Row) -> tuple[int, int, Optional[int]]:
    sid, pn, cid, *_ = r
    return (int(sid), int(pn), cid)


def _page_key(r: Row) -> tuple[int, int]:
    sid, pn, *_ = r
    return (int(sid), int(pn))


def _backend_info(conn: Any, *, preferred_backend: str, base_collection: str) -> tuple[str, str, str]:
    """
    Return (backend, collection_name, distance_metric) from vector_indexes mapping.
    Fallback: assume cosine on base collection if never embedded / mapping missing.
    """
    info = store.get_vector_index(conn, backend=preferred_backend, collection_name=base_collection)
    if info:
        col, dist, *_ = info
        return preferred_backend, str(col), str(dist)
    return preferred_backend, base_collection, "cosine"


def _retrieve_rows(
    *,
    conn: Any,
    question: str,
    mode: str,
    k: int,
    backend: str,
    db_path: Path,
    chroma_db_path: Path,
    flat_index_dir: Path,
    collection: str,
    embedding_model: str,
    distance_cutoff: Optional[float],
) -> tuple[list_cmd[Row], Optional[str]]:
    """
    Run retrieval (keyword/semantic/hybrid) and return (rows, semantic_metric_used).

    semantic_metric_used is one of: "cosine" | "l2" | "ip" when semantic is involved; otherwise None.
    """
    b = (backend or "").strip().lower()
    if b not in {"auto", "chroma", "flat"}:
        raise typer.BadParameter("Invalid --backend. Use: auto | chroma | flat")

    m = (mode or "").strip().lower()
    if m not in {"keyword", "semantic", "hybrid"}:
        raise typer.BadParameter("Invalid --retriever. Use: keyword | semantic | hybrid")

    if m == "keyword":
        kw_rows_raw = store.get_evidence_pages(conn, query=question, limit=k)
        rows_kw: list_cmd[Row] = [
            (int(sid), int(pn), None, None, None, "", str(path), str(text), float(distance), "keyword")
            for sid, pn, path, text, distance in kw_rows_raw
        ]
        return rows_kw[:k], None

    # Resolve which embedded mapping to use (auto tries chroma mapping first, else flat mapping)
    if b == "auto":
        if store.get_vector_index(conn, backend="chroma", collection_name=collection):
            chosen_backend, chosen_collection, dist = _backend_info(conn, preferred_backend="chroma", base_collection=collection)
        else:
            chosen_backend, chosen_collection, dist = _backend_info(conn, preferred_backend="flat", base_collection=collection)
    else:
        chosen_backend, chosen_collection, dist = _backend_info(conn, preferred_backend=b, base_collection=collection)

    vec = make_retriever(
        backend=chosen_backend,
        chroma_db_path=chroma_db_path.expanduser().resolve(),
        flat_index_dir=flat_index_dir.expanduser().resolve(),
        collection=chosen_collection,  # uses embedded metric collection
        embedding_model=embedding_model,
        db_path=db_path,
        distance=dist,  # pass for flat validation / chroma creation
    )

    sem_hits = vec.query(question, k=k, where=None)

    if distance_cutoff is not None:
        sem_hits = [
            h
            for h in sem_hits
            if _cosine_equiv_distance(float(h.distance), dist) <= distance_cutoff
        ]

    sem_rows: list_cmd[Row] = [
        (
            int(h.source_id),
            int(h.page_num),
            _opt_int(getattr(h, "chunk_id", None)),
            _opt_int(getattr(h, "char_start", None)),
            _opt_int(getattr(h, "char_end", None)),
            getattr(h, "fingerprint", None),
            str(h.path),
            str(h.text),
            float(h.distance),  # raw semantic distance from backend
            "semantic",
        )
        for h in sem_hits
    ]

    if m == "semantic":
        # Defensive dedupe on chunk key
        seen: set[tuple[int, int, Optional[int]]] = set()
        out: list_cmd[Row] = []
        for r in sem_rows:
            ck = _chunk_key(r)
            if ck in seen:
                continue
            seen.add(ck)
            out.append(r)
            if len(out) >= k:
                break
        return out, dist

    # Hybrid: semantic first, keyword fill after (page-level)
    kw_rows_raw = store.get_evidence_pages(conn, query=question, limit=max(k * 2, k))
    kw_rows: list_cmd[Row] = [
        (int(sid), int(pn), None, None, None, "", str(path), str(text), float(distance), "keyword")
        for sid, pn, path, text, distance in kw_rows_raw
    ]

    merged: list_cmd[Row] = []
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
            ck = _chunk_key(r)  # (sid, pn, None)
            if ck in seen_chunks:
                continue
            seen_chunks.add(ck)
            merged.append(r)
            if len(merged) >= k:
                break

    # Final defensive dedupe
    seen2: set[tuple[int, int, Optional[int]]] = set()
    out2: list_cmd[Row] = []
    for r in merged:
        ck = _chunk_key(r)
        if ck in seen2:
            continue
        seen2.add(ck)
        out2.append(r)
    return out2[:k], dist


def _load_jsonl(path: Path) -> list_cmd[dict[str, Any]]:
    out: list_cmd[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                out.append(json.loads(s))
            except Exception as e:
                raise ValueError(f"Bad JSONL at {path}:{ln}: {e}\nLine: {s[:200]}") from e
    return out



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
    """
    setup_logging(verbose)
    conn = connect(db_path)
    init_db(conn)

    pdf_dir = pdf_dir.expanduser().resolve()
    logger.info("Scanning %s", pdf_dir)

    n = store.index_pdfs(conn, pdf_dir=pdf_dir, commit_every=commit_every)
    logger.info("Indexed %d PDFs → %s", n, db_path)


@app.command("list")
def list_cmd(
    limit: int = typer.Option(20, "--limit", "-n", min=1, max=200, help="Max rows to display."),
    db_path: Path = typer.Option(DEFAULT_DB, "--db-path", help="SQLite database path."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """List known sources (id + extraction status + path)."""
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
    """Extract per-page text for one paper or all sources."""
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
    """Delete ALL rows from the SQLite DB (sources/pages/pages_fts)."""
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
    """Keyword search over extracted text using SQLite FTS5."""
    setup_logging(verbose)
    conn = connect(db_path)
    init_db(conn)

    rows = store.search_pages(conn, query=query, limit=limit)
    table = Table(title=f"Search: {query!r} (showing {len(rows)})")
    table.add_column("source_id", justify="right")
    table.add_column("page", justify="right")
    table.add_column("file")
    table.add_column("snippet")

    for source_id, page_num, path, snip, _distance in rows:
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
    offline: bool = typer.Option(
        False,
        "--offline/--no-offline",
        help="Force HF/Transformers offline mode (no network). Requires models already cached.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Build/update the semantic vector index from extracted pages."""
    setup_logging(verbose)
    _set_hf_offline(offline)
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


@app.command("eval")
def eval(
    cases: Path = typer.Option(Path("eval/cases.jsonl"), "--cases", help="JSONL file of eval cases."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write JSON report to this path."),
    show: int = typer.Option(30, "--show", help="Show first N rows in summary table."),
    db_path: Path = typer.Option(DEFAULT_DB, "--db-path", help="SQLite database path (sources/pages)."),
    backend: str = typer.Option("auto", "--backend", help="Default backend for semantic/hybrid: auto | chroma | flat"),
    chroma_db_path: Path = typer.Option(DEFAULT_CHROMA_DIR, "--chroma-db-path", help="Chroma persistence directory."),
    flat_index_dir: Path = typer.Option(DEFAULT_FLAT_DIR, "--flat-index-dir", help="Flat index directory."),
    collection: str = typer.Option("paperbrace_pages", "--collection", help="Collection name (backend-specific)."),
    embedding_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--embedding-model"),
    k: int = typer.Option(8, "--k", min=1, max=30, help="Default k (can be overridden per case)."),
    retriever: str = typer.Option("semantic", "--retriever", help="Default mode: keyword|semantic|hybrid (override per case)."),
    distance_cutoff: Optional[float] = typer.Option(
        None,
        "--distance-cutoff",
        help="Default semantic cutoff (cosine-equivalent). Can be overridden per case.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline/--no-offline",
        help="Force HF/Transformers offline mode (no network). Requires models already cached.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Retrieval evaluation harness (v0): page-level hit@k and precision@k.

    Case format (JSONL), minimal:
      {"id":"vap-1","query":"...","gold_pages":[{"source_id":1,"page_num":10}]}

    Optional per-case overrides:
      mode/retriever, k, backend, distance_cutoff
    """
    setup_logging(verbose)
    _set_hf_offline(offline)
    conn = connect(db_path)
    init_db(conn)

    cases = cases.expanduser().resolve()
    if not cases.exists():
        raise typer.BadParameter(f"Cases file not found: {cases}")
    logger.info("Eval cases path: %s", cases)
    items = _load_jsonl(cases)

    results: list_cmd[dict[str, Any]] = []
    sum_hit = 0
    sum_prec = 0.0
    n_labeled = 0

    for c in items:
        cid = str(c.get("id", ""))
        q = str(c.get("query") or c.get("question") or "")
        if not q:
            continue

        mode_c = str(c.get("mode", c.get("retriever", retriever))).strip().lower()
        k_c = int(c.get("k", k))
        backend_c = str(c.get("backend", backend)).strip().lower()
        cutoff_c = c.get("distance_cutoff", distance_cutoff)

        gold_pages = c.get("gold_pages", [])
        gold_set: set[tuple[int, int]] = set()
        for g in gold_pages or []:
            page_num = int(g["page_num"])
            if "source_id" in g and g["source_id"] is not None:
                source_id = int(g["source_id"])
            else:
                # new format: {"file_name": "...pdf", "page_num": N}
                source_id = store.get_source_id_by_filename(conn, str(g["file_name"]))
            gold_set.add((source_id, page_num))

        rows, metric_used = _retrieve_rows(
            conn=conn,
            question=q,
            mode=mode_c,
            k=k_c,
            backend=backend_c,
            db_path=db_path,
            chroma_db_path=chroma_db_path,
            flat_index_dir=flat_index_dir,
            collection=collection,
            embedding_model=embedding_model,
            distance_cutoff=cutoff_c,
        )

        got_pages = {_page_key(r) for r in rows}

        hit_at_k: Optional[int] = None
        precision_at_k: Optional[float] = None
        if gold_set:
            n_labeled += 1
            hit_at_k = 1 if (got_pages & gold_set) else 0
            precision_at_k = len(got_pages & gold_set) / max(len(got_pages), 1)
            sum_hit += hit_at_k
            sum_prec += precision_at_k

        results.append(
            {
                "id": cid,
                "query": q,
                "mode": mode_c,
                "k": k_c,
                "backend": backend_c,
                "distance_cutoff": cutoff_c,
                "semantic_metric_used": metric_used,
                "gold_pages": sorted(gold_set),
                "got_pages": sorted(got_pages),
                "hit_at_k": hit_at_k,
                "precision_at_k": precision_at_k,
            }
        )

    table = Table(title=f"Eval: {cases.name} (cases={len(results)}, labeled={n_labeled})")
    table.add_column("id")
    table.add_column("mode")
    table.add_column("k", justify="right")
    table.add_column("hit@k", justify="right")
    table.add_column("prec@k", justify="right")
    for r in results[: max(show, 0)]:
        table.add_row(
            r["id"],
            r["mode"],
            str(r["k"]),
            "-" if r["hit_at_k"] is None else str(r["hit_at_k"]),
            "-" if r["precision_at_k"] is None else f'{r["precision_at_k"]:.2f}',
        )
    console.print(table)

    summary = {
        "cases_file": str(cases),
        "n_cases": len(results),
        "n_labeled": n_labeled,
        "hit_at_k_avg": (sum_hit / n_labeled) if n_labeled else None,
        "precision_at_k_avg": (sum_prec / n_labeled) if n_labeled else None,
        "results": results,
    }

    if out:
        out = out.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Wrote → %s", out)


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
    flat_index_dir: Path = typer.Option(DEFAULT_FLAT_DIR, "--flat-index-dir", help="Flat index directory."),
    collection: str = typer.Option("paperbrace_pages", "--collection", help="Collection name (backend-specific)."),
    embedding_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--embedding-model"),
    distance_cutoff: Optional[float] = typer.Option(
        None,
        "--distance-cutoff",
        help=(
            "Discard semantic hits with cosine-equivalent distance > this value (lower is better). "
            "Default: None (no cutoff). Balanced: 0.60; Strict: 0.45–0.50; Loose: 0.70–0.80."
        ),
    ),
    debug: bool = typer.Option(
        False,
        "--debug/--no-debug",
        help="Show extra debugging details (e.g., raw bm25, raw semantic distances, backend/metric info).",
    ),
    offline: bool = typer.Option(
        False,
        "--offline/--no-offline",
        help="Force HF/Transformers offline mode (no network). Requires models already cached.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """
    Answer a question using retrieved evidence + a local Hugging Face LLM.

    For pure keyword lookup without an LLM, use `paperbrace search`.
    """
    setup_logging(verbose)
    _set_hf_offline(offline)
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

    try:
        rows, _metric = _retrieve_rows(
            conn=conn,
            question=question,
            mode=mode,
            k=k,
            backend=backend,
            db_path=db_path,
            chroma_db_path=chroma_db_path,
            flat_index_dir=flat_index_dir,
            collection=collection,
            embedding_model=embedding_model,
            distance_cutoff=distance_cutoff,
        )
    except Exception as e:
        logger.error("%s", e)
        raise typer.Exit(code=2)

    items = [
        Evidence(
            source_id=int(sid),
            page_num=int(pn),
            path=str(path),
            text=str(text),
            distance=float(dist),
            chunk_id=_opt_int(cid),
            char_start=_opt_int(start),
            char_end=_opt_int(end),
            fingerprint=fing,
            retrieval=ret,
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
            str(e.chunk_id + 1) if e.chunk_id is not None else _fmt_null(e.chunk_id),  # 1-based for user
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

    blocks: list_cmd[str] = []
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


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    """Paperbrace CLI."""
    return
