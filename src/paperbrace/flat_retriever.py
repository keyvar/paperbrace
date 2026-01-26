from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from paperbrace import store
from paperbrace.retriever import ChunkForIndex, RetrievedChunk
from paperbrace.vector_ops import l2_normalize, Distance

logger = logging.getLogger("paperbrace")

# ---------- id helpers ----------

_CHUNK_MULT = 10_000  # max chunks per page assumed < 10k
_PAGE_MULT = 100_000  # assumes page_num < 100k


def _page_key(source_id: int, page_num: int) -> np.int64:
    """Stable int64 key for (source_id, page_num)."""
    return np.int64(int(source_id) * _PAGE_MULT + int(page_num))


def _fingerprint_u64(s: str) -> np.uint64:
    """
    Deterministic 64-bit fingerprint (numeric, NPZ-friendly).

    We store numeric fingerprints so the .npz can be loaded with allow_pickle=False.
    """
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(h, "little", signed=False))


# ---------- numpy scalar helpers ----------


def _as_int_scalar(x: Any) -> int:
    """Convert numpy scalar/0d array/python int to int."""
    a = np.asarray(x)
    if a.shape == ():
        return int(a.item())
    if a.size == 1:
        return int(a.reshape(()).item())
    raise TypeError(f"Expected scalar int, got shape={a.shape} dtype={a.dtype}")


def _as_str_scalar(x: Any) -> str:
    """Convert numpy scalar/0d array/python str to str."""
    a = np.asarray(x)
    if a.shape == ():
        return str(a.item())
    if a.size == 1:
        return str(a.reshape(()).item())
    raise TypeError(f"Expected scalar str, got shape={a.shape} dtype={a.dtype}")


@dataclass(frozen=True)
class _IndexData:
    ids: np.ndarray        # int64, shape (N,)
    page_key: np.ndarray   # int64, shape (N,)  (source_id/page_num)
    emb: np.ndarray        # float32, shape (N, dim), L2-normalized
    source_id: np.ndarray   # int32, shape (N,)
    page_num: np.ndarray   # int32, shape (N,)
    chunk_id: np.ndarray   # int32, shape (N,)
    char_start: np.ndarray # int32, shape (N,)
    char_end: np.ndarray   # int32, shape (N,)
    fp64: np.ndarray       # uint64, shape (N,)


class FlatRetriever:
    """
    Flat (NumPy) vector retriever with SQLite FTS5 prefilter (chunk-aware).

    Storage:
      - Writes a single .npz file per collection under index_dir.
      - Stores ONLY numeric arrays (no text/path arrays) so allow_pickle=False works.

    Query:
      1) Prefilter with SQLite FTS5 (pages_fts) to get candidate PAGES (up to `prefilter_k_pages`)
         (converted from NL to OR-style via store.nl_to_fts_query).
      2) distance only CHUNKS that belong to those pages by cosine distance (1 - cosine_sim).
      3) Fetch page text/path from SQLite pages+pagers tables, slice chunk by (char_start, char_end),
         return chunk text in RetrievedChunk.text.

    Designed as a robust Windows-friendly fallback when Chroma is hard to install.
    """

    def __init__(
        self,
        *,
        index_dir: Path,
        collection: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sqlite_db_path: Path,
        prefilter_k_pages: int = 5000,
        distance: str = "cosine",
    ) -> None:
        self.index_dir = index_dir.expanduser().resolve()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.collection = collection
        self.embedding_model = embedding_model
        self.sqlite_db_path = sqlite_db_path.expanduser().resolve()
        self.prefilter_k_pages = int(prefilter_k_pages)
        self.distance = (distance or "cosine").strip().lower()
        if self.distance not in {"cosine", "l2", "ip"}:
            raise ValueError("distance must be one of: cosine | l2 | ip")

        self._st = SentenceTransformer(embedding_model)
        self.dim = int(self._st.get_sentence_embedding_dimension())

        self._path = self.index_dir / f"{self.collection}.npz"

        # Create empty index if missing
        if not self._path.exists():
            self._save(
                _IndexData(
                    ids=np.zeros((0,), dtype=np.int64),
                    page_key=np.zeros((0,), dtype=np.int64),
                    emb=np.zeros((0, self.dim), dtype=np.float32),
                    source_id=np.zeros((0,), dtype=np.int32),
                    page_num=np.zeros((0,), dtype=np.int32),
                    chunk_id=np.zeros((0,), dtype=np.int32),
                    char_start=np.zeros((0,), dtype=np.int32),
                    char_end=np.zeros((0,), dtype=np.int32),
                    fp64=np.zeros((0,), dtype=np.uint64),
                )
            )

    # ---------- persistence ----------

    def _load(self) -> _IndexData:
        """
        Load index arrays + metadata.

        Back-compat:
          - If old .npz contains object arrays, retry with allow_pickle=True
            (but you should rebuild with --force afterwards).
          - If old .npz is page-based (no chunk fields), we upgrade on-the-fly
            by creating chunk_id=0 and offsets spanning the full page. (Still works.)
        """
        try:
            z = np.load(self._path, allow_pickle=False)
        except ValueError as e:
            if "Object arrays cannot be loaded" in str(e):
                logger.warning(
                    "Flat index contains object arrays; reloading with allow_pickle=True. "
                    "Rebuild soon with: paperbrace embed --all --backend flat --force"
                )
                z = np.load(self._path, allow_pickle=True)
            else:
                raise

        dim = _as_int_scalar(z.get("dim", np.array(self.dim, dtype=np.int32)))
        model = _as_str_scalar(z.get("embedding_model", np.array(self.embedding_model, dtype="U")))

        if dim != self.dim:
            raise ValueError(f"Flat index dim={dim} but model produces dim={self.dim}. Rebuild flat index.")
        if model != self.embedding_model:
            raise ValueError(
                f"Flat index model={model!r} but configured {self.embedding_model!r}. Rebuild flat index."
            )

        ids = np.asarray(z["ids"]).astype(np.int64, copy=False)
        emb = np.asarray(z["emb"]).astype(np.float32, copy=False)
        source_id = np.asarray(z["source_id"]).astype(np.int32, copy=False)
        page_num = np.asarray(z["page_num"]).astype(np.int32, copy=False)

        # New chunk-aware fields (fallbacks for older files)
        chunk_id = np.asarray(z.get("chunk_id", np.zeros_like(page_num))).astype(np.int32, copy=False)

        # offsets: if missing, treat as whole page chunk
        char_start = np.asarray(z.get("char_start", np.zeros_like(page_num))).astype(np.int32, copy=False)
        char_end = np.asarray(z.get("char_end", np.zeros_like(page_num))).astype(np.int32, copy=False)

        # fingerprint numeric: if missing, zeros
        fp64 = np.asarray(z.get("fp64", np.zeros((ids.shape[0],), dtype=np.uint64))).astype(np.uint64, copy=False)

        # precomputed page_key for fast filtering
        page_key = np.asarray(
            z.get("page_key", (source_id.astype(np.int64) * _PAGE_MULT + page_num.astype(np.int64))),
        ).astype(np.int64, copy=False)

        if emb.size:
            emb = l2_normalize(emb)

        # Sanity lengths
        n = ids.shape[0]
        for name, arr in [
            ("page_key", page_key),
            ("source_id", source_id),
            ("page_num", page_num),
            ("chunk_id", chunk_id),
            ("char_start", char_start),
            ("char_end", char_end),
            ("fp64", fp64),
        ]:
            if arr.shape[0] != n:
                raise ValueError(f"Corrupt flat index: {name} has n={arr.shape[0]} but ids has n={n}")

        return _IndexData(
            ids=ids,
            page_key=page_key,
            emb=emb,
            source_id=source_id,
            page_num=page_num,
            chunk_id=chunk_id,
            char_start=char_start,
            char_end=char_end,
            fp64=fp64,
        )

    def _save(self, data: _IndexData) -> None:
        """Save numeric arrays + scalar metadata (no object arrays)."""
        np.savez_compressed(
            self._path,
            ids=data.ids.astype(np.int64, copy=False),
            page_key=data.page_key.astype(np.int64, copy=False),
            emb=data.emb.astype(np.float32, copy=False),
            source_id=data.source_id.astype(np.int32, copy=False),
            page_num=data.page_num.astype(np.int32, copy=False),
            chunk_id=data.chunk_id.astype(np.int32, copy=False),
            char_start=data.char_start.astype(np.int32, copy=False),
            char_end=data.char_end.astype(np.int32, copy=False),
            fp64=data.fp64.astype(np.uint64, copy=False),
            dim=np.array(self.dim, dtype=np.int32),
            embedding_model=np.array(self.embedding_model, dtype="U"),
        )

    # ---------- protocol methods ----------

    def upsert_source(self, pages: Sequence[ChunkForIndex], **kwargs: Any) -> None:
        """
        Embed and upsert CHUNKS into the flat index.

        Expects ChunkForIndex to be chunk-aware:
          - source_id, page_num, path, text (text = chunk text)
          - chunk_id (int), char_start (int), char_end (int), fingerprint (str)

        If those extra fields are missing, we treat each entry as chunk_id=0 with
        offsets 0..len(text) and fingerprint="".
        """
        if not pages:
            return

        batch_size = int(kwargs.get("batch_size", 64))
        show_pb = bool(kwargs.get("show_progress_bar", False))

        texts = [p.text for p in pages]
        new_emb = self._st.encode(texts, batch_size=batch_size, show_progress_bar=show_pb)
        new_emb = l2_normalize(new_emb)

        new_paper = np.asarray([int(p.source_id) for p in pages], dtype=np.int32)
        new_page = np.asarray([int(p.page_num) for p in pages], dtype=np.int32)

        new_chunk = np.asarray([int(getattr(p, "chunk_id", 0)) for p in pages], dtype=np.int32)
        new_start = np.asarray([int(getattr(p, "char_start", 0)) for p in pages], dtype=np.int32)
        new_end = np.asarray(
            [
                int(getattr(p, "char_end", len(p.text) if p.text is not None else 0))
                for p in pages
            ],
            dtype=np.int32,
        )

        # fingerprint -> numeric
        new_fp = np.asarray(
            [_fingerprint_u64(str(getattr(p, "fingerprint", ""))) for p in pages],
            dtype=np.uint64,
        )

        new_page_key = (new_paper.astype(np.int64) * _PAGE_MULT + new_page.astype(np.int64)).astype(np.int64)
        new_ids = (new_page_key * _CHUNK_MULT + new_chunk.astype(np.int64)).astype(np.int64)

        data = self._load()

        # Remove existing rows with the same ids, then append
        if data.ids.size:
            keep = ~np.isin(data.ids, new_ids)
            out = _IndexData(
                ids=np.concatenate([data.ids[keep], new_ids], axis=0),
                page_key=np.concatenate([data.page_key[keep], new_page_key], axis=0),
                emb=np.concatenate([data.emb[keep], new_emb], axis=0),
                source_id=np.concatenate([data.source_id[keep], new_paper], axis=0),
                page_num=np.concatenate([data.page_num[keep], new_page], axis=0),
                chunk_id=np.concatenate([data.chunk_id[keep], new_chunk], axis=0),
                char_start=np.concatenate([data.char_start[keep], new_start], axis=0),
                char_end=np.concatenate([data.char_end[keep], new_end], axis=0),
                fp64=np.concatenate([data.fp64[keep], new_fp], axis=0),
            )
        else:
            out = _IndexData(
                ids=new_ids,
                page_key=new_page_key,
                emb=new_emb,
                source_id=new_paper,
                page_num=new_page,
                chunk_id=new_chunk,
                char_start=new_start,
                char_end=new_end,
                fp64=new_fp,
            )

        self._save(out)

    def delete_source(self, source_id: int, **kwargs: Any) -> None:
        """Delete all vectors for a given source_id (all chunks)."""
        data = self._load()
        if data.ids.size == 0:
            return
        keep = data.source_id != int(source_id)
        self._save(
            _IndexData(
                ids=data.ids[keep],
                page_key=data.page_key[keep],
                emb=data.emb[keep],
                source_id=data.source_id[keep],
                page_num=data.page_num[keep],
                chunk_id=data.chunk_id[keep],
                char_start=data.char_start[keep],
                char_end=data.char_end[keep],
                fp64=data.fp64[keep],
            )
        )

    # ---------- SQLite helpers ----------

    def _fts_prefilter_pagekeys(
        self, query: str, where: Optional[Dict[str, Any]], limit_pages: int
    ) -> np.ndarray:
        """
        Use SQLite FTS5 to return candidate PAGE keys (source_id/page_num -> page_key).
        """
        fts_query = store.nl_to_fts_query(query)
        if not fts_query.strip():
            return np.zeros((0,), dtype=np.int64)

        conn = sqlite3.connect(str(self.sqlite_db_path))
        try:
            params: List[Any] = [fts_query]
            sql = """
            SELECT pages_fts.source_id, pages_fts.page_num
            FROM pages_fts
            WHERE pages_fts MATCH ?
            """

            if where:
                if "source_id" in where:
                    sql += " AND pages_fts.source_id = ?"
                    params.append(int(where["source_id"]))
                if "page_num" in where:
                    sql += " AND pages_fts.page_num = ?"
                    params.append(int(where["page_num"]))

            sql += " ORDER BY bm25(pages_fts) LIMIT ?"
            params.append(int(limit_pages))

            rows = conn.execute(sql, params).fetchall()
            if not rows:
                return np.zeros((0,), dtype=np.int64)

            return np.asarray([_page_key(int(sid), int(pn)) for sid, pn in rows], dtype=np.int64)
        finally:
            conn.close()

    def _fetch_pages(self, pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[str, str]]:
        """
        Fetch (path, page_text) for a small list of (source_id, page_num) pairs.
        """
        if not pairs:
            return {}

        conn = sqlite3.connect(str(self.sqlite_db_path))
        try:
            clauses: List[str] = []
            params: List[Any] = []
            for sid, pn in pairs:
                clauses.append("(pg.source_id = ? AND pg.page_num = ?)")
                params.extend([int(sid), int(pn)])

            sql = f"""
            SELECT pg.source_id, pg.page_num, p.path, pg.text
            FROM pages pg
            JOIN sources p ON p.id = pg.source_id
            WHERE {" OR ".join(clauses)}
            """
            rows = conn.execute(sql, params).fetchall()

            out: Dict[Tuple[int, int], Tuple[str, str]] = {}
            for sid, pn, path, text in rows:
                out[(int(sid), int(pn))] = (str(path), str(text))
            return out
        finally:
            conn.close()

    # ---------- query ----------

    def query(
        self,
        query: str,
        k: int,
        where: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedChunk]:
        """
        Query top-k CHUNKS using FTS prefilter + cosine distance.

        kwargs:
          - prefilter_k_pages: override default candidate page pool size
          - fallback_full_scan: if True, full scan if prefilter yields none (default True)
        """
        data = self._load()
        if data.ids.size == 0:
            return []

        pre_pages = int(kwargs.get("prefilter_k_pages", self.prefilter_k_pages))
        fallback_full_scan = bool(kwargs.get("fallback_full_scan", True))

        # Candidate pages from FTS
        cand_pages = self._fts_prefilter_pagekeys(query, where=where, limit_pages=pre_pages)

        # Mask candidate chunks by page_key (and optional chunk_id filter)
        if cand_pages.size:
            mask = np.isin(data.page_key, cand_pages)
        else:
            mask = np.zeros((data.ids.shape[0],), dtype=bool)

        if where and "chunk_id" in where:
            mask = mask & (data.chunk_id == int(where["chunk_id"]))

        # If prefilter yields nothing, optionally full-scan (still respects where filters)
        if not mask.any():
            if not fallback_full_scan:
                return []
            mask = np.ones((data.ids.shape[0],), dtype=bool)
            if where:
                if "source_id" in where:
                    mask &= data.source_id == int(where["source_id"])
                if "page_num" in where:
                    mask &= data.page_num == int(where["page_num"])
                if "chunk_id" in where:
                    mask &= data.chunk_id == int(where["chunk_id"])
            if not mask.any():
                return []

        emb = data.emb[mask]
        sids = data.source_id[mask]
        pnums = data.page_num[mask]
        cids = data.chunk_id[mask]
        starts = data.char_start[mask]
        ends = data.char_end[mask]

        q = self._st.encode([query], show_progress_bar=False)
        q = l2_normalize(q)[0]  # (dim,)

        metric = str(kwargs.get("distance", self.distance)).strip().lower()
        if metric not in {"cosine", "l2", "ip"}:
            raise ValueError("distance must be one of: cosine | l2 | ip")
        
        # NOTE: emb is already normalized in _load(); q is normalized above.
        # If later you want *raw* IP/L2 on unnormalized vectors, we’d need to store unnormalized emb too.
        if metric == "cosine":
            dist = Distance.cosine(emb, q)
        elif metric == "ip":
            dist = Distance.inner_product(emb, q)
        else:  # "l2"
            dist = Distance.l2_squared(emb, q)

        n = dist.shape[0]
        kk = min(int(k), n)
        if kk <= 0:
            return []

        idx = np.argpartition(dist, kk - 1)[:kk]
        idx = idx[np.argsort(dist[idx])]  # sorted

        # Fetch required pages once (unique sid/pn)
        top_page_pairs = {(int(sids[i]), int(pnums[i])) for i in idx}
        page_map = self._fetch_pages(list(top_page_pairs))

        out: List[RetrievedChunk] = []
        for i in idx:
            sid = int(sids[i])
            pn = int(pnums[i])
            chunk_id = int(cids[i])
            s = int(starts[i])
            e = int(ends[i])

            path, page_text = page_map.get((sid, pn), ("", ""))
            if not page_text:
                chunk_text = ""
            else:
                # Clamp offsets defensively
                s2 = max(0, min(s, len(page_text)))
                e2 = max(s2, min(e, len(page_text)))
                chunk_text = page_text[s2:e2]

            # If your RetrievedChunk dataclass has chunk fields, include them; otherwise fall back.
            try:
                out.append(
                    RetrievedChunk(
                        source_id=sid,
                        page_num=pn,
                        path=path,
                        text=chunk_text,
                        distance=float(dist[i]),
                        chunk_id=chunk_id,      # type: ignore[arg-type]
                        char_start=s,           # type: ignore[arg-type]
                        char_end=e,             # type: ignore[arg-type]
                    )
                )
            except TypeError:
                out.append(
                    RetrievedChunk(
                        source_id=sid,
                        page_num=pn,
                        path=path,
                        text=chunk_text,
                        distance=float(dist[i]),
                    )
                )

        return out
