from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from paperbrace.retriever import PageForIndex, RetrievedPage
from paperbrace import store

logger = logging.getLogger("paperbrace")


def _pid(paper_id: int, page_num: int) -> int:
    """Stable integer id for a (paper_id, page_num) pair."""
    return int(paper_id) * 100_000 + int(page_num)  # assumes page_num < 100k


def _as_int_scalar(x: Any) -> int:
    """Convert numpy scalar/0d array/python int to int."""
    a = np.asarray(x)
    if a.shape == ():
        return int(a.item())
    # tolerate 1-element arrays
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


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize vectors (float32)."""
    v = v.astype(np.float32, copy=False)
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / denom


@dataclass(frozen=True)
class _IndexData:
    ids: np.ndarray       # int64, shape (N,)
    emb: np.ndarray       # float32, shape (N, dim), L2-normalized
    paper_id: np.ndarray  # int32, shape (N,)
    page_num: np.ndarray  # int32, shape (N,)


class FlatPageRetriever:
    """
    Flat (NumPy) vector retriever with SQLite FTS prefilter.

    Storage:
      - Writes a single .npz file per collection under index_dir.
      - Stores ONLY numeric arrays (ids/emb/paper_id/page_num) + scalar metadata.

    Query:
      1) Prefilter with SQLite FTS5 (pages_fts) to get up to `prefilter_k` candidate pages
         (converted from NL to OR-style via store.nl_to_fts_query).
      2) Score only those candidate vectors by cosine distance (1 - cosine_sim).
      3) Fetch final top-k page text/path from SQLite pages+pagers tables.

    This is designed as a robust Windows-friendly fallback when Chroma is hard to install.
    """

    def __init__(
        self,
        *,
        index_dir: Path,
        collection: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sqlite_db_path: Path,
        prefilter_k: int = 5000,
    ) -> None:
        self.index_dir = index_dir.expanduser().resolve()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.collection = collection
        self.embedding_model = embedding_model
        self.sqlite_db_path = sqlite_db_path.expanduser().resolve()
        self.prefilter_k = int(prefilter_k)

        self._st = SentenceTransformer(embedding_model)
        self.dim = int(self._st.get_sentence_embedding_dimension())

        self._path = self.index_dir / f"{self.collection}.npz"

        # Ensure index exists (or create empty)
        if not self._path.exists():
            self._save(
                _IndexData(
                    ids=np.zeros((0,), dtype=np.int64),
                    emb=np.zeros((0, self.dim), dtype=np.float32),
                    paper_id=np.zeros((0,), dtype=np.int32),
                    page_num=np.zeros((0,), dtype=np.int32),
                )
            )

    # ---------- persistence ----------

    def _load(self) -> Tuple[_IndexData, int, str]:
        """
        Load index arrays + metadata.

        Returns:
          (data, dim, embedding_model)

        Back-compat:
          - If old .npz contains object arrays, retry with allow_pickle=True.
          - We ignore non-numeric keys (like old 'text'/'path') entirely.
        """
        try:
            z = np.load(self._path, allow_pickle=False)
        except ValueError as e:
            # Old index file had object arrays (e.g., embedding_model stored as dtype=object)
            if "Object arrays cannot be loaded" in str(e):
                logger.warning(
                    "Flat index contains object arrays; reloading with allow_pickle=True. "
                    "After this run, rebuild with: paperbrace embed --all --backend flat --force"
                )
                z = np.load(self._path, allow_pickle=True)
            else:
                raise

        dim = _as_int_scalar(z.get("dim", np.array(self.dim, dtype=np.int32)))
        model = _as_str_scalar(z.get("embedding_model", np.array(self.embedding_model, dtype="U")))

        # If the file was created with a different model/dim, fail loudly.
        if dim != self.dim:
            raise ValueError(f"Flat index dim={dim} but model produces dim={self.dim}. Rebuild the flat index.")
        if model != self.embedding_model:
            raise ValueError(
                f"Flat index model={model!r} but you configured {self.embedding_model!r}. Rebuild the flat index."
            )

        ids = np.asarray(z["ids"]).astype(np.int64, copy=False)
        emb = np.asarray(z["emb"]).astype(np.float32, copy=False)
        paper_id = np.asarray(z["paper_id"]).astype(np.int32, copy=False)
        page_num = np.asarray(z["page_num"]).astype(np.int32, copy=False)

        # Ensure normalized (safe even if already normalized)
        if emb.size:
            emb = _l2_normalize(emb)

        data = _IndexData(ids=ids, emb=emb, paper_id=paper_id, page_num=page_num)
        return data, dim, model

    def _save(self, data: _IndexData) -> None:
        """Save numeric arrays + scalar metadata (no object arrays)."""
        np.savez_compressed(
            self._path,
            ids=data.ids.astype(np.int64, copy=False),
            emb=data.emb.astype(np.float32, copy=False),
            paper_id=data.paper_id.astype(np.int32, copy=False),
            page_num=data.page_num.astype(np.int32, copy=False),
            dim=np.array(self.dim, dtype=np.int32),
            embedding_model=np.array(self.embedding_model, dtype="U"),
        )

    # ---------- protocol methods ----------

    def upsert_pages(self, pages: Sequence[PageForIndex], **kwargs: Any) -> None:
        """
        Embed and upsert pages into the flat index.

        Notes:
          - For simplicity we do: load -> remove old ids -> append new -> save.
          - This is fast enough for local libs and keeps code very robust.
        """
        if not pages:
            return

        batch_size = int(kwargs.get("batch_size", 64))
        show_pb = bool(kwargs.get("show_progress_bar", False))

        texts = [p.text for p in pages]
        new_emb = self._st.encode(texts, batch_size=batch_size, show_progress_bar=show_pb)
        new_emb = _l2_normalize(np.asarray(new_emb, dtype=np.float32))

        new_ids = np.asarray([_pid(p.paper_id, p.page_num) for p in pages], dtype=np.int64)
        new_paper = np.asarray([p.paper_id for p in pages], dtype=np.int32)
        new_page = np.asarray([p.page_num for p in pages], dtype=np.int32)

        data, _, _ = self._load()

        # Remove existing rows with the same ids, then append
        if data.ids.size:
            keep = ~np.isin(data.ids, new_ids)
            ids = np.concatenate([data.ids[keep], new_ids], axis=0)
            emb = np.concatenate([data.emb[keep], new_emb], axis=0)
            paper_id = np.concatenate([data.paper_id[keep], new_paper], axis=0)
            page_num = np.concatenate([data.page_num[keep], new_page], axis=0)
        else:
            ids, emb, paper_id, page_num = new_ids, new_emb, new_paper, new_page

        self._save(_IndexData(ids=ids, emb=emb, paper_id=paper_id, page_num=page_num))

    def delete_paper(self, paper_id: int, **kwargs: Any) -> None:
        """Delete all vectors for a given paper_id."""
        data, _, _ = self._load()
        if data.ids.size == 0:
            return
        keep = data.paper_id != int(paper_id)
        self._save(
            _IndexData(
                ids=data.ids[keep],
                emb=data.emb[keep],
                paper_id=data.paper_id[keep],
                page_num=data.page_num[keep],
            )
        )

    def _fts_prefilter_ids(
        self, query: str, where: Optional[Dict[str, Any]], limit: int
    ) -> np.ndarray:
        """
        Use SQLite FTS5 to return candidate flat ids (paper_id/page_num -> pid).
        """
        fts_query = store.nl_to_fts_query(query)
        conn = sqlite3.connect(str(self.sqlite_db_path))
        try:
            params: List[Any] = [fts_query]
            sql = """
            SELECT pages_fts.paper_id, pages_fts.page_num
            FROM pages_fts
            WHERE pages_fts MATCH ?
            """
            if where:
                # minimal where support
                if "paper_id" in where:
                    sql += " AND pages_fts.paper_id = ?"
                    params.append(int(where["paper_id"]))
                if "page_num" in where:
                    sql += " AND pages_fts.page_num = ?"
                    params.append(int(where["page_num"]))

            sql += " ORDER BY bm25(pages_fts) LIMIT ?"
            params.append(int(limit))

            rows = conn.execute(sql, params).fetchall()
            if not rows:
                return np.zeros((0,), dtype=np.int64)

            ids = np.asarray([_pid(int(pid), int(pn)) for pid, pn in rows], dtype=np.int64)
            return ids
        finally:
            conn.close()

    def _fetch_text_for_hits(self, pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[str, str]]:
        """
        Fetch (path, text) from SQLite for a small list of (paper_id, page_num) pairs.
        """
        if not pairs:
            return {}

        conn = sqlite3.connect(str(self.sqlite_db_path))
        try:
            # Build a small OR clause; k <= 30 so this is fine.
            clauses = []
            params: List[Any] = []
            for pid, pn in pairs:
                clauses.append("(pg.paper_id = ? AND pg.page_num = ?)")
                params.extend([int(pid), int(pn)])

            sql = f"""
            SELECT pg.paper_id, pg.page_num, p.path, pg.text
            FROM pages pg
            JOIN papers p ON p.id = pg.paper_id
            WHERE {" OR ".join(clauses)}
            """
            rows = conn.execute(sql, params).fetchall()
            out: Dict[Tuple[int, int], Tuple[str, str]] = {}
            for pid, pn, path, text in rows:
                out[(int(pid), int(pn))] = (str(path), str(text))
            return out
        finally:
            conn.close()

    def query_pages(
        self,
        query: str,
        k: int,
        where: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedPage]:
        """
        Query top-k pages using FTS prefilter + cosine distance.

        kwargs:
          - prefilter_k: override default candidate pool size
          - fallback_full_scan: if True, full scan if prefilter yields none (default True)
        """
        data, _, _ = self._load()
        if data.ids.size == 0:
            return []

        pre_k = int(kwargs.get("prefilter_k", self.prefilter_k))
        fallback_full_scan = bool(kwargs.get("fallback_full_scan", True))

        # Candidate ids from FTS
        cand_ids = self._fts_prefilter_ids(query, where=where, limit=pre_k)
        if cand_ids.size:
            mask = np.isin(data.ids, cand_ids)
        else:
            mask = np.zeros((data.ids.shape[0],), dtype=bool)

        # If prefilter yields nothing, optionally full-scan
        if not mask.any():
            if not fallback_full_scan:
                return []
            mask = np.ones((data.ids.shape[0],), dtype=bool)

        emb = data.emb[mask]
        pids = data.paper_id[mask]
        pnums = data.page_num[mask]

        q = self._st.encode([query], show_progress_bar=False)
        q = _l2_normalize(np.asarray(q, dtype=np.float32))[0]  # (dim,)

        # cosine distance: 1 - dot (since both normalized)
        sims = emb @ q
        dist = 1.0 - sims

        n = dist.shape[0]
        kk = min(int(k), n)
        if kk <= 0:
            return []

        idx = np.argpartition(dist, kk - 1)[:kk]
        idx = idx[np.argsort(dist[idx])]  # sorted by distance

        top_pairs = [(int(pids[i]), int(pnums[i])) for i in idx]
        meta = self._fetch_text_for_hits(top_pairs)

        out: List[RetrievedPage] = []
        for i in idx:
            pid = int(pids[i])
            pn = int(pnums[i])
            path, text = meta.get((pid, pn), ("", ""))
            out.append(
                RetrievedPage(
                    paper_id=pid,
                    page_num=pn,
                    path=path,
                    text=text,
                    score=float(dist[i]),  # lower is better
                )
            )
        return out
