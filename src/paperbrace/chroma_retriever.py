from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Must be set before importing chromadb
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
from sentence_transformers import SentenceTransformer

from paperbrace.retriever import ChunkForIndex, RetrievedChunk
from paperbrace.vector_ops import l2_normalize


class ChromaRetriever:
    """
    Chroma-based retriever (chunk-aware).
    """

    def __init__(
        self,
        *,
        persist_dir: Path,
        collection: str,
        embedding_model: str,
        distance: str = "cosine",
    ) -> None:
        persist_dir = persist_dir.expanduser().resolve()
        persist_dir.mkdir(parents=True, exist_ok=True)
        s = (distance or "cosine").strip().lower()
        if s not in {"cosine", "l2", "ip"}:
            raise ValueError("distance must be one of: cosine | l2 | ip")
        self.distance = s
        self.collection = collection
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection(name=collection, metadata={"hnsw:space": s})
        self._st = SentenceTransformer(embedding_model)
        

    @staticmethod
    def _cid(source_id: int, page_num: int, chunk_id: int) -> str:
        """Stable Chroma id for a specific chunk."""
        return f"{int(source_id)}:{int(page_num)}:{int(chunk_id)}"

    def upsert_source(self, pages: Sequence[ChunkForIndex], **kwargs: Any) -> None:
        """
        Upsert chunk entries for a sequence of pages.

        Expected ChunkForIndex fields (chunk-aware):
          - source_id, page_num, path, text
          - chunk_id (int)
          - char_start (int), char_end (int)
          - fingerprint (str)
        If your ChunkForIndex makes these optional, we fall back to safe defaults.
        """
        if not pages:
            return

        batch_size = int(kwargs.get("batch_size", 64))
        show_pb = bool(kwargs.get("show_progress_bar", False))

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []

        for p in pages:
            # Allow older ChunkForIndex without chunk fields
            chunk_id = int(getattr(p, "chunk_id", 0))
            char_start = int(getattr(p, "char_start", 0))
            char_end = int(getattr(p, "char_end", 0))
            fingerprint = str(getattr(p, "fingerprint", ""))

            ids.append(self._cid(p.source_id, p.page_num, chunk_id))
            docs.append(p.text)
            metas.append(
                {
                    "source_id": int(p.source_id),
                    "page_num": int(p.page_num),
                    "path": str(p.path),
                    "chunk_id": chunk_id,
                    "char_start": char_start,
                    "char_end": char_end,
                    "fingerprint": fingerprint,
                }
            )

        embs = self._st.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=show_pb,
        )
        embs = l2_normalize(embs)
        
        self._col.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs.tolist(),
        )

    def delete_source(self, source_id: int, **kwargs: Any) -> None:
        """
        Delete all chunks for a source_id.
        """
        self._col.delete(where={"source_id": int(source_id)})

    def query(
        self,
        query: str,
        k: int,
        where: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedChunk]:
        """
        Query semantic index and return top-k chunk hits.

        `where` can filter by metadata fields such as:
          - source_id, page_num, chunk_id, path
        """

        q_emb = self._st.encode([query], show_progress_bar=False)
        q_emb = l2_normalize(q_emb)[0]  # (dim,)
        q_emb = q_emb.tolist()


        res = self._col.query(
            query_embeddings=q_emb,
            n_results=int(k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out: List[RetrievedChunk] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                RetrievedChunk(
                    source_id=int(meta.get("source_id")),
                    page_num=int(meta.get("page_num")),
                    chunk_id=int(meta.get("chunk_id")) if meta.get("chunk_id") is not None else None,
                    char_start=int(meta.get("char_start")) if meta.get("char_start") is not None else None,
                    char_end=int(meta.get("char_end")) if meta.get("char_end") is not None else None,
                    fingerprint=str(meta.get("fingerprint")) if meta.get("fingerprint") is not None else None,
                    path=str(meta.get("path")),
                    text=str(doc),         # chunk text
                    distance=float(dist),     # lower is better
                )
            )
        return out
