from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Must be set before importing chromadb
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
from sentence_transformers import SentenceTransformer

from paperbrace.retriever import PageForIndex, RetrievedPage


class ChromaPageRetriever:
    """
    Chroma-based retriever.

    We embed ourselves with SentenceTransformer so we can suppress progress bars,
    and pass embeddings to Chroma (both upsert + query).
    """

    def __init__(
        self,
        *,
        persist_dir: Path,
        collection: str,
        embedding_model: str,
    ) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection(name=collection)
        self._st = SentenceTransformer(embedding_model)

    @staticmethod
    def _pid(paper_id: int, page_num: int) -> str:
        return f"{paper_id}:{page_num}"

    def upsert_pages(self, pages: Sequence[PageForIndex], **kwargs: Any) -> None:
        if not pages:
            return
        ids = [self._pid(p.paper_id, p.page_num) for p in pages]
        docs = [p.text for p in pages]
        metas = [{"paper_id": p.paper_id, "page_num": p.page_num, "path": p.path} for p in pages]

        embs = self._st.encode(docs, show_progress_bar=False)
        self._col.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs.tolist(),
        )

    def delete_paper(self, paper_id: int, **kwargs: Any) -> None:
        self._col.delete(where={"paper_id": int(paper_id)})

    def query_pages(
        self,
        query: str,
        k: int,
        where: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedPage]:
        q_emb = self._st.encode([query], show_progress_bar=False).tolist()

        res = self._col.query(
            query_embeddings=q_emb,
            n_results=int(k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out: List[RetrievedPage] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                RetrievedPage(
                    paper_id=int(meta["paper_id"]),
                    page_num=int(meta["page_num"]),
                    path=str(meta["path"]),
                    text=str(doc),
                    score=float(dist),  # lower is better
                )
            )
        return out
