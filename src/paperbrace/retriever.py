from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE") # turns off telemetry for Chroma DB

import chromadb
from chromadb.utils import embedding_functions


@dataclass(frozen=True)
class PageForIndex:
    """A single page to embed/index in the vector store."""
    paper_id: int
    page_num: int
    path: str
    text: str


@dataclass(frozen=True)
class RetrievedPage:
    """A retrieved page result from the vector store."""
    paper_id: int
    page_num: int
    path: str
    text: str
    score: float  # lower distance or higher similarity depending on backend


class Retriever(Protocol):
    def upsert_pages(self, pages: Sequence[PageForIndex], **kwargs: Any) -> None: ...
    def delete_paper(self, paper_id: int, **kwargs: Any) -> None: ...
    def query_pages(
        self,
        query: str,
        k: int,
        where: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedPage]: ...


class ChromaPageRetriever:
    def __init__(
        self,
        persist_dir: Path,
        collection: str,
        embedding_model: str,
    ) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._col = self._client.get_or_create_collection(
            name=collection,
            embedding_function=ef,
        )

    @staticmethod
    def _pid(paper_id: int, page_num: int) -> str:
        return f"{paper_id}:{page_num}"

    def upsert_pages(self, pages: Sequence[PageForIndex], **kwargs: Any) -> None:
        if not pages:
            return
        ids = [self._pid(p.paper_id, p.page_num) for p in pages]
        docs = [p.text for p in pages]
        metas = [
            {"paper_id": p.paper_id, "page_num": p.page_num, "path": p.path}
            for p in pages
        ]
        self._col.upsert(ids=ids, documents=docs, metadatas=metas)

    def delete_paper(self, paper_id: int, **kwargs: Any) -> None:
        # Deletes all pages for that paper_id (used on re-embed)
        self._col.delete(where={"paper_id": int(paper_id)})

    def query_pages(
        self,
        query: str,
        k: int,
        where: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedPage]:
        q_emb = self._st.encode([query], show_progress_bar=False)
        res = self._col.query(
            query_texts=[query],
            n_results=k,
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
                    score=float(dist),
                )
            )
        return out
