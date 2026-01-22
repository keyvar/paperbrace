from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence


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
    score: float  # distance (lower is better for both backends here)


class Retriever(Protocol):
    """Minimal interface your app uses, regardless of vector backend."""
    def upsert_pages(self, pages: Sequence[PageForIndex], **kwargs: Any) -> None: ...
    def delete_paper(self, paper_id: int, **kwargs: Any) -> None: ...
    def query_pages(
        self,
        query: str,
        k: int,
        where: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedPage]: ...


def make_retriever(
    *,
    backend: str,
    db_path: Optional[Path] = None,
    persist_dir: Optional[Path] = None,
    collection: str = "pages",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    table: str = "pages_vec",
) -> Retriever:
    """
    Factory that lazily imports the requested backend so missing extras don't break imports.

    Args:
        backend: "chroma" or "sqlitevec"
        db_path: SQLite DB path (required for sqlitevec backend)
        persist_dir: directory for Chroma persistence (required for chroma backend)
        collection: collection name (chroma)
        embedding_model: sentence-transformers model id
        table: vec table name (sqlitevec)

    Returns:
        A Retriever instance.
    """
    b = (backend or "").strip().lower()

    if b == "chroma":
        if persist_dir is None:
            raise ValueError("persist_dir is required for backend='chroma'")
        from paperbrace.chroma_backend import ChromaPageRetriever
        return ChromaPageRetriever(
            persist_dir=persist_dir,
            collection=collection,
            embedding_model=embedding_model,
        )

    if b in {"sqlitevec", "sqlite-vec"}:
        if db_path is None:
            raise ValueError("db_path is required for backend='sqlitevec'")
        from paperbrace.sqlitevec_backend import SqliteVecPageRetriever
        return SqliteVecPageRetriever(
            db_path=db_path,
            table=table,
            embedding_model=embedding_model,
        )

    raise ValueError("Unknown retriever backend. Use 'chroma' or 'sqlitevec'.")
