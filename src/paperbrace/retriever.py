# src/paperbrace/retriever.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence


@dataclass(frozen=True)
class PageForIndex:
    """A single page (or chunk) to embed/index in the vector store."""
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


def make_retriever(
    *,
    backend: str,
    chroma_db_path: Optional[Path] = None,
    flat_index_dir: Optional[Path] = None,
    collection: str = "paperbrace_pages",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    db_path: Optional[Path] = None,
) -> Retriever:
    """
    Factory that lazily imports the requested backend so missing extras don't break imports.

    backend:
      - "chroma": requires extra paperbrace[chroma]
      - "flat":   requires extra paperbrace[flat]
      - "auto":   prefer chroma if available, else flat
    """
    b = (backend or "").strip().lower()

    if b in {"", "auto"}:
        # Try chroma first
        try:
            from paperbrace.chroma_retriever import ChromaPageRetriever  # noqa
            if chroma_db_path is None:
                raise ValueError("chroma_db_path is required for backend='chroma'")
            return ChromaPageRetriever(
                persist_dir=chroma_db_path,
                collection=collection,
                embedding_model=embedding_model,
            )
        except ModuleNotFoundError:
            pass

        # Fall back to flat
        try:
            from paperbrace.flat_retriever import FlatPageRetriever  # noqa
            if flat_index_dir is None:
                raise ValueError("flat_index_dir is required for backend='flat'")
            return FlatPageRetriever(
                index_dir=flat_index_dir,
                collection=collection,
                embedding_model=embedding_model,
            )
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "No vector backend is installed.\n\n"
                "Install one of:\n"
                "  pip install -e '.[chroma]'\n"
                "  pip install -e '.[flat]'\n"
            ) from e

    if b == "chroma":
        try:
            from paperbrace.chroma_retriever import ChromaPageRetriever
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Chroma backend selected but chromadb is not installed.\n"
                "Install it with:\n"
                "  pip install -e '.[chroma]'\n"
            ) from e
        if chroma_db_path is None:
            raise ValueError("chroma_db_path is required for backend='chroma'")
        return ChromaPageRetriever(
            persist_dir=chroma_db_path,
            collection=collection,
            embedding_model=embedding_model,
        )

    if b == "flat":
        try:
            from paperbrace.flat_retriever import FlatPageRetriever
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Flat backend selected but numpy/sentence-transformers are not installed.\n"
                "Install it with:\n"
                "  pip install -e '.[flat]'\n"
            ) from e
        if flat_index_dir is None:
            raise ValueError("flat_index_dir is required for backend='flat'")
        return FlatPageRetriever(
            index_dir=flat_index_dir,
            collection=collection,
            embedding_model=embedding_model,
            sqlite_db_path=db_path,
        )

    raise ValueError("Unknown retriever backend. Use 'auto', 'chroma', or 'flat'.")

