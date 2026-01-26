# src/paperbrace/retriever.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence
import logging

logger = logging.getLogger("paperbrace")


@dataclass(frozen=True)
class ChunkForIndex:
    """A single chunk to embed/index in the vector store."""
    source_id: int
    page_num: int
    path: str
    text: str
    chunk_id: int = 0
    char_start: int = 0
    char_end: int = 0
    fingerprint: str = ""


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk result from the vector store."""
    source_id: int
    page_num: int
    path: str
    text: str
    distance: float
    chunk_id: int = 0
    char_start: int = 0
    char_end: int = 0
    fingerprint: str = ""


class Retriever(Protocol):
    def upsert_source(self, pages: Sequence[ChunkForIndex], **kwargs: Any) -> None: ...
    def delete_source(self, source_id: int, **kwargs: Any) -> None: ...
    def query(
        self,
        query: str,
        k: int,
        where: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[RetrievedChunk]: ...


def make_retriever(
    *,
    backend: str,
    chroma_db_path: Optional[Path] = None,
    flat_index_dir: Optional[Path] = None,
    collection: str = "paperbrace_pages",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    db_path: Optional[Path] = None,
    distance: str = "cosine", 
) -> Retriever:
    """
    Factory that lazily imports the requested backend so missing extras don't break imports.

    backend:
      - "chroma": requires extra paperbrace[chroma]
      - "flat":   requires extra paperbrace[flat]
      - "auto":   prefer chroma if available, else flat
    """
    b = (backend or "").strip().lower()
    s = (distance or "cosine").strip().lower()
    if s not in {"cosine", "l2", "ip"}:
        raise ValueError("distance must be one of: cosine | l2 | ip")

    if b in {"", "auto"}:
        # Try chroma first
        try:
            from paperbrace.chroma_retriever import ChromaRetriever  # noqa
            if chroma_db_path is None:
                raise ValueError("chroma_db_path is required for backend='chroma'")
            logger.info(f"Retriever CHROMA (auto) {embedding_model} distance={s}")
            return ChromaRetriever(
                persist_dir=chroma_db_path,
                collection=collection,
                embedding_model=embedding_model,
                distance=s,
            )
        except ModuleNotFoundError:
            pass

        # Fall back to flat
        try:
            from paperbrace.flat_retriever import FlatRetriever  # noqa
            if flat_index_dir is None:
                raise ValueError("flat_index_dir is required for backend='flat'")
            if db_path is None:
                raise ValueError("db_path is required for backend='flat'")
            logger.info(f"Retriever FLAT (auto) {embedding_model} distance {s}")
            return FlatRetriever(
                index_dir=flat_index_dir,
                collection=collection,
                embedding_model=embedding_model,
                sqlite_db_path=db_path,
                distance=s,
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
            from paperbrace.chroma_retriever import ChromaRetriever
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Chroma backend selected but chromadb is not installed.\n"
                "Install it with:\n"
                "  pip install -e '.[chroma]'\n"
            ) from e
        if chroma_db_path is None:
            raise ValueError("chroma_db_path is required for backend='chroma'")
        logger.info(f"Retriever CHROMA {embedding_model} distance {s}")
        return ChromaRetriever(
            persist_dir=chroma_db_path,
            collection=collection,
            embedding_model=embedding_model,
            distance=s,
        )

    if b == "flat":
        try:
            from paperbrace.flat_retriever import FlatRetriever
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Flat backend selected but numpy/sentence-transformers are not installed.\n"
                "Install it with:\n"
                "  pip install -e '.[flat]'\n"
            ) from e
        if flat_index_dir is None:
            raise ValueError("flat_index_dir is required for backend='flat'")
        if db_path is None:
            raise ValueError("db_path is required for backend='flat'")
        logger.info(f"Retriever FLAT {embedding_model} distance {s}")
        return FlatRetriever(
            index_dir=flat_index_dir,
            collection=collection,
            embedding_model=embedding_model,
            sqlite_db_path=db_path,
            distance=s,
        )

    raise ValueError("Unknown retriever backend. Use 'auto', 'chroma', or 'flat'.")

