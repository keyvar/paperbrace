from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    text: str
    char_start: int
    char_end: int
    fingerprint: str  # sha1 of normalized chunk text


def _normalize_for_fingerprint(text: str) -> str:
    """
    Normalize text into a stable token stream for matching.

    Lowercase, keep word characters, collapse whitespace.
    This is intentionally simple and stable across platforms.
    """
    toks = re.findall(r"\w+", (text or "").lower())
    return " ".join(toks)


def fingerprint(text: str) -> str:
    """
    Stable fingerprint for a chunk. Useful later for PDF highlighting / matching.
    """
    norm = _normalize_for_fingerprint(text)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def chunk_text(
    text: str,
    *,
    max_chars: int = 1200,
    overlap: int = 200,
    min_chars: int = 250,
) -> List[Chunk]:
    """
    Chunk a page of extracted text into overlapping character windows.

    - max_chars: target chunk size
    - overlap: overlap between consecutive chunks
    - min_chars: drop tiny trailing chunks unless it's the only one

    Returns:
        List[Chunk] with offsets into the original `text`.
    """
    s = text or ""
    s_len = len(s)
    if s_len == 0:
        return []

    chunks: List[Chunk] = []
    start = 0

    while start < s_len:
        end = min(start + max_chars, s_len)

        # Try to end on a natural boundary (newline preferred, else space)
        if end < s_len:
            nl = s.rfind("\n", start, end)
            sp = s.rfind(" ", start, end)
            cut = max(nl, sp)
            if cut > start + min_chars // 2:
                end = cut

        chunk = s[start:end].strip()
        if chunk:
            # Adjust offsets to match stripped chunk
            left_strip = len(s[start:end]) - len(s[start:end].lstrip())
            right_strip = len(s[start:end]) - len(s[start:end].rstrip())
            cs = start + left_strip
            ce = end - right_strip

            chunks.append(
                Chunk(
                    text=chunk,
                    char_start=cs,
                    char_end=ce,
                    fingerprint=fingerprint(chunk),
                )
            )

        if end >= s_len:
            break

        start = max(0, end - overlap)

    # Drop a tiny last chunk if it’s redundant
    if len(chunks) > 1 and (chunks[-1].char_end - chunks[-1].char_start) < min_chars:
        chunks.pop()

    return chunks
