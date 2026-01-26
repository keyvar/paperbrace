from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Evidence:
    source_id: int
    page_num: int
    path: str
    text: str
    distance: float
    chunk_id: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    fingerprint: Optional[str] = ""
    retrieval: Optional[str] = ""

def to_markdown(
    question: str,
    items: List[Evidence],
    answer: Optional[str] = None,
    max_chars_per_item: int = 1200,
) -> str:
    """
    Render Paperbrace output as Markdown (answer + evidence pack).

    This function supports two modes:

    - Evidence-only (Phase A): if `answer` is None, it renders only the Evidence
      section. This is useful when the user wants retrieval/citations without
      running an LLM.
    - Answer + Evidence (Phase B): if `answer` is provided, it renders an Answer
      section first, followed by the Evidence section.

    Evidence items are rendered as numbered blocks [1], [2], ... so that an LLM
    (or a human) can cite them consistently.

    Args:
        question: The user’s question/query.
        items: Retrieved evidence items (page-level).
        answer: Optional generated answer text. If None, evidence-only output.
        max_chars_per_item: Maximum characters of each evidence page text to include.
            Longer texts are truncated for readability.

    Returns:
        A Markdown string with:
          - Title
          - Question
          - Optional Answer
          - Evidence blocks, each including source_id, page number, filename, and excerpt
          - If `items` is empty, an Evidence section with a "No matching evidence" note.
    """

    lines: list[str] = []
    lines.append("# Paperbrace Answer\n")
    lines.append(f"**Question:** {question}\n")

    if answer is not None:
        lines.append("## Answer\n")
        lines.append(answer.strip() + "\n")

    lines.append("## Evidence\n")
    if not items:
        lines.append("_No matching evidence found._\n")
        return "\n".join(lines)

    for i, e in enumerate(items, start=1):
        fname = Path(e.path).name
        excerpt = (e.text or "").strip()
        if len(excerpt) > max_chars_per_item:
            excerpt = excerpt[:max_chars_per_item] + "…"

        lines.append(f"### [{i}] source_id={e.source_id} page={e.page_num} — {fname}\n")
        lines.append("```text")
        lines.append(excerpt)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
