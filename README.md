# Paperbrace

**Paperbrace** is a local-first literature navigator for a researcher’s PDF library (e.g., a Zotero attachments folder).

## Goals

- Index a local folder of PDFs and enable QA/navigation over the user’s library
- Ground all answers in retrieved passages with page-level citations (“cite-or-refuse”)
- Run locally by default (no uploads, no external services)

Ask things like:
- “Which paper mentions X?”
- “Which paper claims improvement over Y?”
- “Summarize paper Z’s contributions and limitations.”
- “Compare A vs B on assumptions/datasets/metrics.”

**Local-first by design:** everything runs on your machine. No uploads, no servers.

---

## Non-goals

- Running user-provided code
- Opening/executing links
- “Write my paper” mode (this is for reading/understanding/navigation)
- Hosted service operated by us

---

## Current status

**L0 (Search + Cite MVP)**.

- Index a folder of PDFs into a local SQLite DB
- List papers
- Extract per-page text into the DB

---

## Requirements

- macOS / Linux
- Python (project-local via `pyenv local` recommended)
- SQLite (bundled with Python)

---

## Quickstart (dev)

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```
Index PDFs:
```
paperbrace index --pdf-dir /path/to/pdfs --db paperbrace.db
```
List:
```
paperbrace list --db paperbrace.db
```
Extract:
```
paperbrace extract --paper-id 1 --db paperbrace.db
```
## License
TBD (MIT or Apache-2.0)