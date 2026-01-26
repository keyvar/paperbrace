# Development Guide

## Repo layout (src layout)

- `src/paperbrace/` → Python package code
- `tests/` → pytest tests
- `docs/` → documentation

## Python setup

Recommended: pyenv + per-project Python.

Inside repo root:
```bash
pyenv install 3.13.11
pyenv local 3.13.11

Install (editable)
```
From repo root:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
paperbrace --help
```

## Quickstart

After pip install -e ., you should have the CLI entrypoint.

### Index PDFs (metadata only)
```
paperbrace index --pdf-dir /path/to/pdfs --db paperbrace.db -v
```
### List
```
paperbrace list --db paperbrace.db
```
### Extract a single paper
```
paperbrace extract --paper-id 1 --db paperbrace.db
```

### Extract all (build searchable content, recommended):
```
paperbrace extract --all --db paperbrace.db -v
```
Re-extract everything forces rebuild of extracted text + search index.
```
paperbrace extract --all --db paperbrace.db --force -v
```
### Keyword search
```
paperbrace search "your search expression" --db paperbrace.db
```
Example
```
paperbrace search "baseline AND improvement" --db paperbrace.db -n 20
```

### Ask (Model Mode)
Model will auto-download if needed
With keyword retriever (default)
```
paperbrace ask "your question" --db paperbrace.db --k 8 --retriever keyword
```
With model override
```
paperbrace ask "your question" --db paperbrace.db --k 8 --retriever keyword --model Qwen/Qwen2.5-3B-Instruct
```
With semantic retriever
```
paperbrace ask "your question" --db paperbrace.db --retriever semantic
```
You can also pass ```--retriever hybrid``` which appends keyword hits to **after** the list retrieved by semantic search.

Hybrid is useful in a narrow high-value band:

When the query is on-topic, semantic wins, so hybrid adds little. Also, when the query is off-topic, semantic hits with cutoff may return nothing, and hybrid may show irrelevant keyword matches, which should be indentified by the model as “not supported.”

The real justification for hybrid is this specific failure mode:

On-topic query, semantic misses, rare terms / acronyms / exact phrases, numbers / thresholds / formula-like strings, citation-style queries (“Table 2”, “Figure 3”, “AUROC 0.83”), extraction quirks where embeddings blur the exact token match.

optional keyword to write into output file ```--out answer.md```

### Purge
```
paperbrace purge --db paperbrace.db --yes -v
```

## Verbose logs:
```
paperbrace index ... -v
```

## Dependencies
typer, rich, pymupdf

sqlite FTS5 (built into modern SQLite)

embeddings + vector index library (TBD)

## Testing

Install:
```
pip install pytest
```
Run:
```
pytest -q
```
or (no warnings)
```
pytest -q -W ignore::DeprecationWarning
```
Lint/format (optional but recommended)

Lightweight approach:
```
pip install ruff
ruff check .
```