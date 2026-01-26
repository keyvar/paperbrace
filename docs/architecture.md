# Architecture

## High-level components

1) **Ingestor**
- Scans (and later watches) a folder of PDFs
- Tracks file path, mtime, size

2) **Extractor**
- PDF → per-page text
- Preserves page numbers for citations
- (L0+) adds section hints + chunk-to-page mapping

3) **Indexer**
- Stores extracted text + metadata in SQLite
- (L0) Keyword index via SQLite FTS5
- (L0+) Embeddings + vector index (SQLite-based vector, faiss, or similar)

4) **Retriever**
- (L0) Keyword retrieval
- (l0+) Hybrid retrieval (keyword + semantic)

5) **Answerer**
- Generates answers only from retrieved evidence
- If evidence is insufficient: refuses / asks to refine query
- Outputs citations: (source_id, page_num) + excerpt text

6) **UI**
- CLI first
- (l0+) local web UI for search, highlights, export

## Data flow (L0)

PDF folder
→ index metadata into `sources`
→ extract per-page text into `pages`
→ query via FTS and fetch evidence passages
→ answer with citations

## Storage (current)

SQLite tables:
- `sources`: id, path, mtime, size, extracted_mtime, timestamps
- `pages`: (source_id, page_num) → text

Planned (L0):
- `pages_fts` (FTS5) over page text for fast keyword search

Planned (L0+):
- `chunks`: chunk_id, source_id, page_start/end, text
- `embeddings`: chunk_id, vector (plus index)

## Key design constraints

- Never execute user-provided code
- Never open external links automatically
- Default to zero network access
- Traceability: every claim must link to evidence