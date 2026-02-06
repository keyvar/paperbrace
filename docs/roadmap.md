# Roadmap

## L0 — Search + Cite MVP
- [x] Point to local folder of PDFs
- [x] Build a local DB of PDF metadata
- [x] Extract per-page text
- [ ] Keyword retrieval (SQLite FTS5)
- [ ] Ask questions → answer with citations + highlighted excerpts
- [ ] Export answers + citations to Markdown

**Success criteria**
- Search across ~200 PDFs in seconds
- Every answer includes evidence snippets with page numbers

## L1 — Paper-level summaries (PaperCard)

- [ ] PaperCard JSON + rendered view: problem / contributions / method / datasets/metrics / limitations
- [ ] Key claims with supporting evidence

## L2 — Multi-paper queries + comparison tables
- [ ] “Which sources improve over Y?” → ranked list + structured table (with citations)
- [ ] “Compare A vs B” → matrix with citations

## L3 — Router/orchestrator
- [ ] Route user intent to workflows (Locate / Explain / Compare)

## L4 — Evaluation harness
- [ ] Local eval suite (retrieval checks, citation-supported answer rate, regressions)

## L5 — Local citation graph
- [ ] Build citation graph from local metadata (BibTeX/JSON)
- [ ] “Related sources in my library” (citations/authors/keywords + embeddings)
