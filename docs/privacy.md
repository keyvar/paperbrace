# Privacy 

## Default behavior

Paperbrace is designed to run locally on the user’s machine:
- PDFs are read locally
- Extracted text is stored locally (SQLite)

## Optional future integrations
user-initiated, opt in explicitly enabled, no content transmission:
- Zotero
- arXiv metadata lookup
- Semantic Scholar metadata/graph enrichment

## Threat model notes

- Your local machine is the trust boundary
- Protect the SQLite DB like you protect your PDFs (it contains extracted text)
- If you sync your machine/backups to cloud services, your content may be included by your own syncing