# Privacy

## Default behavior

Paperbrace is designed to run locally on the user’s machine:

- PDFs are read locally
- Extracted text is stored locally (SQLite)

## Model downloads (Hugging Face)

Paperbrace uses local Hugging Face models (e.g., SentenceTransformers for embeddings, and optionally a local LLM for answering). If a required model is **not already present in your local Hugging Face cache**, the Hugging Face tooling may download it from the Hugging Face Hub when you run commands like `embed` or `ask`.

Once a model is cached locally, Paperbrace can be run in an offline mode to avoid any network access for model resolution.

### Offline mode

If you enable Paperbrace offline mode (e.g., via a CLI flag in your setup), Paperbrace will set the following environment variables so Hugging Face/Transformers will not attempt network access:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

Offline mode requires that the needed models are already cached locally.

## Optional future integrations

User-initiated, opt-in explicitly enabled, no content transmission:

- Zotero
- arXiv metadata lookup
- Semantic Scholar metadata/graph enrichment

## Threat model notes

- Your local machine is the trust boundary
- Protect the SQLite DB like you protect your PDFs (it contains extracted text)
- If you sync your machine/backups to cloud services, your content may be included by your own syncing

## Running on cloud infrastructure 

Paperbrace is intended for **local use only**. If you choose to run Paperbrace on cloud infrastructure (e.g., a hosted VM, container platform, remote desktop, or managed storage), you are responsible for ensuring your deployment meets all privacy, security, and legal/compliance requirements. This includes access controls, networking, storage encryption, retention, and any applicable regulations for the documents you process.
