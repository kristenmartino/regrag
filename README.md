# RegRAG

Retrieval-augmented generation over FERC regulatory orders, with grounded citations, agentic query decomposition, and an audit trail.

**Status:** in development. See [docs/regrag-case-study.md](docs/regrag-case-study.md) for the design and [docs/implementation-plan.md](docs/implementation-plan.md) for the build plan.

## Repo layout

```
regrag/
├── apps/
│   ├── api/        # FastAPI service: /chat endpoint, LangGraph orchestration
│   ├── ingest/     # corpus pipeline: fetch FERC PDFs, parse, chunk, embed
│   └── web/        # Next.js chat UI
├── packages/
│   └── eval/       # eval harness: 28-question seed set, retrieval/citation/refusal metrics
├── corpus/
│   ├── manifest.yaml   # seed list of FERC documents
│   ├── raw/            # downloaded PDFs (gitignored)
│   └── parsed/         # normalized JSON records (gitignored)
├── docs/           # case study, ingestion plan, diagrams, critique, feasibility, build plan
└── infra/          # SQL schema, Vercel/Railway config
```

## Quick start

Python version is pinned in `.python-version` (currently 3.12.13). Set it up once with `pyenv` (`brew install pyenv && pyenv install 3.12`); subsequent venv creations pick it up automatically when run from the repo root.

Each app has its own `pyproject.toml` and venv:

```bash
# Ingest the seed corpus
cd apps/ingest
python -m venv .venv && source .venv/bin/activate
pip install -e .
regrag-ingest run --manifest ../../corpus/manifest.yaml

# Run the API locally
cd ../../apps/api
python -m venv .venv && source .venv/bin/activate
pip install -e .
uvicorn regrag_api.main:app --reload
```
