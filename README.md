# RegRAG

Retrieval-augmented generation over FERC regulatory orders, with grounded citations, agentic query decomposition, and an audit trail.

**🟢 Live demo:** [regrag.vercel.app](https://regrag.vercel.app) — public, no auth, demo only.
**📄 Case study:** [docs/regrag-case-study.md](docs/regrag-case-study.md)
**📊 Eval results:** [docs/eval-results.md](docs/eval-results.md) — 40 questions, 96.9% retrieval recall / 90.0% refusal accuracy / 95.4% citation faithfulness on the 15-doc corpus.
**🏗 Build plan + deployment:** [docs/implementation-plan.md](docs/implementation-plan.md) · [docs/deployment.md](docs/deployment.md)

## Repo layout

```
regrag/
├── apps/
│   ├── api/        # FastAPI service: /chat endpoint, LangGraph orchestration
│   ├── ingest/     # corpus pipeline: fetch FERC PDFs, parse, chunk, embed
│   └── web/        # Next.js chat UI
├── packages/
│   └── eval/       # eval harness: 40-question set, retrieval/citation/refusal metrics + baselines
├── corpus/
│   ├── manifest.yaml   # seed list of FERC documents
│   ├── raw/            # downloaded PDFs (gitignored)
│   └── parsed/         # normalized JSON records (gitignored)
├── docs/           # case study, ingestion plan, diagrams, critique, feasibility, build plan
└── infra/          # SQL schema, Vercel/Railway config
```

## Quick start

Python version is pinned in `.python-version` (currently 3.12.13). Set it up once with `pyenv` (`brew install pyenv && pyenv install 3.12`); subsequent venv creations pick it up automatically when run from the repo root.

**Env setup** — copy `.env.example` to `.env` at the repo root and fill in `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`, `DATABASE_URL` (see [docs/deployment.md](docs/deployment.md) for where to get these). Both `apps/ingest` and `apps/api` load this file.

Each app has its own `pyproject.toml` and venv:

```bash
# Ingest the seed corpus (15 FERC orders → Neon Postgres + pgvector)
cd apps/ingest
python -m venv .venv && source .venv/bin/activate
pip install -e .
regrag-ingest run --manifest ../../corpus/manifest.yaml

# Run the API locally (FastAPI + LangGraph)
cd ../../apps/api
python -m venv .venv && source .venv/bin/activate
pip install -e .
uvicorn regrag_api.server:app --reload --port 8000

# Run the eval harness (40 questions, ~$3 in Anthropic spend, ~15 min)
cd ../../packages/eval
python -m venv .venv && source .venv/bin/activate
pip install -e .
regrag-eval run

# Run the frontend (Next.js + Tailwind + shadcn/ui)
cd ../../apps/web
pnpm install && pnpm dev   # serves on http://localhost:3000
```

End-to-end smoke test: with the API running, `regrag-chat "What does Order 2222 require for DER aggregation reporting?"` from any venv with `regrag-api` installed.
