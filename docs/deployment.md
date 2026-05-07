# RegRAG Deployment

The demo runs on three services:

| Layer    | Provider | What's deployed                              |
|----------|----------|----------------------------------------------|
| Database | Neon     | Postgres 17 + pgvector (already provisioned) |
| Backend  | Railway  | `apps/api` — FastAPI / LangGraph             |
| Frontend | Vercel   | `apps/web` — Next.js 16 + shadcn/ui          |

Local dev still works as documented in the root README; this doc is only for the public deploy.

---

## 1. Prerequisites

- `railway` CLI installed (`brew install railway`)
- `vercel` CLI installed (`npm i -g vercel`)
- `gh` CLI installed (`brew install gh`) — for the optional GitHub repo step
- A Neon project (already done, connection string in `~/regrag/.env`)
- Voyage AI key + Anthropic key — same ones used in local dev (see [project_regrag_secrets.md](memory) for rotation reminder before public exposure)

---

## 2. Backend → Railway

### 2.1 First-time setup

```bash
cd ~/regrag/apps/api

# Authenticate (opens browser)
railway login

# Create or link a project. First time:
railway init --name regrag-api
# (or `railway link` if you've already created the project in the dashboard)

# Set env vars (one at a time, or via the Railway dashboard)
railway variables set DATABASE_URL="$(grep ^DATABASE_URL ~/regrag/.env | cut -d= -f2-)"
railway variables set VOYAGE_API_KEY="$(grep ^VOYAGE_API_KEY ~/regrag/.env | cut -d= -f2-)"
railway variables set ANTHROPIC_API_KEY="$(grep ^ANTHROPIC_API_KEY ~/regrag/.env | cut -d= -f2-)"

# Set CORS origins (comma-separated). Add the Vercel URL after the first frontend deploy.
railway variables set ALLOWED_ORIGINS="https://regrag.vercel.app"
```

### 2.2 Deploy

```bash
cd ~/regrag/apps/api
railway up --detach
```

Railway builds the Dockerfile, deploys, and assigns a public URL like `https://regrag-api-production.up.railway.app`. Verify:

```bash
curl https://<your-railway-url>/health
# → {"status":"ok"}
```

### 2.3 Cron / corpus updates (optional, not part of demo)

`apps/ingest` is a one-shot CLI, not a long-running service. If you want a weekly corpus-discovery job, schedule it in Railway as a **cron service** (separate service in the same project) running:

```
regrag-ingest run --manifest /app/corpus/manifest.yaml
```

Skip for the demo — the seed corpus is static.

---

## 3. Frontend → Vercel

### 3.1 First-time setup

```bash
cd ~/regrag/apps/web

vercel login
vercel link  # prompts for project name → regrag
```

### 3.2 Set the API URL

```bash
vercel env add NEXT_PUBLIC_API_URL production
# paste the Railway URL from 2.2 (e.g. https://regrag-api-production.up.railway.app)
```

### 3.3 Deploy

```bash
vercel deploy --prod
```

Vercel auto-detects Next.js, builds, and assigns a URL like `https://regrag.vercel.app`.

### 3.4 Update Railway CORS

Once the Vercel URL is final, add it to the backend's CORS allowlist:

```bash
cd ~/regrag/apps/api
railway variables set ALLOWED_ORIGINS="https://regrag.vercel.app"
# (Railway redeploys automatically when env changes)
```

---

## 4. Custom domain (optional)

In the Vercel dashboard → Domains → Add `regrag.kristenmartino.ai`. Vercel shows the CNAME record needed; add it at your DNS provider. Propagation: 5–60 min.

For the API on Railway, add a custom domain similarly (e.g. `api.regrag.kristenmartino.ai`) → CNAME to the Railway-assigned domain. Then update `NEXT_PUBLIC_API_URL` in Vercel and redeploy.

---

## 5. Post-deploy checklist

- [ ] Backend `/health` returns 200
- [ ] Backend `/audit` returns rows (Neon connection works in prod env)
- [ ] Frontend chat page loads
- [ ] One sample query succeeds end-to-end with streaming visible in Pipeline panel
- [ ] One audit row appears in `/audit` after the test query
- [ ] CORS lockdown verified: a curl from a non-allowlisted origin gets blocked

## 6. Operational notes

- **Anthropic spend**: capped at $25/mo per the user's setup. A malicious actor with the demo URL could burn ~500–5,000 queries/mo before hitting the cap (~$0.005–$0.05/query). Watch the Anthropic console; rotate the key + redeploy if you see abnormal usage.
- **Voyage spend**: free tier covers 200M tokens/mo; the corpus + chat usage is well under.
- **Neon**: free tier is 0.5 GB storage + 191 compute hours/mo. Current corpus is ~5 MB; chat traffic is low. Monitor in Neon dashboard.
- **Logs**: `railway logs --tail` for backend; Vercel dashboard for frontend.

## 7. Tearing down

```bash
cd ~/regrag/apps/api && railway down
cd ~/regrag/apps/web && vercel remove regrag --yes
```

Neon project stays — delete from the Neon dashboard if you want to fully clean up.
