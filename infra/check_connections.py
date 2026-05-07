"""Smoke-test all three external services. Run with:
  apps/ingest/.venv/bin/python infra/check_connections.py

Reads .env from apps/ingest/.env. Exits 0 if all three pass, non-zero otherwise.
Prints a clear ✓/✗ per service so you can see which one needs attention.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"

if not ENV_PATH.exists():
    print(f"✗ {ENV_PATH.relative_to(REPO_ROOT)} not found.")
    print(f"  Copy {ENV_PATH.parent / '.env.example'} → {ENV_PATH.name} and fill in values.")
    sys.exit(2)

load_dotenv(ENV_PATH, override=True)  # .env wins over parent shell env

results: list[tuple[str, bool, str]] = []


def check_voyage() -> tuple[bool, str]:
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        return False, "VOYAGE_API_KEY not set in .env"
    try:
        import voyageai
        client = voyageai.Client(api_key=key)
        result = client.embed(["hello world"], model="voyage-3.5-lite", output_dimension=512)
        emb = result.embeddings[0]
        if len(emb) != 512:
            return False, f"unexpected embedding dim: got {len(emb)}, want 512"
        return True, f"voyage-3.5-lite returned {len(emb)}-dim embedding ({result.total_tokens} tokens)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_anthropic() -> tuple[bool, str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return False, "ANTHROPIC_API_KEY not set in .env"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with exactly the word OK."}],
        )
        text = resp.content[0].text.strip()
        return True, f"claude-haiku-4-5 replied: {text!r} (in: {resp.usage.input_tokens}t, out: {resp.usage.output_tokens}t)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_postgres() -> tuple[bool, str]:
    url = os.environ.get("DATABASE_URL")
    if not url:
        return False, "DATABASE_URL not set in .env"
    try:
        import psycopg
        with psycopg.connect(url, autocommit=True, connect_timeout=15) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                pg_version = cur.fetchone()[0].split(",")[0]
                cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                row = cur.fetchone()
                pgvector_installed = row is not None
                cur.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name IN ('documents', 'chunks', 'query_log')
                    ORDER BY table_name
                """)
                tables = [r[0] for r in cur.fetchall()]
        msg = f"connected to {pg_version}; pgvector={'yes' if pgvector_installed else 'NO'}; tables={tables or '(none — apply infra/neon-schema.sql)'}"
        ok = pgvector_installed and len(tables) == 3
        return ok, msg
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


print(f"Checking connections (env: {ENV_PATH.relative_to(REPO_ROOT)})\n")

for name, fn in [("voyage  ", check_voyage), ("anthropic", check_anthropic), ("postgres ", check_postgres)]:
    ok, msg = fn()
    glyph = "✓" if ok else "✗"
    print(f"  {glyph} {name}  {msg}")
    results.append((name.strip(), ok, msg))

failed = [n for n, ok, _ in results if not ok]
if failed:
    print(f"\n{len(failed)}/3 failed: {', '.join(failed)}")
    sys.exit(1)

print("\nAll three connections OK. Ready for embedding + load stages.")
