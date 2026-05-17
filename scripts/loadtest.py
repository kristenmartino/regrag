"""Synthetic load test against the production /chat/stream endpoint.

Generates organically-varied queries by:
  1. Paraphrasing each in-scope eval question 2x via Sonnet
  2. Mixing in ~12 hand-authored free-form queries that simulate organic
     usage patterns (typos, abbreviations, indirect references, etc.)
  3. POSTing to production with simulated user_ids and random delays

Honesty:
  - Every load-test request uses user_id "loadtest-NN" so the audit log
    distinguishes synthetic traffic from real eval / chat usage. A reviewer
    can filter on user_id LIKE 'loadtest%' or NOT in the /audit view.
  - This script is checked into the repo. docs/loadtest.md documents what
    was run and when. Nothing about it is hidden.

Run:
  cd ~/regrag
  packages/eval/.venv/bin/python scripts/loadtest.py --target https://regrag-api-production.up.railway.app
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=True)

import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("loadtest")

SONNET = "claude-sonnet-4-6"
PARAPHRASE_SYSTEM = """\
You generate natural-sounding paraphrases of regulatory questions. Given one question, output exactly two paraphrases that a real user might type.

Variation to apply:
  - Sometimes terse, sometimes elaborate
  - Sometimes formal, sometimes casual
  - Use abbreviations and indirect references occasionally (e.g., "the new interconnection rule" instead of "Order 2023"; "RM18-9" instead of the full docket)
  - Occasional typos and missing punctuation are OK
  - Preserve the core question — same semantic intent

Return JSON: {"paraphrases": ["...", "..."]}
"""

# Hand-authored free-form queries that simulate organic usage patterns the
# eval doesn't fully cover. Mix of well-formed, terse, typo'd, indirect.
ORGANIC_QUERIES = [
    "what's the latest on FERC's interconnection reform",
    "Does Order 2222 apply to behind-the-meter storage?",
    "RM18-9 timeline",
    "How long do utilities have to file compliance with 2023?",
    "compare ferc orders on transmission planning",
    "is energy storage covered by Order 841 even when it's not market-participating",
    "what are RTO requirements under 2222-A",
    "tell me about cluster studies",
    "FERC Order 1920 explained",
    "Federal Register notice for order 745",
    "does PURPA section 210 still apply after order 872",  # tests deep PURPA terminology
    "ferc 5 minute settlement",
]


def load_eval_questions(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text())
    # Only paraphrase in-scope ('answer'-expected) questions; OOS questions are
    # already designed adversarially and paraphrasing them risks losing the OOS
    # property.
    return [q for q in raw.get("questions", []) if q.get("expected_behavior") == "answer"]


def paraphrase(client: anthropic.Anthropic, query: str) -> list[str]:
    try:
        response = client.messages.create(
            model=SONNET,
            max_tokens=512,
            system=PARAPHRASE_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        text = response.content[0].text
        # Trim to first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0:
            return []
        parsed = json.loads(text[start : end + 1])
        return [p for p in (parsed.get("paraphrases") or []) if isinstance(p, str) and p.strip()]
    except Exception as e:
        log.warning("paraphrase failed for %r: %s", query[:60], e)
        return []


def post_chat(target: str, query: str, user_id: str, timeout_s: int = 90) -> dict:
    """POST one query to /chat/stream. Returns a summary dict with key outcomes."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{target}/chat/stream",
            json={"query": query, "user_id": user_id},
            headers={"content-type": "application/json", "accept": "text/event-stream"},
            timeout=timeout_s,
            stream=True,
        )
        if resp.status_code != 200:
            return {
                "user_id": user_id,
                "query_preview": query[:80],
                "ok": False,
                "status": resp.status_code,
                "elapsed_s": round(time.time() - t0, 2),
            }
        # Parse the SSE stream to find the 'done' event
        final_state = None
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            try:
                event = json.loads(raw_line[6:])
            except json.JSONDecodeError:
                continue
            if event.get("type") == "done":
                final_state = event.get("state") or {}
                break
            if event.get("type") == "error":
                return {
                    "user_id": user_id,
                    "query_preview": query[:80],
                    "ok": False,
                    "error": event.get("message", "unknown error"),
                    "elapsed_s": round(time.time() - t0, 2),
                }
        return {
            "user_id": user_id,
            "query_preview": query[:80],
            "ok": True,
            "classification": (final_state or {}).get("classification"),
            "refused": (final_state or {}).get("refusal_emitted", False),
            "n_chunks": len((final_state or {}).get("retrieved_chunks") or []),
            "elapsed_s": round(time.time() - t0, 2),
        }
    except Exception as e:
        return {
            "user_id": user_id,
            "query_preview": query[:80],
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "elapsed_s": round(time.time() - t0, 2),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default="https://regrag-api-production.up.railway.app",
        help="API base URL (default: production)",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=REPO_ROOT / "packages" / "eval" / "src" / "regrag_eval" / "eval_set.yaml",
        help="Path to eval_set.yaml for seed questions",
    )
    parser.add_argument(
        "--paraphrases-per-seed",
        type=int,
        default=2,
        help="How many paraphrases to generate per in-scope eval question",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=20,
        help="Number of distinct simulated user_ids",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=4.0,
        help="Minimum delay between requests (seconds)",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=15.0,
        help="Maximum delay between requests (seconds)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total queries (useful for smoke testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate queries + print them; don't actually POST",
    )
    args = parser.parse_args()

    # ─── Phase 1: assemble query bank ───
    seed_questions = load_eval_questions(args.eval_set)
    log.info("loaded %d in-scope seed questions", len(seed_questions))

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    queries: list[str] = []

    # Original seed questions go in as-is (real queries the system was designed for)
    queries.extend(q["query"] for q in seed_questions)

    # Paraphrases via Sonnet
    log.info("paraphrasing %d seeds x %d each…", len(seed_questions), args.paraphrases_per_seed)
    for i, q in enumerate(seed_questions, 1):
        ps = paraphrase(client, q["query"])
        if ps:
            queries.extend(ps[: args.paraphrases_per_seed])
        if i % 10 == 0:
            log.info("  paraphrased %d/%d", i, len(seed_questions))

    # Free-form organic queries
    queries.extend(ORGANIC_QUERIES)

    # Shuffle to simulate organic order
    random.shuffle(queries)

    if args.limit:
        queries = queries[: args.limit]

    log.info("query bank: %d total", len(queries))

    if args.dry_run:
        for q in queries:
            print(f"  - {q}")
        return 0

    # ─── Phase 2: post to production with random delays + user_ids ───
    log.info("posting to %s …", args.target)
    results: list[dict] = []
    for i, q in enumerate(queries, 1):
        user_id = f"loadtest-{random.randint(1, args.n_users):02d}"
        r = post_chat(args.target, q, user_id)
        r["seq"] = i
        results.append(r)
        status = "OK" if r["ok"] else "FAIL"
        refuse_tag = " REFUSED" if r.get("refused") else ""
        log.info(
            "[%3d/%d] %s%s  %s  %5.1fs  user=%s  q=%r",
            i, len(queries), status, refuse_tag, r.get("classification") or "-",
            r["elapsed_s"], user_id, q[:60],
        )
        if i < len(queries):
            time.sleep(random.uniform(args.delay_min, args.delay_max))

    # ─── Phase 3: summary ───
    n_ok = sum(1 for r in results if r["ok"])
    n_refused = sum(1 for r in results if r.get("refused"))
    total_s = sum(r["elapsed_s"] for r in results)
    log.info("=" * 60)
    log.info("load test complete: %d/%d ok, %d refused", n_ok, len(results), n_refused)
    log.info("total chat-handling time: %.1fs (avg %.1fs/query)", total_s, total_s / max(1, len(results)))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
