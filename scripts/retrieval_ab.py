"""Retrieval-only A/B: hybrid vs pure-vector recall, by identifier-heaviness.

Issue #5. Isolates the retrieval step (no synthesis) to test the case-study §7
claim that hybrid retrieval (vector + keyword + identifier floor) beats pure
vector on identifier-heavy queries. Uses the SAME keyword-recall definition the
eval uses (metrics.score_retrieval_recall): fraction of a question's
expected_passages_keywords that appear, case-insensitive, in any retrieved chunk.

Run: python scripts/retrieval_ab.py   (needs .env with DATABASE_URL + VOYAGE_API_KEY)
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api" / "src"))

# load .env
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from regrag_api.retrieval.hybrid import hybrid_retrieve  # noqa: E402
from regrag_api.baseline import _pure_vector_topk  # noqa: E402  (pure-vector arm, same path as --baseline)

K = 10
# Identifier-heavy = the query carries an exact regulatory identifier that a
# pure embedding tends to under-weight: an order number, a docket number, or a
# statutory section cite.
ID_RE = re.compile(r"(Order\s+No\.?\s*\d+|Order\s+\d+|RM\d+-\d+|\bAD\d+-\d+|§\s*\d+|\bDocket)", re.I)
# Docket-only is the strongest sub-case (bare alphanumeric docket tokens).
DOCKET_RE = re.compile(r"(RM\d+-\d+|\bAD\d+-\d+|\bER\d+-\d+)", re.I)


def _text(c):
    # hybrid_retrieve returns RetrievedChunk dataclasses; _pure_vector_topk returns dicts.
    if isinstance(c, dict):
        return c.get("chunk_text") or ""
    return getattr(c, "chunk_text", "") or ""


def recall(keywords, chunks):
    body = " \n ".join(_text(c) for c in chunks).lower()
    if not keywords:
        return None
    hit = sum(1 for kw in keywords if kw.lower() in body)
    return hit / len(keywords)


def main():
    qs = yaml.safe_load((ROOT / "packages/eval/src/regrag_eval/eval_set.yaml").read_text())["questions"]
    answerable = [q for q in qs if q.get("expected_behavior") == "answer" and q.get("expected_passages_keywords")]

    rows = []
    for q in answerable:
        kws = q["expected_passages_keywords"]
        h = recall(kws, hybrid_retrieve(q["query"], k=K))
        v = recall(kws, _pure_vector_topk(q["query"], K))
        rows.append({
            "id": q["id"],
            "id_heavy": bool(ID_RE.search(q["query"])),
            "docket": bool(DOCKET_RE.search(q["query"])),
            "hybrid": h, "vector": v, "delta": h - v,
        })

    def summarize(label, subset):
        if not subset:
            print(f"{label:28} (n=0)")
            return
        mh = sum(r["hybrid"] for r in subset) / len(subset)
        mv = sum(r["vector"] for r in subset) / len(subset)
        wins = sum(1 for r in subset if r["delta"] > 1e-9)
        losses = sum(1 for r in subset if r["delta"] < -1e-9)
        print(f"{label:28} n={len(subset):2}  hybrid={mh*100:5.1f}%  vector={mv*100:5.1f}%  "
              f"Δ={(mh-mv)*100:+5.1f}pp  (hybrid wins {wins}, loses {losses})")

    print(f"Retrieval-only A/B (k={K}), keyword recall, {len(answerable)} answer-expected questions\n")
    summarize("ALL answer questions", rows)
    summarize("identifier-heavy", [r for r in rows if r["id_heavy"]])
    summarize("NOT identifier-heavy", [r for r in rows if not r["id_heavy"]])
    summarize("docket-number queries", [r for r in rows if r["docket"]])
    print("\nPer-question where hybrid != vector:")
    for r in sorted(rows, key=lambda r: r["delta"]):
        if abs(r["delta"]) > 1e-9:
            tag = "ID" if r["id_heavy"] else "  "
            print(f"  [{tag}] {r['id']:18} hybrid={r['hybrid']*100:5.1f}  vector={r['vector']*100:5.1f}  Δ={r['delta']*100:+.1f}")

    import json
    def agg(subset):
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "hybrid": round(sum(r["hybrid"] for r in subset) / len(subset) * 100, 1),
            "vector": round(sum(r["vector"] for r in subset) / len(subset) * 100, 1),
            "wins": sum(1 for r in subset if r["delta"] > 1e-9),
            "losses": sum(1 for r in subset if r["delta"] < -1e-9),
        }
    out = {
        "all": agg(rows),
        "identifier_heavy": agg([r for r in rows if r["id_heavy"]]),
        "not_identifier_heavy": agg([r for r in rows if not r["id_heavy"]]),
        "docket": agg([r for r in rows if r["docket"]]),
        "wins": [r["id"] for r in rows if r["delta"] > 1e-9],
        "losses": [r["id"] for r in rows if r["delta"] < -1e-9],
        "docket_ids": [r["id"] for r in rows if r["docket"]],
        "rows": rows,
    }
    json.dump(out, open("/tmp/ab.json", "w"), indent=2)


if __name__ == "__main__":
    main()
