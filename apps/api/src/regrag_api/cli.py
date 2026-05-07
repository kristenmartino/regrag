"""regrag-retrieve and regrag-chat CLIs — terminal smoke tests."""

from __future__ import annotations

import logging
from pathlib import Path

import click
from dotenv import load_dotenv

from .orchestration.graph import run as run_graph
from .retrieval.hybrid import hybrid_retrieve
from .retrieval.identifiers import extract_identifiers


def _load_env():
    repo_root = Path(__file__).resolve().parents[4]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


@click.command()
@click.argument("query", nargs=-1, required=True)
@click.option("-k", "--top-k", default=8, help="Number of chunks to return")
@click.option("-v", "--verbose", count=True, help="Increase log verbosity")
def retrieve(query: tuple[str, ...], top_k: int, verbose: int):
    """Run hybrid retrieval against the corpus and print the top results."""
    level = logging.WARNING - (10 * verbose)
    logging.basicConfig(level=max(logging.DEBUG, level), format="%(levelname)s %(message)s")
    _load_env()

    q = " ".join(query)
    ids = extract_identifiers(q)
    print(f"Query: {q}")
    if not ids.is_empty:
        print(f"Identifiers extracted:")
        if ids.orders:     print(f"  orders:     {ids.orders}")
        if ids.dockets:    print(f"  dockets:    {ids.dockets}")
        if ids.ferc_cites: print(f"  FERC cites: {ids.ferc_cites}")
        if ids.usc_cites:  print(f"  USC cites:  {ids.usc_cites}")
        if ids.cfr_cites:  print(f"  CFR cites:  {ids.cfr_cites}")
    print()

    results = hybrid_retrieve(q, k=top_k)
    print(f"Top {len(results)} results:")
    for i, r in enumerate(results, 1):
        marks = []
        if r.vector_rank is not None: marks.append(f"v#{r.vector_rank}")
        if r.keyword_rank is not None: marks.append(f"k#{r.keyword_rank}")
        if r.floor_match: marks.append("floor")
        marks_str = ",".join(marks)

        scores = []
        if r.cosine_sim is not None: scores.append(f"cos={r.cosine_sim:.3f}")
        if r.ts_rank is not None: scores.append(f"ts={r.ts_rank:.3f}")
        scores_str = " ".join(scores)

        preview = r.chunk_text.replace("\n", " ")[:140]
        section = r.section_heading or "?"
        print(
            f"  {i:>2}. [rrf={r.rrf_score:.4f}] [{marks_str}] [{scores_str}]\n"
            f"      {r.chunk_id} ({section})\n"
            f"      {preview}"
        )


@click.command()
@click.argument("query", nargs=-1, required=True)
@click.option("-v", "--verbose", count=True, help="Increase log verbosity")
@click.option("--show-chunks", is_flag=True, help="Print the retrieved chunks before the answer")
def chat(query: tuple[str, ...], verbose: int, show_chunks: bool):
    """Run the full RegRAG flow on a query: classify → (decompose →) retrieve → synthesize → verify."""
    level = logging.WARNING - (10 * verbose)
    logging.basicConfig(level=max(logging.DEBUG, level), format="%(levelname)s %(message)s")
    _load_env()

    q = " ".join(query)
    print(f"Query: {q}\n")
    state = run_graph(q)

    intent = state.get("classification") or "?"
    confidence = state.get("classification_confidence")
    conf_str = f" ({confidence:.2f})" if confidence is not None else ""
    print(f"Classification: {intent}{conf_str}")
    if state.get("sub_queries"):
        print("Sub-queries:")
        for s in state["sub_queries"]:
            print(f"  - {s}")
    chunks = state.get("retrieved_chunks") or []
    print(f"Retrieved chunks: {len(chunks)} (top cosine={state.get('top_cosine_sim'):.3f})")
    if show_chunks:
        for i, c in enumerate(chunks[:8], 1):
            preview = c["chunk_text"].replace("\n", " ")[:120]
            print(f"  {i}. {c['chunk_id']} ({c.get('section_heading')}): {preview}")

    print()
    if state.get("refusal_emitted"):
        print(f"REFUSED ({state.get('refusal_reason')}): {state.get('final_answer') or ''}")
    else:
        print(f"Answer ({state.get('citations_stripped', 0)} citations stripped, "
              f"regen={state.get('regeneration_count', 0)}):")
        print()
        print(state.get("final_answer") or "(no answer)")

    timings = state.get("timings", {})
    total_ms = sum(timings.values())
    print(f"\n— stages: {timings} | total: {total_ms} ms")


if __name__ == "__main__":
    retrieve()
