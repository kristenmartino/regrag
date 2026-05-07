"""regrag-ingest CLI."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from .chunk import chunk_document, chunk_to_dict
from .fetch import FetchResult, fetch_all
from .load import get_conn, upsert_chunks, upsert_document
from .manifest import ManifestEntry, load_manifest
from .parse import parse_pdf, parsed_to_dict


@click.group()
@click.option("-v", "--verbose", count=True, help="Increase log verbosity")
@click.option(
    "--env-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to .env (defaults to apps/ingest/.env relative to repo root)",
)
def cli(verbose: int, env_file: Path | None):
    level = logging.WARNING - (10 * verbose)
    logging.basicConfig(
        level=max(logging.DEBUG, level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if env_file is None:
        # default: ~/regrag/.env (repo root, shared between apps/ingest and apps/api)
        repo_root = Path(__file__).resolve().parents[4]
        env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)


@cli.command()
@click.option(
    "--manifest", "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to corpus/manifest.yaml",
)
@click.option(
    "--corpus-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Root of the corpus tree (defaults to manifest's parent dir)",
)
@click.option("--skip-fetch", is_flag=True, help="Use existing PDFs in corpus/raw/ — don't download")
@click.option("--skip-parse", is_flag=True, help="Stop after fetch — don't parse PDFs")
@click.option("--skip-chunk", is_flag=True, help="Stop after parse — don't chunk")
@click.option("--skip-load", is_flag=True, help="Stop after chunk — don't embed/load to DB")
def run(manifest_path, corpus_root, skip_fetch, skip_parse, skip_chunk, skip_load):
    """Run the full pipeline: discover → fetch → parse → chunk → embed → load."""
    manifest_path = manifest_path.resolve()
    corpus_root = corpus_root or manifest_path.parent
    raw_dir = corpus_root / "raw"
    parsed_dir = corpus_root / "parsed"
    chunks_dir = corpus_root / "chunks"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("regrag_ingest")
    entries = load_manifest(manifest_path)
    log.warning("manifest: %d documents from %s", len(entries), manifest_path)

    fetch_results: dict[str, FetchResult] = {}

    if not skip_fetch:
        log.warning("== FETCH ==")
        for r in fetch_all(entries, raw_dir):
            fetch_results[r.entry.slug] = r
        n_fetched = sum(1 for r in fetch_results.values() if r.status != "cached")
        n_cached = sum(1 for r in fetch_results.values() if r.status == "cached")
        log.warning("fetch complete: %d fetched, %d cached", n_fetched, n_cached)

    if not skip_parse:
        log.warning("== PARSE ==")
        for entry in entries:
            raw_path = raw_dir / f"{entry.slug}.pdf"
            if not raw_path.exists():
                log.error("missing raw PDF: %s", raw_path)
                continue
            parsed = parse_pdf(raw_path)
            out_path = parsed_dir / f"{entry.slug}.json"
            out_path.write_text(json.dumps(parsed_to_dict(parsed), indent=2))
            log.info(
                "parsed: %s → %d sections, %d footnotes",
                entry.slug, len(parsed.sections), len(parsed.footnotes),
            )

    if skip_chunk:
        return

    log.warning("== CHUNK ==")
    chunks_per_doc: dict[str, list] = {}
    for entry in entries:
        parsed_path = parsed_dir / f"{entry.slug}.json"
        if not parsed_path.exists():
            continue
        parsed = json.loads(parsed_path.read_text())
        accession = entry.accession_number or entry.slug
        chunks = chunk_document(parsed, accession=accession)
        chunks_per_doc[entry.slug] = chunks
        chunks_path = chunks_dir / f"{entry.slug}.json"
        chunks_path.write_text(json.dumps([chunk_to_dict(c) for c in chunks], indent=2))
        sizes = [len(c.chunk_text) for c in chunks]
        log.warning(
            "chunked: %s → %d chunks, avg %d chars, max %d chars",
            entry.slug, len(chunks),
            sum(sizes) // max(1, len(sizes)),
            max(sizes) if sizes else 0,
        )

    if skip_load:
        return

    log.warning("== LOAD ==")
    if not fetch_results:
        # populate from disk if --skip-fetch was used
        from hashlib import sha256
        for entry in entries:
            raw_path = raw_dir / f"{entry.slug}.pdf"
            if not raw_path.exists():
                continue
            h = sha256(raw_path.read_bytes()).hexdigest()
            fetch_results[entry.slug] = FetchResult(entry, raw_path, h, "cached", 0)

    conn = get_conn()
    total_embedded = 0
    total_skipped = 0
    try:
        for entry in entries:
            fr = fetch_results.get(entry.slug)
            if not fr:
                log.error("no fetch result for %s — skipping load", entry.slug)
                continue
            chunks = chunks_per_doc.get(entry.slug, [])
            if not chunks:
                log.warning("no chunks for %s — skipping", entry.slug)
                continue
            upsert_document(conn, entry, fr)
            n_emb, n_skip = upsert_chunks(conn, chunks, skip_existing=True)
            conn.commit()
            log.warning(
                "loaded: %s → %d chunks (embedded %d, skipped %d)",
                entry.slug, len(chunks), n_emb, n_skip,
            )
            total_embedded += n_emb
            total_skipped += n_skip
    finally:
        conn.close()
    log.warning("load complete: %d new chunks embedded, %d skipped", total_embedded, total_skipped)


if __name__ == "__main__":
    cli()
