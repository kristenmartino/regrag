"""Stage 5: embed chunks via Voyage AI's voyage-3.5-lite at 512 dimensions.

Batched at 128 (Voyage's tolerated default). Idempotent at the call-site level
(load.py skips chunks whose chunk_content_hash + embedding_model already match).
"""

from __future__ import annotations

import logging
import os

import voyageai

EMBEDDING_MODEL = "voyage-3.5-lite"
EMBEDDING_DIM = 512
BATCH_SIZE = 128

log = logging.getLogger(__name__)


def embed_texts(texts: list[str], *, input_type: str = "document") -> list[list[float]]:
    """Embed a list of strings, returning one 512-dim vector per input."""
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY not set in environment")
    client = voyageai.Client(api_key=api_key)
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = client.embed(
            batch,
            model=EMBEDDING_MODEL,
            output_dimension=EMBEDDING_DIM,
            input_type=input_type,
        )
        out.extend(result.embeddings)
        log.info(
            "embedded batch %d/%d (%d texts, %d tokens)",
            i // BATCH_SIZE + 1,
            (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE,
            len(batch),
            result.total_tokens,
        )
    return out
