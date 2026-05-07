"""Shared Anthropic client + JSON-output parsing helpers."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

import anthropic

CLASSIFIER_MODEL = "claude-haiku-4-5"
SYNTHESIS_MODEL = "claude-sonnet-4-6"
DECOMPOSER_MODEL = "claude-sonnet-4-6"


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def parse_json_response(text: str) -> dict:
    """Extract a JSON object from a model response. Tolerates ```json fences,
    leading/trailing whitespace, and pre/post commentary.

    Raises ValueError with the original text if no JSON object is found.
    """
    stripped = text.strip()
    fence_match = _FENCE_RE.match(stripped)
    if fence_match:
        return json.loads(fence_match.group(1))
    # try the whole string first
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # fall back: find the first balanced { ... } block
    start = stripped.find("{")
    if start < 0:
        raise ValueError(f"no JSON object found in response: {text!r}")
    depth = 0
    for i in range(start, len(stripped)):
        if stripped[i] == "{":
            depth += 1
        elif stripped[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start:i + 1])
    raise ValueError(f"unbalanced braces in response: {text!r}")
