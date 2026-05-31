"""Structured (JSON) logging for the API (issue #7).

`configure_logging()` installs a JSON formatter on the root logger by default so
Railway logs are queryable by field; `REGRAG_LOG_FORMAT=text` opts into the prior
human-readable format for local dev. Stdlib only — no extra dependency.

Log calls pass structured fields via the standard `extra={}` kwarg; the formatter
emits them as top-level JSON keys. NEVER pass raw query / answer / prompt / chunk
text in `extra` — logs are metadata only (issues #7/#9).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

# Standard LogRecord attributes; anything else on a record is treated as an
# `extra` field and emitted as a JSON key.
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
    "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, msg + any `extra` fields."""

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                out[key] = value
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


_TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Install the structured (or text) formatter on the root logger.

    REGRAG_LOG_FORMAT=json (default) → JSON lines; =text → human-readable.
    """
    fmt = os.environ.get("REGRAG_LOG_FORMAT", "json").strip().lower()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_TEXT_FORMAT) if fmt == "text" else JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
