"""Unified logging configuration for all ingestion pipelines."""

from __future__ import annotations

import json
import logging


def configure_logging(level: str = "INFO") -> None:
    """Set up root logging with a consistent format across all pipelines."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def emit_status(logger: logging.Logger, status: str, **details: object) -> None:
    """Log a structured JSON status line (used by Binance WebSocket listeners)."""
    payload = {"type": "status", "status": status, **details}
    logger.info(json.dumps(payload, separators=(",", ":")))
