"""Sentiment Scorer — runs FinBERT on Tavily search results and writes numeric
scores to TimescaleDB.

The model ``ProsusAI/finbert`` is a finance-domain BERT that classifies text
into *positive*, *negative*, or *neutral* and returns calibrated probabilities.
We convert these into a single compound score in [−1, +1]:

    score = p(positive) − p(negative)

The model is loaded lazily on first use to avoid slowing down import time.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.db.db import execute_query

LOGGER = logging.getLogger("sentiment_scorer")

# ── Lazy model loading ──────────────────────────────────────────

_pipeline = None


def _get_pipeline():
    """Lazily load the FinBERT sentiment pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    LOGGER.info("loading FinBERT sentiment model (first use) …")
    try:
        from transformers import pipeline as hf_pipeline

        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,  # return all 3 class probabilities
            truncation=True,
            max_length=512,
        )
        LOGGER.info("FinBERT model loaded successfully")
    except Exception:
        LOGGER.exception("failed to load FinBERT — sentiment scoring disabled")
        _pipeline = None

    return _pipeline


# ── Scoring helpers ─────────────────────────────────────────────


def _score_text(text: str) -> dict[str, float]:
    """Return ``{"positive": p, "negative": p, "neutral": p}`` for one text."""
    pipe = _get_pipeline()
    if pipe is None:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    try:
        result = pipe(text[:512])  # truncate to model max length
        # result is [[{"label": "positive", "score": 0.9}, ...]]
        scores = {item["label"]: item["score"] for item in result[0]}
        return scores
    except Exception:
        LOGGER.exception("FinBERT inference failed for text: %s…", text[:60])
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}


def _compound_score(probs: dict[str, float]) -> float:
    """Convert 3-class probabilities to a single score in [−1, +1]."""
    return probs.get("positive", 0.0) - probs.get("negative", 0.0)


# ── Public API ──────────────────────────────────────────────────

INSERT_SQL = """
INSERT INTO sentiment_scores
    (time, symbol, score, article_count, positive_ratio,
     negative_ratio, avg_relevance, top_headline)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (time, symbol) DO UPDATE SET
    score          = EXCLUDED.score,
    article_count  = EXCLUDED.article_count,
    positive_ratio = EXCLUDED.positive_ratio,
    negative_ratio = EXCLUDED.negative_ratio,
    avg_relevance  = EXCLUDED.avg_relevance,
    top_headline   = EXCLUDED.top_headline;
"""


def score_and_store(record: dict[str, Any]) -> None:
    """Score a Tavily sentiment record and write to TimescaleDB.

    Parameters
    ----------
    record : dict
        A record dict as built by ``tavily_ingestion.py``, containing at
        minimum ``token``, ``timestamp``, and ``results`` (list of dicts
        with ``content``, ``score``, ``title``).
    """
    symbol = record.get("token", "UNKNOWN")
    timestamp_str = record.get("timestamp")
    results = record.get("results", [])

    if not results:
        LOGGER.debug("no results to score for %s", symbol)
        return

    # Parse timestamp
    try:
        if timestamp_str:
            ts = datetime.fromisoformat(
                timestamp_str.replace("Z", "+00:00")
            )
        else:
            ts = datetime.now(timezone.utc)
    except (ValueError, TypeError):
        ts = datetime.now(timezone.utc)

    # Score each article / tweet
    positive_count = 0
    negative_count = 0
    relevance_sum = 0.0
    compound_sum = 0.0
    top_headline = ""
    top_relevance = 0.0

    for item in results:
        content = str(item.get("content") or "")
        if not content.strip():
            continue

        probs = _score_text(content)
        compound = _compound_score(probs)
        compound_sum += compound

        if compound > 0.1:
            positive_count += 1
        elif compound < -0.1:
            negative_count += 1

        relevance = float(item.get("score") or 0.0)
        relevance_sum += relevance

        title = str(item.get("title") or "")
        if relevance > top_relevance:
            top_relevance = relevance
            top_headline = title

    n = len(results)
    avg_compound = compound_sum / n if n > 0 else 0.0
    positive_ratio = positive_count / n if n > 0 else 0.0
    negative_ratio = negative_count / n if n > 0 else 0.0
    avg_relevance = relevance_sum / n if n > 0 else 0.0

    row = (
        ts,
        symbol,
        round(avg_compound, 6),
        n,
        round(positive_ratio, 4),
        round(negative_ratio, 4),
        round(avg_relevance, 4),
        top_headline[:500] if top_headline else None,
    )

    try:
        execute_query(INSERT_SQL, row)
        LOGGER.info(
            "sentiment scored: symbol=%s score=%.4f articles=%d pos=%.0f%% neg=%.0f%%",
            symbol,
            avg_compound,
            n,
            positive_ratio * 100,
            negative_ratio * 100,
        )
    except Exception:
        LOGGER.exception("failed to write sentiment score for %s", symbol)
