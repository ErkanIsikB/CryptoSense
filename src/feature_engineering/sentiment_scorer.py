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
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

LOGGER = logging.getLogger("sentiment_scorer")

# ── Lazy model loading ─────────────────────────────────────────

_pipeline = None


def _get_pipeline():
    """Lazily load the FinBERT sentiment pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    LOGGER.info("loading FinBERT sentiment model (first use) …")
    # noinspection PyBroadException
    try:
        from transformers import pipeline as hf_pipeline

        _pipeline = hf_pipeline(# type: ignore
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            revision="refs/pr/29",  # <--- THE MAGIC FIX: Pull from the safe PR branch
            top_k=None,  # return all 3 class probabilities
            truncation=True,
            max_length=512,
            device=0
        )
        LOGGER.info("FinBERT model loaded successfully")
    except Exception:
        LOGGER.exception("failed to load FinBERT — sentiment scoring disabled")
        _pipeline = None

    return _pipeline


# ── Scoring helpers ────────────────────────────────────────────


def score_texts_batched(texts: list[str]) -> list[dict[str, float]]:
    """Score a whole list of texts at once using KeyDataset for maximum GPU efficiency."""
    pipe = _get_pipeline()
    if pipe is None or not texts:
        return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0} for _ in texts]

    # noinspection PyBroadException
    try:
        truncated = [t[:512] for t in texts]

        # 1. Create the dataset
        dataset = Dataset.from_dict({"text": truncated})

        parsed = []

        # 2. Use KeyDataset to stream batches directly to the VRAM without list conversion
        batch_results = pipe(KeyDataset(dataset, "text"), batch_size=64)

        for result in batch_results:
            # type: ignore
            scores = {item["label"]: float(item["score"]) for item in result}  # type: ignore
            parsed.append(scores)

        return parsed
    except Exception:
        LOGGER.exception("FinBERT batch inference failed")
        return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0} for _ in texts]


def compound_score(probs: dict[str, float]) -> float:
    """Convert 3-class probabilities to a single score in [−1, +1]."""
    return probs.get("positive", 0.0) - probs.get("negative", 0.0)


# ── Public API ─────────────────────────────────────────────────

INSERT_SQL = """
INSERT INTO tweet_sentiment_5m
    (bucket, symbol, avg_score, tweet_count, positive_count,
     negative_count, neutral_count, max_score, min_score, sample_tweet)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (bucket, symbol) DO UPDATE SET
    avg_score      = EXCLUDED.avg_score,
    tweet_count    = EXCLUDED.tweet_count,
    positive_count = EXCLUDED.positive_count,
    negative_count = EXCLUDED.negative_count,
    neutral_count  = EXCLUDED.neutral_count,
    max_score      = EXCLUDED.max_score,
    min_score      = EXCLUDED.min_score,
    sample_tweet   = EXCLUDED.sample_tweet;
"""


def score_and_store(record: dict[str, Any]) -> None:
    """Score a Tavily sentiment record and write to TimescaleDB."""
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

    # 1. Gather all valid text into a single batch
    valid_items = []
    texts_to_score = []
    for item in results:
        content = str(item.get("content") or "")
        if content.strip():
            valid_items.append(item)
            texts_to_score.append(content)

    if not valid_items:
        LOGGER.debug("no valid text to score for %s", symbol)
        return

    # 2. Fire the GPU exactly ONE time for the whole batch
    batch_scores = score_texts_batched(texts_to_score)

    # 3. Calculate the math
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    max_score = -999.0
    min_score = 999.0
    compound_sum = 0.0
    sample_tweet = ""
    top_relevance = -1.0

    for item, probs in zip(valid_items, batch_scores):
        compound = compound_score(probs)
        compound_sum += compound

        if compound > 0.1:
            positive_count += 1
        elif compound < -0.1:
            negative_count += 1
        else:
            neutral_count += 1

        max_score = max(max_score, compound)
        min_score = min(min_score, compound)

        relevance = float(item.get("score") or 0.0)
        content = str(item.get("content") or "")
        if relevance > top_relevance:
            top_relevance = relevance
            sample_tweet = content

    n = len(valid_items)
    avg_compound = compound_sum / n if n > 0 else 0.0

    row = (
        ts,
        symbol,
        round(avg_compound, 6),
        n,
        positive_count,
        negative_count,
        neutral_count,
        round(max_score, 6) if max_score != -999.0 else None,
        round(min_score, 6) if min_score != 999.0 else None,
        sample_tweet[:500] if sample_tweet else None,
    )

    # noinspection PyBroadException
    try:
        execute_query(INSERT_SQL, row)
        LOGGER.info(
            "sentiment scored and saved: symbol=%s avg_score=%.4f articles=%d pos=%d neg=%d neu=%d",
            symbol,
            avg_compound,
            n,
            positive_count,
            negative_count,
            neutral_count,
        )
    except Exception:
        LOGGER.exception("failed to write sentiment score for %s", symbol)
