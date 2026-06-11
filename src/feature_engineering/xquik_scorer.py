"""XQuik Scorer — runs FinBERT on XQuik search results and writes numeric
scores to TimescaleDB.

Uses the shared :mod:`src.feature_engineering.finbert` engine for inference
and applies XQuik/tweet-specific enrichment (source credibility, weighting)
before persisting to ``tweet_sentiment_5m``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.db.db import execute_query
from src.feature_engineering.finbert import compound_score, score_texts_batched
from src.feature_engineering.source_credibility import (
    calculate_weighted_sentiment,
    enrich_scored_item,
    extract_author_handle,
)

LOGGER = logging.getLogger("xquik_scorer")

# ── Public API ─────────────────────────────────────────────────

INSERT_SQL = """
INSERT INTO tweet_sentiment_5m
    (bucket, symbol, avg_score, tweet_count, positive_count,
     negative_count, neutral_count, max_score, min_score, sample_tweet,
     weighted_avg_score, total_source_weight, tier1_count, tier2_count,
     tier3_count, economy_news_count, turkish_economy_count, unknown_count)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (bucket, symbol) DO UPDATE SET
    avg_score      = EXCLUDED.avg_score,
    tweet_count    = EXCLUDED.tweet_count,
    positive_count = EXCLUDED.positive_count,
    negative_count = EXCLUDED.negative_count,
    neutral_count  = EXCLUDED.neutral_count,
    max_score      = EXCLUDED.max_score,
    min_score      = EXCLUDED.min_score,
    sample_tweet   = EXCLUDED.sample_tweet,
    weighted_avg_score       = EXCLUDED.weighted_avg_score,
    total_source_weight      = EXCLUDED.total_source_weight,
    tier1_count              = EXCLUDED.tier1_count,
    tier2_count              = EXCLUDED.tier2_count,
    tier3_count              = EXCLUDED.tier3_count,
    economy_news_count       = EXCLUDED.economy_news_count,
    turkish_economy_count    = EXCLUDED.turkish_economy_count,
    unknown_count            = EXCLUDED.unknown_count;
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
    enriched_items: list[dict[str, Any]] = []

    for item, probs in zip(valid_items, batch_scores):
        compound = compound_score(probs)
        compound_sum += compound
        enrich_scored_item(item, compound, extract_author_handle(item))
        enriched_items.append(item)

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
    weighted_stats = calculate_weighted_sentiment(enriched_items)
    weighted_avg = weighted_stats["weighted_avg_sentiment"]

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
        round(weighted_avg, 6),
        round(float(weighted_stats["total_source_weight"]), 6),
        int(weighted_stats["tier1_count"]),
        int(weighted_stats["tier2_count"]),
        int(weighted_stats["tier3_count"]),
        int(weighted_stats["economy_news_count"]),
        int(weighted_stats["turkish_economy_count"]),
        int(weighted_stats["unknown_count"]),
    )

    # noinspection PyBroadException
    try:
        execute_query(INSERT_SQL, row)
        LOGGER.info(
            "sentiment scored and saved: symbol=%s avg_score=%.4f weighted_avg=%.4f "
            "articles=%d total_weight=%.2f pos=%d neg=%d neu=%d tiers=%s",
            symbol,
            avg_compound,
            weighted_avg,
            n,
            weighted_stats["total_source_weight"],
            positive_count,
            negative_count,
            neutral_count,
            {
                "tier1": weighted_stats["tier1_count"],
                "tier2": weighted_stats["tier2_count"],
                "tier3": weighted_stats["tier3_count"],
                "economy_news": weighted_stats["economy_news_count"],
                "turkish_economy": weighted_stats["turkish_economy_count"],
                "unknown": weighted_stats["unknown_count"],
            },
        )
    except Exception:
        LOGGER.exception("failed to write sentiment score for %s", symbol)
