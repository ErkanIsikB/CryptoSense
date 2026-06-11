"""News RSS Aggregator — buckets scored news articles into 5-minute windows.

Scored articles from :mod:`src.feature_engineering.news_rss_scorer` are
assigned to the current 5-minute bucket and upserted into
``news_sentiment_5m``; the ON CONFLICT clause keeps a running average when
multiple poll cycles land in the same bucket.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from src.db.db import execute_batch_async
from src.feature_engineering.news_rss_scorer import ScoredArticle

LOGGER = logging.getLogger("news_rss_aggregator")

WINDOW_S = 300  # 5-minute buckets

INSERT_SQL = """
    INSERT INTO news_sentiment_5m
        (bucket, symbol, avg_score, news_count, positive_count, negative_count, neutral_count, sample_headline)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (bucket, symbol) DO UPDATE SET
        avg_score = (news_sentiment_5m.avg_score * news_sentiment_5m.news_count + EXCLUDED.avg_score) / (news_sentiment_5m.news_count + 1),
        news_count = news_sentiment_5m.news_count + 1,
        positive_count = news_sentiment_5m.positive_count + EXCLUDED.positive_count,
        negative_count = news_sentiment_5m.negative_count + EXCLUDED.negative_count,
        neutral_count = news_sentiment_5m.neutral_count + EXCLUDED.neutral_count,
        sample_headline = EXCLUDED.sample_headline;
"""


def build_rows(articles: list[ScoredArticle]) -> list[tuple[Any, ...]]:
    """Convert scored articles into per-symbol rows for the current bucket."""
    rows: list[tuple[Any, ...]] = []
    bucket_ts = time.time() - (time.time() % WINDOW_S)
    bucket_dt = datetime.fromtimestamp(bucket_ts, tz=timezone.utc)

    for article in articles:
        for symbol in article.symbols:
            positive = 1 if article.score > 0.2 else 0
            negative = 1 if article.score < -0.2 else 0
            neutral = 1 if -0.2 <= article.score <= 0.2 else 0

            rows.append((
                bucket_dt,
                symbol,
                article.score,
                1,  # news_count
                positive,
                negative,
                neutral,
                article.text[:500],  # sample headline with source and description
            ))

    return rows


async def aggregate_and_store(articles: list[ScoredArticle]) -> None:
    """Bucket scored articles and upsert them into ``news_sentiment_5m``."""
    rows = build_rows(articles)
    if not rows:
        return

    try:
        await execute_batch_async(INSERT_SQL, rows)
        LOGGER.info("Flushed %d news sentiment records to DB.", len(rows))
    except Exception as e:
        LOGGER.error("Failed to flush news sentiment to DB: %s", e)
