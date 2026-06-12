"""XQuik Aggregator — buckets individual tweet scores into 5-minute windows.

Works identically to ``TradeAggregator``: incoming scored tweets are buffered
by ``(symbol, 5-min bucket)``.  When a new tweet arrives in a later bucket,
completed buckets are flushed to ``tweet_sentiment_5m``.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.core.config import settings
from src.db.db import execute_batch
from src.feature_engineering.source_credibility import empty_tier_counts, tier_count_field

LOGGER = logging.getLogger("xquik_aggregator")

WINDOW_S: int = settings.AGGREGATION_WINDOW_SECONDS  # 300 = 5 min


@dataclass
class SentimentAccumulator:
    """Mutable accumulator for a single (symbol, bucket)."""

    bucket_ts: float
    symbol: str
    score_sum: float = 0.0
    weighted_score_sum: float = 0.0
    total_source_weight: float = 0.0
    count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    max_score: float = -math.inf
    min_score: float = math.inf
    sample_tweet: str = ""
    _best_engagement: float = field(default=-1.0, repr=False)
    tier_counts: dict[str, int] = field(default_factory=empty_tier_counts)

    def add(
        self,
        score: float,
        tweet_text: str = "",
        engagement: float = 0.0,
        source_weight: float = 1.0,
        source_tier: str = "unknown",
    ) -> None:
        try:
            source_weight = max(float(source_weight or 1.0), 0.0)
        except (TypeError, ValueError):
            source_weight = 1.0
        self.score_sum += score
        self.weighted_score_sum += score * source_weight
        self.total_source_weight += source_weight
        self.count += 1
        self.tier_counts[tier_count_field(source_tier)] += 1

        if score > 0.1:
            self.positive_count += 1
        elif score < -0.1:
            self.negative_count += 1
        else:
            self.neutral_count += 1

        self.max_score = max(self.max_score, score)
        self.min_score = min(self.min_score, score)

        # Keep the tweet with the highest engagement as sample
        if engagement > self._best_engagement and tweet_text:
            self._best_engagement = engagement
            self.sample_tweet = tweet_text

    def to_row(self) -> tuple[Any, ...]:
        avg = self.score_sum / self.count if self.count > 0 else 0.0
        weighted_avg = (
            self.weighted_score_sum / self.total_source_weight
            if self.total_source_weight > 0
            else 0.0
        )
        bucket_dt = datetime.fromtimestamp(self.bucket_ts, tz=timezone.utc)
        LOGGER.debug(
            "sentiment bucket ready: symbol=%s bucket=%s avg=%.4f weighted_avg=%.4f "
            "count=%d total_weight=%.2f tiers=%s",
            self.symbol,
            bucket_dt.isoformat(),
            avg,
            weighted_avg,
            self.count,
            self.total_source_weight,
            self.tier_counts,
        )
        return (
            bucket_dt,
            self.symbol,
            round(avg, 6),
            self.count,
            self.positive_count,
            self.negative_count,
            self.neutral_count,
            round(self.max_score, 6) if self.max_score != -math.inf else None,
            round(self.min_score, 6) if self.min_score != math.inf else None,
            self.sample_tweet[:500] if self.sample_tweet else None,
            round(weighted_avg, 6),
            round(self.total_source_weight, 6),
            self.tier_counts["tier1_count"],
            self.tier_counts["tier2_count"],
            self.tier_counts["tier3_count"],
            self.tier_counts["economy_news_count"],
            self.tier_counts["turkish_economy_count"],
            self.tier_counts["unknown_count"],
        )


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


def _bucket_start(epoch_s: float) -> float:
    return epoch_s - (epoch_s % WINDOW_S)


class SentimentAggregator:
    """Thread-safe 5-minute tweet sentiment aggregator."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[str, float], SentimentAccumulator] = {}
        self._lock = threading.Lock()
        self.system_boot_time = datetime.now(timezone.utc)
        self._last_flushed_bucket = _bucket_start(self.system_boot_time.timestamp())

    def add(
        self,
        symbol: str,
        score: float,
        tweet_time_s: float,
        tweet_text: str = "",
        engagement: float = 0.0,
        source_weight: float = 1.0,
        source_tier: str = "unknown",
    ) -> None:
        bucket_ts = _bucket_start(tweet_time_s)
        key = (symbol, bucket_ts)

        with self._lock:
            acc = self._buckets.get(key)
            if acc is None:
                acc = SentimentAccumulator(bucket_ts=bucket_ts, symbol=symbol)
                self._buckets[key] = acc
            acc.add(score, tweet_text, engagement, source_weight, source_tier)

        self.maybe_flush(tweet_time_s)

    def flush_all(self) -> None:
        """Force-flush all open buckets (shutdown)."""
        with self._lock:
            rows = []
            for acc in self._buckets.values():
                if acc.bucket_ts >= self.system_boot_time.timestamp():
                    if acc.count > 0:
                        rows.append(acc.to_row())
                    else:
                        bucket_dt = datetime.fromtimestamp(acc.bucket_ts, tz=timezone.utc)
                        rows.append((
                            bucket_dt,
                            acc.symbol,
                            0.0,
                            0,
                            0,
                            0,
                            0,
                            None,
                            None,
                            None,
                            0.0,
                            0.0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ))
            self._buckets.clear()
        self._write(rows, synchronous=True)

    def maybe_flush(self, current_time_s: float) -> None:
        now_bucket = _bucket_start(current_time_s)
        to_flush: list[tuple[Any, ...]] = []

        with self._lock:
            # Union configured tokens with core assets to ensure no anomaly-engine feeds are skipped
            tracked_tokens = settings.ENABLED_TOKENS | {"BTC", "ETH", "SOL", "BNB", "AVAX"}

            # 1. Determine which bucket timestamps are fully completed and flush them
            t = self._last_flushed_bucket + WINDOW_S
            while t < now_bucket:
                # Generate a row for every tracked symbol
                for symbol in tracked_tokens:
                    key = (symbol, t)
                    acc = self._buckets.pop(key, None)
                    if acc is not None and acc.count > 0:
                        to_flush.append(acc.to_row())
                    else:
                        # Falling back to a 0-tweet record
                        bucket_dt = datetime.fromtimestamp(t, tz=timezone.utc)
                        to_flush.append((
                            bucket_dt,
                            symbol,
                            0.0,  # avg_score
                            0,    # tweet_count
                            0,    # positive_count
                            0,    # negative_count
                            0,    # neutral_count
                            None, # max_score
                            None, # min_score
                            None, # sample_tweet
                            0.0,  # weighted_avg_score
                            0.0,  # total_source_weight
                            0,    # tier1_count
                            0,    # tier2_count
                            0,    # tier3_count
                            0,    # economy_news_count
                            0,    # turkish_economy_count
                            0,    # unknown_count
                        ))
                t += WINDOW_S
            
            # Update last flushed bucket to the latest completed bucket
            if now_bucket - WINDOW_S > self._last_flushed_bucket:
                self._last_flushed_bucket = now_bucket - WINDOW_S

            # 2. Extract late-arriving tweets for already completed buckets, and pop/discard pre-boot stale buckets
            first_valid_bucket = _bucket_start(self.system_boot_time.timestamp()) + WINDOW_S
            stale_keys = [k for k in self._buckets if k[1] < now_bucket]
            for key in stale_keys:
                bucket_ts = key[1]
                if bucket_ts >= first_valid_bucket:
                    # Flush late-arriving tweet accumulations to overwrite the 0-tweet fallback row
                    acc = self._buckets.pop(key)
                    if acc.count > 0:
                        to_flush.append(acc.to_row())
                else:
                    # Pre-boot bucket is truncated/distorted, pop and discard it
                    self._buckets.pop(key, None)

        self._write(to_flush)

    @staticmethod
    def _write(rows: list[tuple[Any, ...]], synchronous: bool = False) -> None:
        if not rows:
            return

        def run_in_background() -> None:
            try:
                execute_batch(INSERT_SQL, rows)
                LOGGER.info("flushed %d tweet sentiment bucket(s) to DB", len(rows))
            except Exception:
                LOGGER.exception("failed to flush tweet sentiment buckets")

        if synchronous:
            run_in_background()
        else:
            threading.Thread(target=run_in_background, daemon=True).start()
