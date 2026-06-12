"""Orderbook Aggregator — condenses orderbook snapshots into 5-minute metrics.

For every snapshot arriving within a 5-minute window, we compute:

* **spread** — best_ask − best_bid
* **mid_price** — (best_ask + best_bid) / 2
* **bid_depth / ask_depth** — total qty on each side
* **imbalance** — (bid_depth − ask_depth) / (bid_depth + ask_depth)

When the window closes, the running *averages* of these metrics are written as
a single row to ``orderbook_snapshots_5m``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.core.config import settings
from src.feature_engineering.base_aggregator import BaseTimeBucketAggregator

LOGGER = logging.getLogger("orderbook_aggregator")

WINDOW_S: int = settings.AGGREGATION_WINDOW_SECONDS


@dataclass
class OrderbookAccumulator:
    """Running averages for a single (symbol, bucket)."""

    bucket_ts: float
    symbol: str
    sum_spread: float = 0.0
    sum_mid_price: float = 0.0
    sum_bid_depth: float = 0.0
    sum_ask_depth: float = 0.0
    sum_imbalance: float = 0.0
    count: int = 0

    def add(self, bids: list[list[str]], asks: list[list[str]]) -> None:
        if not bids or not asks:
            return

        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
        except (IndexError, ValueError):
            return

        bid_depth = sum(float(level[1]) for level in bids if len(level) >= 2)
        ask_depth = sum(float(level[1]) for level in asks if len(level) >= 2)

        spread = best_ask - best_bid
        mid_price = (best_ask + best_bid) / 2.0
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

        self.sum_spread += spread
        self.sum_mid_price += mid_price
        self.sum_bid_depth += bid_depth
        self.sum_ask_depth += ask_depth
        self.sum_imbalance += imbalance
        self.count += 1

    def to_row(self) -> tuple[Any, ...]:
        n = max(self.count, 1)
        bucket_dt = datetime.fromtimestamp(self.bucket_ts, tz=timezone.utc)
        return (
            bucket_dt,
            self.symbol,
            self.sum_spread / n,
            self.sum_mid_price / n,
            self.sum_bid_depth / n,
            self.sum_ask_depth / n,
            self.sum_imbalance / n,
            self.count,
        )


INSERT_SQL = """
INSERT INTO orderbook_snapshots_5m
    (bucket, symbol, avg_spread, avg_mid_price, avg_bid_depth,
     avg_ask_depth, avg_imbalance, snapshot_count)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (bucket, symbol) DO UPDATE SET
    avg_spread     = EXCLUDED.avg_spread,
    avg_mid_price  = EXCLUDED.avg_mid_price,
    avg_bid_depth  = EXCLUDED.avg_bid_depth,
    avg_ask_depth  = EXCLUDED.avg_ask_depth,
    avg_imbalance  = EXCLUDED.avg_imbalance,
    snapshot_count = EXCLUDED.snapshot_count;
"""


class OrderbookAggregator(BaseTimeBucketAggregator):
    """Thread-safe 5-minute orderbook metric aggregator."""

    def __init__(self) -> None:
        super().__init__(
            window_s=WINDOW_S,
            insert_sql=INSERT_SQL,
            entity_name="orderbook snapshot(s)",
            logger=LOGGER,
        )

    def add(
        self,
        symbol: str,
        event_time_ms: int,
        bids: list[list[str]],
        asks: list[list[str]],
    ) -> None:
        bucket_ts = self._bucket_start(event_time_ms)
        key = (symbol, bucket_ts)

        with self._lock:
            acc = self._buckets.get(key)
            if acc is None:
                acc = OrderbookAccumulator(bucket_ts=bucket_ts, symbol=symbol)
                self._buckets[key] = acc
            acc.add(bids, asks)

        self.maybe_flush(event_time_ms)

    def _should_flush(self, acc: OrderbookAccumulator) -> bool:
        return acc.count > 0
