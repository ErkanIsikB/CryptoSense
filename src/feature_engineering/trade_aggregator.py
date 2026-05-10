"""Trade Aggregator — buffers Binance aggTrade events into 5-minute OHLCV candles.

Each incoming ``TradeEvent`` is bucketed by its trade timestamp into a 5-minute
window.  When a new event arrives that belongs to a *later* bucket, all
completed buckets are flushed to the database.  A periodic timer also flushes
stale buckets for low-activity symbols.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.core.config import settings
from src.db.db import execute_batch

LOGGER = logging.getLogger("trade_aggregator")

WINDOW_S: int = settings.AGGREGATION_WINDOW_SECONDS  # default 300 = 5 min


@dataclass
class CandleAccumulator:
    """Mutable accumulator for a single (symbol, bucket) candle."""

    bucket_ts: float  # Unix epoch of the bucket start
    symbol: str
    open: float = 0.0
    high: float = -math.inf
    low: float = math.inf
    close: float = 0.0
    volume: float = 0.0
    quote_volume: float = 0.0
    trade_count: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    _first_trade_time_ms: int = field(default=0, repr=False)

    def add(self, price: float, qty: float, trade_time_ms: int, is_buyer_maker: bool) -> None:
        quote = price * qty

        if self.trade_count == 0 or trade_time_ms < self._first_trade_time_ms:
            self.open = price
            self._first_trade_time_ms = trade_time_ms

        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price  # always updated; last call = last trade
        self.volume += qty
        self.quote_volume += quote
        self.trade_count += 1

        if is_buyer_maker:
            # is_buyer_maker=True → seller is the taker → sell volume
            self.sell_volume += qty
        else:
            self.buy_volume += qty

    def to_row(self) -> tuple[Any, ...]:
        net_trade = self.buy_volume - self.sell_volume
        vwap = self.quote_volume / self.volume if self.volume > 0 else None
        bucket_dt = datetime.fromtimestamp(self.bucket_ts, tz=timezone.utc)
        return (
            bucket_dt,
            self.symbol,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.quote_volume,
            self.trade_count,
            self.buy_volume,
            self.sell_volume,
            net_trade,
            vwap,
        )


INSERT_SQL = """
INSERT INTO trade_candles_5m
    (bucket, symbol, open, high, low, close, volume, quote_volume,
     trade_count, buy_volume, sell_volume, net_trade, vwap)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (bucket, symbol) DO UPDATE SET
    open         = EXCLUDED.open,
    high         = EXCLUDED.high,
    low          = EXCLUDED.low,
    close        = EXCLUDED.close,
    volume       = EXCLUDED.volume,
    quote_volume = EXCLUDED.quote_volume,
    trade_count  = EXCLUDED.trade_count,
    buy_volume   = EXCLUDED.buy_volume,
    sell_volume  = EXCLUDED.sell_volume,
    net_trade    = EXCLUDED.net_trade,
    vwap         = EXCLUDED.vwap;
"""


def _bucket_start(epoch_ms: int) -> float:
    """Return the Unix epoch (seconds) of the 5-min bucket that *epoch_ms* belongs to."""
    epoch_s = epoch_ms / 1000.0
    return epoch_s - (epoch_s % WINDOW_S)


class TradeAggregator:
    """Thread-safe 5-minute OHLCV aggregator.

    Call :meth:`add` with each trade event.  Completed buckets are
    automatically flushed to TimescaleDB.
    """

    def __init__(self) -> None:
        self._buckets: dict[tuple[str, float], CandleAccumulator] = {}
        self._lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────

    def add(
        self,
        symbol: str,
        price: float,
        qty: float,
        trade_time_ms: int,
        is_buyer_maker: bool,
    ) -> None:
        bucket_ts = _bucket_start(trade_time_ms)
        key = (symbol, bucket_ts)

        with self._lock:
            acc = self._buckets.get(key)
            if acc is None:
                acc = CandleAccumulator(bucket_ts=bucket_ts, symbol=symbol)
                self._buckets[key] = acc
            acc.add(price, qty, trade_time_ms, is_buyer_maker)

        # Flush any buckets that are fully in the past
        self._maybe_flush(trade_time_ms)

    def flush_all(self) -> None:
        """Force-flush every open bucket (used at shutdown)."""
        with self._lock:
            rows = [acc.to_row() for acc in self._buckets.values() if acc.trade_count > 0]
            self._buckets.clear()

        self._write(rows)

    # ── Internal helpers ────────────────────────────────────────

    def _maybe_flush(self, current_time_ms: int) -> None:
        now_bucket = _bucket_start(current_time_ms)
        to_flush: list[tuple[Any, ...]] = []

        with self._lock:
            stale_keys = [
                key for key in self._buckets
                if key[1] < now_bucket  # strictly older bucket
            ]
            for key in stale_keys:
                acc = self._buckets.pop(key)
                if acc.trade_count > 0:
                    to_flush.append(acc.to_row())

        self._write(to_flush)

    @staticmethod
    def _write(rows: list[tuple[Any, ...]]) -> None:
        if not rows:
            return
        try:
            execute_batch(INSERT_SQL, rows)
            LOGGER.info("flushed %d trade candle(s) to DB", len(rows))
        except Exception:
            LOGGER.exception("failed to flush trade candles")
