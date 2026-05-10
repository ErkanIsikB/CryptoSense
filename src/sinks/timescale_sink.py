"""TimescaleDB Sink — writes engineered data to the database.

This sink is used as a lightweight adapter that routes records to
the appropriate feature-engineering module for aggregation and DB writes.
It implements ``BaseSink`` so it can be used as a drop-in replacement for
``JsonlFileSink`` in the existing pipeline code.

The actual aggregation and DB writes are handled by:
- ``TradeAggregator`` for trade events
- ``OrderbookAggregator`` for orderbook snapshots
- ``sentiment_scorer`` for sentiment records
"""

from __future__ import annotations

import logging
from typing import Any

from src.sinks.base import BaseSink
from src.feature_engineering.trade_aggregator import TradeAggregator
from src.feature_engineering.orderbook_aggregator import OrderbookAggregator
from src.feature_engineering.sentiment_scorer import score_and_store

LOGGER = logging.getLogger("timescale_sink")


class TimescaleSink(BaseSink):
    """Routes incoming records to the correct aggregator/writer."""

    def __init__(self) -> None:
        self._trade_aggregator = TradeAggregator()
        self._orderbook_aggregator = OrderbookAggregator()

    async def write(self, key: str, record: dict[str, Any]) -> None:
        record_type = record.get("type", "")

        if record_type == "aggTrade":
            self._trade_aggregator.add(
                symbol=record["symbol"],
                price=record["price"],
                qty=record["qty"],
                trade_time_ms=record["trade_time_ms"],
                is_buyer_maker=record["is_buyer_maker"],
            )

        elif record_type in ("depth_snapshot", "depthUpdate"):
            self._orderbook_aggregator.add(
                symbol=record["symbol"],
                event_time_ms=record["event_time_ms"],
                bids=record.get("bids", []),
                asks=record.get("asks", []),
            )

        elif record.get("event_type") == "sentiment":
            score_and_store(record)

        else:
            LOGGER.warning("unknown record type: key=%s type=%s", key, record_type)

    async def close(self) -> None:
        """Flush remaining aggregator buffers and close DB pool."""
        LOGGER.info("flushing all aggregators before shutdown")
        self._trade_aggregator.flush_all()
        self._orderbook_aggregator.flush_all()
        LOGGER.info("TimescaleSink closed")

    @property
    def trade_aggregator(self) -> TradeAggregator:
        return self._trade_aggregator

    @property
    def orderbook_aggregator(self) -> OrderbookAggregator:
        return self._orderbook_aggregator
