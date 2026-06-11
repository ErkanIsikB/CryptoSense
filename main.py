"""CryptoSense — unified data-ingestion orchestrator.

Starts all pipelines concurrently and handles graceful shutdown.
Data flows through feature engineering aggregators into TimescaleDB.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from src.core.config import settings
from src.core.utils.logging import configure_logging
from src.core.utils.retraining_scheduler import (
    shutdown_retraining_scheduler,
    start_retraining_scheduler,
)
from src.core.utils.signals import setup_signals

from src.data_sources.binancewebsocket.ws_trades_ingestion import start_trade_stream
from src.data_sources.binancewebsocket.ws_orderbook_ingestion import start_orderbook_stream
from src.data_sources.xquik.xquik_ingestion import start_xquik_sentiment_stream
from src.data_sources.news_rss_ingestion import start_news_rss_stream
from src.data_sources.bitquery.cex_flow_ingestion import start_cex_flow_stream

from src.models.anomaly_pipeline import start_anomaly_stream
from src.models.llm_pipeline import start_llm_decision_stream

from src.sinks.timescale_sink import TimescaleSink
from src.db.db import run_migration, close_pool

LOGGER = logging.getLogger("orchestrator")


@dataclass(frozen=True)
class Pipeline:
    name: str
    starter: Callable[..., Awaitable[None]] | Callable[..., None]
    is_async: bool = True


async def _run_all() -> None:
    configure_logging(settings.LOG_LEVEL)

    stop = asyncio.Event()
    setup_signals(stop)

    # ── Database setup ──────────────────────────────────────────
    if settings.DB_URL:
        # noinspection PyBroadException
        try:
            LOGGER.info("running TimescaleDB schema migration")
            run_migration()
            LOGGER.info("schema migration completed")
        except Exception:
            LOGGER.exception("schema migration failed — DB writes will fail")
    else:
        LOGGER.warning("DB_URL not set — data will only be written to JSONL files")

    # ── Create shared TimescaleDB sink ──────────────────────────
    timescale_sink = TimescaleSink() if settings.DB_URL else None

    LOGGER.info("starting all ingestion pipelines")

    tasks: list[asyncio.Task[object]] = []

    # 1. Binance Trades — uses TimescaleSink for 5-min OHLCV aggregation
    trade_task = asyncio.create_task(
        start_trade_stream(stop, sink=timescale_sink),
        name="binance_trades",
    )
    tasks.append(trade_task)

    # 2. Binance Orderbook — uses TimescaleSink for 5-min metric aggregation
    orderbook_task = asyncio.create_task(
        start_orderbook_stream(stop, sink=timescale_sink),
        name="binance_orderbook",
    )
    tasks.append(orderbook_task)

    # 3. XQuik Tweet Sentiment — keyword monitors + FinBERT scoring
    if settings.XQUIK_API:
        xquik_task = asyncio.create_task(
            start_xquik_sentiment_stream(stop),
            name="xquik_sentiment",
        )
        tasks.append(xquik_task)
    else:
        LOGGER.warning("XQUIK_API not set — tweet sentiment pipeline disabled")

    # 3.5 Institutional News RSS Sentiment (Phase 3)
    news_task = asyncio.create_task(
        start_news_rss_stream(stop),
        name="news_rss_sentiment",
    )
    tasks.append(news_task)

    # 4. CEX Flows — Bitquery HTTP polling every 5 minutes
    if settings.BITQUERY_API_KEY:
        cex_task = asyncio.create_task(
            start_cex_flow_stream(stop),
            name="cex_flows",
        )
        tasks.append(cex_task)
    else:
        LOGGER.warning("BITQUERY_API_KEY not set — CEX flow pipeline disabled")

    # 5. AI Anomaly Engine — Runs inference every 5 mins from DB (NEW)
    if settings.DB_URL:
        anomaly_task = asyncio.create_task(
            start_anomaly_stream(stop),
            name="anomaly_engine",
        )
        tasks.append(anomaly_task)
    else:
        LOGGER.warning("DB_URL not set — anomaly engine disabled")

    # 6. LLM Decision Engine — Runs clock-aligned market briefings from DB (NEW)
    if settings.DB_URL:
        llm_task = asyncio.create_task(
            start_llm_decision_stream(stop),
            name="llm_decision_engine",
        )
        tasks.append(llm_task)
    else:
        LOGGER.warning("DB_URL not set — LLM decision engine disabled")

    try:
        if settings.RETRAIN_ENABLED and settings.DB_URL:
            start_retraining_scheduler()
        elif settings.RETRAIN_ENABLED:
            LOGGER.warning("DB_URL not set — scheduled retraining disabled")

        await stop.wait()
    finally:
        LOGGER.info("shutdown signal received — stopping all pipelines")
        stop.set()
        
        # 1. Stop Ingestion Entrances & Streams First
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # 2. Introduce a Short Settling Grace Period (allow background threads to exit connection contexts)
        await asyncio.sleep(1.0)

        # 3. Tear Down Sinks & Trigger/Await Final Flushes (flushes run synchronously)
        if timescale_sink is not None:
            await timescale_sink.close()

        # 4. Stop scheduled jobs before closing database connections
        shutdown_retraining_scheduler()

        # 5. Kill the Connection Pool LAST
        if settings.DB_URL:
            close_pool()

        LOGGER.info("all pipelines stopped")


def main() -> None:
    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
