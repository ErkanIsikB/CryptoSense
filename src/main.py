"""CryptoSense — unified data-ingestion orchestrator.

Starts all four pipelines concurrently and handles graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from src.core.config import settings
from src.core.utils.logging import configure_logging
from src.core.utils.signals import setup_signals

from src.data_sources.binancewebsocket.ws_trades_ingestion import start_trade_stream
from src.data_sources.binancewebsocket.ws_orderbook_ingestion import start_orderbook_stream
from src.data_sources.tavily.tavily_ingestion import start_sentiment_stream

LOGGER = logging.getLogger("orchestrator")


@dataclass(frozen=True)
class Pipeline:
    name: str
    starter: Callable[[asyncio.Event], Awaitable[None]] | Callable[[asyncio.Event], None]
    is_async: bool = True


PIPELINES: tuple[Pipeline, ...] = (
    Pipeline(name="binance_trades", starter=start_trade_stream, is_async=True),
    Pipeline(name="binance_orderbook", starter=start_orderbook_stream, is_async=True),
    Pipeline(name="sentiment", starter=start_sentiment_stream, is_async=False),
)


async def _run_all() -> None:
    configure_logging(settings.LOG_LEVEL)

    stop = asyncio.Event()
    setup_signals(stop)

    LOGGER.info("starting all ingestion pipelines")

    tasks: list[asyncio.Task[object]] = []
    for pipeline in PIPELINES:
        if pipeline.is_async:
            task = asyncio.create_task(pipeline.starter(stop), name=pipeline.name)
        else:
            task = asyncio.create_task(asyncio.to_thread(pipeline.starter, stop), name=pipeline.name)
        tasks.append(task)

    try:
        await stop.wait()
    finally:
        LOGGER.info("shutdown signal received — stopping all pipelines")
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        LOGGER.info("all pipelines stopped")


def main() -> None:
    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
