"""Binance spot orderbook REST snapshot ingestion pipeline."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass

import requests

from src.core.config import settings
from src.sinks.base import BaseSink
from src.sinks.jsonl_sink import JsonlFileSink
from src.core.utils.logging import emit_status

STREAM_NAME = "depth_snapshot"
QUEUE_MAXSIZE = 50_000
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 20.0

LOGGER = logging.getLogger("binance_orderbook_stream")

OUTPUT_DIR = settings.DATA_DIR / "orderbook"


@dataclass(frozen=True)
class OrderbookEvent:
    symbol: str
    event_time_ms: int
    first_update_id: int
    final_update_id: int
    bids: list[list[str]]
    asks: list[list[str]]


class RateLimitError(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__("rate limited")
        self.retry_after = retry_after


def _build_depth_rest_url() -> str:
    return f"{settings.BINANCE_REST_BASE.rstrip('/')}/api/v3/depth"


def parse_depth_snapshot(symbol: str, payload: dict[str, object]) -> OrderbookEvent | None:
    try:
        now_ms = int(time.time() * 1000)
        last_update_id = int(payload["lastUpdateId"])
        bids_raw = payload["bids"]
        asks_raw = payload["asks"]
    except (KeyError, TypeError, ValueError):
        return None

    if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
        return None

    bids = [level for level in bids_raw if isinstance(level, list) and len(level) >= 2]
    asks = [level for level in asks_raw if isinstance(level, list) and len(level) >= 2]

    return OrderbookEvent(
        symbol=symbol.upper(),
        event_time_ms=now_ms,
        first_update_id=last_update_id,
        final_update_id=last_update_id,
        bids=bids,
        asks=asks,
    )


async def _fetch_snapshot(symbol: str) -> OrderbookEvent | None:
    url = _build_depth_rest_url()
    params = {
        "symbol": symbol.upper(),
        "limit": settings.ORDERBOOK_DEPTH_LIMIT,
    }

    def request_once() -> requests.Response:
        return requests.get(url, params=params, timeout=settings.ORDERBOOK_REST_TIMEOUT_S)

    response = await asyncio.to_thread(request_once)

    if response.status_code == 429:
        retry_after_raw = response.headers.get("Retry-After", "1")
        try:
            retry_after = float(retry_after_raw)
        except ValueError:
            retry_after = 1.0
        raise RateLimitError(max(retry_after, 1.0))

    response.raise_for_status()
    return parse_depth_snapshot(symbol, response.json())


async def _listen_orderbook(
    symbols: tuple[str, ...],
    queue: asyncio.Queue[OrderbookEvent],
    stop: asyncio.Event,
) -> None:
    backoff = INITIAL_BACKOFF_S

    while not stop.is_set():
        try:
            emit_status(LOGGER, "polling", endpoint=_build_depth_rest_url(), depth=settings.ORDERBOOK_DEPTH_LIMIT)
            for symbol in symbols:
                if stop.is_set():
                    break
                event = await _fetch_snapshot(symbol)
                if event is not None:
                    await queue.put(event)
                await asyncio.sleep(settings.ORDERBOOK_SYMBOL_PAUSE_S)

            backoff = INITIAL_BACKOFF_S
            await asyncio.sleep(settings.ORDERBOOK_POLL_INTERVAL_S)

        except asyncio.CancelledError:
            raise
        except RateLimitError as exc:
            wait_s = max(exc.retry_after, settings.ORDERBOOK_POLL_INTERVAL_S)
            emit_status(LOGGER, "rate_limited", wait_s=round(wait_s, 2))
            await asyncio.sleep(wait_s)
        except (requests.RequestException, ValueError) as exc:
            emit_status(LOGGER, "reconnecting", backoff=round(backoff, 2), error=str(exc))
            await asyncio.sleep(backoff + random.uniform(0.0, 0.5))
            backoff = min(MAX_BACKOFF_S, backoff * 2)


async def _write_orderbook(
    queue: asyncio.Queue[OrderbookEvent],
    stop: asyncio.Event,
    sink: BaseSink,
) -> None:
    while not stop.is_set() or not queue.empty():
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

        try:
            await sink.write(
                event.symbol,
                {
                    "type": STREAM_NAME,
                    "source": "binance_rest",
                    "symbol": event.symbol,
                    "event_time_ms": event.event_time_ms,
                    "first_update_id": event.first_update_id,
                    "final_update_id": event.final_update_id,
                    "bids": event.bids,
                    "asks": event.asks,
                },
            )
        finally:
            queue.task_done()


async def start_orderbook_stream(stop: asyncio.Event) -> None:
    """Public entry point — run the Binance orderbook depth pipeline."""
    symbols = settings.BINANCE_SYMBOLS
    queue: asyncio.Queue[OrderbookEvent] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    sink = JsonlFileSink(OUTPUT_DIR)

    listener_task = asyncio.create_task(_listen_orderbook(symbols, queue, stop))
    writer_task = asyncio.create_task(_write_orderbook(queue, stop, sink))

    try:
        await stop.wait()
    finally:
        listener_task.cancel()
        await asyncio.gather(listener_task, return_exceptions=True)
        await queue.join()
        writer_task.cancel()
        await asyncio.gather(writer_task, return_exceptions=True)
        await sink.close()