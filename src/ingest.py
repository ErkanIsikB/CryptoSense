"""Binance Futures aggregate-trade stream ingestion pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from config import settings
from sinks.base import BaseSink
from sinks.jsonl_sink import JsonlFileSink
from utils.logging import emit_status

STREAM_NAME = "aggTrade"
QUEUE_MAXSIZE = 10_000
ROLLING_WINDOW_S = 60.0
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 15.0
OPEN_TIMEOUT_S = 15.0
CLOSE_TIMEOUT_S = 5.0
RECEIVE_TIMEOUT_S = 600.0
SESSION_ROTATE_AFTER_S = 23 * 60 * 60 + 50 * 60

LOGGER = logging.getLogger("binance_futures_market_stream")

OUTPUT_DIR = settings.DATA_DIR / "trades"


@dataclass(frozen=True)
class TradeEvent:
    event_time_ms: int
    trade_time_ms: int
    symbol: str
    price: float
    qty: float
    aggregate_trade_id: int
    is_buyer_maker: bool


class RollingVolume:
    def __init__(self, window_s: float = ROLLING_WINDOW_S) -> None:
        self.window_s = window_s
        self._events: Deque[tuple[float, float]] = deque()
        self._sum_qty = 0.0

    def add(self, ts_s: float, qty: float) -> None:
        self._events.append((ts_s, qty))
        self._sum_qty += qty
        self._evict(ts_s)

    def value(self, now_s: float) -> float:
        self._evict(now_s)
        return self._sum_qty

    def _evict(self, now_s: float) -> None:
        cutoff = now_s - self.window_s
        while self._events and self._events[0][0] < cutoff:
            _, qty = self._events.popleft()
            self._sum_qty -= qty


def _build_market_stream_url(symbols: tuple[str, ...]) -> str:
    streams = "/".join(f"{symbol.lower()}@{STREAM_NAME}" for symbol in symbols)
    return f"{settings.BINANCE_MARKET_WS_BASE}/stream?streams={streams}"


def parse_agg_trade(message: dict[str, object]) -> TradeEvent | None:
    data = message.get("data")
    if not isinstance(data, dict) or data.get("e") != "aggTrade":
        return None
    try:
        return TradeEvent(
            event_time_ms=int(data["E"]),
            trade_time_ms=int(data["T"]),
            symbol=str(data["s"]),
            price=float(data["p"]),
            qty=float(data["q"]),
            aggregate_trade_id=int(data["a"]),
            is_buyer_maker=bool(data["m"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


async def _listen(
    symbols: tuple[str, ...],
    queue: asyncio.Queue[TradeEvent],
    stop: asyncio.Event,
) -> None:
    url = _build_market_stream_url(symbols)
    backoff = INITIAL_BACKOFF_S

    while not stop.is_set():
        connected_at = time.monotonic()
        try:
            emit_status(LOGGER, "connecting", url=url)
            async with websockets.connect(
                url,
                ping_interval=None,
                ping_timeout=None,
                open_timeout=OPEN_TIMEOUT_S,
                close_timeout=CLOSE_TIMEOUT_S,
                max_queue=QUEUE_MAXSIZE,
            ) as websocket:
                emit_status(LOGGER, "connected", url=url)
                backoff = INITIAL_BACKOFF_S

                while not stop.is_set():
                    session_age_s = time.monotonic() - connected_at
                    if session_age_s >= SESSION_ROTATE_AFTER_S:
                        emit_status(LOGGER, "rotating_connection", age_s=round(session_age_s, 2))
                        await websocket.close(code=1000, reason="proactive reconnect before 24h limit")
                        break

                    timeout_s = min(RECEIVE_TIMEOUT_S, max(1.0, SESSION_ROTATE_AFTER_S - session_age_s))
                    raw_message = await asyncio.wait_for(websocket.recv(), timeout=timeout_s)
                    message = json.loads(raw_message)

                    event = parse_agg_trade(message)
                    if event is not None:
                        await queue.put(event)

        except asyncio.CancelledError:
            raise
        except (ConnectionClosed, InvalidStatus, OSError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
            emit_status(LOGGER, "reconnecting", url=url, backoff=round(backoff, 2), error=str(exc))

        if stop.is_set():
            break

        await asyncio.sleep(backoff + random.uniform(0.0, 0.5))
        backoff = min(MAX_BACKOFF_S, backoff * 2)


async def _aggregator(
    queue: asyncio.Queue[TradeEvent],
    stop: asyncio.Event,
    sink: BaseSink,
) -> None:
    rolling_volumes: dict[str, RollingVolume] = {}

    while not stop.is_set() or not queue.empty():
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue

        try:
            now_s = event.trade_time_ms / 1000
            volume = rolling_volumes.setdefault(event.symbol, RollingVolume())
            volume.add(now_s, event.qty)

            await sink.write(
                event.symbol,
                {
                    "type": STREAM_NAME,
                    "symbol": event.symbol,
                    "event_time_ms": event.event_time_ms,
                    "trade_time_ms": event.trade_time_ms,
                    "price": event.price,
                    "qty": event.qty,
                    "aggregate_trade_id": event.aggregate_trade_id,
                    "is_buyer_maker": event.is_buyer_maker,
                    "volume_1m": volume.value(now_s),
                },
            )
        finally:
            queue.task_done()


async def start_trade_stream(stop: asyncio.Event) -> None:
    """Public entry point — run the Binance aggregate-trade pipeline."""
    symbols = settings.BINANCE_SYMBOLS
    queue: asyncio.Queue[TradeEvent] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    sink = JsonlFileSink(OUTPUT_DIR)

    listener_task = asyncio.create_task(_listen(symbols, queue, stop))
    aggregator_task = asyncio.create_task(_aggregator(queue, stop, sink))

    try:
        await stop.wait()
    finally:
        listener_task.cancel()
        await asyncio.gather(listener_task, return_exceptions=True)
        await queue.join()
        aggregator_task.cancel()
        await asyncio.gather(aggregator_task, return_exceptions=True)
        await sink.close()