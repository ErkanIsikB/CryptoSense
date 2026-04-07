"""Binance spot orderbook (depth) stream ingestion pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from config import settings
from sinks.base import BaseSink
from sinks.jsonl_sink import JsonlFileSink
from utils.logging import emit_status

STREAM_NAME = "depth@100ms"
QUEUE_MAXSIZE = 50_000
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 20.0
CLOSE_TIMEOUT_S = 5.0
PING_INTERVAL_S = 20.0
PING_TIMEOUT_S = 20.0
SESSION_ROTATE_AFTER_S = 23 * 60 * 60 + 50 * 60

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


def _build_depth_stream_url(base: str, symbols: tuple[str, ...]) -> str:
    streams = "/".join(f"{symbol.lower()}@{STREAM_NAME}" for symbol in symbols)
    return f"{base.rstrip('/')}/stream?streams={streams}"


def parse_depth_event(message: dict[str, object]) -> OrderbookEvent | None:
    data = message.get("data")
    if not isinstance(data, dict) or data.get("e") != "depthUpdate":
        return None
    try:
        symbol = str(data["s"])
        event_time_ms = int(data["E"])
        first_update_id = int(data["U"])
        final_update_id = int(data["u"])
        bids_raw = data["b"]
        asks_raw = data["a"]
    except (KeyError, TypeError, ValueError):
        return None

    if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
        return None

    bids = [level for level in bids_raw if isinstance(level, list) and len(level) >= 2]
    asks = [level for level in asks_raw if isinstance(level, list) and len(level) >= 2]

    return OrderbookEvent(
        symbol=symbol,
        event_time_ms=event_time_ms,
        first_update_id=first_update_id,
        final_update_id=final_update_id,
        bids=bids,
        asks=asks,
    )


async def _listen_orderbook(
    symbols: tuple[str, ...],
    queue: asyncio.Queue[OrderbookEvent],
    stop: asyncio.Event,
) -> None:
    bases = settings.BINANCE_WS_BASES
    base_index = 0
    backoff = INITIAL_BACKOFF_S

    while not stop.is_set():
        base = bases[base_index % len(bases)]
        base_index += 1
        url = _build_depth_stream_url(base, symbols)
        connected_at = time.monotonic()

        try:
            emit_status(LOGGER, "connecting", url=url)
            async with websockets.connect(
                url,
                open_timeout=settings.WS_OPEN_TIMEOUT_S,
                close_timeout=CLOSE_TIMEOUT_S,
                ping_interval=PING_INTERVAL_S,
                ping_timeout=PING_TIMEOUT_S,
                max_queue=QUEUE_MAXSIZE,
            ) as websocket:
                emit_status(LOGGER, "connected", url=url)
                backoff = INITIAL_BACKOFF_S

                while not stop.is_set():
                    age_s = time.monotonic() - connected_at
                    if age_s >= SESSION_ROTATE_AFTER_S:
                        emit_status(LOGGER, "rotating_connection", age_s=round(age_s, 2), url=url)
                        await websocket.close(code=1000, reason="proactive reconnect before 24h limit")
                        break

                    raw_message = await websocket.recv()
                    message = json.loads(raw_message)
                    event = parse_depth_event(message)
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
                    "type": "depthUpdate",
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