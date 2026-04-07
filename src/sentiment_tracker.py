"""Periodic crypto sentiment tracker using the Tavily search API."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import requests
import schedule

from config import settings
from sinks.base import BaseSink
from sinks.jsonl_sink import JsonlFileSink

LOGGER = logging.getLogger("sentiment_tracker")

TOKENS: list[dict[str, str]] = [
    {"name": "Bitcoin", "symbol": "BTC"},
    {"name": "Ethereum", "symbol": "ETH"},
    {"name": "Solana", "symbol": "SOL"},
    {"name": "BNB", "symbol": "BNB"},
    {"name": "Avalanche", "symbol": "AVAX"},
]

OUTPUT_DIR = settings.DATA_DIR / "sentiment"


def build_query(token_name: str, token_symbol: str) -> str:
    return (
        f"{token_name} {token_symbol} latest news OR market sentiment OR "
        f"breaking OR price moving events OR FUD OR FOMO"
    )


def _request_with_retry(payload: dict[str, Any]) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(settings.SENTIMENT_MAX_RETRIES + 1):
        try:
            response = requests.post(
                settings.TAVILY_API_URL, json=payload, timeout=settings.SENTIMENT_TIMEOUT_S
            )
            if response.status_code == 429:
                wait_seconds = min(30, 2**attempt)
                LOGGER.warning("tavily_rate_limited wait_s=%s attempt=%s", wait_seconds, attempt + 1)
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= settings.SENTIMENT_MAX_RETRIES:
                break
            wait_seconds = min(20, 2**attempt)
            LOGGER.warning("tavily_retry wait_s=%s attempt=%s error=%s", wait_seconds, attempt + 1, exc)
            time.sleep(wait_seconds)

    raise RuntimeError(f"Tavily request failed after retries: {last_error}")


def fetch_token_sentiment(token: dict[str, str]) -> dict[str, Any] | None:
    payload = {
        "api_key": settings.TAVILY_API_KEY,
        "query": build_query(token["name"], token["symbol"]),
        "topic": "news",
        "search_depth": settings.SENTIMENT_SEARCH_DEPTH,
        "include_answer": settings.SENTIMENT_INCLUDE_ANSWER,
        "include_images": settings.SENTIMENT_INCLUDE_IMAGES,
    }

    try:
        response = _request_with_retry(payload)
        data = response.json()
    except Exception as exc:
        LOGGER.error("tavily_fetch_failed token=%s error=%s", token["symbol"], exc)
        return None

    return {
        "event_type": "sentiment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "token": token["symbol"],
        "query": payload["query"],
        "answer": data.get("answer"),
        "results": data.get("results", []),
        "response_time": data.get("response_time"),
    }


def _append_sentiment(sink: BaseSink, record: dict[str, Any]) -> None:
    asyncio.run(sink.write("sentiment", record))


def _fetch_crypto_sentiment_cycle(sink: BaseSink) -> None:
    if not settings.TAVILY_API_KEY:
        LOGGER.error("missing_env_var name=TAVILY_API_KEY")
        return

    selected_tokens = TOKENS[: settings.SENTIMENT_MAX_TOKENS_PER_CYCLE]
    LOGGER.info("sentiment_cycle_start tokens=%s", [t["symbol"] for t in selected_tokens])

    for token in selected_tokens:
        record = fetch_token_sentiment(token)
        if record is None:
            continue
        _append_sentiment(sink, record)
        LOGGER.info("sentiment_written token=%s", token["symbol"])


def start_sentiment_stream(stop: asyncio.Event) -> None:
    """Public entry point — blocks until *stop* is set.

    Designed to be called via ``asyncio.to_thread()`` from the
    orchestrator so that the synchronous schedule loop does not block
    the event loop.
    """
    sink = JsonlFileSink(OUTPUT_DIR)

    _fetch_crypto_sentiment_cycle(sink)
    schedule.every(settings.SENTIMENT_INTERVAL_MINUTES).minutes.do(_fetch_crypto_sentiment_cycle, sink)

    try:
        while not stop.is_set():
            schedule.run_pending()
            time.sleep(30)
    finally:
        asyncio.run(sink.close())
