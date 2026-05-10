"""XQuik Ingestion — real-time X (Twitter) keyword monitoring for crypto sentiment.

Flow:
1. On startup, ensure keyword monitors exist for each tracked coin
2. Every 5 minutes, poll the events API for new tweets
3. Score each tweet through FinBERT
4. Feed scored tweets into SentimentAggregator (5-min buckets → TimescaleDB)

Uses the XQuik REST API:
- POST /api/v1/monitors/keywords — create keyword monitor
- GET  /api/v1/monitors/keywords — list existing monitors
- GET  /api/v1/events            — poll captured tweet events
- DELETE /api/v1/monitors/keywords/{id} — cleanup on shutdown
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from src.core.config import settings
from src.feature_engineering.sentiment_scorer import _score_text, _compound_score
from src.feature_engineering.sentiment_aggregator import SentimentAggregator

LOGGER = logging.getLogger("xquik_ingestion")

XQUIK_BASE = "https://xquik.com/api/v1"

# Keyword queries per tracked coin (X search syntax)
KEYWORD_QUERIES: dict[str, str] = {
    "BTC": '$BTC OR #Bitcoin OR "bitcoin"',
    "ETH": '$ETH OR #Ethereum OR "ethereum"',
    "SOL": '$SOL OR #Solana OR "solana"',
    "BNB": '$BNB OR "bnb" OR "binance coin"',
    "AVAX": '$AVAX OR #Avalanche OR "avalanche crypto"',
}


def _headers() -> dict[str, str]:
    return {
        "x-api-key": settings.XQUIK_API,
        "Content-Type": "application/json",
    }


# ── Monitor management ──────────────────────────────────────────


async def _list_keyword_monitors(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    """List all existing keyword monitors."""
    resp = await client.get(
        f"{XQUIK_BASE}/monitors/keywords",
        headers=_headers(),
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("monitors", [])


async def _create_keyword_monitor(
    client: httpx.AsyncClient, query: str
) -> dict[str, Any]:
    """Create a new keyword monitor."""
    resp = await client.post(
        f"{XQUIK_BASE}/monitors/keywords",
        headers=_headers(),
        json={
            "query": query,
            "eventTypes": ["tweet.new", "tweet.quote"],
        },
    )
    resp.raise_for_status()
    return resp.json()


async def _delete_keyword_monitor(client: httpx.AsyncClient, monitor_id: str) -> None:
    """Delete a keyword monitor."""
    resp = await client.delete(
        f"{XQUIK_BASE}/monitors/keywords/{monitor_id}",
        headers=_headers(),
    )
    resp.raise_for_status()


async def _set_monitor_active(
    client: httpx.AsyncClient, monitor_id: str, active: bool
) -> None:
    """Pause or unpause a keyword monitor.

    - ``active=True``  → unpause (resume monitoring, costs credits)
    - ``active=False`` → pause   (stop monitoring, saves credits)
    """
    resp = await client.patch(
        f"{XQUIK_BASE}/monitors/keywords/{monitor_id}",
        headers=_headers(),
        json={"isActive": active},
    )
    resp.raise_for_status()


async def _pause_all_monitors(
    client: httpx.AsyncClient, symbol_monitors: dict[str, str]
) -> None:
    """Pause all keyword monitors to stop credit consumption."""
    for symbol, mid in symbol_monitors.items():
        try:
            await _set_monitor_active(client, mid, active=False)
            LOGGER.info("paused monitor: symbol=%s id=%s", symbol, mid)
        except Exception:
            LOGGER.exception("failed to pause monitor %s (id=%s)", symbol, mid)
        await asyncio.sleep(0.12)


async def ensure_keyword_monitors(client: httpx.AsyncClient) -> dict[str, str]:
    """Ensure keyword monitors exist for all tracked coins.

    Returns a mapping of ``{symbol: monitor_id}``.
    """
    existing = await _list_keyword_monitors(client)
    existing_queries = {m["query"]: m["id"] for m in existing}

    symbol_to_monitor: dict[str, str] = {}

    for symbol, query in KEYWORD_QUERIES.items():
        if query in existing_queries:
            mid = existing_queries[query]
            symbol_to_monitor[symbol] = mid
            LOGGER.info("monitor exists: symbol=%s id=%s query=%s", symbol, mid, query)

            # Check if paused and unpause
            for m in existing:
                if m["id"] == mid and not m.get("isActive", True):
                    try:
                        await _set_monitor_active(client, mid, active=True)
                        LOGGER.info("unpaused monitor: symbol=%s id=%s", symbol, mid)
                    except Exception:
                        LOGGER.exception("failed to unpause monitor %s", symbol)
                    await asyncio.sleep(0.12)
                    break
        else:
            try:
                result = await _create_keyword_monitor(client, query)
                mid = result["id"]
                symbol_to_monitor[symbol] = mid
                LOGGER.info(
                    "monitor created: symbol=%s id=%s query=%s", symbol, mid, query
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 409:
                    # Already exists (race or query variation)
                    LOGGER.info("monitor already exists for %s", symbol)
                    # Re-fetch to get the ID
                    refreshed = await _list_keyword_monitors(client)
                    for m in refreshed:
                        if m["query"] == query:
                            symbol_to_monitor[symbol] = m["id"]
                            break
                else:
                    LOGGER.error(
                        "failed to create monitor for %s: %s %s",
                        symbol,
                        exc.response.status_code,
                        exc.response.text,
                    )
            # Small delay to avoid rate limits (10 req/s)
            await asyncio.sleep(0.15)

    return symbol_to_monitor


# ── Event polling ───────────────────────────────────────────────

# Track the last seen event ID per monitor to avoid processing duplicates
_last_event_ids: dict[str, str] = {}


async def _poll_events(
    client: httpx.AsyncClient,
    symbol: str,
    monitor_id: str,
) -> list[dict[str, Any]]:
    """Poll new events for a given keyword monitor.

    Returns a list of tweet event dicts.
    """
    params: dict[str, Any] = {
        "keywordMonitorId": monitor_id,
        "limit": 100,
    }

    # Use cursor to get only new events
    last_cursor = _last_event_ids.get(monitor_id)
    if last_cursor:
        params["after"] = last_cursor

    all_events: list[dict[str, Any]] = []

    try:
        resp = await client.get(
            f"{XQUIK_BASE}/events",
            headers=_headers(),
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

        events = data.get("events", [])
        all_events.extend(events)
        LOGGER.info(
            "polled %s: %d events (hasMore=%s)",
            symbol, len(events), data.get("hasMore"),
        )

        # If there are more pages, keep fetching
        while data.get("hasMore"):
            next_cursor = data.get("nextCursor")
            if not next_cursor:
                break
            params["after"] = next_cursor
            resp = await client.get(
                f"{XQUIK_BASE}/events",
                headers=_headers(),
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            events = data.get("events", [])
            all_events.extend(events)
            LOGGER.info(
                "polled %s (page): +%d events (hasMore=%s)",
                symbol, len(events), data.get("hasMore"),
            )

        # Update cursor to the latest event
        if all_events:
            # The nextCursor from the last page
            final_cursor = data.get("nextCursor")
            if final_cursor:
                _last_event_ids[monitor_id] = final_cursor

    except httpx.HTTPStatusError as exc:
        LOGGER.error(
            "event poll failed: symbol=%s monitor=%s status=%s body=%s",
            symbol,
            monitor_id,
            exc.response.status_code,
            exc.response.text[:200],
        )
    except Exception:
        LOGGER.exception("event poll error: symbol=%s", symbol)

    return all_events


# ── Scoring & aggregation ───────────────────────────────────────


def _score_tweet(text: str) -> float:
    """Score a single tweet through FinBERT. Returns compound score [-1, +1]."""
    probs = _score_text(text)
    return _compound_score(probs)


def _parse_event_time(event: dict[str, Any]) -> float:
    """Extract tweet timestamp as Unix epoch seconds."""
    # Events have 'occurredAt' in ISO format
    occurred_at = event.get("occurredAt", "")
    if occurred_at:
        try:
            dt = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, TypeError):
            pass

    # Fallback: check data.createdAt
    data = event.get("data", {})
    created_at = data.get("createdAt", "")
    if created_at:
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, TypeError):
            pass

    return time.time()


async def _poll_and_score_cycle(
    client: httpx.AsyncClient,
    symbol_monitors: dict[str, str],
    aggregator: SentimentAggregator,
) -> int:
    """Run one poll-score-aggregate cycle for all symbols.

    Returns total number of tweets scored.
    """
    total_scored = 0

    for symbol, monitor_id in symbol_monitors.items():
        events = await _poll_events(client, symbol, monitor_id)
        if not events:
            LOGGER.info("no new tweets for %s", symbol)
            continue

        scored = 0
        for event in events:
            data = event.get("data", {})
            text = data.get("text", "")
            if not text or not text.strip():
                continue

            # Skip retweets (they just duplicate text)
            if data.get("isRetweet"):
                continue

            tweet_time_s = _parse_event_time(event)
            score = _score_tweet(text)

            # Use likes+retweets as engagement metric for sample tweet selection
            engagement = float(
                data.get("likeCount", 0) or 0
            ) + float(
                data.get("retweetCount", 0) or 0
            )

            aggregator.add(
                symbol=symbol,
                score=score,
                tweet_time_s=tweet_time_s,
                tweet_text=text[:500],
                engagement=engagement,
            )
            scored += 1

        if scored > 0:
            LOGGER.info(
                "scored %d tweet(s) for %s from %d events", scored, symbol, len(events)
            )
        total_scored += scored

        # Small delay between symbols to stay within rate limits
        await asyncio.sleep(0.12)

    return total_scored


# ── Public entry point ──────────────────────────────────────────


async def start_xquik_sentiment_stream(stop: asyncio.Event) -> None:
    """Public entry point — runs XQuik keyword monitoring pipeline.

    1. Ensures keyword monitors exist for all tracked coins
    2. Polls events every 5 minutes
    3. Scores tweets through FinBERT
    4. Aggregates into 5-minute buckets → TimescaleDB
    """
    if not settings.XQUIK_API:
        LOGGER.error("XQUIK_API not set — tweet sentiment pipeline disabled")
        return

    aggregator = SentimentAggregator()

    LOGGER.info(
        "XQuik sentiment stream starting (poll every %ds)",
        settings.XQUIK_POLL_INTERVAL_S,
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Ensure monitors exist (and unpause if paused)
        try:
            symbol_monitors = await ensure_keyword_monitors(client)
            LOGGER.info(
                "keyword monitors ready: %s",
                {s: mid for s, mid in symbol_monitors.items()},
            )
        except Exception:
            LOGGER.exception("failed to set up keyword monitors")
            return

        if not symbol_monitors:
            LOGGER.error("no keyword monitors could be created — aborting")
            return

        # Step 2: Poll loop
        try:
            while not stop.is_set():
                try:
                    total = await _poll_and_score_cycle(client, symbol_monitors, aggregator)
                    if total > 0:
                        LOGGER.info("poll cycle complete: %d tweets scored total", total)
                    else:
                        LOGGER.debug("poll cycle complete: no new tweets")
                except Exception:
                    LOGGER.exception("XQuik poll cycle failed")

                # Wait for next cycle or until stopped
                try:
                    await asyncio.wait_for(
                        stop.wait(), timeout=settings.XQUIK_POLL_INTERVAL_S
                    )
                except asyncio.TimeoutError:
                    pass  # normal — poll again
        finally:
            # Step 3: Pause monitors on shutdown to save credits
            LOGGER.info("pausing all keyword monitors to save credits")
            await _pause_all_monitors(client, symbol_monitors)

    # Flush remaining data on shutdown
    aggregator.flush_all()
    LOGGER.info("XQuik sentiment stream stopped")
