"""XQuik Ingestion — real-time X (Twitter) keyword monitoring for crypto sentiment.

Flow:
1. On startup, ensure keyword monitors exist for each tracked coin
2. Fast-forward cursors to ignore old backlog
3. Every 5 minutes, poll the events API for strictly new tweets
4. Score each tweet through FinBERT (Batched for GPU efficiency)
5. Feed scored tweets into SentimentAggregator (5-min buckets → TimescaleDB)

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
from datetime import datetime
from typing import Any

import httpx

from src.core.config import settings
from src.feature_engineering.sentiment_scorer import score_texts_batched, compound_score
from src.feature_engineering.sentiment_aggregator import SentimentAggregator
from src.feature_engineering.source_credibility import (
    enrich_scored_item,
    extract_author_handle,
    resolve_source_credibility,
    tier_count_field,
)

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
    """Pause or unpause a keyword monitor."""
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
    """Ensure keyword monitors exist for all tracked coins."""
    existing = await _list_keyword_monitors(client)
    existing_queries = {m["query"]: m["id"] for m in existing}

    symbol_to_monitor: dict[str, str] = {}

    for symbol, query in KEYWORD_QUERIES.items():
        if query in existing_queries:
            mid = existing_queries[query]
            symbol_to_monitor[symbol] = mid
            LOGGER.info("monitor exists: symbol=%s id=%s query=%s", symbol, mid, query)

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
                    LOGGER.info("monitor already exists for %s", symbol)
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
            await asyncio.sleep(0.15)

    return symbol_to_monitor


# ── Event polling ───────────────────────────────────────────────

_last_seen_event_id: dict[str, str] = {}


async def _fast_forward_cursors(
    client: httpx.AsyncClient,
    symbol_monitors: dict[str, str],
) -> None:
    """Record the newest event ID for each monitor without processing anything."""
    for symbol, monitor_id in symbol_monitors.items():
        try:
            resp = await client.get(
                f"{XQUIK_BASE}/events",
                headers=_headers(),
                params={"keywordMonitorId": monitor_id, "limit": 1},
            )
            resp.raise_for_status()
            data = resp.json()
            events = data.get("events", [])
            if events:
                newest_id = str(events[0].get("id", ""))
                _last_seen_event_id[monitor_id] = newest_id
                LOGGER.info(
                    "fast-forwarded %s: newest event id=%s (will skip all older)",
                    symbol, newest_id,
                )
            else:
                LOGGER.info("fast-forwarded %s: no events yet", symbol)
        except Exception:
            LOGGER.exception("fast-forward failed for %s", symbol)
        await asyncio.sleep(0.12)


async def _poll_events(
    client: httpx.AsyncClient,
    symbol: str,
    monitor_id: str,
) -> list[dict[str, Any]]:
    """Poll strictly new events for a given keyword monitor."""
    last_seen = _last_seen_event_id.get(monitor_id)
    new_events: list[dict[str, Any]] = []
    found_seen = False
    params: dict[str, Any] = {
        "keywordMonitorId": monitor_id,
        "limit": 100,
    }

    try:
        resp = await client.get(
            f"{XQUIK_BASE}/events",
            headers=_headers(),
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

        events = data.get("events", [])

        for evt in events:
            evt_id = str(evt.get("id", ""))
            if evt_id == last_seen:
                found_seen = True
                break
            new_events.append(evt)

        LOGGER.info(
            "polled %s: %d new events from %d fetched (hitSeen=%s, hasMore=%s)",
            symbol, len(new_events), len(events), found_seen, data.get("hasMore"),
        )

        page = 0
        max_pages = 5
        while not found_seen and data.get("hasMore") and page < max_pages:
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

            for evt in events:
                evt_id = str(evt.get("id", ""))
                if evt_id == last_seen:
                    found_seen = True
                    break
                new_events.append(evt)

            LOGGER.info(
                "polled %s (page %d): +%d events (hitSeen=%s)",
                symbol, page + 1, len(events), found_seen,
            )
            page += 1

        if new_events:
            newest_id = str(new_events[0].get("id", ""))
            _last_seen_event_id[monitor_id] = newest_id

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

    new_events.reverse()
    return new_events


# ── Scoring & aggregation ───────────────────────────────────────


def _parse_event_time(event: dict[str, Any]) -> float:
    """Extract tweet timestamp as Unix epoch seconds."""
    occurred_at = event.get("occurredAt", "")
    if occurred_at:
        try:
            dt = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, TypeError):
            pass

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
    """Run one poll-score-aggregate cycle for all symbols using Batched GPU Inference."""
    total_scored = 0

    for symbol, monitor_id in symbol_monitors.items():
        events = await _poll_events(client, symbol, monitor_id)
        if not events:
            LOGGER.info("no new tweets for %s", symbol)
            continue

        valid_events = []
        texts_to_score = []

        for event in events:
            data = event.get("data", {})
            text = data.get("text", "")

            if not text or not text.strip():
                continue
            if data.get("isRetweet"):
                continue

            valid_events.append(event)
            texts_to_score.append(text)

        if not valid_events:
            continue

        # Fire the GPU exactly ONE time for the whole batch (offloaded to thread)
        batch_scores = await asyncio.to_thread(score_texts_batched, texts_to_score)

        scored = 0
        tier_counts = {
            "tier1_count": 0,
            "tier2_count": 0,
            "tier3_count": 0,
            "economy_news_count": 0,
            "turkish_economy_count": 0,
            "unknown_count": 0,
        }
        for event, probs in zip(valid_events, batch_scores):
            data = event.get("data", {})
            text = data.get("text", "")

            tweet_time_s = _parse_event_time(event)
            score = compound_score(probs)
            author_handle = extract_author_handle(data, event)
            credibility = resolve_source_credibility(author_handle)
            enrich_scored_item(data, score, author_handle)
            tier_counts[tier_count_field(credibility.source_tier)] += 1

            engagement = float(data.get("likeCount", 0) or 0) + float(data.get("retweetCount", 0) or 0)

            aggregator.add(
                symbol=symbol,
                score=score,
                tweet_time_s=tweet_time_s,
                tweet_text=text[:500],
                engagement=engagement,
                source_weight=credibility.source_weight,
                source_tier=credibility.source_tier,
            )
            scored += 1

        if scored > 0:
            LOGGER.info(
                "scored %d tweet(s) for %s from %d events source_tiers=%s",
                scored,
                symbol,
                len(events),
                tier_counts,
            )
        total_scored += scored

        await asyncio.sleep(0.12)

    return total_scored


# ── Public entry point ──────────────────────────────────────────


async def start_xquik_sentiment_stream(stop: asyncio.Event) -> None:
    """Public entry point — runs XQuik keyword monitoring pipeline."""
    if not settings.XQUIK_API:
        LOGGER.error("XQUIK_API not set — tweet sentiment pipeline disabled")
        return

    aggregator = SentimentAggregator()

    LOGGER.info(
        "XQuik sentiment stream starting (poll every %ds)",
        settings.XQUIK_POLL_INTERVAL_S,
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
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

        LOGGER.info("fast-forwarding event cursors to skip old tweets")
        await _fast_forward_cursors(client, symbol_monitors)

        async def periodic_flusher() -> None:
            while not stop.is_set():
                try:
                    await asyncio.sleep(15.0)
                    now_s = time.time()
                    # Passively flushes any stale buckets whose logical end times are in the past
                    aggregator._maybe_flush(now_s)
                except Exception as e:
                    LOGGER.error("error in periodic sentiment flusher: %s", e)

        flusher_task = asyncio.create_task(periodic_flusher())

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

                try:
                    await asyncio.wait_for(
                        stop.wait(), timeout=settings.XQUIK_POLL_INTERVAL_S
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            LOGGER.info("pausing all keyword monitors to save credits")
            flusher_task.cancel()
            await asyncio.gather(flusher_task, return_exceptions=True)
            await _pause_all_monitors(client, symbol_monitors)

    aggregator.flush_all()
    LOGGER.info("XQuik sentiment stream stopped")
