"""XQuik Ingestion — real-time X (Twitter) keyword monitoring for crypto sentiment.

Flow:
1. On startup, ensure keyword monitors exist for each tracked coin
2. Fast-forward cursors to ignore old backlog
3. Every 5 minutes, poll the events API for strictly new tweets
4. Filter out retweets and hashtag-stuffed "crypto news" tweets that are
   not actually about the tracked coin
5. Score each tweet through FinBERT (Batched for GPU efficiency)
6. Feed scored tweets into SentimentAggregator (5-min buckets → TimescaleDB)

Uses the XQuik REST API:
- POST /api/v1/monitors/keywords — create keyword monitor
- GET  /api/v1/monitors/keywords — list existing monitors
- GET  /api/v1/events            — poll captured tweet events
- DELETE /api/v1/monitors/keywords/{id} — cleanup on shutdown
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any

import httpx

from src.core.config import settings
from src.models.sentiment_models import score_tweets_batched, compound_score, is_english
from src.feature_engineering.xquik_aggregator import SentimentAggregator
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


# ── Tweet relevance filtering ───────────────────────────────────
#
# Keyword monitors also match generic "crypto news" tweets that append a wall
# of coin tags (… #BTC #ETH #SOL #DOGE #crypto) to content that is not about
# the tracked coin at all. Those tweets pollute the per-symbol sentiment, so
# a tweet is only kept if its prose (text minus tags/links/mentions) actually
# mentions the coin, or if the coin is not just one tag in a multi-coin blast.

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_TAG_RE = re.compile(r"[#$][A-Za-z][A-Za-z0-9_]*")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# Terms that prove the prose is genuinely about the coin
_SYMBOL_TERMS: dict[str, re.Pattern[str]] = {
    "BTC": re.compile(r"\b(btc|bitcoin)\b", re.IGNORECASE),
    "ETH": re.compile(r"\b(eth|ethereum|ether)\b", re.IGNORECASE),
    "SOL": re.compile(r"\b(sol|solana)\b", re.IGNORECASE),
    "BNB": re.compile(r"\b(bnb|binance coin)\b", re.IGNORECASE),
    "AVAX": re.compile(r"\b(avax|avalanche)\b", re.IGNORECASE),
}

# Hashtags that name a topic rather than a specific coin — these don't count
# towards the multi-coin tag blast detection
_GENERIC_TAGS = {
    "crypto", "cryptocurrency", "cryptocurrencies", "cryptonews", "cryptotwitter",
    "blockchain", "web3", "defi", "nft", "nfts", "altcoin", "altcoins",
    "memecoin", "memecoins", "trading", "investing", "news", "breaking",
    "bullish", "bearish", "bullrun", "hodl", "fintech", "markets", "airdrop",
    "giveaway", "presale",
}

# A tweet whose prose never mentions the coin is dropped when it tags more
# than this many distinct coins (the aggregator/news-bot signature) …
MAX_OFFTOPIC_COIN_TAGS = 2
# … or when almost nothing is left after stripping tags, links, and mentions.
MIN_PROSE_WORDS = 3


def _is_offtopic_news_tweet(symbol: str, text: str) -> bool:
    """True when a tweet only touches ``symbol`` through hashtag/cashtag spam."""
    symbol_terms = _SYMBOL_TERMS.get(symbol)
    if symbol_terms is None:
        return False

    prose = _TAG_RE.sub(" ", _MENTION_RE.sub(" ", _URL_RE.sub(" ", text)))

    # The tweet body genuinely talks about the coin → keep it
    if symbol_terms.search(prose):
        return False

    # The coin appears only as a tag among several other coins → news blast
    tags = {t[1:].lower() for t in _TAG_RE.findall(text)}
    coin_tags = tags - _GENERIC_TAGS
    if len(coin_tags) > MAX_OFFTOPIC_COIN_TAGS:
        return True

    # Nothing of substance left once tags/links/mentions are stripped
    if len(_WORD_RE.findall(prose)) < MIN_PROSE_WORDS:
        return True

    return False


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


async def _fetch_events_page(
    client: httpx.AsyncClient,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Helper to fetch a single page of events from the XQuik API."""
    resp = await client.get(
        f"{XQUIK_BASE}/events",
        headers=_headers(),
        params=params,
    )
    resp.raise_for_status()
    return resp.json()


async def _fast_forward_cursors(
    client: httpx.AsyncClient,
    symbol_monitors: dict[str, str],
) -> None:
    """Record the newest event ID for each monitor without processing anything."""
    for symbol, monitor_id in symbol_monitors.items():
        try:
            data = await _fetch_events_page(
                client,
                {"keywordMonitorId": monitor_id, "limit": 1},
            )
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
        page = 0
        max_pages = 6  # Page 0 (initial) + up to 5 additional pages
        while page < max_pages:
            data = await _fetch_events_page(client, params)
            events = data.get("events", [])

            for evt in events:
                evt_id = str(evt.get("id", ""))
                if evt_id == last_seen:
                    found_seen = True
                    break
                new_events.append(evt)

            if page == 0:
                LOGGER.info(
                    "polled %s: %d new events from %d fetched (hitSeen=%s, hasMore=%s)",
                    symbol, len(new_events), len(events), found_seen, data.get("hasMore"),
                )
            else:
                LOGGER.info(
                    "polled %s (page %d): +%d events (hitSeen=%s)",
                    symbol, page, len(events), found_seen,
                )

            if found_seen or not data.get("hasMore"):
                break

            next_cursor = data.get("nextCursor")
            if not next_cursor:
                break
            params["after"] = next_cursor
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
        offtopic = 0

        non_english = 0

        for event in events:
            data = event.get("data", {})
            text = data.get("text", "")

            if not text or not text.strip():
                continue
            if data.get("isRetweet"):
                continue
            if _is_offtopic_news_tweet(symbol, text):
                offtopic += 1
                continue
            if not is_english(text):
                non_english += 1
                continue

            valid_events.append(event)
            texts_to_score.append(text)

        if offtopic or non_english:
            LOGGER.info(
                "filtered %d off-topic + %d non-English tweet(s) for %s",
                offtopic, non_english, symbol,
            )

        if not valid_events:
            continue

        # Fire the GPU exactly ONE time for the whole batch (offloaded to thread)
        batch_scores = await asyncio.to_thread(score_tweets_batched, texts_to_score)

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
                    aggregator.maybe_flush(now_s)
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
