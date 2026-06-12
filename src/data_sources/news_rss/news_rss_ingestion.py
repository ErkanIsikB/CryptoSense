"""Reliable News RSS Ingestion Pipeline (Phase 3).

Fetches latest news from Tier-1 crypto sources (CoinDesk, CoinTelegraph)
to provide institutional sentiment validation against retail Twitter sentiment.

Ingestion only: this module fetches and deduplicates articles, then hands
them to ``news_rss_scorer`` (FinBERT) and ``news_rss_aggregator`` (5-min
buckets → TimescaleDB) in ``src.feature_engineering``.
"""

import asyncio
import logging
import re
import time
from xml.etree import ElementTree

import httpx

from src.feature_engineering.news_rss_scorer import score_articles
from src.feature_engineering.news_rss_aggregator import aggregate_and_store

LOGGER = logging.getLogger("news_rss")

# Tier-1 Crypto News RSS Feeds
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptopanic.com/news/rss/",
    "https://cryptoslate.com/feed/",
    "https://www.newsbtc.com/feed/",
]

_last_seen_titles: set[str] = set()


def _source_name_for_feed(feed_url: str) -> str:
    if "cointelegraph" in feed_url: return "CoinTelegraph"
    if "coindesk" in feed_url: return "CoinDesk"
    if "cryptopanic" in feed_url: return "CryptoPanic"
    if "cryptoslate" in feed_url: return "CryptoSlate"
    if "newsbtc" in feed_url: return "NewsBTC"
    return "News"


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)
_ITEM_RE = re.compile(r"<item>(.*?)</item>", re.DOTALL)
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL)
_DESC_RE = re.compile(r"<description>(.*?)</description>", re.DOTALL)


def _parse_xml_lenient(content: bytes) -> list[dict[str, str]]:
    """Parse RSS feed XML, falling back to regex extraction if malformed."""
    try:
        root = ElementTree.fromstring(content)
        items = []
        for item in root.findall(".//item"):
            items.append({
                "title": item.findtext("title") or "",
                "description": item.findtext("description") or ""
            })
        return items
    except ElementTree.ParseError as err:
        LOGGER.warning("Strict XML parsing failed. Falling back to regex extraction: %s", err)
        text = content.decode("utf-8", errors="ignore")
        items = []
        
        def clean_xml_field(val: str) -> str:
            cdata_match = _CDATA_RE.search(val)
            if cdata_match:
                val = cdata_match.group(1)
            # Unescape common entities
            val = val.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            val = val.replace("&quot;", '"').replace("&apos;", "'").replace("&#8217;", "'")
            return val.strip()

        for item_content in _ITEM_RE.findall(text):
            t_match = _TITLE_RE.search(item_content)
            d_match = _DESC_RE.search(item_content)
            
            title = clean_xml_field(t_match.group(1)) if t_match else ""
            desc = clean_xml_field(d_match.group(1)) if d_match else ""
            if title or desc:
                items.append({"title": title, "description": desc})
        return items


async def _fetch_new_articles(client: httpx.AsyncClient) -> list[str]:
    """Fetch all RSS feeds and return texts of articles not seen before."""
    new_articles = []

    for feed_url in RSS_FEEDS:
        source_name = _source_name_for_feed(feed_url)

        try:
            resp = await client.get(feed_url, timeout=10.0)
            resp.raise_for_status()

            items = _parse_xml_lenient(resp.content)
            for item in items:
                title = item.get("title") or ""
                description = item.get("description") or ""

                if not title or title in _last_seen_titles:
                    continue

                # We found a new article!
                _last_seen_titles.add(title)

                # Keep memory bound
                if len(_last_seen_titles) > 1000:
                    _last_seen_titles.clear()

                # Clean up description (remove all HTML elements, attributes, and tags cleanly)
                desc_clean = _HTML_TAG_RE.sub(" ", description).strip()

                # Combine source, title, and description for maximum context
                full_text = f"[{source_name}] {title} - {desc_clean}"
                new_articles.append(full_text)
        except Exception as e:
            LOGGER.warning("Failed to fetch RSS feed %s: %s", feed_url, e)

    return new_articles


async def fetch_and_process_rss(client: httpx.AsyncClient) -> None:
    """Fetch RSS feeds, then hand new articles to the scorer and aggregator."""
    new_articles = await _fetch_new_articles(client)

    if not new_articles:
        LOGGER.info("No new institutional news articles found.")
        return

    LOGGER.info("Found %d new articles. Scoring with FinBERT...", len(new_articles))
    scored = await asyncio.to_thread(score_articles, new_articles)
    await aggregate_and_store(scored)


async def start_news_rss_stream(stop_event: asyncio.Event) -> None:
    """Background task to poll RSS feeds every 5 minutes."""
    LOGGER.info("Institutional News RSS stream started.")

    # Pre-fill the cache to avoid scoring hundreds of old articles on startup
    async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
        for feed_url in RSS_FEEDS:
            try:
                resp = await client.get(feed_url, timeout=10.0)
                items = _parse_xml_lenient(resp.content)
                for item in items:
                    title = item.get("title")
                    if title:
                        _last_seen_titles.add(title)
            except Exception:
                pass

    while not stop_event.is_set():
        try:
            async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
                await fetch_and_process_rss(client)
        except Exception as e:
            LOGGER.exception("News RSS loop error: %s", e)

        # Sleep until the next 5-minute bucket (plus 10 seconds to allow news to publish)
        now = time.time()
        sleep_time = 300 - (now % 300) + 10
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=sleep_time)
        except asyncio.TimeoutError:
            pass
