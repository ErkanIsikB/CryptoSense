"""Reliable News RSS Ingestion Pipeline (Phase 3).

Fetches latest news from Tier-1 crypto sources (CoinDesk, CoinTelegraph)
to provide institutional sentiment validation against retail Twitter sentiment.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import httpx

from src.core.config import settings
from src.feature_engineering.sentiment_scorer import score_texts_batched, compound_score
from src.db.db import execute_batch_async

LOGGER = logging.getLogger("news_rss")

# Tier-1 Crypto News RSS Feeds
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptopanic.com/news/rss/",
    "https://cryptoslate.com/feed/",
    "https://www.newsbtc.com/feed/",
]

# Simple keyword matching for symbol attribution
SYMBOL_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "BNB": ["binance", "bnb"],
    "AVAX": ["avalanche", "avax"],
}

_last_seen_titles: set[str] = set()

def _get_symbols_from_text(text: str) -> list[str]:
    """Identify which tracked symbols are mentioned in the news title/summary."""
    text_lower = text.lower()
    found = []
    for symbol, keywords in SYMBOL_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(symbol)
    
    # If no specific coin is mentioned, attribute to BTC as the market proxy
    if not found:
        return ["BTC"]
    return found

async def fetch_and_process_rss(client: httpx.AsyncClient) -> None:
    """Fetch RSS feeds, score with FinBERT, and save to database."""
    new_articles = []

    for feed_url in RSS_FEEDS:
        source_name = "News"
        if "cointelegraph" in feed_url: source_name = "CoinTelegraph"
        elif "coindesk" in feed_url: source_name = "CoinDesk"
        elif "cryptopanic" in feed_url: source_name = "CryptoPanic"
        elif "cryptoslate" in feed_url: source_name = "CryptoSlate"
        elif "newsbtc" in feed_url: source_name = "NewsBTC"

        try:
            resp = await client.get(feed_url, timeout=10.0)
            resp.raise_for_status()
            
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item"):
                title = item.findtext("title") or ""
                description = item.findtext("description") or ""
                
                if not title or title in _last_seen_titles:
                    continue
                
                # We found a new article!
                _last_seen_titles.add(title)
                
                # Keep memory bound
                if len(_last_seen_titles) > 1000:
                    _last_seen_titles.clear()

                # Clean up description (remove basic HTML tags if any)
                desc_clean = description.replace("<p>", "").replace("</p>", "").replace("<br>", " ")
                
                # Combine source, title, and description for maximum context
                full_text = f"[{source_name}] {title} - {desc_clean}"
                new_articles.append(full_text)
        except Exception as e:
            LOGGER.warning("Failed to fetch RSS feed %s: %s", feed_url, e)

    if not new_articles:
        LOGGER.info("No new institutional news articles found.")
        return

    LOGGER.info("Found %d new articles. Scoring with FinBERT...", len(new_articles))
    
    # Process each new article
    rows = []
    bucket_ts = time.time() - (time.time() % 300)
    bucket_dt = datetime.fromtimestamp(bucket_ts, tz=timezone.utc)
    
    try:
        # Score with FinBERT in a single batch
        batch_scores = score_texts_batched(new_articles)
    except Exception as e:
        LOGGER.error("FinBERT batch scoring failed: %s", e)
        batch_scores = [{"positive": 0.0, "negative": 0.0, "neutral": 1.0} for _ in new_articles]

    for text_payload, scores in zip(new_articles, batch_scores):
        try:
            score = compound_score(scores)
            symbols = _get_symbols_from_text(text_payload)
            
            for symbol in symbols:
                positive = 1 if score > 0.2 else 0
                negative = 1 if score < -0.2 else 0
                neutral = 1 if -0.2 <= score <= 0.2 else 0
                
                rows.append((
                    bucket_dt,
                    symbol,
                    score,
                    1, # news_count
                    positive,
                    negative,
                    neutral,
                    text_payload[:500] # sample headline with source and description
                ))
        except Exception as e:
            LOGGER.error("FinBERT scoring or processing failed for news article: %s", e)

    # Insert into DB (Aggregating simple rows)
    if rows:
        sql = """
            INSERT INTO news_sentiment_5m
                (bucket, symbol, avg_score, news_count, positive_count, negative_count, neutral_count, sample_headline)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (bucket, symbol) DO UPDATE SET
                avg_score = (news_sentiment_5m.avg_score * news_sentiment_5m.news_count + EXCLUDED.avg_score) / (news_sentiment_5m.news_count + 1),
                news_count = news_sentiment_5m.news_count + 1,
                positive_count = news_sentiment_5m.positive_count + EXCLUDED.positive_count,
                negative_count = news_sentiment_5m.negative_count + EXCLUDED.negative_count,
                neutral_count = news_sentiment_5m.neutral_count + EXCLUDED.neutral_count,
                sample_headline = EXCLUDED.sample_headline;
        """
        try:
            await execute_batch_async(sql, rows)
            LOGGER.info("Flushed %d news sentiment records to DB.", len(rows))
        except Exception as e:
            LOGGER.error("Failed to flush news sentiment to DB: %s", e)


async def start_news_rss_stream(stop_event: asyncio.Event) -> None:
    """Background task to poll RSS feeds every 5 minutes."""
    LOGGER.info("Institutional News RSS stream started.")
    
    # Pre-fill the cache to avoid scoring hundreds of old articles on startup
    async with httpx.AsyncClient(follow_redirects=True) as client:
        for feed_url in RSS_FEEDS:
            try:
                resp = await client.get(feed_url, timeout=10.0)
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item"):
                    title = item.findtext("title")
                    if title:
                        _last_seen_titles.add(title)
            except Exception:
                pass

    while not stop_event.is_set():
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
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
