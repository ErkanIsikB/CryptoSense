"""News RSS Scorer — runs FinBERT on institutional news articles.

Uses the shared :mod:`src.models.sentiment_models` engine's FinBERT
pipeline for inference (optimised for formal financial text) and
attributes each article to the tracked symbols it mentions. Scored
articles are handed to :mod:`src.feature_engineering.news_rss_aggregator`
for bucketing and persistence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.models.sentiment_models import compound_score, score_news_batched

LOGGER = logging.getLogger("news_rss_scorer")

import re

# Strict regex matching with word boundaries to prevent false positives (e.g. matching "sol" inside "solution")
SYMBOL_KEYWORDS_RE = {
    "BTC": re.compile(r"\b(bitcoin|btc)\b", re.IGNORECASE),
    "ETH": re.compile(r"\b(ethereum|eth|ether)\b", re.IGNORECASE),
    "SOL": re.compile(r"\b(solana|sol)\b", re.IGNORECASE),
    "BNB": re.compile(r"\b(binance|bnb)\b", re.IGNORECASE),
    "AVAX": re.compile(r"\b(avalanche|avax)\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class ScoredArticle:
    """A news article with its FinBERT compound score and attributed symbols."""

    text: str
    score: float
    symbols: list[str]


def _get_symbols_from_text(text: str) -> list[str]:
    """Identify which tracked symbols are mentioned in the news title/summary using regex word boundaries."""
    found = []
    for symbol, pattern in SYMBOL_KEYWORDS_RE.items():
        if pattern.search(text):
            found.append(symbol)

    # If no specific coin is mentioned, attribute to BTC as the market proxy
    if not found:
        return ["BTC"]
    return found


def score_articles(texts: list[str]) -> list[ScoredArticle]:
    """Score a batch of article texts with FinBERT and attribute symbols."""
    if not texts:
        return []

    try:
        # Score with FinBERT in a single batch
        batch_scores = score_news_batched(texts)
    except Exception as e:
        LOGGER.error("FinBERT batch scoring failed: %s", e)
        batch_scores = [{"positive": 0.0, "negative": 0.0, "neutral": 1.0} for _ in texts]

    scored: list[ScoredArticle] = []
    for text, probs in zip(texts, batch_scores):
        try:
            scored.append(
                ScoredArticle(
                    text=text,
                    score=compound_score(probs),
                    symbols=_get_symbols_from_text(text),
                )
            )
        except Exception as e:
            LOGGER.error("FinBERT scoring or processing failed for news article: %s", e)

    return scored
