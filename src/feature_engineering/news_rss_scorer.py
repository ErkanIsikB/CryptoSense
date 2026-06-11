"""News RSS Scorer — runs FinBERT on institutional news articles.

Uses the shared :mod:`src.feature_engineering.finbert` engine for inference
and attributes each article to the tracked symbols it mentions. Scored
articles are handed to :mod:`src.feature_engineering.news_rss_aggregator`
for bucketing and persistence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.feature_engineering.finbert import compound_score, score_texts_batched

LOGGER = logging.getLogger("news_rss_scorer")

# Simple keyword matching for symbol attribution
SYMBOL_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "BNB": ["binance", "bnb"],
    "AVAX": ["avalanche", "avax"],
}


@dataclass(frozen=True)
class ScoredArticle:
    """A news article with its FinBERT compound score and attributed symbols."""

    text: str
    score: float
    symbols: list[str]


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


def score_articles(texts: list[str]) -> list[ScoredArticle]:
    """Score a batch of article texts with FinBERT and attribute symbols."""
    if not texts:
        return []

    try:
        # Score with FinBERT in a single batch
        batch_scores = score_texts_batched(texts)
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
