"""Dual sentiment engine — FinBERT (news) + CryptoBERT (tweets).

Two finance-domain BERT models are loaded lazily on first use:

* **FinBERT** (``ProsusAI/finbert``) — trained on Reuters financial news.
  Best for formal, well-structured text such as earnings reports and
  news headlines.  Labels: *positive*, *negative*, *neutral*.

* **CryptoBERT** (``ElKulako/cryptobert``) — pre-trained on 3.2M crypto
  tweets, then fine-tuned on ~2M StockTwits posts.  Best for informal,
  slang-heavy social-media text.  Labels: *Bullish*, *Bearish*, *Neutral*
  (normalised to *positive*, *negative*, *neutral* by this module).

Both models produce a probability dict ``{positive, negative, neutral}``
which is converted into a single compound score in [−1, +1]:

    score = p(positive) − p(negative)

Source-specific scorers import from here so models are loaded exactly once
per process.

Additionally, :func:`is_english` provides lightweight language detection
(via ``langdetect``) so that non-English tweets can be filtered before
scoring.
"""

from __future__ import annotations

import logging
import re

from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

LOGGER = logging.getLogger(__name__)

# ── Lazy model loading ─────────────────────────────────────────

_finbert_pipeline = None
_cryptobert_pipeline = None


def _get_finbert_pipeline():
    """Lazily load the FinBERT sentiment pipeline (for news articles)."""
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline

    LOGGER.info("loading FinBERT sentiment model (first use) …")
    # noinspection PyBroadException
    try:
        from transformers import pipeline as hf_pipeline

        _finbert_pipeline = hf_pipeline(  # type: ignore
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            revision="refs/pr/29",
            top_k=None,
            truncation=True,
            max_length=512,
            device=0,
        )
        LOGGER.info("FinBERT model loaded successfully")
    except Exception:
        LOGGER.exception("failed to load FinBERT — news sentiment scoring disabled")
        _finbert_pipeline = None

    return _finbert_pipeline


def _get_cryptobert_pipeline():
    """Lazily load the CryptoBERT sentiment pipeline (for crypto tweets)."""
    global _cryptobert_pipeline
    if _cryptobert_pipeline is not None:
        return _cryptobert_pipeline

    LOGGER.info("loading CryptoBERT sentiment model (first use) …")
    # noinspection PyBroadException
    try:
        from transformers import pipeline as hf_pipeline

        _cryptobert_pipeline = hf_pipeline(  # type: ignore
            "sentiment-analysis",
            model="ElKulako/cryptobert",
            tokenizer="ElKulako/cryptobert",
            top_k=None,
            truncation=True,
            max_length=512,
            device=0,
        )
        LOGGER.info("CryptoBERT model loaded successfully")
    except Exception:
        LOGGER.exception("failed to load CryptoBERT — tweet sentiment scoring disabled")
        _cryptobert_pipeline = None

    return _cryptobert_pipeline


# ── Label normalisation ────────────────────────────────────────

# CryptoBERT labels → standard labels used throughout the codebase
_CRYPTOBERT_LABEL_MAP = {
    "Bullish": "positive",
    "Bearish": "negative",
    "Neutral": "neutral",
}


def _normalise_cryptobert(scores: dict[str, float]) -> dict[str, float]:
    """Map CryptoBERT labels (Bullish/Bearish/Neutral) to standard names."""
    return {
        _CRYPTOBERT_LABEL_MAP.get(k, k): v
        for k, v in scores.items()
    }


# ── Scoring helpers ────────────────────────────────────────────

_NEUTRAL_FALLBACK = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}


def _score_batched(texts: list[str], pipe, normalise_fn=None) -> list[dict[str, float]]:
    """Score a batch of texts through a HF pipeline with optional label normalisation."""
    if pipe is None or not texts:
        return [_NEUTRAL_FALLBACK.copy() for _ in texts]

    # noinspection PyBroadException
    try:
        truncated = [t[:512] for t in texts]
        dataset = Dataset.from_dict({"text": truncated})

        parsed = []
        batch_results = pipe(KeyDataset(dataset, "text"), batch_size=64)

        for result in batch_results:
            scores = {item["label"]: float(item["score"]) for item in result}  # type: ignore
            if normalise_fn:
                scores = normalise_fn(scores)
            parsed.append(scores)

        return parsed
    except Exception:
        LOGGER.exception("batch inference failed")
        return [_NEUTRAL_FALLBACK.copy() for _ in texts]


def score_news_batched(texts: list[str]) -> list[dict[str, float]]:
    """Score news articles with FinBERT (optimised for formal financial text)."""
    return _score_batched(texts, _get_finbert_pipeline())


def score_tweets_batched(texts: list[str]) -> list[dict[str, float]]:
    """Score crypto tweets with CryptoBERT (optimised for informal crypto language)."""
    return _score_batched(texts, _get_cryptobert_pipeline(), _normalise_cryptobert)


# Backward compatibility — existing code that imports score_texts_batched
# will continue to work, falling through to FinBERT.
score_texts_batched = score_news_batched


def compound_score(probs: dict[str, float]) -> float:
    """Convert 3-class probabilities to a single score in [−1, +1]."""
    return probs.get("positive", 0.0) - probs.get("negative", 0.0)


# ── Language detection ─────────────────────────────────────────

# Regex patterns for stripping language-neutral noise
_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_TAG_RE = re.compile(r"[#$][A-Za-z][A-Za-z0-9_]*")

def is_english(text: str) -> bool:
    """Return True if *text* is detected as English.

    Cleans language-neutral noise (URLs, mentions, hashtags, and cashtags)
    and uses ``langdetect`` for lightweight detection on the remaining prose.

    Returns True on detection failure (fail-open) to avoid accidentally dropping
    legitimate English tweets that are too short to classify.
    """
    if not text or not text.strip():
        return False

    try:
        # 2. Strip language-neutral noise to leave only actual prose
        cleaned = _TAG_RE.sub(" ", _MENTION_RE.sub(" ", _URL_RE.sub(" ", text))).strip()

        # If there's no prose left at all, fail-open (assume English)
        if not cleaned:
            return True

        from langdetect import detect
        return detect(cleaned) == "en"
    except Exception:
        # Fail-open: if detection fails, assume English
        return True
