"""FinBERT engine — shared sentiment model used by all source-specific scorers.

The model ``ProsusAI/finbert`` is a finance-domain BERT that classifies text
into *positive*, *negative*, or *neutral* and returns calibrated probabilities.
We convert these into a single compound score in [−1, +1]:

    score = p(positive) − p(negative)

The model is loaded lazily on first use to avoid slowing down import time.
Source-specific scorers (``xquik_scorer``, ``news_rss_scorer``) import from
here so the model is loaded exactly once per process.
"""

from __future__ import annotations

import logging

from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

LOGGER = logging.getLogger("finbert")

# ── Lazy model loading ─────────────────────────────────────────

_pipeline = None


def _get_pipeline():
    """Lazily load the FinBERT sentiment pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    LOGGER.info("loading FinBERT sentiment model (first use) …")
    # noinspection PyBroadException
    try:
        from transformers import pipeline as hf_pipeline

        _pipeline = hf_pipeline(# type: ignore
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            revision="refs/pr/29",  # <--- THE MAGIC FIX: Pull from the safe PR branch
            top_k=None,  # return all 3 class probabilities
            truncation=True,
            max_length=512,
            device=0
        )
        LOGGER.info("FinBERT model loaded successfully")
    except Exception:
        LOGGER.exception("failed to load FinBERT — sentiment scoring disabled")
        _pipeline = None

    return _pipeline


# ── Scoring helpers ────────────────────────────────────────────


def score_texts_batched(texts: list[str]) -> list[dict[str, float]]:
    """Score a whole list of texts at once using KeyDataset for maximum GPU efficiency."""
    pipe = _get_pipeline()
    if pipe is None or not texts:
        return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0} for _ in texts]

    # noinspection PyBroadException
    try:
        truncated = [t[:512] for t in texts]

        # 1. Create the dataset
        dataset = Dataset.from_dict({"text": truncated})

        parsed = []

        # 2. Use KeyDataset to stream batches directly to the VRAM without list conversion
        batch_results = pipe(KeyDataset(dataset, "text"), batch_size=64)

        for result in batch_results:
            # type: ignore
            scores = {item["label"]: float(item["score"]) for item in result}  # type: ignore
            parsed.append(scores)

        return parsed
    except Exception:
        LOGGER.exception("FinBERT batch inference failed")
        return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0} for _ in texts]


def compound_score(probs: dict[str, float]) -> float:
    """Convert 3-class probabilities to a single score in [−1, +1]."""
    return probs.get("positive", 0.0) - probs.get("negative", 0.0)
