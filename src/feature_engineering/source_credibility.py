"""Source credibility lookup and weighted sentiment helpers.

The resolver is intentionally independent of the sentiment model. It maps
known X/Twitter handles to moderate credibility weights and falls back to an
included ``unknown`` tier so ordinary accounts are never dropped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("source_credibility")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "core" / "config" / "twitter_source_tiers.json"
MAX_DEFAULT_SOURCE_WEIGHT = 3.0
KNOWN_TIERS = (
    "tier1",
    "tier2",
    "tier3",
    "economy_news_sources",
    "turkish_economy_sources",
    "unknown",
)
TIER_COUNT_FIELDS = {
    "tier1": "tier1_count",
    "tier2": "tier2_count",
    "tier3": "tier3_count",
    "economy_news_sources": "economy_news_count",
    "turkish_economy_sources": "turkish_economy_count",
    "unknown": "unknown_count",
}


@dataclass(frozen=True)
class SourceCredibility:
    author_handle: str | None
    source_tier: str
    source_weight: float
    source_focus: str | None = None
    source_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_twitter_handle(handle: Any) -> str | None:
    """Return a canonical lower-case handle with ``@`` or ``None``."""
    if handle is None:
        return None

    normalized = str(handle).strip()
    if not normalized:
        return None

    if normalized.startswith("https://") or normalized.startswith("http://"):
        normalized = normalized.rstrip("/").split("/")[-1]

    normalized = normalized.lstrip("@").strip()
    if not normalized:
        return None

    return f"@{normalized.lower()}"


def display_twitter_handle(handle: Any) -> str | None:
    normalized = normalize_twitter_handle(handle)
    if normalized is None:
        return None
    return f"@{normalized[1:]}"


def _configured_display_handle(handle: Any) -> str | None:
    normalized = normalize_twitter_handle(handle)
    if normalized is None:
        return None

    raw = str(handle).strip()
    if raw.startswith("https://"):
        return display_twitter_handle(raw)
    raw = raw if raw.startswith("@") else f"@{raw}"
    return raw


def _safe_weight(raw_weight: Any, default: float = 1.0) -> float:
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(weight, MAX_DEFAULT_SOURCE_WEIGHT))


@lru_cache(maxsize=1)
def _load_source_index(config_path: str = str(DEFAULT_CONFIG_PATH)) -> dict[str, SourceCredibility]:
    path = Path(config_path)
    try:
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        LOGGER.warning("source tier config not found at %s; all sources will be unknown", path)
        return {}
    except json.JSONDecodeError:
        LOGGER.exception("source tier config is invalid JSON at %s; all sources will be unknown", path)
        return {}

    source_index: dict[str, SourceCredibility] = {}
    for tier_name, tier_config in config.get("tiers", {}).items():
        tier_weight = _safe_weight(tier_config.get("default_weight"), default=1.0)
        for account in tier_config.get("accounts", []):
            handle = normalize_twitter_handle(account.get("handle"))
            if handle is None:
                continue
            source_index[handle] = SourceCredibility(
                author_handle=_configured_display_handle(account.get("handle")),
                source_tier=tier_name,
                source_weight=tier_weight,
                source_focus=account.get("focus"),
                source_reason=account.get("reason"),
            )

    LOGGER.info("loaded %d source credibility account(s) from %s", len(source_index), path)
    return source_index


def resolve_source_credibility(author_handle: Any) -> SourceCredibility:
    """Resolve an author handle to credibility metadata.

    Unknown, missing, or malformed handles are kept with weight 1.0.
    """
    normalized = normalize_twitter_handle(author_handle)
    if normalized is not None:
        source = _load_source_index().get(normalized)
        if source is not None:
            return source

    return SourceCredibility(
        author_handle=display_twitter_handle(author_handle),
        source_tier="unknown",
        source_weight=1.0,
        source_focus=None,
        source_reason="Default unknown source",
    )


def extract_author_handle(*records: Any) -> str | None:
    """Best-effort author handle extraction across common provider shapes."""
    direct_fields = (
        "author_handle",
        "authorHandle",
        "authorUsername",
        "author_username",
        "username",
        "userName",
        "screen_name",
        "screenName",
        "handle",
    )
    nested_fields = ("author", "user", "creator", "account")

    for record in records:
        if not isinstance(record, dict):
            continue

        for field in direct_fields:
            handle = normalize_twitter_handle(record.get(field))
            if handle is not None:
                return handle

        for field in nested_fields:
            nested = record.get(field)
            if not isinstance(nested, dict):
                continue
            for nested_field in direct_fields:
                handle = normalize_twitter_handle(nested.get(nested_field))
                if handle is not None:
                    return handle

    return None


def enrich_scored_item(item: dict[str, Any], raw_sentiment_score: float, author_handle: Any = None) -> dict[str, Any]:
    """Add source credibility fields to a scored tweet/article-like item."""
    resolved_handle = author_handle if author_handle is not None else extract_author_handle(item)
    credibility = resolve_source_credibility(resolved_handle)
    item["author_handle"] = credibility.author_handle
    item["source_tier"] = credibility.source_tier
    item["source_weight"] = credibility.source_weight
    item["source_focus"] = credibility.source_focus
    item["source_reason"] = credibility.source_reason
    item["raw_sentiment_score"] = raw_sentiment_score
    item["weighted_sentiment_contribution"] = raw_sentiment_score * credibility.source_weight
    return item


def empty_tier_counts() -> dict[str, int]:
    return {field: 0 for field in TIER_COUNT_FIELDS.values()}


def tier_count_field(source_tier: str | None) -> str:
    return TIER_COUNT_FIELDS.get(source_tier or "unknown", "unknown_count")


def calculate_weighted_sentiment(scored_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute raw and weighted aggregate sentiment for already-scored items."""
    if not scored_items:
        return {
            "unweighted_avg_sentiment": 0.0,
            "weighted_avg_sentiment": 0.0,
            "tweet_count": 0,
            "total_source_weight": 0.0,
            **empty_tier_counts(),
        }

    raw_sum = 0.0
    weighted_sum = 0.0
    total_weight = 0.0
    counts = empty_tier_counts()

    for item in scored_items:
        raw_score = float(item.get("raw_sentiment_score", item.get("score", 0.0)) or 0.0)
        weight = _safe_weight(item.get("source_weight"), default=1.0)
        raw_sum += raw_score
        weighted_sum += raw_score * weight
        total_weight += weight
        counts[tier_count_field(item.get("source_tier"))] += 1

    n = len(scored_items)
    return {
        "unweighted_avg_sentiment": raw_sum / n,
        "weighted_avg_sentiment": weighted_sum / total_weight if total_weight > 0 else 0.0,
        "tweet_count": n,
        "total_source_weight": total_weight,
        **counts,
    }
