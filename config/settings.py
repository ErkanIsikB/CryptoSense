"""
Centralized settings — loaded once from the project-root .env file.

Every configurable value lives here so that individual modules never call
``os.getenv`` or ``load_dotenv`` directly.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Project paths ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"

# ── Binance (shared by trade & orderbook streams) ──────────────────────
BINANCE_SYMBOLS: tuple[str, ...] = tuple(
    part.strip().lower()
    for part in os.getenv("BINANCE_SYMBOLS", "btcusdt,ethusdt,solusdt,bnbusdt,avaxusdt").split(",")
    if part.strip()
)

BINANCE_MARKET_WS_BASE: str = os.getenv(
    "BINANCE_MARKET_WS_BASE", "wss://fstream.binance.com/market"
)

BINANCE_WS_BASES: tuple[str, ...] = tuple(
    part.strip().rstrip("/")
    for part in os.getenv(
        "BINANCE_WS_BASES",
        "wss://stream.binance.com:9443,wss://stream.binance.com:443,wss://data-stream.binance.vision",
    ).split(",")
    if part.strip()
) or (
    "wss://stream.binance.com:9443",
    "wss://stream.binance.com:443",
    "wss://data-stream.binance.vision",
)

WS_OPEN_TIMEOUT_S: float = float(os.getenv("WS_OPEN_TIMEOUT_S", "20"))

# ── Bitquery ────────────────────────────────────────────────────────────
BITQUERY_API_KEY: str = os.getenv("BITQUERY_API_KEY", "").strip()
BITQUERY_STREAM_URL: str = os.getenv(
    "BITQUERY_STREAM_URL", "wss://streaming.bitquery.io/graphql"
).strip()
WHALE_USD_THRESHOLD: float = float(os.getenv("WHALE_USD_THRESHOLD", "500000"))
TRADE_MIN_USD: float = float(os.getenv("TRADE_MIN_USD", "10000"))
BACKFILL_MINUTES: int = int(os.getenv("BACKFILL_MINUTES", "3"))
MAX_ACTIVE_STREAMS: int = int(os.getenv("MAX_ACTIVE_STREAMS", "0"))

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


def _parse_csv_set(raw: str | None, defaults: set[str]) -> set[str]:
    """Parse a comma-separated env var into a set of upper-cased strings."""
    if raw is None:
        return defaults
    normalized = {part.strip().upper() for part in raw.split(",") if part.strip()}
    return normalized or defaults


ENABLED_TOKENS: set[str] = _parse_csv_set(
    os.getenv("ENABLED_TOKENS"), {"BTC", "ETH", "SOL", "BNB", "AVAX"}
)
ENABLED_CATEGORIES: set[str] = {
    v.lower()
    for v in _parse_csv_set(
        os.getenv("ENABLED_CATEGORIES"), {"TRANSFERS", "TRADES"}
    )
}

# ── Sentiment (Tavily) ─────────────────────────────────────────────────
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "").strip()
TAVILY_API_URL: str = os.getenv("TAVILY_API_URL", "https://api.tavily.com/search")
SENTIMENT_INTERVAL_MINUTES: int = int(os.getenv("SENTIMENT_INTERVAL_MINUTES", "120"))
SENTIMENT_SEARCH_DEPTH: str = os.getenv("SENTIMENT_SEARCH_DEPTH", "basic").strip() or "basic"
SENTIMENT_MAX_TOKENS_PER_CYCLE: int = max(1, int(os.getenv("SENTIMENT_MAX_TOKENS_PER_CYCLE", "2")))
SENTIMENT_TIMEOUT_S: float = float(os.getenv("SENTIMENT_TIMEOUT_S", "20"))
SENTIMENT_INCLUDE_ANSWER: bool = (
    os.getenv("SENTIMENT_INCLUDE_ANSWER", "true").strip().lower() in {"1", "true", "yes"}
)
SENTIMENT_INCLUDE_IMAGES: bool = (
    os.getenv("SENTIMENT_INCLUDE_IMAGES", "false").strip().lower() in {"1", "true", "yes"}
)
SENTIMENT_MAX_RETRIES: int = max(0, int(os.getenv("SENTIMENT_MAX_RETRIES", "2")))
