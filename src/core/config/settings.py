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
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "scripts" / "data"

# ── Binance (shared by trade & orderbook streams) ──────────────────────
BINANCE_SYMBOLS: tuple[str, ...] = tuple(
    part.strip().lower()
    for part in os.getenv("BINANCE_SYMBOLS", "btcusdt,ethusdt,solusdt,bnbusdt,avaxusdt").split(",")
    if part.strip()
)

BINANCE_MARKET_WS_BASE: str = os.getenv(
    "BINANCE_MARKET_WS_BASE", "wss://fstream.binance.com/market"
)

BINANCE_REST_BASE: str = os.getenv("BINANCE_REST_BASE", "https://api.binance.com")

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

ORDERBOOK_DEPTH_LIMIT: int = int(os.getenv("ORDERBOOK_DEPTH_LIMIT", "20"))
ORDERBOOK_POLL_INTERVAL_S: float = float(os.getenv("ORDERBOOK_POLL_INTERVAL_S", "2.0"))
ORDERBOOK_SYMBOL_PAUSE_S: float = float(os.getenv("ORDERBOOK_SYMBOL_PAUSE_S", "0.15"))
ORDERBOOK_REST_TIMEOUT_S: float = float(os.getenv("ORDERBOOK_REST_TIMEOUT_S", "10"))

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


def _parse_csv_set(raw: str | None, defaults: set[str]) -> set[str]:
    """Parse a comma-separated env var into a set of upper-cased strings."""
    if raw is None:
        return defaults
    normalized = {part.strip().upper() for part in raw.split(",") if part.strip()}
    return normalized or defaults


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_tuple(raw: str | None, defaults: tuple[str, ...], *, upper: bool = False) -> tuple[str, ...]:
    if raw is None:
        return defaults
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not parts:
        return defaults
    if upper:
        return tuple(part.upper() for part in parts)
    return parts


ENABLED_TOKENS: set[str] = _parse_csv_set(
    os.getenv("ENABLED_TOKENS"), {"BTC", "ETH", "SOL", "BNB", "AVAX"}
)
ENABLED_CATEGORIES: set[str] = {
    v.lower()
    for v in _parse_csv_set(
        os.getenv("ENABLED_CATEGORIES"), {"TRANSFERS", "TRADES"}
    )
}

# ── Bitquery ──────────────────────────────────────────────────────────
BITQUERY_API_KEY: str = os.getenv("BITQUERY_API_KEY", "").strip()

# ── Database (TimescaleDB) ────────────────────────────────────
DB_URL: str = os.getenv("DB_URL", "").strip()

# ── XQuik (X/Twitter Sentiment) ──────────────────────────────
XQUIK_API: str = os.getenv("XQUIK_API", "").strip()
XQUIK_POLL_INTERVAL_S: int = int(os.getenv("XQUIK_POLL_INTERVAL_S", "300"))

# ── Aggregation ───────────────────────────────────────────────
AGGREGATION_WINDOW_SECONDS: int = int(os.getenv("AGGREGATION_WINDOW_SECONDS", "300"))

# ── CEX Flow (Bitquery) ──────────────────────────────────────
CEX_FLOW_POLL_INTERVAL_S: int = int(os.getenv("CEX_FLOW_POLL_INTERVAL_S", "300"))
CEX_FLOW_NETWORKS: tuple[str, ...] = tuple(
    part.strip().lower()
    for part in os.getenv("CEX_FLOW_NETWORKS", "eth,bsc,solana").split(",")
    if part.strip()
)
CEX_FLOW_TIMEOUT_S: float = float(os.getenv("CEX_FLOW_TIMEOUT_S", "30"))

# ── Optional JSONL backup alongside DB writes ────────────────
ENABLE_JSONL_BACKUP: bool = (
    os.getenv("ENABLE_JSONL_BACKUP", "false").strip().lower() in {"1", "true", "yes"}
)

# ── Scheduled anomaly-model retraining ────────────────────────
RETRAIN_ENABLED: bool = _parse_bool(os.getenv("RETRAIN_ENABLED"), False)
RETRAIN_INTERVAL_DAYS: int = max(1, int(os.getenv("RETRAIN_INTERVAL_DAYS", "14")))
RETRAIN_LOOKBACK_DAYS: int = max(1, int(os.getenv("RETRAIN_LOOKBACK_DAYS", "14")))
RETRAIN_DEVICE: str = os.getenv("RETRAIN_DEVICE", "auto").strip().lower() or "auto"
RETRAIN_TIMEZONE: str = os.getenv("RETRAIN_TIMEZONE", "UTC").strip() or "UTC"
RETRAIN_MISFIRE_GRACE_SECONDS: int = max(
    1, int(os.getenv("RETRAIN_MISFIRE_GRACE_SECONDS", "3600"))
)
RETRAIN_SYMBOLS: tuple[str, ...] = _parse_csv_tuple(
    os.getenv("RETRAIN_SYMBOLS"),
    ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT"),
    upper=True,
)
RETRAIN_OUTPUT_DIR: Path = Path(
    os.getenv("RETRAIN_OUTPUT_DIR", str(PROJECT_ROOT / "src" / "models" / "saved_weights"))
)
