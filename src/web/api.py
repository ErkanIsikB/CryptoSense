"""CryptoSense FastAPI Backend — Exposes real-time Time-series and LLM metrics from TimescaleDB."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.db.db import execute_query_fetch

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("api")

app = FastAPI(
    title="CryptoSense Core API",
    description="REST backend exposing real-time trade candles, orderbooks, CEX flows, anomalies, and LLM scores.",
    version="1.0.0",
)

# Enable CORS for the Streamlit dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Database Queries ────────────────────────────────────────────────


SQL_GET_LATEST_METRICS = """
WITH finalized_buckets AS (
    SELECT bucket, symbol FROM trade_candles_5m
    INTERSECT
    SELECT bucket, symbol FROM orderbook_snapshots_5m
    INTERSECT
    SELECT bucket, symbol || 'USDT' AS symbol FROM tweet_sentiment_5m
)
SELECT 
    fb.bucket AS bucket,
    fb.symbol AS symbol,
    t.close AS close_price,
    t.volume AS volume_5m,
    t.vwap AS vwap,
    t.net_trade AS net_trade,
    o.avg_spread AS spread,
    o.avg_mid_price AS mid_price,
    o.avg_bid_depth AS bid_depth,
    o.avg_ask_depth AS ask_depth,
    o.avg_imbalance AS imbalance,
    s.avg_score AS sentiment_score,
    s.tweet_count AS tweet_count,
    COALESCE(c.net_flow_usd, 0.0) AS net_cex_flow_usd,
    COALESCE(a.mse_score, 0.0) AS mse_score,
    COALESCE(a.is_anomaly, FALSE) AS is_anomaly,
    COALESCE(a.severity, 'NORMAL') AS severity
FROM finalized_buckets fb
JOIN trade_candles_5m t ON t.bucket = fb.bucket AND t.symbol = fb.symbol
JOIN orderbook_snapshots_5m o ON o.bucket = fb.bucket AND o.symbol = fb.symbol
JOIN tweet_sentiment_5m s ON s.bucket = fb.bucket AND s.symbol = REPLACE(fb.symbol, 'USDT', '')
LEFT JOIN (
    SELECT bucket, TRIM(symbol) as symbol, SUM(net_flow_usd) as net_flow_usd 
    FROM cex_flows_5m 
    GROUP BY bucket, symbol
) c ON c.bucket = fb.bucket AND c.symbol = TRIM(REPLACE(fb.symbol, 'USDT', ''))
LEFT JOIN ai_anomalies_5m a ON a.bucket = fb.bucket AND a.symbol = REPLACE(fb.symbol, 'USDT', '')
ORDER BY bucket DESC, symbol ASC
LIMIT 50;
"""

SQL_GET_LATEST_HEALTH_SCORES = """
SELECT bucket, symbol, health_score, reasoning, explanation, model_name, latency_ms, input_payload
FROM llm_health_scores
ORDER BY bucket DESC, symbol ASC
LIMIT 25;
"""

SQL_GET_ACTIVE_ANOMALIES = """
SELECT bucket, symbol, mse_score, is_anomaly, severity, llm_payload
FROM ai_anomalies_5m
WHERE is_anomaly = TRUE
ORDER BY bucket DESC, symbol ASC
LIMIT 30;
"""


# ── REST API Endpoints ───────────────────────────────────────────────


@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/latest-metrics")
def get_latest_metrics() -> list[dict[str, Any]]:
    """Retrieve the latest finalized bucket metrics across all symbols."""
    try:
        rows = execute_query_fetch(SQL_GET_LATEST_METRICS)
        result = []
        for r in rows:
            result.append(
                {
                    "bucket": r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]),
                    "symbol": r[1],
                    "close_price": round(r[2], 2) if r[2] else 0.0,
                    "volume_5m": round(r[3], 2) if r[3] else 0.0,
                    "vwap": round(r[4], 2) if r[4] else 0.0,
                    "net_trade": round(r[5], 2) if r[5] else 0.0,
                    "spread": round(r[6], 4) if r[6] else 0.0,
                    "mid_price": round(r[7], 2) if r[7] else 0.0,
                    "bid_depth": round(r[8], 2) if r[8] else 0.0,
                    "ask_depth": round(r[9], 2) if r[9] else 0.0,
                    "imbalance": round(r[10], 3) if r[10] else 0.0,
                    "sentiment_score": round(r[11], 3) if r[11] else 0.0,
                    "tweet_count": int(r[12]) if r[12] else 0,
                    "net_cex_flow_usd": round(r[13], 2) if r[13] else 0.0,
                    "mse_score": round(r[14], 6) if r[14] else 0.0,
                    "is_anomaly": bool(r[15]),
                    "severity": r[16],
                }
            )
        return result
    except Exception as exc:
        LOGGER.exception("Failed to fetch latest metrics: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database fetch failure: {exc}")


@app.get("/api/health-scores")
def get_health_scores() -> list[dict[str, Any]]:
    """Retrieve recent LLM qualitative ratings and reasoning blocks."""
    try:
        rows = execute_query_fetch(SQL_GET_LATEST_HEALTH_SCORES)
        result = []
        for r in rows:
            result.append(
                {
                    "bucket": r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]),
                    "symbol": r[1],
                    "health_score": int(r[2]) if r[2] is not None else 50,
                    "reasoning": r[3],
                    "explanation": r[4],
                    "model_name": r[5],
                    "latency_ms": int(r[6]) if r[6] else 0,
                    "input_payload": r[7] if isinstance(r[7], list) else [],
                }
            )
        return result
    except Exception as exc:
        LOGGER.exception("Failed to fetch health scores: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database fetch failure: {exc}")


@app.get("/api/anomalies")
def get_active_anomalies() -> list[dict[str, Any]]:
    """List historical anomaly alerts detected by the LSTM Autoencoder."""
    try:
        rows = execute_query_fetch(SQL_GET_ACTIVE_ANOMALIES)
        result = []
        for r in rows:
            result.append(
                {
                    "bucket": r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]),
                    "symbol": r[1],
                    "mse_score": round(r[2], 6) if r[2] else 0.0,
                    "is_anomaly": bool(r[3]),
                    "severity": r[4],
                    "llm_payload": r[5] if isinstance(r[5], dict) else {},
                }
            )
        return result
    except Exception as exc:
        LOGGER.exception("Failed to fetch anomalies: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database fetch failure: {exc}")
