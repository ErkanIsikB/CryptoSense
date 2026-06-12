"""CryptoSense FastAPI Backend — Exposes real-time Time-series and LLM metrics from TimescaleDB."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.db.db import execute_query_fetch_async

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
LIMIT 250;
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
async def get_latest_metrics() -> list[dict[str, Any]]:
    """Retrieve the latest finalized bucket metrics across all symbols, with EWMA sentiment."""
    try:
        rows = await execute_query_fetch_async(SQL_GET_LATEST_METRICS)
        if not rows:
            return []

        # Convert to Pandas to compute the EWMA per symbol
        cols = [
            "bucket", "symbol", "close_price", "volume_5m", "vwap", "net_trade",
            "spread", "mid_price", "bid_depth", "ask_depth", "imbalance",
            "sentiment_score", "tweet_count", "net_cex_flow_usd", "mse_score",
            "is_anomaly", "severity"
        ]
        df = pd.DataFrame(rows, columns=cols)
        df["bucket"] = pd.to_datetime(df["bucket"])

        # Sort chronologically per symbol to calculate correct EWMA values
        df = df.sort_values(by=["symbol", "bucket"]).reset_index(drop=True)
        df["sentiment_score"] = df.groupby("symbol")["sentiment_score"].transform(
            lambda x: x.ewm(span=6, adjust=False).mean()
        )

        # Sort back to return latest metrics first
        df = df.sort_values(by=["bucket", "symbol"], ascending=[False, True])
        latest_df = df.head(50)

        result = []
        for _, r in latest_df.iterrows():
            result.append(
                {
                    "bucket": r["bucket"].isoformat() if hasattr(r["bucket"], "isoformat") else str(r["bucket"]),
                    "symbol": r["symbol"],
                    "close_price": round(float(r["close_price"]), 2) if r["close_price"] else 0.0,
                    "volume_5m": round(float(r["volume_5m"]), 2) if r["volume_5m"] else 0.0,
                    "vwap": round(float(r["vwap"]), 2) if r["vwap"] else 0.0,
                    "net_trade": round(float(r["net_trade"]), 2) if r["net_trade"] else 0.0,
                    "spread": round(float(r["spread"]), 4) if r["spread"] else 0.0,
                    "mid_price": round(float(r["mid_price"]), 2) if r["mid_price"] else 0.0,
                    "bid_depth": round(float(r["bid_depth"]), 2) if r["bid_depth"] else 0.0,
                    "ask_depth": round(float(r["ask_depth"]), 2) if r["ask_depth"] else 0.0,
                    "imbalance": round(float(r["imbalance"]), 3) if r["imbalance"] else 0.0,
                    "sentiment_score": round(float(r["sentiment_score"]), 3) if r["sentiment_score"] else 0.0,
                    "tweet_count": int(r["tweet_count"]) if r["tweet_count"] else 0,
                    "net_cex_flow_usd": round(float(r["net_cex_flow_usd"]), 2) if r["net_cex_flow_usd"] else 0.0,
                    "mse_score": round(float(r["mse_score"]), 6) if r["mse_score"] else 0.0,
                    "is_anomaly": bool(r["is_anomaly"]),
                    "severity": r["severity"],
                }
            )
        return result
    except Exception as exc:
        LOGGER.exception("Failed to fetch latest metrics: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database fetch failure: {exc}")


@app.get("/api/health-scores")
async def get_health_scores() -> list[dict[str, Any]]:
    """Retrieve recent LLM qualitative ratings and reasoning blocks."""
    try:
        rows = await execute_query_fetch_async(SQL_GET_LATEST_HEALTH_SCORES)
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
async def get_active_anomalies() -> list[dict[str, Any]]:
    """List historical anomaly alerts detected by the LSTM Autoencoder."""
    try:
        rows = await execute_query_fetch_async(SQL_GET_ACTIVE_ANOMALIES)
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
