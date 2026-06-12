"""CryptoSense LLM Decision Pipeline — Real-time Market Health & Trajectory Engine.

Queries TimescaleDB for 12-candle historical sequence data, runs structured Qwen 2.5
inference via Ollama, and persists qualitative metrics & briefings to the database.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Literal

import ollama
from pydantic import BaseModel, Field

from src.db.db import execute_query_async, execute_query_fetch_async

LOGGER = logging.getLogger("llm_pipeline")

TRACKED_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "AVAX"]


# ── 1. Structured Output Schema Definition ──────────────────────────


class CryptoSenseBrief(BaseModel):
    """Pydantic schema to enforce strict, schema-locked token output from Ollama."""

    primary_metric_driver: Literal[
        "volume_spike", "liquidity_flight", "sentiment_shift", "on_chain_whale_flow", "none"
    ]
    market_trajectory_summary: str = Field(
        ...,
        description="A strict, factual 3-sentence quantitative summary explaining the WHY behind the deterministic health score.",
    )
    trustworthiness_classification: Literal[
        "HIGH_CONVICTION", "LOW_TRUST_SPECULATIVE", "LIQUIDITY_EXHAUSTION", "STABLE_BASELINE"
    ]


def calculate_deterministic_health_score(ctx: dict[str, Any]) -> int:
    """Calculate Market Health Score (0-100) deterministically from raw payloads."""
    score = 50.0
    latest = ctx.get("raw_payload", {})
    orderbook = latest.get("orderbook", {})
    sentiment = latest.get("sentiment", {})
    on_chain = latest.get("on_chain", {})
    alert = ctx.get("macro_alert", {})

    # 1. Market Imbalance Impact (-15 to +15) [Weight: 30%]
    imbalance = orderbook.get("avg_imbalance", 0.0)
    score += imbalance * 15.0

    # 2. Sentiment Impact (-20 to +20) [Weight: 40%]
    retail_score = sentiment.get("retail_avg_score", 0.0)
    inst_score = sentiment.get("institutional_avg_score", 0.0)
    
    # 70% Institutional News, 30% Retail Twitter (Anti-Fake News mechanism)
    blended_sentiment = (inst_score * 0.70) + (retail_score * 0.30)
    score += blended_sentiment * 20.0

    # 3. On-chain Impact (-15 to +15) [Weight: 30%]
    # Scale continuously using tanh. Bearish when positive (CEX inflow), bullish when negative.
    flow = on_chain.get("net_cex_flow_usd", 0.0)
    # Scale factor of $2.5M USD represents a significant 5-min flow
    flow_impact = -15.0 * math.tanh(flow / 2_500_000.0)
    score += flow_impact

    # 4. Dynamic Anomaly Penalty / Volatility Boost
    # Scale impact dynamically based on reconstruction error (MSE) relative to the optimal threshold
    if alert.get("is_anomaly"):
        mse = alert.get("mse_score", 0.0)
        threshold = alert.get("threshold", 0.008)
        if threshold <= 0:
            threshold = 0.008
        
        # Calculate dynamic magnitude based on the excess ratio above the threshold
        # This scales smoothly from 0.0 up to a cap of 25.0 (reached when MSE is double the threshold)
        ratio = mse / threshold
        excess_ratio = ratio - 1.0
        magnitude = min(25.0, 25.0 * excess_ratio)
        
        # Extract features for direction scoring
        market_data = latest.get("market_data", {})
        net_trade = market_data.get("net_trade", 0.0)
        close_price = market_data.get("close_price", 0.0)
        vwap = market_data.get("vwap", 0.0)
        
        # Compute ternary direction flags (1.0 = Bullish, -1.0 = Bearish, 0.0 = Neutral)
        f_price = 1.0 if close_price > vwap else -1.0 if close_price < vwap else 0.0
        f_trade = 1.0 if net_trade > 0.0 else -1.0 if net_trade < 0.0 else 0.0
        f_sentiment = 1.0 if blended_sentiment > 0.1 else -1.0 if blended_sentiment < -0.1 else 0.0
        
        # Combine using a weighted direction score (perfectly bounded between -1.0 and +1.0)
        # 40% Price Momentum, 30% Net Trade Momentum, 30% Sentiment Momentum
        direction_score = (f_price * 0.40) + (f_trade * 0.30) + (f_sentiment * 0.30)
        
        if direction_score > 0:
            # Bullish anomaly (volatility pump/surge): rewards health score (scaled by 50% max magnitude)
            score += magnitude * direction_score * 0.5
        else:
            # Bearish anomaly (panic sell/dump): penalizes health score (scales up to 100% max magnitude)
            score += magnitude * direction_score
        
    return int(max(0.0, min(100.0, score)))


# ── 2. Prompts Configuration ─────────────────────────────────────────


SYSTEM_PROMPT = """You are an Expert Portfolio Manager and Macro Analyst for CryptoSense. 
Your single job is to interpret a 12-candle data sequence and a deterministically calculated Health Score (0-100).

CRITICAL INSTRUCTIONS FOR FACTUAL DETERMINISM:
1. You DO NOT calculate the health score. The system provides it. Your job is to EXPLAIN WHY the score is what it is using traditional trading principles (Liquidity, Whale Flows, Sentiment Shifts).
2. Grounding Barrier: You are strictly forbidden from hallucinating data. Only use the metrics provided in the JSON payload.
3. Fallback Grounding: If an incoming feature reads 0.0 (such as net_cex_flow_usd or tweet_count), treat this as an active state of silence or no-activity.
4. Formatting: You must respond exclusively using the structured JSON keys provided."""


def build_user_prompt(symbol: str, ctx: dict[str, Any], health_score: int) -> str:
    """Construct the standardized, factually bounded prompt context for Ollama."""
    alert = ctx["macro_alert"]
    bucket_dt = ctx["latest_bucket"]
    bucket_str = bucket_dt.isoformat() if hasattr(bucket_dt, "isoformat") else str(bucket_dt)

    return f"""======================================================================
GLOBAL MACRO EVALUATION ALERT HEADER
Target Asset: {symbol}
Live Interval Boundary: {bucket_str}
DETERMINISTIC HEALTH SCORE: {health_score}/100
AI Engine Anomaly Status: {alert['is_anomaly']}
Reconstruction Error (MSE): {alert['mse_score']:.6f}
======================================================================

The PyTorch LSTM model evaluated the exact 1-hour chronological history leading up to this interval. 

Analyze the raw feature matrix below to diagnose why the algorithmic health score is {health_score}/100. Apply portfolio management principles to explain this score to a client.

[RAW 1-HOUR DATA TIMELINE (Oldest to Newest)]
{json.dumps(ctx['clean_sequence'], indent=2)}

CORE DIRECTIVES:
1. Formulate your short explanation exactly in a 3-sentence trajectory overview style for an executive project report. Focus entirely on the delta shift of the final 5-minute window compared to the preceding 11 data rows."""


# ── 3. Data Sifting & Failsafe Database Ingestion ───────────────────


async def fetch_12_candle_sequence(
    symbol: str, max_retries: int = 10, retry_delay: float = 2.0
) -> dict[str, Any] | None:
    """Fetch the past 12 consecutive buckets for a symbol, applying the polling failsafe."""
    # Calculate the mathematically expected latest completed 5-minute bucket
    now_epoch = time.time()
    expected_ts = now_epoch - (now_epoch % 300) - 300
    expected_dt = datetime.fromtimestamp(expected_ts, tz=timezone.utc)

    sql = """
        SELECT bucket, symbol, llm_payload 
        FROM ai_anomalies_5m 
        WHERE symbol = %s 
        ORDER BY bucket DESC 
        LIMIT 12;
    """

    for attempt in range(max_retries):
        try:
            rows = await execute_query_fetch_async(sql, (symbol,))
            if not rows or len(rows) < 12:
                # Require full 1-hour context to form sequence
                return None

            latest_db_bucket = rows[0][0]
            if latest_db_bucket.tzinfo is None:
                latest_db_bucket = latest_db_bucket.replace(tzinfo=timezone.utc)

            # Database has caught up to our expected bucket mark
            if latest_db_bucket >= expected_dt:
                break

            LOGGER.warning(
                "⏳ [Attempt %d/%d] Bucket %s not found in DB yet for %s (Latest in DB: %s). Retrying...",
                attempt + 1,
                max_retries,
                expected_dt.strftime("%H:%M:%S"),
                symbol,
                latest_db_bucket.strftime("%H:%M:%S"),
            )
            await asyncio.sleep(retry_delay)
        except Exception as err:
            LOGGER.error("Error during CEX / Anomaly fetch retry cycle: %s", err)
            await asyncio.sleep(retry_delay)
    else:
        # Failsafe threshold exceeded: skip processing to prevent timeline-drift duplicates
        LOGGER.error(
            "❌ [Failsafe] Expected bucket %s did not commit in time for %s. Skipping cycle.",
            expected_dt.strftime("%H:%M:%S"),
            symbol,
        )
        return None

    # Reverse database rows to chronological order (oldest to newest)
    rows = rows[::-1]

    clean_sequence = []
    latest_bucket = None
    macro_alert_header = None
    latest_raw_payload = None

    for i, row in enumerate(rows):
        bucket, db_symbol, payload = row
        if isinstance(payload, str):
            payload = json.loads(payload)

        # Temporal Overlap Fix: Omission Logic for Rows 1-11
        if i < 11:
            if "AI_ENGINE" in payload:
                del payload["AI_ENGINE"]
            clean_sequence.append(payload)

        # Master Alert Isolation for Row 12 (The latest candle)
        else:
            latest_bucket = bucket
            latest_raw_payload = payload.copy()
            ai_engine = payload.pop("AI_ENGINE", {})
            macro_alert_header = {
                "is_anomaly": ai_engine.get("is_statistical_anomaly", False),
                "mse_score": ai_engine.get("reconstruction_error", 0.0),
                "threshold": ai_engine.get("optimal_threshold", 0.008),
                "severity": ai_engine.get("severity", "NORMAL"),
            }
            clean_sequence.append(payload)

    return {
        "latest_bucket": latest_bucket,
        "macro_alert": macro_alert_header,
        "clean_sequence": clean_sequence,
        "raw_payload": latest_raw_payload,
    }


# ── 4. Asynchronous Ollama Inference ───────────────────────────────────


async def run_qwen_inference(symbol: str, ctx: dict[str, Any], health_score: int) -> dict[str, Any]:
    """Execute structured Qwen 2.5 local model inference via Ollama."""
    user_prompt = build_user_prompt(symbol, ctx, health_score)
    start_time = time.time()

    def call_ollama() -> dict[str, Any]:
        return ollama.chat(
            model="qwen2.5:7b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format=CryptoSenseBrief.model_json_schema(),
            options={"temperature": 0.0, "top_p": 0.1},
        )

    try:
        # Offload synchronous Ollama CPU/GPU client connection block to thread worker
        response = await asyncio.to_thread(call_ollama)
        raw_json = response["message"]["content"]
        parsed_result = json.loads(raw_json)
    except Exception as exc:
        LOGGER.exception("Ollama inference or parsing failure: %s", exc)
        parsed_result: dict[str, Any] = {
            "primary_metric_driver": "none",
            "market_trajectory_summary": f"Fallback error: Failed to parse Qwen JSON output. {exc}",
            "trustworthiness_classification": "LOW_TRUST_SPECULATIVE",
        }

    parsed_result["latency_ms"] = int((time.time() - start_time) * 1000)
    return parsed_result


# ── 5. Database Persistence ─────────────────────────────────────────


async def write_health_score(
    bucket: datetime, symbol: str, result: dict[str, Any], clean_sequence: list[dict[str, Any]], deterministic_score: int
) -> None:
    """Save the health score, executive summary, and 12-candle sequence payload."""
    sql = """
        INSERT INTO llm_health_scores
            (bucket, symbol, health_score, reasoning, explanation,
             model_name, latency_ms, input_payload)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (bucket, symbol) DO UPDATE SET
            health_score  = EXCLUDED.health_score,
            reasoning     = EXCLUDED.reasoning,
            explanation   = EXCLUDED.explanation,
            model_name    = EXCLUDED.model_name,
            latency_ms    = EXCLUDED.latency_ms,
            input_payload = EXCLUDED.input_payload;
    """
    reasoning_str = f"Driver: {result.get('primary_metric_driver')} | Trust: {result.get('trustworthiness_classification')}"

    try:
        await execute_query_async(
            sql,
            (
                bucket,
                symbol,
                deterministic_score,
                reasoning_str,
                result.get("market_trajectory_summary", ""),
                "qwen2.5:7b",
                result.get("latency_ms", 0),
                json.dumps(clean_sequence),
            ),
        )
    except Exception:
        LOGGER.exception("Failed to write llm_health_score to DB")


# ── 6. Asynchronous Unified Execution Loop ──────────────────────────


async def start_llm_decision_stream(stop_event: asyncio.Event) -> None:
    """Public background task — runs synchronized LLM briefings every 5 minutes."""
    LOGGER.info("CryptoSense LLM Decision Stream started (aligned to 5-minute clock boundary)")

    while not stop_event.is_set():
        # 1. Align loop execution to exactly 35 seconds past the 5-minute mark
        current_time = time.time()
        seconds_until_next_bucket = 300 - (current_time % 300)
        adaptive_sleep_duration = seconds_until_next_bucket + 35

        LOGGER.info(
            "⏰ LLM Engine syncing. Sleeping for %ds until next uncorrupted bucket mark.",
            int(adaptive_sleep_duration),
        )

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=adaptive_sleep_duration)
        except asyncio.TimeoutError:
            pass  # Sleep elapsed naturally — run inference

        if stop_event.is_set():
            break

        # 2. Iterate through symbols and execute briefings
        LOGGER.info("⏰ Waking up LLM Engine. Initiating chronological briefings...")
        try:
            for symbol in TRACKED_SYMBOLS:
                ctx = await fetch_12_candle_sequence(symbol)
                if ctx is None:
                    continue

                deterministic_score = calculate_deterministic_health_score(ctx)

                LOGGER.info("🧠 Running structured Qwen 2.5 inference for %s...", symbol)
                result = await run_qwen_inference(symbol, ctx, deterministic_score)

                LOGGER.info(
                    "📊 %s brief completed in %dms. Det. Score: %d/100 | Driver: %s",
                    symbol,
                    result.get("latency_ms", 0),
                    deterministic_score,
                    result.get("primary_metric_driver", "none"),
                )

                # Persist score and clean sequence block
                await write_health_score(ctx["latest_bucket"], symbol, result, ctx["clean_sequence"], deterministic_score)

        except Exception as exc:
            LOGGER.exception("Error in LLM decision cycle: %s", exc)
