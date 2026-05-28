"""CryptoSense LLM Decision Pipeline — Real-time Market Health & Trajectory Engine.

Queries TimescaleDB for 12-candle historical sequence data, runs structured Qwen 2.5
inference via Ollama, and persists qualitative metrics & briefings to the database.
"""

from __future__ import annotations

import asyncio
import json
import logging
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

    market_health_score: int = Field(
        ...,
        description="An integer from 0 to 100 representing structural stability (0=Total Liquidity/Sentiment Collapse, 100=Perfect Symmetrical Health).",
    )
    primary_metric_driver: Literal[
        "volume_spike", "liquidity_flight", "sentiment_shift", "on_chain_whale_flow", "none"
    ]
    market_trajectory_summary: str = Field(
        ...,
        description="A strict, factual 3-sentence quantitative summary explaining how the metric vectors shifted or escalated over the last hour.",
    )
    trustworthiness_classification: Literal[
        "HIGH_CONVICTION", "LOW_TRUST_SPECULATIVE", "LIQUIDITY_EXHAUSTION", "STABLE_BASELINE"
    ]


# ── 2. Prompts Configuration ─────────────────────────────────────────


SYSTEM_PROMPT = """You are the automated macro analytical sub-module for the CryptoSense pipeline. Your single job is to interpret an incoming time-series matrix containing exactly 12 chronological slices of 5-minute payload blocks.

CRITICAL INSTRUCTIONS FOR FACTUAL DETERMINISM:
1. Grounding Barrier: You are strictly forbidden from extrapolating, predicting, or projecting future asset performance. Only describe what has factually occurred within the 1-hour boundaries provided.
2. Value Locking: Never cite a metric, volume figure, or price point that is not explicitly written inside the data context string.
3. Fallback Grounding: If an incoming feature reads 0.0 (such as net_cex_flow_usd or tweet_count), treat this as an active state of silence or no-activity. Do not assume or guess why it is zero; simply state that the metric was inactive for that bucket block.
4. Formatting: You must respond exclusively using the structured JSON keys provided. Do not append introduction remarks, conversational greetings, or concluding pleasantries."""


def build_user_prompt(symbol: str, ctx: dict[str, Any]) -> str:
    """Construct the standardized, factually bounded prompt context for Ollama."""
    alert = ctx["macro_alert"]
    bucket_dt = ctx["latest_bucket"]
    bucket_str = bucket_dt.isoformat() if hasattr(bucket_dt, "isoformat") else str(bucket_dt)

    return f"""======================================================================
GLOBAL MACRO EVALUATION ALERT HEADER (LATEST TIMESTAMP ONLY)
Target Asset: {symbol}
Live Interval Boundary: {bucket_str}
AI Engine Anomaly Status: {alert['is_anomaly']}
Reconstruction Error (MSE): {alert['mse_score']:.6f} (System Threshold: 0.008)
======================================================================

The PyTorch LSTM model evaluated the exact 1-hour chronological history leading up to this interval and triggered the master engine metrics shown above. 

Analyze the raw feature matrix below to diagnose what structural shift caused the machine learning model to register this specific reconstruction error footprint. Notice that historical anomaly tracking has been stripped to preserve timeline purity:

[RAW 1-HOUR DATA TIMELINE (Oldest to Newest)]
{json.dumps(ctx['clean_sequence'], indent=2)}

CORE DIRECTIVES:
1. Calculate a Market Health Score from 0 to 100. Deduct points heavily if orderbook ask/bid depths hollow out below baseline averages while price experiences sudden volatility, or if social volume indicators drop off during market movements.
2. Formulate your short explanation exactly in a 3-sentence trajectory overview style for an executive project report. Focus entirely on the delta shift of the final 5-minute window compared to the preceding 11 data rows."""


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


async def run_qwen_inference(symbol: str, ctx: dict[str, Any]) -> dict[str, Any]:
    """Execute structured Qwen 2.5 local model inference via Ollama."""
    user_prompt = build_user_prompt(symbol, ctx)
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
        parsed_result = {
            "market_health_score": 50,
            "primary_metric_driver": "none",
            "market_trajectory_summary": f"Fallback error: Failed to parse Qwen JSON output. {exc}",
            "trustworthiness_classification": "LOW_TRUST_SPECULATIVE",
        }

    parsed_result["latency_ms"] = int((time.time() - start_time) * 1000)
    return parsed_result


# ── 5. Database Persistence ─────────────────────────────────────────


async def write_health_score(
    bucket: datetime, symbol: str, result: dict[str, Any], clean_sequence: list[dict[str, Any]]
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
                max(0, min(100, result.get("market_health_score", 50))),
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

                LOGGER.info("🧠 Running structured Qwen 2.5 inference for %s...", symbol)
                result = await run_qwen_inference(symbol, ctx)

                LOGGER.info(
                    "📊 %s brief completed in %dms. Score: %d/100 | Driver: %s",
                    symbol,
                    result.get("latency_ms", 0),
                    result.get("market_health_score", 50),
                    result.get("primary_metric_driver", "none"),
                )

                # Persist score and clean sequence block
                await write_health_score(ctx["latest_bucket"], symbol, result, ctx["clean_sequence"])

        except Exception as exc:
            LOGGER.exception("Error in LLM decision cycle: %s", exc)
