import asyncio
import logging
import json
import os
import time
from pathlib import Path

import torch
import pandas as pd
import numpy as np

from src.core.config import settings
from src.db.db import execute_query_fetch_async, execute_query_async
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.model_registry import ModelRegistry

LOGGER = logging.getLogger("anomaly_pipeline")

TRACKED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT"]

SEQ_LEN = 12
LATENT_DIM = 10
ANOMALY_THRESHOLD = 0.008

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WEIGHTS_DIR = PROJECT_ROOT / "src" / "models" / "saved_weights"

SQL_FETCH_LATEST = """
WITH finalized_buckets AS (
    SELECT bucket, symbol FROM trade_candles_5m
    INTERSECT
    SELECT bucket, symbol FROM orderbook_snapshots_5m
    INTERSECT
    SELECT bucket, symbol || 'USDT' AS symbol FROM tweet_sentiment_5m
)
SELECT 
    fb.bucket AS final_bucket,
    t.open, 
    t.high, 
    t.low, 
    t.close, 
    t.volume, 
    t.trade_count, 
    t.net_trade, 
    t.vwap,
    o.avg_spread, 
    o.avg_mid_price, 
    o.avg_bid_depth, 
    o.avg_ask_depth, 
    o.avg_imbalance,
    s.avg_score, 
    s.tweet_count, 
    s.positive_count, 
    s.negative_count,
    COALESCE(c.net_flow_usd, 0.0) AS net_flow_usd
FROM finalized_buckets fb
JOIN trade_candles_5m t ON t.bucket = fb.bucket AND t.symbol = fb.symbol
JOIN orderbook_snapshots_5m o ON o.bucket = fb.bucket AND o.symbol = fb.symbol
JOIN tweet_sentiment_5m s ON s.bucket = fb.bucket AND s.symbol = REPLACE(fb.symbol, 'USDT', '')
LEFT JOIN (
    SELECT bucket, TRIM(symbol) as symbol, SUM(net_flow_usd) as net_flow_usd 
    FROM cex_flows_5m 
    GROUP BY bucket, symbol
) c ON c.bucket = fb.bucket AND c.symbol = TRIM(REPLACE(fb.symbol, 'USDT', ''))
WHERE fb.symbol = %s
ORDER BY final_bucket DESC
LIMIT %s;
"""

async def fetch_and_scale_latest_window(target_symbol: str, base_symbol: str, scaler_params: dict) -> tuple[torch.Tensor | None, dict | None]:
    rows = await execute_query_fetch_async(SQL_FETCH_LATEST, (target_symbol, SEQ_LEN))

    if not rows or len(rows) < SEQ_LEN:
        LOGGER.warning(f"📭 Not enough history in DB for {target_symbol} yet. Need {SEQ_LEN} rows, got {len(rows) if rows else 0}. Skipping.")
        return None, None

    rows = rows[::-1]
    sql_columns = [
        "bucket", "open", "high", "low", "close", "volume", "trade_count",
        "net_trade", "vwap", "avg_spread", "avg_mid_price", "avg_bid_depth",
        "avg_ask_depth", "avg_imbalance", "avg_score", "tweet_count",
        "positive_count", "negative_count", "net_flow_usd"
    ]
    df = pd.DataFrame(rows, columns=sql_columns)
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True)

    # Safely reorder columns to match scaler JSON feature mapping exactly
    df = df[["bucket"] + scaler_params["features"]]

    time_diffs = df["bucket"].diff().dropna()
    if (time_diffs > pd.Timedelta(minutes=5)).any():
        LOGGER.warning(f"⏳ Data timeline gap detected for {target_symbol} in the last hour. Skipping inference until timeline heals.")
        return None, None

    latest_data_dict = df.iloc[-1].to_dict()

    df_scaled = df.copy()
    for col in scaler_params["features"]:
        min_v = scaler_params["mins"][col]
        max_v = scaler_params["maxs"][col]
        range_v = max_v - min_v if (max_v - min_v) != 0 else 1.0
        df_scaled[col] = (df[col] - min_v) / range_v

    feature_matrix = df_scaled[scaler_params["features"]].to_numpy(dtype=np.float32)
    input_tensor = torch.tensor(feature_matrix).unsqueeze(0)

    return input_tensor, latest_data_dict


def _latest_versioned_artifacts(symbol: str) -> tuple[Path, Path] | None:
    symbol_dir = WEIGHTS_DIR / symbol.upper()
    if not symbol_dir.exists():
        return None

    version_dirs = sorted(
        (path for path in symbol_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for version_dir in version_dirs:
        model_path = version_dir / "model.pt"
        scaler_path = version_dir / "scaler.json"
        if model_path.exists() and scaler_path.exists():
            return model_path, scaler_path

    return None


def _legacy_artifacts(symbol: str) -> tuple[Path, Path]:
    base_sym = symbol.replace("USDT", "").lower()
    return (
        WEIGHTS_DIR / f"lstm_autoencoder_{base_sym}.pt",
        WEIGHTS_DIR / f"scaler_params_{base_sym}.json",
    )


def _resolve_artifact_paths(symbol: str) -> tuple[Path, Path]:
    versioned = _latest_versioned_artifacts(symbol)
    if versioned is not None:
        return versioned
    return _legacy_artifacts(symbol)


def _resolve_model_device() -> torch.device:
    configured = settings.RETRAIN_DEVICE
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"RETRAIN_DEVICE={configured!r} requested but CUDA is unavailable.")
    return torch.device(configured)


def load_model_into_registry(symbol: str, device: torch.device) -> bool:
    model_path, scaler_path = _resolve_artifact_paths(symbol)

    if not model_path.exists() or not scaler_path.exists():
        LOGGER.error(f"Missing weights for {symbol}. Skipping this coin.")
        return False

    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler_params = json.load(f)

    input_dim = len(scaler_params["features"])
    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=LATENT_DIM, seq_len=SEQ_LEN)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    ModelRegistry.register(symbol, model, scaler_params)
    LOGGER.info("Loaded anomaly model for %s from %s", symbol, model_path)
    return True


def run_model_inference(model: LSTMAutoencoder, input_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        reconstructed = model(input_tensor)
        mse = torch.mean((input_tensor - reconstructed) ** 2).item()
    return mse


async def start_anomaly_stream(stop_event: asyncio.Event) -> None:
    LOGGER.info("Starting Multi-Coin AI Anomaly Detection Engine...")
    device = _resolve_model_device()

    loaded_symbols = []
    for symbol in TRACKED_SYMBOLS:
        if load_model_into_registry(symbol, device):
            loaded_symbols.append(symbol)

    if not loaded_symbols:
        LOGGER.error("No models loaded. Shutting down anomaly pipeline.")
        return

    while not stop_event.is_set():
        try:
            # 1. Execute inference across all models in high-speed succession
            for target_symbol in TRACKED_SYMBOLS:
                base_symbol = target_symbol.replace("USDT", "")
                model, scaler_params = ModelRegistry.get(target_symbol)

                if model is None or scaler_params is None:
                    continue

                input_tensor, latest_data = await fetch_and_scale_latest_window(
                    target_symbol, base_symbol, scaler_params
                )

                if input_tensor is None or latest_data is None:
                    continue

                model_device = next(model.parameters()).device
                input_tensor = input_tensor.to(model_device)

                # Offload PyTorch inference computation to a worker thread
                mse = await asyncio.to_thread(run_model_inference, model, input_tensor)

                threshold = scaler_params.get("optimal_threshold", ANOMALY_THRESHOLD)
                is_anomaly = mse > threshold

                if is_anomaly:
                    LOGGER.warning(f"🚨 {target_symbol} ANOMALY! MSE: {mse:.6f} | threshold: {threshold:.6f} 🚨")
                else:
                    LOGGER.info(f"{target_symbol} heartbeat normal. MSE: {mse:.6f} | threshold: {threshold:.6f}")

                llm_payload = {
                    "timestamp": latest_data["bucket"].isoformat(),
                    "symbol": target_symbol,
                    "market_data": {
                        "close_price": round(latest_data["close"], 2),
                        "volume_5m": round(latest_data["volume"], 2),
                        "vwap": round(latest_data["vwap"], 2),
                        "net_trade": round(latest_data["net_trade"], 2)
                    },
                    "orderbook": {
                        "avg_spread": round(latest_data["avg_spread"], 4),
                        "avg_imbalance": round(latest_data["avg_imbalance"], 3),
                        "bid_depth": round(latest_data["avg_bid_depth"], 2),
                        "ask_depth": round(latest_data["avg_ask_depth"], 2)
                    },
                    "sentiment": {
                        "avg_score": round(latest_data["avg_score"], 3),
                        "tweet_count": int(latest_data["tweet_count"]),
                        "positive_count": int(latest_data["positive_count"]),
                        "negative_count": int(latest_data["negative_count"])
                    },
                    "on_chain": {
                        "net_cex_flow_usd": round(latest_data["net_flow_usd"], 2)
                    },
                    "AI_ENGINE": {
                        "reconstruction_error": round(mse, 6),
                        "is_statistical_anomaly": is_anomaly,
                        "severity": "CRITICAL" if mse > (threshold * 2) else "HIGH" if is_anomaly else "NORMAL"
                    }
                }

                insert_sql = """
                             INSERT INTO ai_anomalies_5m
                                 (bucket, symbol, mse_score, is_anomaly, severity, llm_payload)
                             VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (bucket, symbol) DO UPDATE 
                                 SET mse_score = EXCLUDED.mse_score,
                                 is_anomaly = EXCLUDED.is_anomaly,
                                 severity = EXCLUDED.severity,
                                 llm_payload = EXCLUDED.llm_payload;
                             """
                await execute_query_async(insert_sql, (
                    latest_data["bucket"], base_symbol, mse, is_anomaly,
                    llm_payload["AI_ENGINE"]["severity"], json.dumps(llm_payload)
                ))

            # 2. Outdent Alignment Sleep (Runs exactly once AFTER processing all 5 coins)
            current_time = time.time()
            seconds_until_next_bucket = 300 - (current_time % 300)
            
            # Bump padding from + 5 to + 25 to allow slower NLP and CEX poller tasks to settle
            adaptive_sleep_duration = seconds_until_next_bucket + 25

            LOGGER.info(f"⏰ Syncing radar array. Sleeping for {int(adaptive_sleep_duration)}s until next uncorrupted database mark.")
            await asyncio.sleep(adaptive_sleep_duration)

        except asyncio.CancelledError:
            break
        except Exception as e:
            LOGGER.exception(f"Error in anomaly pipeline: {e}")

    LOGGER.info("Anomaly pipeline stopped.")
