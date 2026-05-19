import asyncio
import logging
import json
import os
from pathlib import Path

import torch
import pandas as pd
import numpy as np

from src.db.db import execute_query_fetch, execute_query
from src.models.lstm_autoencoder import LSTMAutoencoder

LOGGER = logging.getLogger("anomaly_pipeline")

SYMBOL = "BTC"
SEQ_LEN = 12
INPUT_DIM = 19
LATENT_DIM = 10  # Must match the hidden_dim we trained with!
ANOMALY_THRESHOLD = 0.005  # Trigger if error is roughly double our 0.002 training loss

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "saved_weights" / f"lstm_autoencoder_{SYMBOL.lower()}.pt"
SCALER_PATH = PROJECT_ROOT / "src" / "models" / "saved_weights" / f"scaler_params_{SYMBOL.lower()}.json"

# The exact same SQL logic used in training, but fetching only the latest 12 rows
SQL_FETCH_LATEST = """
                   SELECT COALESCE(t.bucket, o.bucket, s.bucket, c.bucket) AS final_bucket, \
                          COALESCE(t.open, 0.0) as open, 
    COALESCE(t.high, 0.0) as high, 
    COALESCE(t.low, 0.0) as low, 
    COALESCE(t.close, 0.0) as close, 
    COALESCE(t.volume, 0.0) as volume, 
    COALESCE(t.trade_count, 0) as trade_count, 
    COALESCE(t.net_trade, 0.0) as net_trade, 
    COALESCE(t.vwap, 0.0) as vwap,
    COALESCE(o.avg_spread, 0.0) as avg_spread, 
    COALESCE(o.avg_mid_price, 0.0) as avg_mid_price, 
    COALESCE(o.avg_bid_depth, 0.0) as avg_bid_depth, 
    COALESCE(o.avg_ask_depth, 0.0) as avg_ask_depth, 
    COALESCE(o.avg_imbalance, 0.0) as avg_imbalance,
    COALESCE(s.avg_score, 0.0) as avg_score, 
    COALESCE(s.tweet_count, 0) as tweet_count, 
    COALESCE(s.positive_count, 0) as positive_count, 
    COALESCE(s.negative_count, 0) as negative_count,
    COALESCE(c.net_flow_usd, 0.0) as net_flow_usd
                   FROM trade_candles_5m t
                       FULL OUTER JOIN orderbook_snapshots_5m o
                   ON t.bucket = o.bucket AND t.symbol = o.symbol
                       FULL OUTER JOIN tweet_sentiment_5m s
                       ON COALESCE (t.bucket, o.bucket) = s.bucket AND COALESCE (t.symbol, o.symbol) = s.symbol
                       FULL OUTER JOIN (
                       SELECT bucket, symbol, SUM (net_flow_usd) as net_flow_usd
                       FROM cex_flows_5m
                       GROUP BY bucket, symbol
                       ) c
                       ON COALESCE (t.bucket, o.bucket, s.bucket) = c.bucket
                       AND COALESCE (t.symbol, o.symbol, s.symbol) = c.symbol
                   WHERE COALESCE (t.symbol, o.symbol, s.symbol, c.symbol) = %s
                   ORDER BY final_bucket DESC
                       LIMIT %s; \
                   """


def fetch_and_scale_latest_window(scaler_params: dict) -> tuple[torch.Tensor | None, dict | None]:
    """Fetches the last 12 buckets and applies the exact training MinMax scaling."""
    rows = execute_query_fetch(SQL_FETCH_LATEST, (SYMBOL, SEQ_LEN))

    if not rows or len(rows) < SEQ_LEN:
        return None, None

    # Reverse rows because we fetched DESC, but LSTM needs chronological ASC
    rows = rows[::-1]

    columns = ["bucket"] + scaler_params["features"]
    df = pd.DataFrame(rows, columns=columns)
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True)

    # Check for downtime gaps in the 1-hour window
    time_diffs = df["bucket"].diff().dropna()
    if (time_diffs > pd.Timedelta(minutes=5)).any():
        LOGGER.warning("Data gap detected in the last hour. Skipping inference until timeline heals.")
        return None, None

    latest_data_dict = df.iloc[-1].to_dict()

    # Apply JSON MinMax Math
    df_scaled = df.copy()
    for col in scaler_params["features"]:
        min_v = scaler_params["mins"][col]
        max_v = scaler_params["maxs"][col]
        range_v = max_v - min_v if (max_v - min_v) != 0 else 1.0
        df_scaled[col] = (df[col] - min_v) / range_v

    feature_matrix = df_scaled[scaler_params["features"]].to_numpy(dtype=np.float32)
    # Shape: [Batch=1, SeqLen=12, Features=19]
    input_tensor = torch.tensor(feature_matrix).unsqueeze(0)

    return input_tensor, latest_data_dict


async def start_anomaly_stream(stop_event: asyncio.Event) -> None:
    LOGGER.info("Starting AI Anomaly Detection Engine...")

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        LOGGER.error("Model weights or scaler missing. Run train_anomaly_detector.py first.")
        return

    # Load JSON Scaler Profile
    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        scaler_params = json.load(f)

    # Load PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(input_dim=INPUT_DIM, hidden_dim=LATENT_DIM, seq_len=SEQ_LEN)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    while not stop_event.is_set():
        try:
            # Wake up every 5 minutes
            await asyncio.sleep(300)

            input_tensor, latest_data = fetch_and_scale_latest_window(scaler_params)

            if input_tensor is None or latest_data is None:
                continue

            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                reconstructed = model(input_tensor)
                mse = torch.mean((input_tensor - reconstructed) ** 2).item()

            is_anomaly = mse > ANOMALY_THRESHOLD

            if is_anomaly:
                LOGGER.warning("🚨 ANOMALY DETECTED! MSE: %.6f 🚨", mse)
            else:
                LOGGER.info("Market heartbeat normal. MSE: %.6f", mse)

            # LLM Decision Engine Payload
            llm_payload = {
                "timestamp": latest_data["bucket"].isoformat(),
                "symbol": SYMBOL,
                "market_data": {
                    "close_price": round(latest_data["close"], 2),
                    "volume_5m": round(latest_data["volume"], 2),
                    "orderbook_imbalance": round(latest_data["avg_imbalance"], 3)
                },
                "sentiment": {
                    "avg_score": round(latest_data["avg_score"], 3),
                    "tweet_count": int(latest_data["tweet_count"])
                },
                "AI_ENGINE": {
                    "reconstruction_error": round(mse, 6),
                    "is_statistical_anomaly": is_anomaly,
                    "severity": "HIGH" if mse > (ANOMALY_THRESHOLD * 2) else "NORMAL"
                }
            }

            print("\n--- LLM DECISION ENGINE PAYLOAD ---")
            print(json.dumps(llm_payload, indent=2))
            print("-----------------------------------\n")

            # --- NEW: Save the AI's prediction to TimescaleDB ---
            insert_sql = """
                         INSERT INTO ai_anomalies_5m
                             (bucket, symbol, mse_score, is_anomaly, severity, llm_payload)
                         VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (bucket, symbol) DO \
                         UPDATE \
                             SET mse_score = EXCLUDED.mse_score, \
                             is_anomaly = EXCLUDED.is_anomaly, \
                             severity = EXCLUDED.severity, \
                             llm_payload = EXCLUDED.llm_payload; \
                         """

            # Use execute_query to write to the DB safely
            execute_query(
                insert_sql,
                (
                    latest_data["bucket"],
                    SYMBOL,
                    mse,
                    is_anomaly,
                    llm_payload["AI_ENGINE"]["severity"],
                    json.dumps(llm_payload)
                )
            )
            LOGGER.debug("AI prediction successfully written to database.")

        except asyncio.CancelledError:
            break
        except Exception as e:
            LOGGER.exception("Error in anomaly pipeline: %s", e)

    LOGGER.info("Anomaly pipeline stopped.")