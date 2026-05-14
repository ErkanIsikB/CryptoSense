import asyncio
import logging
import json
import pickle
from pathlib import Path

import torch
import pandas as pd

from src.db.db import execute_query_fetch
from src.models.lstm_autoencoder import LSTMAutoencoder

LOGGER = logging.getLogger("anomaly_pipeline")

SYMBOL = "BTCUSDT"
SEQ_LEN = 12  # 60 mins of context at 5-min buckets


def fetch_latest_window() -> pd.DataFrame | None:
    """Fetches exactly the last SEQ_LEN buckets from the DB."""
    sql = """
          SELECT t.bucket, \
                 t.close, \
                 t.volume, \
                 t.vwap, \
                 o.avg_spread, \
                 o.avg_imbalance, \
                 COALESCE(s.avg_score, 0)    as avg_score, \
                 COALESCE(s.tweet_count, 0)  as tweet_count, \
                 COALESCE(c.net_flow_usd, 0) as net_flow_usd
          FROM trade_candles_5m t
                   LEFT JOIN orderbook_snapshots_5m o ON t.bucket = o.bucket AND t.symbol = o.symbol
                   LEFT JOIN tweet_sentiment_5m s ON t.bucket = s.bucket AND REPLACE(t.symbol, 'USDT', '') = s.symbol
                   LEFT JOIN (SELECT bucket, symbol, SUM(net_flow_usd) as net_flow_usd \
                              FROM cex_flows_5m \
                              GROUP BY bucket, symbol) c \
                             ON t.bucket = c.bucket AND REPLACE(t.symbol, 'USDT', '') = c.symbol
          WHERE t.symbol = %s
          ORDER BY t.bucket DESC
              LIMIT %s \
          """
    # Fetch extra row to calculate pct_change properly
    rows = execute_query_fetch(sql, (SYMBOL, SEQ_LEN + 1))

    if len(rows) < SEQ_LEN + 1:
        return None

    columns = [
        'bucket', 'close', 'volume', 'vwap',
        'avg_spread', 'avg_imbalance', 'avg_score',
        'tweet_count', 'net_flow_usd'
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.sort_values('bucket', ascending=True, inplace=True)

    df['price_change_pct'] = df['close'].pct_change()
    df['vwap_dev'] = (df['close'] - df['vwap']) / df['vwap']
    df.dropna(inplace=True)

    return df


async def start_anomaly_stream(stop_event: asyncio.Event) -> None:
    LOGGER.info("Starting AI Anomaly Detection Engine...")

    scaler_path = Path("scripts/data/anomalies/anomaly_scaler.pkl")
    model_path = Path("scripts/data/anomalies/anomaly_model.pth")

    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        model = LSTMAutoencoder(num_features=8)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    except FileNotFoundError:
        LOGGER.error("Anomaly models missing. Run train_anomaly_detector.py first.")
        return

    ANOMALY_THRESHOLD = 0.5  # Adjust based on your training loss distribution

    while not stop_event.is_set():
        try:
            # Wake up every 5 minutes (300 seconds)
            await asyncio.sleep(300)

            df = fetch_latest_window()
            if df is None:
                continue

            features = df[[
                'price_change_pct', 'volume', 'vwap_dev',
                'avg_spread', 'avg_imbalance',
                'avg_score', 'tweet_count', 'net_flow_usd'
            ]].values

            scaled_data = scaler.transform(features)
            input_tensor = torch.from_numpy(scaled_data).float().unsqueeze(0)

            with torch.no_grad():
                reconstructed = model(input_tensor)
                mse = torch.mean((input_tensor - reconstructed) ** 2).item()

            is_anomaly = mse > ANOMALY_THRESHOLD

            # --- Format for the LLM Decision Engine ---
            latest_row = df.iloc[-1]

            llm_payload = {
                "timestamp": latest_row['bucket'].isoformat(),
                "symbol": SYMBOL,
                "market_data": {
                    "price_change_5m_pct": round(latest_row['price_change_pct'] * 100, 3),
                    "volume_5m": round(latest_row['volume'], 2),
                    "orderbook_imbalance": round(latest_row['avg_imbalance'], 3)
                },
                "sentiment": {
                    "avg_score": round(latest_row['avg_score'], 3),
                    "tweet_count": int(latest_row['tweet_count'])
                },
                "on_chain": {
                    "net_cex_flow_usd": round(latest_row['net_flow_usd'], 2)
                },
                "AI_ANOMALY_ENGINE": {
                    "reconstruction_error": round(mse, 4),
                    "is_statistical_anomaly": is_anomaly,
                    "severity": "HIGH" if mse > (ANOMALY_THRESHOLD * 2) else "NORMAL"
                }
            }

            if is_anomaly:
                LOGGER.warning(f"ANOMALY DETECTED! MSE: {mse:.4f}")

            # For now, print the JSON for the LLM.
            # In Phase 3, you will pipe this dict directly into your Llama 3.1 prompt.
            print("\n--- LLM DECISION ENGINE PAYLOAD ---")
            print(json.dumps(llm_payload, indent=2))
            print("-----------------------------------\n")

        except asyncio.CancelledError:
            break
        except Exception as e:
            LOGGER.exception(f"Error in anomaly pipeline: {e}")

    LOGGER.info("Anomaly pipeline stopped.")