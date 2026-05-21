"""CryptoSense Model Training Script.

Surgically filters clean historical data, reconstructs continuous running blocks
while honoring downtime gaps, normalizes features via MinMax mapping, and trains
an unsupervised LSTM Autoencoder to learn baseline market dynamics.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Final

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.db.db import execute_query_fetch
from src.models.lstm_autoencoder import LSTMAutoencoder

# --- System & Logging Configurations ---
import sys

# ANSI color code for Purple/Magenta
PURPLE = "\033[95m"
RESET = "\033[0m"

# Custom formatter to inject color
class PurpleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return f"{PURPLE}{super().format(record)}{RESET}"

# Configure logging to use stdout instead of stderr
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

# Remove default handlers if they exist
if logger.hasHandlers():
    logger.handlers.clear()

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(PurpleFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
logger.addHandler(handler)

# Keep the module-level LOGGER variable
LOGGER = logger

# --- Constants & Tuning Hyperparameters ---
TARGET_SYMBOL = "AVAXUSDT"
BASE_SYMBOL = TARGET_SYMBOL.replace("USDT", "")
SEQUENCE_LENGTH: Final[int] = 12  # 12 rows = Exactly 60 minutes
BUCKET_DELTA_MINUTES: Final[int] = 5
EPOCHS: Final[int] = 100
BATCH_SIZE: Final[int] = 32
LEARNING_RATE: Final[float] = 0.001
LATENT_DIM: Final[int] = 10

# Dynamically calculate the absolute path to the main project folder
PROJECT_ROOT: Final[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model & Parameter Save Coordinates
MODEL_DIR: Final[str] = os.path.join(PROJECT_ROOT, "src", "models", "saved_weights")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH: Final[str] = os.path.join(MODEL_DIR, f"lstm_autoencoder_{BASE_SYMBOL.lower()}.pt")
SCALER_SAVE_PATH: Final[str] = os.path.join(MODEL_DIR, f"scaler_params_{BASE_SYMBOL.lower()}.json")

# --- Structured SQL Query Formulation ---
SQL_DATA_EXTRACT: Final[str] = """
SELECT 
    COALESCE(t.bucket, o.bucket, s.bucket, c.bucket) AS final_bucket,
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
    ON COALESCE(t.bucket, o.bucket) = s.bucket 
    AND REPLACE(COALESCE(t.symbol, o.symbol), 'USDT', '') = s.symbol
FULL OUTER JOIN (
    SELECT bucket, symbol, SUM(net_flow_usd) as net_flow_usd 
    FROM cex_flows_5m 
    GROUP BY bucket, symbol
) c 
    ON COALESCE(t.bucket, o.bucket, s.bucket) = c.bucket 
    AND REPLACE(COALESCE(t.symbol, o.symbol), 'USDT', '') = c.symbol
WHERE COALESCE(t.symbol, o.symbol) = %s 
   OR s.symbol = %s 
   OR c.symbol = %s
ORDER BY final_bucket ASC;
"""

# ── Core Engineering Implementation Logic ───────────────────────

def fetch_and_clean_dataframe() -> pd.DataFrame:
    """Query TimescaleDB, apply UTC localization, and handle column formats."""
    LOGGER.info("Querying TimescaleDB for clean %s dataset...", TARGET_SYMBOL)
    raw_rows = execute_query_fetch(SQL_DATA_EXTRACT, (TARGET_SYMBOL, BASE_SYMBOL, BASE_SYMBOL))
    print(f"SQL gave us these {len(raw_rows[0])} items:", raw_rows[0])
    if not raw_rows:
        raise ValueError(f"Zero clean training samples found for {TARGET_SYMBOL} after the cutoff timestamp.")

    columns = [
        "bucket", "open", "high", "low", "close", "volume", "trade_count", "net_trade", "vwap",
        "avg_spread", "avg_mid_price", "avg_bid_depth", "avg_ask_depth", "avg_imbalance",
        "avg_score", "tweet_count", "positive_count", "negative_count", "net_flow_usd"
    ]

    df = pd.DataFrame(raw_rows, columns=columns)
    df["bucket"] = pd.to_datetime(df["bucket"], utc=True)
    df = df.sort_values("bucket").reset_index(drop=True)
    LOGGER.info("Extracted %d valid running buckets from DB.", len(df))
    return df


def extract_continuous_sequences(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment sequences into uninterrupted 1-hour windows based on temporal proximity."""
    # Isolate feature arrays away from structural timestamps
    feature_columns = [col for col in df.columns if col != "bucket"]

    # Custom Vectorized MinMax Scaler implementation matching Pandas 3.0 behaviors
    min_vals = df[feature_columns].min()
    max_vals = df[feature_columns].max()

    # Safely handle dead/unchanged column channels to avoid dividing by zero
    range_vals = (max_vals - min_vals).replace(0.0, 1.0)

    # Apply standard normalization matrix mapping
    df_scaled = df.copy()
    df_scaled[feature_columns] = (df[feature_columns] - min_vals) / range_vals

    # Package parameters for Live Anomaly Ingestion Scripts
    scaler_params = {
        "features": feature_columns,
        "mins": min_vals.to_dict(),
        "maxs": max_vals.to_dict()
    }

    sequences: list[np.ndarray] = []
    current_block: list[np.ndarray] = []
    last_timestamp: datetime | None = None

    feature_matrix = df_scaled[feature_columns].to_numpy()
    timestamps = df_scaled["bucket"].tolist()

    # Continuous Running Evaluation Engine
    for i, current_ts in enumerate(timestamps):
        if last_timestamp is None:
            current_block.append(feature_matrix[i])
        else:
            time_delta = current_ts - last_timestamp
            # If step gap matches exact interval, chain remains continuous
            if time_delta <= timedelta(minutes=BUCKET_DELTA_MINUTES):
                current_block.append(feature_matrix[i])
            else:
                # Downtime detected. Slice completed blocks into 1-hour windows
                if len(current_block) >= SEQUENCE_LENGTH:
                    for start in range(len(current_block) - SEQUENCE_LENGTH + 1):
                        sequences.append(np.array(current_block[start: start + SEQUENCE_LENGTH]))
                current_block = [feature_matrix[i]]

        last_timestamp = current_ts

    # Flush trailing active block
    if len(current_block) >= SEQUENCE_LENGTH:
        for start in range(len(current_block) - SEQUENCE_LENGTH + 1):
            sequences.append(np.array(current_block[start: start + SEQUENCE_LENGTH]))

    final_sequences = np.array(sequences, dtype=np.float32)
    LOGGER.info("Extracted %d clean sliding sequence windows of length %d.", len(final_sequences), SEQUENCE_LENGTH)
    return final_sequences, scaler_params


def main() -> None:
    # 1. Pipeline Extraction & In-Memory Verification
    df = fetch_and_clean_dataframe()
    X_train_np, scaler_params = extract_continuous_sequences(df)

    if len(X_train_np) == 0:
        LOGGER.error("Insufficient continuous timeline blocks to build a sequence. Aborting training.")
        return

    # Save Scaler Metadata configs for exact live inference scaling matches
    with open(SCALER_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=4)
    LOGGER.info("Scaler normalization parameters exported to %s", SCALER_SAVE_PATH)

    # 2. PyTorch Setup & Dataset Construction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Executing training loop on: %s", device)

    X_tensor = torch.tensor(X_train_np)
    dataset = TensorDataset(X_tensor, X_tensor)  # Unsupervised Target matches Input exactly
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X_train_np.shape[2]
    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=LATENT_DIM, seq_len=SEQUENCE_LENGTH).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Model Optimization Training Loop
    model.train()
    LOGGER.info("Beginning LSTM Autoencoder optimization sequence...")

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        avg_loss = epoch_loss / len(dataset)
        if epoch % 5 == 0 or epoch == 1:
            LOGGER.info("Epoch [%d/%d] | Reconstruction Loss: %.6f", epoch, EPOCHS, avg_loss)

    # 4. Save Final Serialized Weights
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    LOGGER.info("LSTM Model successfully exported to %s", MODEL_SAVE_PATH)
    LOGGER.info("Training cycle completed safely.")


if __name__ == "__main__":
    # Execution entry via Python module namespace: python -m scripts.train_anomaly_detector
    main()