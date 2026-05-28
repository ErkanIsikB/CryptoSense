"""Reusable anomaly detector training service.

This module intentionally mirrors the original ``scripts/train_anomaly_detector.py``
training path so manual training and scheduled retraining use the same logic.
"""

from __future__ import annotations

import json
import logging
import gc
import shutil
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.core.config import settings
from src.db.db import execute_query_fetch
from src.models.lstm_autoencoder import LSTMAutoencoder
from src.models.model_registry import ModelRegistry

LOGGER = logging.getLogger("model_training")

# --- Constants & Tuning Hyperparameters ---
TARGET_SYMBOL = "AVAXUSDT"
SEQUENCE_LENGTH: Final[int] = 12  # 12 rows = Exactly 60 minutes
BUCKET_DELTA_MINUTES: Final[int] = 5
EPOCHS: Final[int] = 100
BATCH_SIZE: Final[int] = 32
LEARNING_RATE: Final[float] = 0.001
LATENT_DIM: Final[int] = 10

MODEL_DIR: Final[Path] = settings.PROJECT_ROOT / "src" / "models" / "saved_weights"

# --- Structured SQL Query Formulation ---
SQL_DATA_EXTRACT: Final[str] = """
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
  AND fb.bucket >= NOW() - %s::interval
ORDER BY final_bucket ASC;
"""


@dataclass(frozen=True)
class TrainingArtifacts:
    symbol: str
    version_dir: Path
    model_path: Path
    scaler_path: Path


def _base_symbol(target_symbol: str) -> str:
    return target_symbol.replace("USDT", "")


def _artifact_dir(output_root: Path, target_symbol: str, artifact_date: date | None) -> Path:
    version_date = artifact_date or datetime.now().date()
    return output_root / target_symbol.upper() / version_date.isoformat()


def _resolve_training_device() -> torch.device:
    configured = settings.RETRAIN_DEVICE
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"RETRAIN_DEVICE={configured!r} requested but CUDA is unavailable.")
    return torch.device(configured)


def _write_artifacts_atomically(
    *,
    output_root: Path,
    version_dir: Path,
    model: LSTMAutoencoder,
    scaler_params: dict[str, Any],
) -> None:
    tmp_dir = output_root / f".{version_dir.parent.name}-{version_dir.name}-{uuid.uuid4().hex}.tmp"
    backup_dir = output_root / f".{version_dir.parent.name}-{version_dir.name}-{uuid.uuid4().hex}.bak"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=False)
        tmp_model_path = tmp_dir / "model.pt"
        tmp_scaler_path = tmp_dir / "scaler.json"

        with open(tmp_scaler_path, "w", encoding="utf-8") as f:
            json.dump(scaler_params, f, indent=4)
        torch.save(model.state_dict(), tmp_model_path)

        version_dir.parent.mkdir(parents=True, exist_ok=True)
        if version_dir.exists():
            version_dir.replace(backup_dir)
        tmp_dir.replace(version_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if backup_dir.exists() and not version_dir.exists():
            backup_dir.replace(version_dir)
        raise
    else:
        shutil.rmtree(backup_dir, ignore_errors=True)


def fetch_and_clean_dataframe(
    target_symbol: str = TARGET_SYMBOL,
    lookback_days: int = settings.RETRAIN_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Query TimescaleDB, apply UTC localization, and handle column formats."""
    lookback_interval = f"{lookback_days} days"
    LOGGER.info("Querying TimescaleDB for clean %s dataset over %s...", target_symbol, lookback_interval)
    raw_rows = execute_query_fetch(SQL_DATA_EXTRACT, (target_symbol, lookback_interval))
    if not raw_rows:
        raise ValueError(f"Zero clean training samples found for {target_symbol} in the last {lookback_days} days.")

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


def _calibrate_and_update_scaler_params(
    model: LSTMAutoencoder,
    X_train_np: np.ndarray,
    df: pd.DataFrame,
    scaler_params: dict[str, Any],
    device: torch.device,
) -> None:
    """Run ROC threshold calibration on trained model and append results to scaler_params."""
    try:
        LOGGER.info("Starting ROC threshold calibration on trained model...")
        
        # 1. Generate statistical proxy labels using Z-scores
        close = df["close"].to_numpy()
        returns = np.zeros_like(close)
        returns[1:] = np.log(close[1:] / close[:-1])
        
        volume = df["volume"].to_numpy()
        imbalance = df["avg_imbalance"].to_numpy()
        net_flow = df["net_flow_usd"].to_numpy()
        
        def compute_z_scores(arr: np.ndarray) -> np.ndarray:
            mean = np.mean(arr)
            std = np.std(arr)
            std = std if std != 0 else 1.0
            return np.abs((arr - mean) / std)
            
        z_returns = compute_z_scores(returns)
        z_volume = compute_z_scores(volume)
        z_imbalance = compute_z_scores(imbalance)
        z_flow = compute_z_scores(net_flow)
        
        proxy_labels = ((z_returns > 2.5) | 
                        (z_volume > 2.5) | 
                        (z_imbalance > 2.5) | 
                        (z_flow > 2.5)).astype(int)
                        
        # 2. Extract labels aligned to sequences
        seq_len = SEQUENCE_LENGTH
        labels = []
        timestamps = df["bucket"].tolist()
        current_block_labels = []
        last_timestamp = None
        
        for i, current_ts in enumerate(timestamps):
            lbl = proxy_labels[i]
            if last_timestamp is None:
                current_block_labels.append(lbl)
            else:
                time_delta = current_ts - last_timestamp
                if time_delta <= timedelta(minutes=5):  # BUCKET_DELTA_MINUTES
                    current_block_labels.append(lbl)
                else:
                    if len(current_block_labels) >= seq_len:
                        for start in range(len(current_block_labels) - seq_len + 1):
                            labels.append(current_block_labels[start + seq_len - 1])
                    current_block_labels = [lbl]
            last_timestamp = current_ts
            
        if len(current_block_labels) >= seq_len:
            for start in range(len(current_block_labels) - seq_len + 1):
                labels.append(current_block_labels[start + seq_len - 1])
                
        y_eval = np.array(labels, dtype=np.int32)
        
        # Alignment check
        if len(X_train_np) != len(y_eval):
            LOGGER.warning("Mismatch in sequences and labels during calibration. Truncating to match.")
            min_len = min(len(X_train_np), len(y_eval))
            X_train_np = X_train_np[:min_len]
            y_eval = y_eval[:min_len]
            
        if len(X_train_np) == 0:
            LOGGER.warning("No sequences available for ROC calibration.")
            scaler_params["optimal_threshold"] = 0.008
            scaler_params["auc_score"] = 0.5
            return

        # 3. Inference to get MSE scores
        model.eval()
        mse_scores = []
        X_tensor = torch.tensor(X_train_np).to(device)
        
        with torch.no_grad():
            batch_size = 64
            for start_idx in range(0, len(X_tensor), batch_size):
                batch_x = X_tensor[start_idx : start_idx + batch_size]
                reconstructed = model(batch_x)
                batch_mse = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2)).cpu().numpy()
                mse_scores.extend(batch_mse)
                
        y_scores = np.array(mse_scores, dtype=np.float32)
        
        # 4. ROC calculations
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_scores = y_scores[desc_score_indices]
        y_eval = y_eval[desc_score_indices]
        
        tps = np.cumsum(y_eval)
        fps = np.cumsum(1 - y_eval)
        
        thresholds = y_scores
        tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
        fpr = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps)
        
        tpr = np.r_[0.0, tpr]
        fpr = np.r_[0.0, fpr]
        thresholds = np.r_[thresholds[0] + 1e-5, thresholds]
        
        # Native trapezoidal implementation to avoid numpy 2.0+ deprecation/removal of np.trapz
        auc_score = float(np.sum((tpr[1:] + tpr[:-1]) * 0.5 * (fpr[1:] - fpr[:-1])))
        
        # Youden's J statistic
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        optimal_threshold = float(thresholds[best_idx])
        
        LOGGER.info(f"ROC calibration completed. Calculated AUC: {auc_score:.5f}, Dynamic Threshold: {optimal_threshold:.6f}")
        scaler_params["optimal_threshold"] = optimal_threshold
        scaler_params["auc_score"] = auc_score
        
    except Exception as e:
        LOGGER.exception(f"Failed to run ROC threshold calibration: {e}")
        # Default safe fallbacks
        scaler_params["optimal_threshold"] = 0.008
        scaler_params["auc_score"] = 0.5


def train_symbol_model(
    target_symbol: str = TARGET_SYMBOL,
    *,
    output_root: Path | str = MODEL_DIR,
    artifact_date: date | None = None,
    hot_swap: bool = True,
    lookback_days: int = settings.RETRAIN_LOOKBACK_DAYS,
) -> TrainingArtifacts | None:
    # 1. Pipeline Extraction & In-Memory Verification
    output_root = Path(output_root)
    version_dir = _artifact_dir(output_root, target_symbol, artifact_date)
    model_save_path = version_dir / "model.pt"
    scaler_save_path = version_dir / "scaler.json"

    df = fetch_and_clean_dataframe(target_symbol, lookback_days)
    X_train_np, scaler_params = extract_continuous_sequences(df)

    if len(X_train_np) == 0:
        LOGGER.error("Insufficient continuous timeline blocks to build a sequence. Aborting training.")
        return None

    # 2. PyTorch Setup & Dataset Construction
    device = _resolve_training_device()
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
    _calibrate_and_update_scaler_params(model, X_train_np, df, scaler_params, device)

    try:
        _write_artifacts_atomically(
            output_root=output_root,
            version_dir=version_dir,
            model=model,
            scaler_params=scaler_params,
        )
    except Exception:
        LOGGER.exception("Failed to write model artifacts atomically for %s", target_symbol)
        raise

    LOGGER.info("Scaler normalization parameters exported to %s", scaler_save_path)
    LOGGER.info("LSTM Model successfully exported to %s", model_save_path)

    model.eval()
    if hot_swap:
        model.to(device)
        ModelRegistry.hot_swap(target_symbol, model, scaler_params)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOGGER.info("Hot-swapped live model registry for %s", target_symbol)

    LOGGER.info("Training cycle completed safely.")
    return TrainingArtifacts(
        symbol=target_symbol,
        version_dir=version_dir,
        model_path=model_save_path,
        scaler_path=scaler_save_path,
    )
