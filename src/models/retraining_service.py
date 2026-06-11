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
from src.models.anomaly_pipeline import resolve_model_device, SQL_COLUMNS

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
    COALESCE(c.net_flow_usd, 0.0) AS net_flow_usd,
    COALESCE(ns.avg_score, 0.0) AS news_avg_score
FROM finalized_buckets fb
JOIN trade_candles_5m t ON t.bucket = fb.bucket AND t.symbol = fb.symbol
JOIN orderbook_snapshots_5m o ON o.bucket = fb.bucket AND o.symbol = fb.symbol
JOIN tweet_sentiment_5m s ON s.bucket = fb.bucket AND s.symbol = REPLACE(fb.symbol, 'USDT', '')
LEFT JOIN (
    SELECT bucket, TRIM(symbol) as symbol, SUM(net_flow_usd) as net_flow_usd 
    FROM cex_flows_5m 
    GROUP BY bucket, symbol
) c ON c.bucket = fb.bucket AND c.symbol = TRIM(REPLACE(fb.symbol, 'USDT', ''))
LEFT JOIN news_sentiment_5m ns ON ns.bucket = fb.bucket AND ns.symbol = REPLACE(fb.symbol, 'USDT', '')
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


def generate_proxy_labels(df: pd.DataFrame, z_threshold: float = 3.0) -> np.ndarray:
    """Generate statistical proxy labels (0 or 1) using Z-score outliers on key channels."""
    # Compute 5m return
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
    z_volume = compute_z_scores(volume)  # Fixed bug: use volume instead of z_returns
    z_imbalance = compute_z_scores(imbalance)
    z_flow = compute_z_scores(net_flow)
    
    proxy_labels = ((z_returns > z_threshold) | 
                    (z_volume > z_threshold) | 
                    (z_imbalance > z_threshold) | 
                    (z_flow > z_threshold)).astype(int)
    return proxy_labels


def slice_continuous_windows(
    timestamps: list[Any],
    feature_matrix: np.ndarray,
    seq_len: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Slice feature matrix into continuous sequence windows based on BUCKET_DELTA_MINUTES."""
    sequences: list[np.ndarray] = []
    current_block: list[np.ndarray] = []
    last_timestamp = None

    for i, current_ts in enumerate(timestamps):
        if last_timestamp is None:
            current_block.append(feature_matrix[i])
        else:
            time_delta = current_ts - last_timestamp
            if time_delta <= timedelta(minutes=BUCKET_DELTA_MINUTES):
                current_block.append(feature_matrix[i])
            else:
                if len(current_block) >= seq_len:
                    for start in range(len(current_block) - seq_len + 1):
                        sequences.append(np.array(current_block[start: start + seq_len]))
                current_block = [feature_matrix[i]]
        last_timestamp = current_ts

    if len(current_block) >= seq_len:
        for start in range(len(current_block) - seq_len + 1):
            sequences.append(np.array(current_block[start: start + seq_len]))

    return np.array(sequences, dtype=np.float32)


def align_labels_to_sequences(
    df: pd.DataFrame,
    proxy_labels: np.ndarray,
    seq_len: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Align individual row labels to sequence windows by mapping sequence label to the last row."""
    labels = []
    current_block_labels = []
    last_timestamp = None
    timestamps = df["bucket"].tolist()

    for i, current_ts in enumerate(timestamps):
        lbl = proxy_labels[i]
        if last_timestamp is None:
            current_block_labels.append(lbl)
        else:
            time_delta = current_ts - last_timestamp
            if time_delta <= timedelta(minutes=BUCKET_DELTA_MINUTES):
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

    return np.array(labels, dtype=np.int32)


def compute_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Receiver Operating Characteristic (ROC) curve metrics natively."""
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    
    thresholds = y_scores
    tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
    fpr = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps)
    
    # Prepend 0 to fpr, tpr and append a dummy threshold to match scikit-learn standard
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[thresholds[0] + 1e-5, thresholds]
    
    return fpr, tpr, thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Calculate Area Under the Curve (AUC) using the trapezoidal rule."""
    return float(np.sum((tpr[1:] + tpr[:-1]) * 0.5 * (fpr[1:] - fpr[:-1])))


def run_evaluation_inference(
    model: nn.Module,
    x_eval: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run model inference on evaluation sequences to get reconstructed MSE scores."""
    model.eval()
    mse_scores = []
    X_tensor = torch.tensor(x_eval).to(device)
    
    with torch.no_grad():
        batch_size = 64
        for start_idx in range(0, len(X_tensor), batch_size):
            batch_x = X_tensor[start_idx : start_idx + batch_size]
            reconstructed = model(batch_x)
            batch_mse = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2)).cpu().numpy()
            mse_scores.extend(batch_mse)
            
    return np.array(mse_scores, dtype=np.float32)


def calculate_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, int]:
    """Calculate the optimal threshold using Youden's J statistic, returning threshold, AUC, fpr, tpr, thresholds, and best_idx."""
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_scores)
    auc_score = compute_auc(fpr, tpr)
    
    # Youden's J statistic
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresholds[best_idx])
    
    return optimal_threshold, auc_score, fpr, tpr, thresholds, best_idx


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

    df = pd.DataFrame(raw_rows, columns=SQL_COLUMNS)
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

    feature_matrix = df_scaled[feature_columns].to_numpy()
    timestamps = df_scaled["bucket"].tolist()

    final_sequences = slice_continuous_windows(timestamps, feature_matrix, SEQUENCE_LENGTH)
    LOGGER.info("Extracted %d clean sliding sequence windows of length %d.", len(final_sequences), SEQUENCE_LENGTH)
    return final_sequences, scaler_params


def _calibrate_and_update_scaler_params(
    model: LSTMAutoencoder,
    x_train_np: np.ndarray,
    df: pd.DataFrame,
    scaler_params: dict[str, Any],
    device: torch.device,
) -> None:
    """Run ROC threshold calibration on trained model and append results to scaler_params."""
    try:
        LOGGER.info("Starting ROC threshold calibration on trained model...")
        
        # 1. Generate statistical proxy labels using Z-scores
        proxy_labels = generate_proxy_labels(df, z_threshold=2.5)
                        
        # 2. Extract labels aligned to sequences
        y_eval = align_labels_to_sequences(df, proxy_labels, SEQUENCE_LENGTH)
        
        # Alignment check
        if len(x_train_np) != len(y_eval):
            LOGGER.warning("Mismatch in sequences and labels during calibration. Truncating to match.")
            min_len = min(len(x_train_np), len(y_eval))
            x_train_np = x_train_np[:min_len]
            y_eval = y_eval[:min_len]
            
        if len(x_train_np) == 0:
            LOGGER.warning("No sequences available for ROC calibration.")
            scaler_params["optimal_threshold"] = 0.008
            scaler_params["auc_score"] = 0.5
            return

        # 3. Inference & ROC Threshold Calibration
        y_scores = run_evaluation_inference(model, x_train_np, device)
        optimal_threshold, auc_score, _, _, _, _ = calculate_optimal_threshold(y_eval, y_scores)
        
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
    device = resolve_model_device()
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
