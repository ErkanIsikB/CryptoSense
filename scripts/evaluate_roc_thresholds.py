"""CryptoSense Offline ROC & AUC Threshold Calibrator.

This script fetches historical feature data from TimescaleDB, applies statistical
Z-score proxy anomaly labeling to historical rows, runs LSTM Autoencoder inference
to calculate reconstruction MSEs, and performs a native ROC/AUC analysis to find the
mathematically optimal anomaly threshold using Youden's J statistic.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Configure path alignment
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.config import settings
from src.models.retraining_service import fetch_and_clean_dataframe, extract_continuous_sequences, TARGET_SYMBOL
from src.models.anomaly_pipeline import _resolve_artifact_paths, _resolve_model_device
from src.models.lstm_autoencoder import LSTMAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
LOGGER = logging.getLogger("roc_evaluator")


def generate_proxy_labels(df: pd.DataFrame, z_threshold: float = 2.5) -> np.ndarray:
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
    z_volume = compute_z_scores(z_returns)  # Z-score of volume
    z_imbalance = compute_z_scores(imbalance)
    z_flow = compute_z_scores(net_flow)
    
    # Label is 1 if any Z-score exceeds the threshold
    proxy_labels = ((z_returns > z_threshold) | 
                    (z_volume > z_threshold) | 
                    (z_imbalance > z_threshold) | 
                    (z_flow > z_threshold)).astype(int)
    return proxy_labels


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
    # Native trapezoidal implementation to avoid numpy 2.0+ deprecation/removal of np.trapz
    return float(np.sum((tpr[1:] + tpr[:-1]) * 0.5 * (fpr[1:] - fpr[:-1])))


def run_roc_calibration(
    symbol: str,
    lookback_days: int = 14,
    z_threshold: float = 2.5
) -> dict[str, Any] | None:
    """Evaluate current model's performance on historical data and find the optimal threshold."""
    LOGGER.info(f"Starting ROC evaluation for {symbol} over past {lookback_days} days...")
    
    # 1. Resolve artifact paths for model weights & scaler
    try:
        model_path, scaler_path = _resolve_artifact_paths(symbol)
    except Exception as e:
        LOGGER.error(f"Could not find model artifacts for {symbol}: {e}")
        return None
        
    if not model_path.exists() or not scaler_path.exists():
        LOGGER.error(f"Missing weights or scaler files at: {model_path} / {scaler_path}")
        return None

    # 2. Fetch historical feature matrix
    try:
        df = fetch_and_clean_dataframe(symbol, lookback_days)
    except Exception as e:
        LOGGER.error(f"Failed to fetch data from TimescaleDB: {e}")
        return None

    # Generate proxy ground-truth labels for each row
    proxy_labels = generate_proxy_labels(df, z_threshold)
    LOGGER.info(f"Generated statistical proxy labels: {np.sum(proxy_labels)} anomalies out of {len(df)} rows.")

    # 3. Load scaler parameters & sequence length
    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler_params = json.load(f)
        
    # Standard sequence extraction
    feature_columns = scaler_params["features"]
    min_vals = pd.Series(scaler_params["mins"])
    max_vals = pd.Series(scaler_params["maxs"])
    range_vals = (max_vals - min_vals).replace(0.0, 1.0)

    # Manual continuous segment scaling to match production retraining
    df_scaled = df.copy()
    df_scaled[feature_columns] = (df[feature_columns] - min_vals) / range_vals

    seq_len = 12
    sequences = []
    labels = []
    
    feature_matrix = df_scaled[feature_columns].to_numpy()
    timestamps = df_scaled["bucket"].tolist()
    
    # Re-slice sequences matching production continuous extraction
    # and map sequence label to the current (last) element in the sequence
    current_block_features = []
    current_block_labels = []
    last_timestamp = None
    
    for i, current_ts in enumerate(timestamps):
        lbl = proxy_labels[i]
        feat = feature_matrix[i]
        
        if last_timestamp is None:
            current_block_features.append(feat)
            current_block_labels.append(lbl)
        else:
            time_delta = current_ts - last_timestamp
            if time_delta <= pd.Timedelta(minutes=5):
                current_block_features.append(feat)
                current_block_labels.append(lbl)
            else:
                if len(current_block_features) >= seq_len:
                    for start in range(len(current_block_features) - seq_len + 1):
                        sequences.append(np.array(current_block_features[start: start + seq_len]))
                        labels.append(current_block_labels[start + seq_len - 1])
                current_block_features = [feat]
                current_block_labels = [lbl]
                
        last_timestamp = current_ts
        
    if len(current_block_features) >= seq_len:
        for start in range(len(current_block_features) - seq_len + 1):
            sequences.append(np.array(current_block_features[start: start + seq_len]))
            labels.append(current_block_labels[start + seq_len - 1])

    X_eval = np.array(sequences, dtype=np.float32)
    y_eval = np.array(labels, dtype=np.int32)
    
    if len(X_eval) == 0:
        LOGGER.error("Zero sequence windows found in the historical data.")
        return None

    LOGGER.info(f"Formulated {len(X_eval)} continuous sequences with {np.sum(y_eval)} target anomaly occurrences.")

    # 4. Initialize model & load state dict
    device = _resolve_model_device()
    input_dim = len(feature_columns)
    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=10, seq_len=seq_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 5. Evaluate Reconstruction MSE Errors
    mse_scores = []
    X_tensor = torch.tensor(X_eval).to(device)
    
    with torch.no_grad():
        # Process in batches to save memory
        batch_size = 64
        for start_idx in range(0, len(X_tensor), batch_size):
            batch_x = X_tensor[start_idx : start_idx + batch_size]
            reconstructed = model(batch_x)
            # Calculate row-wise mean squared error
            batch_mse = torch.mean((batch_x - reconstructed) ** 2, dim=(1, 2)).cpu().numpy()
            mse_scores.extend(batch_mse)

    y_scores = np.array(mse_scores, dtype=np.float32)

    # 6. ROC Metric Computation
    fpr, tpr, thresholds = compute_roc_curve(y_eval, y_scores)
    auc_score = compute_auc(fpr, tpr)
    
    # 7. Locate optimal threshold using Youden's J statistic
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresholds[best_idx])
    max_j = float(j_scores[best_idx])
    best_tpr = float(tpr[best_idx])
    best_fpr = float(fpr[best_idx])

    LOGGER.info("=== ROC Threshold Calibration Results ===")
    LOGGER.info(f"Coin Symbol:         {symbol}")
    LOGGER.info(f"Area Under Curve (AUC): {auc_score:.5f}")
    LOGGER.info(f"Optimal Threshold:   {optimal_threshold:.6f}")
    LOGGER.info(f"Youden's J Max Score:{max_j:.5f}")
    LOGGER.info(f"Sensitivity (TPR):   {best_tpr * 100:.2f}%")
    LOGGER.info(f"False Alarm Rate (FPR): {best_fpr * 100:.2f}%")
    
    return {
        "symbol": symbol,
        "auc_score": auc_score,
        "optimal_threshold": optimal_threshold,
        "max_j": max_j,
        "tpr": best_tpr,
        "fpr": best_fpr
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ROC thresholds dynamically.")
    parser.add_argument("--symbol", type=str, default=TARGET_SYMBOL, help="Target crypto symbol")
    parser.add_argument("--lookback", type=int, default=14, help="Historical lookback days")
    parser.add_argument("--zscore", type=float, default=2.5, help="Z-score proxy outlier threshold")
    args = parser.parse_args()

    results = run_roc_calibration(args.symbol, args.lookback, args.zscore)
    if results is None:
        LOGGER.error("ROC calibration failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
