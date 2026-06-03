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

from src.models.retraining_service import (
    fetch_and_clean_dataframe,
    TARGET_SYMBOL,
    generate_proxy_labels,
    slice_continuous_windows,
    align_labels_to_sequences,
    run_evaluation_inference,
    calculate_optimal_threshold
)
from src.models.anomaly_pipeline import resolve_artifact_paths, resolve_model_device
from src.models.lstm_autoencoder import LSTMAutoencoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
LOGGER = logging.getLogger("roc_evaluator")





def run_roc_calibration(
    symbol: str,
    lookback_days: int = 14,
    z_threshold: float = 2.5
) -> dict[str, Any] | None:
    """Evaluate current model's performance on historical data and find the optimal threshold."""
    LOGGER.info(f"Starting ROC evaluation for {symbol} over past {lookback_days} days...")
    
    # 1. Resolve artifact paths for model weights & scaler
    try:
        model_path, scaler_path = resolve_artifact_paths(symbol)
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
    feature_matrix = df_scaled[feature_columns].to_numpy()
    timestamps = df_scaled["bucket"].tolist()
    
    X_eval = slice_continuous_windows(timestamps, feature_matrix, seq_len)
    y_eval = align_labels_to_sequences(df, proxy_labels, seq_len)
    
    if len(X_eval) == 0:
        LOGGER.error("Zero sequence windows found in the historical data.")
        return None

    LOGGER.info(f"Formulated {len(X_eval)} continuous sequences with {np.sum(y_eval)} target anomaly occurrences.")

    # 4. Initialize model & load state dict
    device = resolve_model_device()
    input_dim = len(feature_columns)
    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=10, seq_len=seq_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # 5. Evaluate Reconstruction MSE Errors & ROC calculations
    y_scores = run_evaluation_inference(model, X_eval, device)
    optimal_threshold, auc_score, fpr, tpr, thresholds, best_idx = calculate_optimal_threshold(y_eval, y_scores)
    
    max_j = float(tpr[best_idx] - fpr[best_idx])
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
