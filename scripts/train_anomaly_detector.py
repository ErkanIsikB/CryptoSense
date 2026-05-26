"""CryptoSense Model Training Script.

Thin wrapper around the reusable retraining service.
"""

from __future__ import annotations

import logging
import sys

from src.models.retraining_service import TARGET_SYMBOL, train_symbol_model

PURPLE = "\033[95m"
RESET = "\033[0m"


class PurpleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return f"{PURPLE}{super().format(record)}{RESET}"


logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(PurpleFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
logger.addHandler(handler)


def main() -> None:
    train_symbol_model(TARGET_SYMBOL)


if __name__ == "__main__":
    # Execution entry via Python module namespace: python -m scripts.train_anomaly_detector
    main()
