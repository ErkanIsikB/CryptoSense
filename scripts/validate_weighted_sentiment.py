"""Validate the source-weighted sentiment calculation with a small example.

Run from the project root:

    python3 scripts/validate_weighted_sentiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering.source_credibility import (  # noqa: E402
    calculate_weighted_sentiment,
    enrich_scored_item,
)


def main() -> None:
    examples = [
        {"author_handle": "@CoinDesk", "score": 0.8},
        {"author_handle": "@VitalikButerin", "score": 0.6},
        {"author_handle": "@randomuser123", "score": -0.5},
    ]

    enriched = [
        enrich_scored_item(item, item["score"], item["author_handle"])
        for item in examples
    ]
    stats = calculate_weighted_sentiment(enriched)

    expected_unweighted = 0.3
    expected_weighted = 3.7 / 7.0

    print("Weighted sentiment validation")
    for item in enriched:
        print(
            f"{item['author_handle']}: tier={item['source_tier']} "
            f"weight={item['source_weight']} raw={item['raw_sentiment_score']} "
            f"weighted_contribution={item['weighted_sentiment_contribution']:.4f}"
        )

    print(f"unweighted_avg={stats['unweighted_avg_sentiment']:.4f}")
    print(f"weighted_avg={stats['weighted_avg_sentiment']:.4f}")
    print(f"total_source_weight={stats['total_source_weight']:.1f}")

    assert abs(stats["unweighted_avg_sentiment"] - expected_unweighted) < 0.0001
    assert abs(stats["weighted_avg_sentiment"] - expected_weighted) < 0.0001
    assert stats["tier1_count"] == 2
    assert stats["unknown_count"] == 1
    print("OK")


if __name__ == "__main__":
    main()
