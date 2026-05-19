"""Integration test: push mock trade and orderbook data through aggregators into DB."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.db.db import execute_query_fetch, close_pool
from src.feature_engineering.trade_aggregator import TradeAggregator
from src.feature_engineering.orderbook_aggregator import OrderbookAggregator

def test_trade_aggregator():
    print("🧪 Testing Trade Aggregator...")
    agg = TradeAggregator()

    # Create a bucket in the past (so it will flush)
    # Use a fixed timestamp that maps to a known 5-min bucket
    base_ms = 1700000000000  # some past timestamp

    # Add some trades to bucket 1
    agg.add("BTCUSDT", 71000.0, 0.5, base_ms + 0, False)       # buy
    agg.add("BTCUSDT", 71050.0, 0.3, base_ms + 1000, True)     # sell
    agg.add("BTCUSDT", 71100.0, 0.2, base_ms + 2000, False)    # buy
    agg.add("BTCUSDT", 70900.0, 0.1, base_ms + 3000, True)     # sell

    # Add a trade to a LATER bucket to trigger flush of bucket 1
    agg.add("BTCUSDT", 71200.0, 0.01, base_ms + 400_000, False)

    # Also flush the remaining bucket
    agg.flush_all()

    # Check DB
    rows = execute_query_fetch("""
        SELECT symbol, open, high, low, close, volume, trade_count, 
               buy_volume, sell_volume, net_trade, vwap
        FROM trade_candles_5m
        WHERE symbol = 'BTCUSDT'
        ORDER BY bucket ASC
        LIMIT 5;
    """)

    print(f"  Rows in DB: {len(rows)}")
    for row in rows:
        print(f"  {row[0]}: O={row[1]} H={row[2]} L={row[3]} C={row[4]} "
              f"V={row[5]:.3f} trades={row[6]} buy={row[7]:.3f} sell={row[8]:.3f} "
              f"net={row[9]:.3f} vwap={row[10]:.2f}")

    assert len(rows) >= 1, "Expected at least 1 candle row"
    print("  ✅ Trade Aggregator OK\n")


def test_orderbook_aggregator():
    print("🧪 Testing Orderbook Aggregator...")
    agg = OrderbookAggregator()

    base_ms = 1700000000000

    # Snapshot 1 in bucket 1
    agg.add("ETHUSDT", base_ms,
            bids=[["2200.00", "100.0"], ["2199.00", "50.0"]],
            asks=[["2201.00", "80.0"], ["2202.00", "40.0"]])

    # Snapshot 2 in bucket 1
    agg.add("ETHUSDT", base_ms + 60000,
            bids=[["2200.50", "120.0"], ["2199.50", "60.0"]],
            asks=[["2201.50", "90.0"], ["2202.50", "30.0"]])

    # Snapshot in later bucket to trigger flush
    agg.add("ETHUSDT", base_ms + 400_000,
            bids=[["2203.00", "100.0"]],
            asks=[["2204.00", "80.0"]])

    agg.flush_all()

    rows = execute_query_fetch("""
        SELECT symbol, avg_spread, avg_mid_price, avg_bid_depth, 
               avg_ask_depth, avg_imbalance, snapshot_count
        FROM orderbook_snapshots_5m
        WHERE symbol = 'ETHUSDT'
        ORDER BY bucket ASC
        LIMIT 5;
    """)

    print(f"  Rows in DB: {len(rows)}")
    for row in rows:
        print(f"  {row[0]}: spread={row[1]:.4f} mid={row[2]:.2f} "
              f"bid_depth={row[3]:.2f} ask_depth={row[4]:.2f} "
              f"imbalance={row[5]:.4f} snapshots={row[6]}")

    assert len(rows) >= 1, "Expected at least 1 orderbook row"
    print("  ✅ Orderbook Aggregator OK\n")


if __name__ == "__main__":
    try:
        test_trade_aggregator()
        test_orderbook_aggregator()
        print("🎉 All integration tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        close_pool()
