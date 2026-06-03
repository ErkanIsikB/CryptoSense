"""Integration test: push mock trade and orderbook data through aggregators into DB.

This test is transaction-isolated. All changes are rolled back at completion,
ensuring zero test data pollution in the database.
"""
import sys
import os
import threading
from contextlib import contextmanager
from unittest.mock import patch


# Adjust path to find src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.db import execute_query_fetch, close_pool, get_pool
from src.feature_engineering.trade_aggregator import TradeAggregator
from src.feature_engineering.orderbook_aggregator import OrderbookAggregator

@contextmanager
def transaction_isolated_context():
    """Context manager to run tests within a database transaction that is rolled back.
    
    It also forces background threads to run synchronously on the main thread.
    """
    pool = get_pool()
    conn = pool.getconn()
    conn.autocommit = False
    
    original_commit = conn.commit
    original_rollback = conn.rollback
    
    # Prevent committing to the DB
    conn.commit = lambda: None
    
    @contextmanager
    def mock_get_connection():
        yield conn

    # SyncThread executes the thread target synchronously on the main thread
    original_thread = threading.Thread
    class SyncThread(original_thread):
        def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
            super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}
            
        def start(self):
            if self._target:
                self._target(*self._args, **self._kwargs)

    with patch("src.db.db.get_connection", side_effect=mock_get_connection), \
         patch("threading.Thread", SyncThread):
        try:
            conn.rollback()  # clear any stale transaction
            
            # Clean target buckets inside the transaction first
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM trade_candles_5m WHERE symbol = 'BTCUSDT' AND bucket >= '2023-11-14 22:00:00+00' AND bucket <= '2023-11-14 22:30:00+00';"
                )
                cur.execute(
                    "DELETE FROM orderbook_snapshots_5m WHERE symbol = 'ETHUSDT' AND bucket >= '2023-11-14 22:00:00+00' AND bucket <= '2023-11-14 22:30:00+00';"
                )
            
            yield conn
        finally:
            try:
                original_rollback()
            except Exception as rollback_err:
                print(f"Error during rollback: {rollback_err}")
            
            conn.commit = original_commit
            conn.rollback = original_rollback
            pool.putconn(conn)


def test_trade_aggregator():
    print("🧪 Testing Trade Aggregator...")
    agg = TradeAggregator()

    # Create a bucket in the past (so it will flush)
    # Use a fixed timestamp that maps to a known 5-min bucket
    base_ms = 1700000000000  # Nov 14, 2023 22:13:20 UTC

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
          AND bucket >= '2023-11-14 22:00:00+00'
          AND bucket <= '2023-11-14 22:30:00+00'
        ORDER BY bucket ASC;
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
          AND bucket >= '2023-11-14 22:00:00+00'
          AND bucket <= '2023-11-14 22:30:00+00'
        ORDER BY bucket ASC;
    """)

    print(f"  Rows in DB: {len(rows)}")
    for row in rows:
        print(f"  {row[0]}: spread={row[1]:.4f} mid={row[2]:.2f} "
              f"bid_depth={row[3]:.2f} ask_depth={row[4]:.2f} "
              f"imbalance={row[5]:.4f} snapshots={row[6]}")

    assert len(rows) >= 1, "Expected at least 1 orderbook row"
    print("  ✅ Orderbook Aggregator OK\n")


def run_integration_tests():
    with transaction_isolated_context():
        test_trade_aggregator()
        test_orderbook_aggregator()


if __name__ == "__main__":
    try:
        run_integration_tests()
        print("🎉 All integration tests passed (transaction rolled back successfully)!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        close_pool()
