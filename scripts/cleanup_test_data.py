"""Clean up test data from integration tests."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.db.db import execute_query, close_pool

if __name__ == "__main__":
    try:
        execute_query("DELETE FROM trade_candles_5m WHERE bucket < '2024-01-01';")
        execute_query("DELETE FROM orderbook_snapshots_5m WHERE bucket < '2024-01-01';")
        print("✅ Test data cleaned up")
    finally:
        close_pool()
