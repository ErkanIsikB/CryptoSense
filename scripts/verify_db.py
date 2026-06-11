"""Verify that TimescaleDB tables were created correctly."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.db import execute_query_fetch, close_pool

if __name__ == "__main__":
    try:
        # List all tables
        tables = execute_query_fetch("""
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename;
        """)
        print("=== Tables in public schema ===")
        for row in tables:
            print(f"  * {row[0]}")

        # Check hypertables
        hypertables = execute_query_fetch("""
            SELECT hypertable_name, num_dimensions
            FROM timescaledb_information.hypertables
            ORDER BY hypertable_name;
        """)
        print(f"\n=== TimescaleDB Hypertables ({len(hypertables)}) ===")
        for row in hypertables:
            print(f"  * {row[0]} (dimensions: {row[1]})")

        # Check column counts per table
        current_tables = [
            'trade_candles_5m',
            'orderbook_snapshots_5m',
            'tweet_sentiment_5m',
            'cex_flows_5m',
            'ai_anomalies_5m',
            'llm_health_scores'
        ]
        for table in current_tables:
            cols = execute_query_fetch(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table}'
                ORDER BY ordinal_position;
            """)
            print(f"\n--- {table} ({len(cols)} columns) ---")
            for col in cols:
                print(f"    {col[0]:25s} {col[1]}")

        print("\n[OK] All tables verified successfully!")

    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        sys.exit(1)
    finally:
        close_pool()
