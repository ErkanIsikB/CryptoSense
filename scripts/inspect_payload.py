import sys
import os
import json

# Root path fix
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.db import execute_query_fetch, close_pool

if __name__ == "__main__":
    try:
        # Fetch the single most recent active anomaly payload for EACH token
        # DISTINCT ON (symbol) filters down to one row per token key
        rows = execute_query_fetch("""
                                   SELECT DISTINCT
                                   ON (symbol) symbol, bucket, llm_payload
                                   FROM ai_anomalies_5m
                                   ORDER BY symbol, bucket DESC;
                                   """)

        if rows:
            print(f"\n🔍 INSPECTING LATEST LIVE PAYLOADS ({len(rows)} Assets with Anomalies)")
            print("=" * 60)

            for symbol, bucket, payload in rows:
                print(f"\n🪙 ASSET SHIELD: {symbol} | ⏰ TIMEFRAME MARK: {bucket}")
                print("-" * 60)

                # If it's a string, load it. If it's already a dict (psycopg2 auto-unpack), use it.
                payload_dict = json.loads(payload) if isinstance(payload, str) else payload

                # Pretty print the JSON with color-friendly indentation
                print(json.dumps(payload_dict, indent=4))
                print("-" * 60)

            print("=" * 60)
        else:
            print("\n📭 No active anomalies found across any of the tokens in the database.")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        close_pool()