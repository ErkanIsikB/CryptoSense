import sys
import os
import json

# Root path fix
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.db import execute_query_fetch, close_pool

if __name__ == "__main__":
    try:
        # Fetch only the single most recent critical anomaly payload from the DB
        row = execute_query_fetch("""
                                  SELECT symbol, bucket, llm_payload
                                  FROM ai_anomalies_5m
                                  WHERE is_anomaly = TRUE
                                  ORDER BY bucket DESC LIMIT 1;
                                  """)

        if row:
            symbol, bucket, payload = row[0]
            print(f"\n🔍 INSPECTING LATEST LIVE PAYLOAD FOR: {symbol} ({bucket})")
            print("=" * 60)

            # If it's a string, load it. If it's already a dict (psycopg2 auto-unpack), use it.
            payload_dict = json.loads(payload) if isinstance(payload, str) else payload

            # Pretty print the JSON with color-friendly indentation
            print(json.dumps(payload_dict, indent=4))
            print("=" * 60)
        else:
            print("\n📭 No anomalies found in the database yet.")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        close_pool()