import sys
import os

# Root path fix
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.db import execute_query_fetch, close_pool

def main():
    try:
        # Fetch the most recent health score record for EACH token
        # ORDER BY symbol, bucket DESC makes sure DISTINCT ON retrieves the latest per symbol
        rows = execute_query_fetch("""
            SELECT DISTINCT ON (symbol) 
                symbol, bucket, health_score, reasoning, explanation, model_name, latency_ms
            FROM llm_health_scores
            ORDER BY symbol, bucket DESC;
        """)

        if rows:
            print("\n📋 LATEST LLM HEALTH SCORES FOR ALL TOKENS")
            print("=" * 80)
            for symbol, bucket, health_score, reasoning, explanation, model_name, latency_ms in rows:
                print(f"🪙 Token: {symbol} | Health Score: {health_score}/100 | Time: {bucket}")
                print(f"🤖 Model: {model_name} | Latency: {latency_ms}ms")
                print(f"🧠 Reasoning Summary: {reasoning}")
                print(f"📝 LLM Briefing Explanation:")
                print(f"   {explanation}")
                print("-" * 80)
            print("=" * 80)
        else:
            print("\n📭 No health scores found in the database.")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        close_pool()

if __name__ == "__main__":
    main()
