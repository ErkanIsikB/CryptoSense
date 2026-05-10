"""Check what's in all tables right now."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.db.db import execute_query_fetch, close_pool

if __name__ == "__main__":
    try:
        # Tweet sentiment (5-min buckets from XQuik)
        rows = execute_query_fetch("""
            SELECT bucket, symbol, avg_score, tweet_count,
                   positive_count, negative_count, neutral_count,
                   max_score, min_score, LEFT(sample_tweet, 60) as sample
            FROM tweet_sentiment_5m
            ORDER BY bucket DESC
            LIMIT 15;
        """)
        print(f"\n🐦 tweet_sentiment_5m ({len(rows)} rows):")
        for r in rows:
            print(f"  {r[0]} | {r[1]:5s} | avg={r[2]:+.4f} | tweets={r[3]:3d} | +{r[4]} -{r[5]} ~{r[6]} | max={r[7]:+.4f} min={r[8]:+.4f} | {r[9]}")

        # Trade candles
        rows = execute_query_fetch("""
            SELECT bucket, symbol, open, high, low, close, volume, trade_count, net_trade, vwap
            FROM trade_candles_5m
            ORDER BY bucket DESC
            LIMIT 10;
        """)
        print(f"\n📊 trade_candles_5m ({len(rows)} rows):")
        for r in rows:
            print(f"  {r[0]} | {r[1]:10s} | O={r[2]:.2f} H={r[3]:.2f} L={r[4]:.2f} C={r[5]:.2f} | V={r[6]:.4f} trades={r[7]} net={r[8]:+.4f} vwap={r[9]:.2f}")

        # Orderbook
        rows = execute_query_fetch("""
            SELECT bucket, symbol, avg_spread, avg_mid_price, avg_bid_depth, avg_ask_depth, avg_imbalance, snapshot_count
            FROM orderbook_snapshots_5m
            ORDER BY bucket DESC
            LIMIT 10;
        """)
        print(f"\n📊 orderbook_snapshots_5m ({len(rows)} rows):")
        for r in rows:
            print(f"  {r[0]} | {r[1]:10s} | spread={r[2]:.4f} mid={r[3]:.2f} bid_d={r[4]:.2f} ask_d={r[5]:.2f} imb={r[6]:+.4f} snaps={r[7]}")

        # CEX flows
        rows = execute_query_fetch("""
            SELECT bucket, symbol, network, inflow_usd, outflow_usd, net_flow_usd, inflow_tx_count, outflow_tx_count
            FROM cex_flows_5m
            ORDER BY bucket DESC
            LIMIT 10;
        """)
        print(f"\n📊 cex_flows_5m ({len(rows)} rows):")
        for r in rows:
            print(f"  {r[0]} | {r[1]:5s} | {r[2]:10s} | in=${r[3]:,.0f} out=${r[4]:,.0f} net=${r[5]:+,.0f} | in_tx={r[6]} out_tx={r[7]}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        close_pool()
