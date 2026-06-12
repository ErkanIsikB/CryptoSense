"""
WebSocket Stream 1: DEX Whale Trades for BTC(WBTC), ETH(WETH), SOL, BNB(WBNB), AVAX(WAVAX)
"""
import asyncio
import json
import logging

import websockets

from src.core.config import settings
from src.sinks.jsonl_sink import JsonlFileSink

BITQUERY_WS_URL = "wss://streaming.bitquery.io/graphql"

QUERY = """
subscription {
  Trading {
    Trades(
      where: {
        AmountsInUsd: {Base: {gt: 100000}}
        Pair: {
          Token: {
            Id: {
              in: [
                "bid:solana:So11111111111111111111111111111111111111112",
                "bid:ethereum:0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                "bid:bsc:0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
                "bid:avalanche:0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",
                "bid:ethereum:0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
              ]
            }
          }
        }
      }
    ) {
      Side
      Trader { Address }
      AmountsInUsd { Base Quote }
      Block { Time Timestamp }
      Pair {
        Market { Network }
        Token { Symbol }
        QuoteToken { Symbol }
      }
      TransactionHeader { Sender To }
    }
  }
}
"""

async def run_ws_whale_trades():
    output_dir = settings.DATA_DIR / "whale_trades"
    sink = JsonlFileSink(output_dir)
    backoff = 1

    while True:
        try:
            async with websockets.connect(
                BITQUERY_WS_URL, 
                subprotocols=[websockets.Subprotocol("graphql-ws")],
              additional_headers={"Authorization": f"Bearer {settings.BITQUERY_API_KEY}"}
            ) as ws:
                
                # GraphQL WS Protocol Init
                await ws.send(json.dumps({"type": "connection_init"}))
                
                # Wait for connection_ack and start subscription
                async for message in ws:
                    init_response = json.loads(message)
                    if init_response.get("type") == "connection_ack":
                        await ws.send(json.dumps({
                            "id": "whale_trades_stream",
                            "type": "start",
                            "payload": {"query": QUERY}
                        }))
                        backoff = 1
                        logging.info("Connected and subscribed to Bitquery Whale Trades Stream")
                        break
                
                # Listen to data
                async for message in ws:
                  data = json.loads(message)
                  if data.get("type") == "data":
                    await sink.write("all_tokens_whale_trades", data["payload"]["data"])
                  elif data.get("type") == "error":
                    logging.error(f"Subscription error: {data}")
                        
        except Exception as e:
            logging.error(f"Whale Trades stream disconnected: {e}. Reconnecting in {backoff} seconds...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_ws_whale_trades())