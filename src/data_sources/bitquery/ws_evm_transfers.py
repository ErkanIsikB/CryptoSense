"""
WebSocket Stream 2: Ethereum EVM Transfers for WETH and WBTC
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
  EVM(network: eth) {
    Transfers(
      where: {
        Transfer: {
          AmountInUSD: { ge: "100000" }
          Currency: {
            SmartContract: {
              in: [
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", # WETH
                "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"  # WBTC
              ]
            }
          }
        }
      }
    ) {
      Transfer { Amount AmountInUSD Sender Receiver Currency { Symbol } }
      Block { Time Number }
      Transaction { Hash }
    }
  }
}
"""

async def run_ws_evm_transfers():
    output_dir = settings.DATA_DIR / "transfers"
    sink = JsonlFileSink(output_dir)
    backoff = 1

    while True:
        try:
            async with websockets.connect(
                BITQUERY_WS_URL, 
                subprotocols=["graphql-ws"],
              additional_headers={"Authorization": f"Bearer {settings.BITQUERY_API_KEY}"}
            ) as ws:
                
                await ws.send(json.dumps({"type": "connection_init"}))
                
                async for message in ws:
                    init_response = json.loads(message)
                    if init_response.get("type") == "connection_ack":
                        await ws.send(json.dumps({
                            "id": "evm_transfers_stream",
                            "type": "start",
                            "payload": {"query": QUERY}
                        }))
                        backoff = 1
                        logging.info("Connected to EVM Transfers Stream")
                        break

                async for message in ws:
                  data = json.loads(message)
                  if data.get("type") == "data":
                    await sink.write("eth_wbtc_transfers", data["payload"]["data"])
                        
        except Exception as e:
            logging.error(f"EVM Transfers stream disconnected: {e}. Reconnecting in {backoff} seconds...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_ws_evm_transfers())