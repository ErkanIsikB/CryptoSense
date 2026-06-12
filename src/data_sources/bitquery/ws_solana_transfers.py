"""
WebSocket Stream 3: Solana Transfers for SOL
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
  Solana {
    Transfers(
      where: {
        Transfer: {
          AmountInUSD: { ge: "100000" }
          Currency: { MintAddress: { is: "So11111111111111111111111111111111111111112" } }
        }
      }
    ) {
      Transfer { Amount AmountInUSD Sender Receiver }
      Block { Time Slot }
      Transaction { Signature }
    }
  }
}
"""

async def run_ws_solana_transfers():
    output_dir = settings.DATA_DIR / "transfers"
    sink = JsonlFileSink(output_dir)
    backoff = 1

    while True:
        try:
            async with websockets.connect(
                BITQUERY_WS_URL, 
                subprotocols=[websockets.Subprotocol("graphql-ws")],
              additional_headers={"Authorization": f"Bearer {settings.BITQUERY_API_KEY}"}
            ) as ws:
                
                await ws.send(json.dumps({"type": "connection_init"}))
                
                async for message in ws:
                    init_response = json.loads(message)
                    if init_response.get("type") == "connection_ack":
                        await ws.send(json.dumps({
                            "id": "solana_transfers_stream",
                            "type": "start",
                            "payload": {"query": QUERY}
                        }))
                        backoff = 1
                        logging.info("Connected to Solana Transfers Stream")
                        break

                async for message in ws:
                  data = json.loads(message)
                  if data.get("type") == "data":
                    await sink.write("solana_transfers", data["payload"]["data"])
                        
        except Exception as e:
            logging.error(f"Solana Transfers stream disconnected: {e}. Reconnecting in {backoff} seconds...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_ws_solana_transfers())