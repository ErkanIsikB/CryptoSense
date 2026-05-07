"""
HTTP Polling: BSC (BNB) and Avalanche (AVAX) Transfers
Runs periodically to save stream limits.
"""
import asyncio
import logging

import httpx

from src.core.config import settings
from src.sinks.jsonl_sink import JsonlFileSink

HTTP_URL = "https://streaming.bitquery.io/graphql"
POLLING_INTERVAL = 300  # 5 Dakikada bir çalışacak

BSC_QUERY = """
{
  EVM(network: bsc, dataset: realtime) {
    Transfers(
      limit: { count: 100 }
      orderBy: { descending: Block_Time }
      where: {
        Block: { Time: { since_relative: { minutes_ago: 5 } } }
        Transfer: {
          AmountInUSD: { ge: "100000" }
          Currency: { SmartContract: { is: "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c" } }
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

AVAX_QUERY = """
{
  EVM(network: avalanche, dataset: realtime) {
    Transfers(
      limit: { count: 100 }
      orderBy: { descending: Block_Time }
      where: {
        Block: { Time: { since_relative: { minutes_ago: 5 } } }
        Transfer: {
          AmountInUSD: { ge: "100000" }
          Currency: { SmartContract: { is: "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7" } }
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

async def fetch_and_save(client, query, sink, network_name, key):
    headers = {
      "Authorization": f"Bearer {settings.BITQUERY_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = await client.post(HTTP_URL, json={"query": query}, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        data_block = data.get("data")
        if data_block:
          evm_data = data_block.get("EVM")
          if evm_data and isinstance(evm_data, dict):
            transfers = evm_data.get("Transfers")
            if transfers:
              await sink.write(key, evm_data)
              logging.info(f"Successfully polled and saved {network_name} transfers.")
              return

        logging.info(f"No large transfers found for {network_name} in the last 5 minutes.")
            
    except Exception as e:
        logging.error(f"Polling failed for {network_name}: {str(e)}")

async def run_http_polling():
    output_dir = settings.DATA_DIR / "transfers"
    sink = JsonlFileSink(output_dir)

    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            logging.info("Starting polling cycle for BSC and AVAX...")
            await asyncio.gather(
                fetch_and_save(client, BSC_QUERY, sink, "BSC", "bnb_transfers"),
                fetch_and_save(client, AVAX_QUERY, sink, "AVALANCHE", "avax_transfers"),
            )
            logging.info(f"Polling cycle complete. Waiting {POLLING_INTERVAL} seconds.")
            await asyncio.sleep(POLLING_INTERVAL)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_http_polling())