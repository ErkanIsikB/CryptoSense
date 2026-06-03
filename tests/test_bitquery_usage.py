"""Bitquery Integration Test.

Runs Bitquery WebSocket streams (whale trades, EVM transfers, Solana transfers)
and HTTP polling tasks concurrently for a configured duration to verify integration,
network connectivity, and Bitquery credit consumption.
"""
import asyncio
import logging

# Import Bitquery data source runners
from src.data_sources.bitquery.ws_whale_trades import run_ws_whale_trades
from src.data_sources.bitquery.ws_evm_transfers import run_ws_evm_transfers
from src.data_sources.bitquery.ws_solana_transfers import run_ws_solana_transfers
from src.data_sources.bitquery.http_polling import run_http_polling

# Test duration set to 5 minutes (300 seconds)
TEST_DURATION = 300 

async def run_all_tasks():
    """Start all data streams concurrently."""
    await asyncio.gather(
        run_ws_whale_trades(),
        run_ws_evm_transfers(),
        run_ws_solana_transfers(),
        run_http_polling()
    )

async def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("⏳ Starting Bitquery Usage Test...")
    logging.info(f"Please note your current Bitquery credits. The test will run for {TEST_DURATION / 60:.1f} minutes.")
    
    try:
        # Wrap tasks inside wait_for to automatically shut down after TEST_DURATION
        await asyncio.wait_for(run_all_tasks(), timeout=TEST_DURATION)
    except asyncio.TimeoutError:
        logging.info("✅ Test duration completed! All streams and connections have been shut down safely.")
        logging.info("You can verify the test by checking the Bitquery dashboard and subtracting your starting points from your current balance.")
    except Exception as e:
        logging.error(f"Unexpected error during test: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("🛑 Test stopped manually by the user.")