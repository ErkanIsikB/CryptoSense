# Bitquery V2 Streaming Ingestion Engine

Lightweight Python engine for real-time whale transfer and DEX trade ingestion across BTC, ETH, SOL, BNB, and AVAX, with structured JSON output under `raw_data/`.

## What it does

- Uses GraphQL subscriptions over WebSocket (`Bitquery V2 Streaming API`)
- Captures:
  - Whale transfers (`AmountInUSD > WHALE_USD_THRESHOLD`)
  - DEX trades (with side, protocol, amount, price)
- Adds `flow_hint` for future inflow/outflow analysis
- Handles reconnects, exponential backoff, and HTTP `429` retry delays
- Writes newline-delimited JSON objects into date/token/category files

## Project structure

- `main.py` — executable entrypoint
- `src/bitquery_stream_engine.py` — ingestion engine
- `raw_data/` — output folder
- `.env.example` — required environment variables
- `tests/test_engine_utils.py` — small sanity tests

## Setup

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Update `.env` and set `BITQUERY_API_KEY`.

## Run

```zsh
python main.py
```

## Low-credit mode (recommended)

Use conservative settings in `.env`:

- `WHALE_USD_THRESHOLD=1500000`
- `TRADE_MIN_USD=100000`
- `BACKFILL_MINUTES=0`
- `ENABLED_TOKENS=ETH,SOL`
- `ENABLED_CATEGORIES=transfers`
- `MAX_ACTIVE_STREAMS=2`

When you want broader coverage, increase gradually (for example add `BTC` first, then `trades`).

## Output format

Files are created in this shape:

- `raw_data/2026-03-31_ETH_transfers.json`
- `raw_data/2026-03-31_ETH_trades.json`

Each line is a standalone JSON object for easy downstream delivery.

## Tiny viewer

Quickly inspect fetched rows from `raw_data/`:

```zsh
/Users/basar/bitquerydatas/.venv/bin/python viewer.py --limit 20
/Users/basar/bitquerydatas/.venv/bin/python viewer.py --date 2026-03-31 --token ETH --category transfers --limit 10
/Users/basar/bitquerydatas/.venv/bin/python viewer.py --token SOL --category trades --pretty --limit 5
```

## Notes

- Query field names can vary by Bitquery schema version and chain dataset naming.
- If your endpoint uses slightly different field names, adjust query blocks in `src/bitquery_stream_engine.py`.
