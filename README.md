# CryptoSense — Multi-Source Ingestion Engine

CryptoSense ingests live crypto market data from multiple providers and writes append-only JSONL output for downstream processing.

## What this project does

It runs three pipelines concurrently:

- Binance aggregate trades (`aggTrade`)
- Binance orderbook REST snapshots (`/api/v3/depth`, `limit=20`)
- Tavily sentiment polling

Each pipeline writes stream records in append mode to files under `scripts/data/`.

## Architecture (second-pass refactor)

### 1) Modular pipeline layout

- `src/data_sources/binancewebsocket/ws_trades_ingestion.py` → Binance trades
- `src/data_sources/binancewebsocket/ws_orderbook_ingestion.py` → Binance orderbook
- `src/data_sources/tavily/tavily_ingestion.py` → Tavily sentiment

### 2) Unified sink abstraction (database-ready)

- `src/sinks/base.py` defines `BaseSink`
- `src/sinks/jsonl_sink.py` provides `JsonlFileSink`

All pipelines write through this sink contract, so a future database sink can be added without changing source-specific ingestion logic.

### 3) Flexible orchestrator

`src/main.py` uses a pipeline registry (`PIPELINES`) instead of hardcoded launch logic. Adding a new source means:

1. Create a new `start_*_stream` module in `src/`
2. Register one entry in `PIPELINES`

## Project structure

```text
.
├── scripts/
│   ├── sinks/
│   │   ├── base.py
│   │   └── jsonl_sink.py
│   └── data/
│       ├── trades/
│       ├── orderbook/
│       └── sentiment/
├── src/
│   ├── main.py
│   ├── core/
│   │   ├── config/
│   │   │   └── settings.py
│   │   └── utils/
│   │       ├── logging.py
│   │       └── signals.py
│   ├── data_sources/
│   │   ├── binancewebsocket/
│   │   │   ├── ws_trades_ingestion.py
│   │   │   └── ws_orderbook_ingestion.py
│   │   └── tavily/
│   │       └── tavily_ingestion.py
│   └── sinks/
│       ├── base.py
│       └── jsonl_sink.py
└── requirements.txt
```

## Setup

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Update `.env` with valid API keys.

## Run

```zsh
python -m src.main
```

Stop with `Ctrl+C` (graceful shutdown is handled).

## Output behavior

- Output files are stable and append-only (`*.jsonl`)
- Existing files are reused between runs
- No per-run file recreation logic

Current file pattern:

- `scripts/data/trades/<symbol>.jsonl`
- `scripts/data/orderbook/<symbol>.jsonl`
- `scripts/data/sentiment/sentiment.jsonl`

## Future database integration

To add a database writer later:

1. Implement `BaseSink` in a new file (example: `src/sinks/postgres_sink.py`)
2. Keep pipeline logic unchanged
3. Swap sink construction in pipeline entrypoints (or inject via settings/factory)

This keeps ingestion, transport, and persistence concerns separated.

## Adding a new data source

1. Add a new pipeline module in `src/`
2. Use `BaseSink` for persistence
3. Register pipeline in `src/main.py` `PIPELINES`
4. Add source-specific settings in `src/core/config/settings.py`

## Tests

No test suite is currently included.
