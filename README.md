# CryptoSense — Multi-Source Ingestion Engine

CryptoSense ingests live crypto market data from multiple providers and writes append-only JSONL output for downstream processing.

## What this project does

It runs four pipelines concurrently:

- Binance aggregate trades (`aggTrade`)
- Binance orderbook depth (`depth@100ms`)
- Bitquery V2 GraphQL streaming (whale transfers + DEX trades)
- Tavily sentiment polling

Each pipeline writes stream records in append mode to files under `data/`.

## Architecture (second-pass refactor)

### 1) Modular pipeline layout

- `src/ingest.py` → Binance trades
- `src/orderbook_ingest.py` → Binance orderbook
- `src/bitquery_stream_engine.py` → Bitquery streams
- `src/sentiment_tracker.py` → Tavily sentiment

### 2) Unified sink abstraction (database-ready)

- `sinks/base.py` defines `BaseSink`
- `sinks/jsonl_sink.py` provides `JsonlFileSink`

All pipelines write through this sink contract, so a future database sink can be added without changing source-specific ingestion logic.

### 3) Flexible orchestrator

`main.py` uses a pipeline registry (`PIPELINES`) instead of hardcoded launch logic. Adding a new source means:

1. Create a new `start_*_stream` module in `src/`
2. Register one entry in `PIPELINES`

## Project structure

```text
.
├── main.py
├── config/
│   └── settings.py
├── sinks/
│   ├── base.py
│   └── jsonl_sink.py
├── src/
│   ├── ingest.py
│   ├── orderbook_ingest.py
│   ├── bitquery_stream_engine.py
│   └── sentiment_tracker.py
├── utils/
│   ├── logging.py
│   └── signals.py
├── data/
│   ├── trades/
│   ├── orderbook/
│   ├── bitquery/
│   └── sentiment/
├── tests/
│   └── test_engine_utils.py
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
python main.py
```

Stop with `Ctrl+C` (graceful shutdown is handled).

## Output behavior

- Output files are stable and append-only (`*.jsonl`)
- Existing files are reused between runs
- No per-run file recreation logic

Current file pattern:

- `data/trades/<symbol>.jsonl`
- `data/orderbook/<symbol>.jsonl`
- `data/bitquery/<TOKEN>_<category>.jsonl`
- `data/sentiment/sentiment.jsonl`

## Future database integration

To add a database writer later:

1. Implement `BaseSink` in a new file (example: `sinks/postgres_sink.py`)
2. Keep pipeline logic unchanged
3. Swap sink construction in pipeline entrypoints (or inject via settings/factory)

This keeps ingestion, transport, and persistence concerns separated.

## Adding a new data source

1. Add a new pipeline module in `src/`
2. Use `BaseSink` for persistence
3. Register pipeline in `main.py` `PIPELINES`
4. Add source-specific settings in `config/settings.py`

## Tests

```zsh
python -m unittest discover -s tests -v
```
