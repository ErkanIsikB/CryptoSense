# CryptoSense — Real-Time Crypto Data Pipeline

CryptoSense is a high-performance data engineering pipeline that ingests, processes, and persists multi-source cryptocurrency market data into TimescaleDB. It captures trades, orderbook depth, social sentiment from X (Twitter), and CEX fund flows — all aligned to 5-minute windows for downstream analysis and LLM consumption.

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│                      CryptoSense Pipeline                     │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐ │
│  │   Binance    │   │   XQuik     │   │    Bitquery         │ │
│  │  WebSocket   │   │  REST API   │   │    REST API         │ │
│  │  (Futures)   │   │  (X/Twitter)│   │   (CEX Flows)       │ │
│  └──────┬───┬──┘   └──────┬──────┘   └──────────┬──────────┘ │
│         │   │              │                     │            │
│    aggTrade depth      tweets              transfers          │
│         │   │              │                     │            │
│  ┌──────▼───▼──────────────▼─────────────────────▼──────────┐ │
│  │              Feature Engineering Layer                    │ │
│  │                                                           │ │
│  │  TradeAggregator  OrderbookAggregator  SentimentAggregator│ │
│  │  (5-min OHLCV)    (5-min averages)     (5-min FinBERT)    │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                  │
│  ┌──────────────────────────▼────────────────────────────────┐ │
│  │                    TimescaleDB                            │ │
│  │                                                           │ │
│  │  trade_candles_5m │ orderbook_snapshots_5m                │ │
│  │  tweet_sentiment_5m │ cex_flows_5m                        │ │
│  └───────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

## Data Sources

### 1. Binance Futures — Trade Stream (`aggTrade`)
Real-time aggregated trade events via WebSocket from Binance Futures.

- **Symbols:** BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, AVAXUSDT
- **Endpoint:** `wss://fstream.binance.com/market/stream`
- **Frequency:** Real-time (continuous stream)

### 2. Binance — Orderbook Depth
REST polling of order book snapshots (top 20 bid/ask levels).

- **Endpoint:** `https://api.binance.com/api/v3/depth`
- **Frequency:** Every ~4 seconds

### 3. XQuik — X (Twitter) Sentiment
Real-time keyword monitoring on X/Twitter using [XQuik](https://xquik.com) platform. Captures tweets matching crypto-specific keyword queries and scores them through FinBERT.

- **Endpoint:** `https://xquik.com/api/v1`
- **Frequency:** Keyword monitors check every 1 second (server-side); events are polled every 5 minutes
- **Scoring:** ProsusAI/FinBERT sentiment model (positive/negative/neutral → compound score [-1, +1])
- **Keyword Queries:**
  - BTC: `$BTC OR #Bitcoin OR "bitcoin"`
  - ETH: `$ETH OR #Ethereum OR "ethereum"`
  - SOL: `$SOL OR #Solana OR "solana"`
  - BNB: `$BNB OR "bnb" OR "binance coin"`
  - AVAX: `$AVAX OR #Avalanche OR "avalanche crypto"`

### 4. Bitquery — CEX Fund Flows
HTTP polling for large transfers involving known CEX hot wallets across Ethereum, BSC, and Solana.

- **Endpoint:** `https://streaming.bitquery.io/graphql`
- **Networks:** `eth`, `bsc`, `solana`
- **Frequency:** Every 5 minutes
- **Filter:** Server-side CEX address filtering (inflows = Receiver is CEX, outflows = Sender is CEX)

## Database Schema (TimescaleDB)

All tables use 5-minute bucketed timestamps for temporal alignment.

### `trade_candles_5m` — OHLCV Candles
| Column | Type | Description |
|--------|------|-------------|
| `bucket` | TIMESTAMPTZ | 5-minute window start |
| `symbol` | TEXT | e.g. BTCUSDT |
| `open/high/low/close` | FLOAT | Price OHLC |
| `volume` | FLOAT | Total volume |
| `quote_volume` | FLOAT | Quote asset volume |
| `trade_count` | INT | Number of trades |
| `buy_volume/sell_volume` | FLOAT | Directional volume |
| `net_trade` | FLOAT | buy_volume - sell_volume |
| `vwap` | FLOAT | Volume-weighted average price |

### `orderbook_snapshots_5m` — Depth Metrics
| Column | Type | Description |
|--------|------|-------------|
| `bucket` | TIMESTAMPTZ | 5-minute window start |
| `symbol` | TEXT | e.g. BTCUSDT |
| `avg_spread` | FLOAT | Average bid-ask spread |
| `avg_mid_price` | FLOAT | Average mid price |
| `avg_bid_depth/avg_ask_depth` | FLOAT | Average depth per side |
| `avg_imbalance` | FLOAT | (bid - ask) / (bid + ask) |
| `snapshot_count` | INT | Snapshots in window |

### `tweet_sentiment_5m` — X/Twitter Sentiment (via XQuik)
| Column | Type | Description |
|--------|------|-------------|
| `bucket` | TIMESTAMPTZ | 5-minute window start |
| `symbol` | TEXT | e.g. BTC |
| `avg_score` | FLOAT | Mean FinBERT compound score [-1, +1] |
| `tweet_count` | INT | Total tweets in window |
| `positive_count` | INT | Tweets with score > +0.1 |
| `negative_count` | INT | Tweets with score < -0.1 |
| `neutral_count` | INT | Remaining tweets |
| `max_score/min_score` | FLOAT | Extremes in window |
| `sample_tweet` | TEXT | Highest-engagement tweet text |

### `cex_flows_5m` — Exchange Fund Flows
| Column | Type | Description |
|--------|------|-------------|
| `bucket` | TIMESTAMPTZ | 5-minute window start |
| `symbol` | TEXT | e.g. ETH |
| `network` | TEXT | e.g. ethereum, bsc |
| `inflow_usd/outflow_usd` | FLOAT | USD value of flows |
| `net_flow_usd` | FLOAT | inflow - outflow |
| `inflow_tx_count/outflow_tx_count` | INT | Transaction counts |

## Setup

### Prerequisites
- Python 3.11+
- TimescaleDB instance (we use [Timescale Cloud](https://www.timescale.com/cloud))
- API keys: XQuik, Bitquery

### 1. Clone & Create Virtual Environment
```bash
git clone <repository-url>
cd CryptoSense
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- `psycopg2-binary` — PostgreSQL/TimescaleDB driver
- `transformers` + `torch` — FinBERT sentiment model
- `httpx` — Async HTTP client for XQuik and Bitquery APIs
- `websockets` — Binance WebSocket client

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# ── Database ──────────────────────────────────────────
DB_URL=postgres://user:pass@host:port/dbname?sslmode=require

# ── XQuik (X/Twitter Sentiment) ──────────────────────
XQUIK_API=xq_your_api_key_here

# ── Bitquery (CEX Flows) ─────────────────────────────
BITQUERY_API_KEY=your_bitquery_key

# ── Binance ───────────────────────────────────────────
BINANCE_SYMBOLS=btcusdt,ethusdt,solusdt,bnbusdt,avaxusdt

# ── Logging ───────────────────────────────────────────
LOG_LEVEL=INFO
```

### 4. Run Schema Migration
```bash
python run_migration.py
```
This creates all hypertables and indices. Safe to run multiple times (idempotent).

## Running

### Start the Pipeline
```bash
python -m src.main
```

On startup you will see:
1. **Schema migration** — ensures all tables exist
2. **XQuik keyword monitors** — creates/reuses monitors for each coin
3. **Binance WebSocket** — connects to futures trade stream
4. **Orderbook polling** — starts depth snapshot polling
5. **Bitquery CEX flows** — starts fund flow polling

### Check Live Data
```bash
python check_live_data.py
```
Displays the latest rows from all database tables.

### Stop the Pipeline
Press `Ctrl+C`. The pipeline will:
1. Stop all data streams
2. **Pause all XQuik keyword monitors** (saves credits while not running)
3. Flush any remaining buffered data to the database
4. Close the connection pool

> **Note:** On next startup, paused monitors are automatically unpaused.

## How It Works

### Feature Engineering — 5-Minute Aggregation

All raw data streams are processed through stateful aggregators before writing to the database:

1. **TradeAggregator** — Buffers individual trades by `(symbol, 5-min bucket)`. When a trade arrives in a new bucket, the completed bucket is flushed as an OHLCV candle with volume, VWAP, and net trade metrics.

2. **OrderbookAggregator** — Accumulates orderbook snapshots and computes running averages for spread, mid-price, depth, and imbalance ratio. Flushes on bucket boundary.

3. **SentimentAggregator** — Receives FinBERT-scored tweets from XQuik keyword monitors. Tracks score distribution (positive/negative/neutral counts, min/max), selects highest-engagement tweet as sample. Flushes per-symbol aggregated sentiment on bucket boundary.

### XQuik Integration Flow

```
Startup
  └─→ GET /monitors/keywords (list existing)
  └─→ POST /monitors/keywords (create missing ones)
  └─→ monitors are now active (checking X every 1 second)

Every 5 minutes:
  └─→ GET /events?keywordMonitorId=N&limit=100 (per coin)
  └─→ Paginate if hasMore=true
  └─→ For each tweet: FinBERT scoring → SentimentAggregator
  └─→ Aggregator auto-flushes completed 5-min buckets to DB
```

### Temporal Alignment

All data types share the same 5-minute bucket boundaries:
- Trade candles: `trade_candles_5m.bucket`
- Orderbook: `orderbook_snapshots_5m.bucket`
- Sentiment: `tweet_sentiment_5m.bucket`
- CEX flows: `cex_flows_5m.bucket`

This makes it trivial to JOIN across tables for a complete market snapshot:

```sql
SELECT
    t.bucket,
    t.symbol,
    t.close as price,
    t.volume,
    t.net_trade,
    t.vwap,
    o.avg_imbalance as orderbook_imbalance,
    o.avg_spread,
    s.avg_score as sentiment,
    s.tweet_count,
    s.positive_count,
    s.negative_count
FROM trade_candles_5m t
LEFT JOIN orderbook_snapshots_5m o
    ON t.bucket = o.bucket AND t.symbol = o.symbol
LEFT JOIN tweet_sentiment_5m s
    ON t.bucket = s.bucket AND REPLACE(t.symbol, 'USDT', '') = s.symbol
WHERE t.symbol = 'BTCUSDT'
ORDER BY t.bucket DESC
LIMIT 10;
```

## Cost Considerations

### XQuik
- Active keyword monitors: **21 credits/hour each**
- 5 monitors = **105 credits/hour** (~2,520 credits/day)
- Event reading: **free** (included in monitor billing)

### Bitquery
- GraphQL queries: metered per API plan

### Binance
- WebSocket `aggTrade` stream: **free** (no API key required)
- REST orderbook depth: **free** (no API key required)

## Project Structure

```
CryptoSense/
├── src/
│   ├── main.py                          # Orchestrator — starts all pipelines
│   ├── core/
│   │   ├── config/
│   │   │   └── settings.py              # Centralized configuration from .env
│   │   └── utils/
│   │       ├── logging.py               # Structured logging setup
│   │       └── signals.py               # Graceful shutdown signal handling
│   ├── data_sources/
│   │   ├── binancewebsocket/
│   │   │   ├── ws_trades_ingestion.py   # Binance Futures aggTrade WebSocket
│   │   │   └── ws_orderbook_ingestion.py# Orderbook depth REST polling
│   │   ├── xquik/
│   │   │   └── xquik_ingestion.py       # XQuik keyword monitor + event polling
│   │   ├── bitquery/
│   │   │   └── cex_flow_ingestion.py    # Bitquery CEX flow GraphQL polling
│   │   └── tavily/
│   │       └── tavily_ingestion.py      # Legacy Tavily web sentiment (disabled)
│   ├── feature_engineering/
│   │   ├── trade_aggregator.py          # 5-min OHLCV candle aggregation
│   │   ├── orderbook_aggregator.py      # 5-min orderbook metric aggregation
│   │   ├── sentiment_aggregator.py      # 5-min tweet sentiment aggregation
│   │   └── sentiment_scorer.py          # FinBERT model loading & inference
│   ├── sinks/
│   │   └── timescale_sink.py            # Routes data to appropriate aggregators
│   └── db/
│       ├── db.py                        # Connection pool & query helpers
│       └── db_schema.sql                # TimescaleDB schema migration
├── run_migration.py                     # Standalone migration runner
├── check_live_data.py                   # Database inspection utility
├── requirements.txt
├── .env                                 # API keys & config (git-ignored)
└── .gitignore
```

## Tracked Symbols

| Symbol | Pair | Trade Stream | Orderbook | Sentiment | CEX Flows |
|--------|------|:---:|:---:|:---:|:---:|
| BTC | BTCUSDT | ✅ | ✅ | ✅ | ✅ (eth) |
| ETH | ETHUSDT | ✅ | ✅ | ✅ | ✅ (eth, bsc) |
| SOL | SOLUSDT | ✅ | ✅ | ✅ | ✅ (solana) |
| BNB | BNBUSDT | ✅ | ✅ | ✅ | — |
| AVAX | AVAXUSDT | ✅ | ✅ | ✅ | — |

## License

This project is for educational and research purposes.
