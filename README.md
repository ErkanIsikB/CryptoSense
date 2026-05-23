# CryptoSense — Real-Time Crypto Data Pipeline & AI Anomaly Detection Engine

CryptoSense is a high-performance, production-grade data engineering pipeline that ingests, processes, and persists multi-source cryptocurrency market data, sentiment, and on-chain fund flows into TimescaleDB. 

In addition to temporal alignment and real-time ingestion, CryptoSense features a **state-of-the-art AI Anomaly Detection Engine** powered by unsupervised deep learning (LSTM Autoencoders). It analyzes 5-minute bucketed market, sentiment, and wallet transfer snapshots to detect deviations from baseline market dynamics and outputs highly structured payloads optimized for downstream LLM reasoning.

---

## 🏗️ Architecture Overview

```mermaid
graph TD
    %% Ingestion Layer
    subgraph Ingestion ["Ingestion Layer"]
        BWS[Binance WebSocket <br/> Trades & Orderbook]
        XQ[XQuik REST API <br/> Twitter Tweets]
        B_CEX[Bitquery HTTP Poll <br/> 5m CEX Fund Flows]
        B_WS[Bitquery WebSockets <br/> whale_trades, evm, solana]
        B_HTTP[Bitquery HTTP Poll <br/> bnb & avax transfers]
    end

    %% Aggregation & Scoring
    subgraph Processing ["Feature Engineering & ML Layer"]
        TA[TradeAggregator <br/> 5-min OHLCV & Volume]
        OA[OrderbookAggregator <br/> 5-min Average Depth & Spread]
        SA[SentimentAggregator <br/> 5-min FinBERT Sentiment]
        FS[FinBERT Model <br/> scoring compound scores]
        JSONL[Local JSONL Backup <br/> Raw Ingress Files]
    end

    %% Database Layer
    subgraph Storage ["TimescaleDB Database"]
        T_TC[(trade_candles_5m)]
        T_OS[(orderbook_snapshots_5m)]
        T_TS[(tweet_sentiment_5m)]
        T_CF[(cex_flows_5m)]
        T_ANOM[(ai_anomalies_5m)]
    end

    %% AI Anomaly Detection Engine
    subgraph ML_Engine ["AI Anomaly Detection Engine (PyTorch)"]
        AP[Anomaly Pipeline <br/> Runs every 5 mins]
        LSTM[LSTM Autoencoder <br/> 19 Features, 10 Latent Dim]
        SCALER[Dynamic MinMax <br/> Normalizer per coin]
    end

    %% Flows
    BWS --> TA
    BWS --> OA
    XQ --> FS --> SA
    B_CEX --> T_CF
    B_WS --> JSONL
    B_HTTP --> JSONL
    
    TA --> T_TC
    OA --> T_OS
    SA --> T_TS
    
    T_TC --> AP
    T_OS --> AP
    T_TS --> AP
    T_CF --> AP
    
    AP --> SCALER --> LSTM
    LSTM -->|MSE Error > Threshold| AP
    AP -->|Upsert anomaly metrics & LLM payload| T_ANOM

    style ML_Engine fill:#2d1a47,stroke:#953df4,stroke-width:2px,color:#fff
    style Ingestion fill:#1b2a47,stroke:#3d8bf4,stroke-width:1px,color:#fff
    style Storage fill:#1a3c2a,stroke:#3df48b,stroke-width:1px,color:#fff
    style Processing fill:#3a3a1a,stroke:#f4d63d,stroke-width:1px,color:#fff
```

### Dynamic Ingestion & Machine Learning Loops
1. **Pipeline Execution & Thread Offloading**: Live trades (Binance), orderbook snapshot averages, sentiment models (XQuik/FinBERT), and exchange transfers are continuously aggregated into 5-minute buckets and committed directly to TimescaleDB. All synchronous database flushes and heavy deep learning scoring (FinBERT & PyTorch LSTM) are offloaded to background threads and daemon tasks, keeping the `asyncio` event loop 100% fluid and non-blocking.
2. **Dynamic DB CTE Barrier Sync & CEX Flow LEFT JOIN**: To prevent partial or "ghost" data entries from corrupting model normalization, the Anomaly Engine queries using a strict 3-table `INTERSECT` CTE barrier across high-frequency feeds (`trade_candles_5m`, `orderbook_snapshots_5m`, and `tweet_sentiment_5m`). The lower-frequency on-chain CEX flows (`cex_flows_5m`) are joined optionally via a `LEFT JOIN` and wrapped in `COALESCE` to default missing metrics to `0.0`. This ensures absolute timeline alignment without locking out anomaly detection during periods of transfer sparsity (e.g. for AVAX, whose exchange transfers are tracked as wrapped/pegged tokens on BSC and Ethereum).
3. **Thread-Safe Pooling & Teardown Security**: Database operations are protected by thread-safe PgPool async query wrappers guarded by an `asyncio.Semaphore(10)` matched to database pool limits. Teardown lifecycles implement a top-down closing hierarchy with a 1.0s settling grace period and synchronous shutdown flushes to prevent psycopg2 pool errors and preserve final in-RAM data buckets.
4. **AI Anomaly Engine**: Every 5 minutes, the engine queries the database, constructs a chronological 1-hour window (12 sequential 5-minute buckets) of 19 market and sentiment features. It maps database fields explicitly by column name to avoid positional scaling corruption, scales the metrics dynamically via trained MinMax matrices, and inputs them into coin-specific LSTM Autoencoder models.
5. **Statistical Outliers**: Reconstruction loss (MSE) is evaluated against a tuned volatility threshold. Severe errors trigger warnings and generate structured JSON payloads containing localized contextual states for instantaneous consumption by LLM agents.

---

## 📡 Data Extraction & Ingestion

CryptoSense implements a dual-method data extraction pipeline to optimize both real-time ingestion fidelity and API credit conservation.

### 1. Market Data (Binance Futures & REST)
- **Aggregated Trades (`aggTrade`)**: Real-time trade events ingested over WebSockets from Binance Futures (`wss://fstream.binance.com/market/stream`).
  - *Tracked Pairs*: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `AVAXUSDT`.
- **Orderbook Depth**: REST polling of active order book snapshots (top 20 bid/ask levels) every ~2.0 seconds to track depth and compute imbalance indexes.

### 2. Social Sentiment (XQuik & FinBERT)
- **X (Twitter) Monitoring**: Real-time keyword monitoring via the [XQuik](https://xquik.com) platform, checking keyword filters every 1 second server-side.
- **Sentiment Inference**: Events are polled every 5 minutes and run locally through a pipeline-integrated `ProsusAI/FinBERT` Hugging Face model to score sentiment on a `[-1.0, +1.0]` (negative to positive) compound scale.

### 3. On-Chain Exchange & Whale Flow (Bitquery GraphQL v2)
Bitquery integration is heavily optimized using both HTTP polling and subscription mechanisms:
- **CEX Flow Ingestion (`cex_flow_ingestion.py`)**: Runs every 5 minutes using HTTP GraphQL POSTs to compile aggregate exchange inflows and outflows for Ethereum, BSC, and Solana (`CEX_FLOW_NETWORKS=eth,bsc,solana`) using predefined CEX hot-wallet coordinates. Wrapped and pegged AVAX exchange flows (e.g. Binance-Peg AVAX on BSC, WAVAX on Ethereum) are tracked dynamically within the active BSC and Ethereum streams, conserving API limits while preserving model richness.
- **WebSocket Whale Trades (`ws_whale_trades.py`)**: Real-time subscription to DEX trades exceeding $100,000 in volume for our tracked assets.
- **WebSocket Transfer Streams (`ws_evm_transfers.py` & `ws_solana_transfers.py`)**: Real-time subscription to large transfers (> $100,000) for Ethereum/EVM and Solana networks.
- **Optimized Polling (`http_polling.py`)**: Periodic REST polling (5-minute interval) for BSC and Avalanche transfers to bypass WebSocket stream limits and conserve valuable API credits.

All raw streams from Whale WebSockets and polling are safely flushed to structured local JSONL files in the background.

---

## 💾 Database Schema (TimescaleDB)

TimescaleDB manages temporal data alignment seamlessly. All core tables are initialized as **hypertables** with a chunk interval of 1 day and optimized query indices.

### 1. `trade_candles_5m` — OHLCV & Volume Metrics
| Column | Type | Description |
| :--- | :--- | :--- |
| `bucket` 🔑 | TIMESTAMPTZ | 5-minute bucket start timestamp |
| `symbol` 🔑 | TEXT | Cryptocurrency futures symbol (e.g. `BTCUSDT`) |
| `open` / `high` / `low` / `close` | DOUBLE PRECISION | Token trade price metrics in bucket |
| `volume` | DOUBLE PRECISION | Total token trade volume in bucket |
| `quote_volume` | DOUBLE PRECISION | Quote asset volume (USDT) |
| `trade_count` | INTEGER | Number of distinct trades in bucket |
| `buy_volume` / `sell_volume` | DOUBLE PRECISION | Directional buying/selling volumes |
| `net_trade` | DOUBLE PRECISION | Net buyer volume (`buy_volume - sell_volume`) |
| `vwap` | DOUBLE PRECISION | Volume-Weighted Average Price |

### 2. `orderbook_snapshots_5m` — Market Depth Metrics
| Column | Type | Description |
| :--- | :--- | :--- |
| `bucket` 🔑 | TIMESTAMPTZ | 5-minute bucket start timestamp |
| `symbol` 🔑 | TEXT | Trading pair symbol |
| `avg_spread` | DOUBLE PRECISION | Average bid-ask spread in bucket |
| `avg_mid_price` | DOUBLE PRECISION | Average mid price in bucket |
| `avg_bid_depth` / `avg_ask_depth` | DOUBLE PRECISION | Average order volume on bid and ask sides |
| `avg_imbalance` | DOUBLE PRECISION | Average imbalance ratio: `(bid - ask) / (bid + ask)` |
| `snapshot_count` | INTEGER | Total book snapshots captured in bucket |

### 3. `tweet_sentiment_5m` — X/Twitter Sentiment Metrics
| Column | Type | Description |
| :--- | :--- | :--- |
| `bucket` 🔑 | TIMESTAMPTZ | 5-minute bucket start timestamp |
| `symbol` 🔑 | TEXT | Unified token symbol (e.g. `BTC`) |
| `avg_score` | DOUBLE PRECISION | Average FinBERT score `[-1, +1]` |
| `tweet_count` | INTEGER | Total scored tweets matching keywords |
| `positive_count` | INTEGER | Tweets with compound score > `+0.1` |
| `negative_count` | INTEGER | Tweets with compound score < `-0.1` |
| `neutral_count` | INTEGER | Tweets scoring between `-0.1` and `+0.1` |
| `max_score` / `min_score` | DOUBLE PRECISION | Extremes of FinBERT scores observed in bucket |
| `sample_tweet` | TEXT | Text of the tweet with highest community engagement |

### 4. `cex_flows_5m` — Exchange Fund Flow Metrics
| Column | Type | Description |
| :--- | :--- | :--- |
| `bucket` 🔑 | TIMESTAMPTZ | 5-minute bucket start timestamp |
| `symbol` 🔑 | TEXT | Unified token symbol (e.g. `ETH`) |
| `network` 🔑 | TEXT | Blockchain network (e.g. `ethereum`, `bsc`, `solana`) |
| `inflow_amount` 🆕 | DOUBLE PRECISION | Cumulative volume of tokens moving into CEX wallets |
| `inflow_usd` | DOUBLE PRECISION | Cumulative USD value of CEX inflows |
| `outflow_amount` 🆕 | DOUBLE PRECISION | Cumulative volume of tokens moving out of CEX wallets |
| `outflow_usd` | DOUBLE PRECISION | Cumulative USD value of CEX outflows |
| `net_flow_usd` | DOUBLE PRECISION | Net flow in USD (`inflow_usd - outflow_usd`) |
| `inflow_tx_count` / `outflow_tx_count` | INTEGER | Transaction counts per inflow/outflow direction |

### 5. `ai_anomalies_5m` 🆕 — Deep Learning Engine Outputs
| Column | Type | Description |
| :--- | :--- | :--- |
| `bucket` 🔑 | TIMESTAMPTZ | 5-minute bucket start timestamp |
| `symbol` 🔑 | TEXT | Base asset symbol (e.g. `BTC`) |
| `mse_score` | DOUBLE PRECISION | Mean Squared Error (reconstruction loss) from Autoencoder |
| `is_anomaly` | BOOLEAN | `TRUE` if `mse_score` exceeds threshold (`0.008`) |
| `severity` | TEXT | Severity ranking (`HIGH` if `mse_score > threshold * 2`, else `NORMAL`) |
| `llm_payload` | JSONB | Complete JSON package ready for LLM consumption and reasoning |

---

## 🧠 AI Anomaly Detection Engine

The AI Anomaly Detection Engine monitors the pipeline for multi-dimensional anomalies (price fluctuations, abrupt orderbook depth changes, sentiment shifts, and high-volume whale transfers) simultaneously.

### 1. Deep Learning Model Architecture
- **Model Type**: Unsupervised LSTM Autoencoder (`src/models/lstm_autoencoder.py`).
- **Layers**:
  - **Encoder**: An LSTM layer mapping input temporal features into a lower-dimensional latent space.
  - **Decoder**: An LSTM layer that takes the latent vector, repeats it over the sequence length, reconstructs the inputs, and maps them back to the original dimensions via a Linear projection.
- **Input Dimensions**: Sequence Length = `12` (representing exactly 1 hour of 5-minute buckets). Feature Dimensions = `19` (covering price metrics, volumes, spread, bid/ask depth, social sentiment, tweet counts, and net USD CEX flows).
- **Latent Bottleneck**: `10` dimensions (`LATENT_DIM = 10`).

### 2. Normalization & Inference Pipeline (`src/models/anomaly_pipeline.py`)
- **Startup**: Dynamically loads PyTorch models (`lstm_autoencoder_<symbol>.pt`) and companion MinMax scaler files (`scaler_params_<symbol>.json`) for all monitored assets into RAM.
- **Temporal Check**: Ensures sequential continuity. If any downtime gap (> 5 minutes) is discovered within the current 12-bucket window, inference is skipped to prevent invalid statistical projections.
- **Inference Execution**: Wakes up every 5 minutes, constructs the normalized sequence using the loaded coin-specific scaler parameters, and runs PyTorch model evaluations to compute reconstruction error (MSE).
- **Threshold Matching**: Classified as an anomaly if the MSE reconstruction loss exceeds the volatility baseline of `0.008`.
- **Database Upsert**: Saves anomaly statuses, scores, and writes a detailed JSON payload (`llm_payload`) containing structured statistical context directly to `ai_anomalies_5m`.

### 3. LLM-Ready Payload Structure
When an anomaly is flagged, the JSON payload in `llm_payload` takes the following format, ready for direct injection into AI agent prompts:

```json
{
  "timestamp": "2026-05-21T19:45:00Z",
  "symbol": "BTCUSDT",
  "market_data": {
    "close_price": 71250.00,
    "volume_5m": 345.20,
    "vwap": 71245.50,
    "net_trade": 12.50
  },
  "orderbook": {
    "avg_spread": 0.0500,
    "avg_imbalance": 0.452,
    "bid_depth": 1250000.00,
    "ask_depth": 850000.00
  },
  "sentiment": {
    "avg_score": -0.420,
    "tweet_count": 89,
    "positive_count": 15,
    "negative_count": 52
  },
  "on_chain": {
    "net_cex_flow_usd": -125000.00
  },
  "AI_ENGINE": {
    "reconstruction_error": 0.018542,
    "is_statistical_anomaly": true,
    "severity": "HIGH"
  }
}
```

---

## 🛠️ Project Structure

```
CryptoSense/
├── main.py                           # Unified system orchestrator (starts all pipelines)
├── requirements.txt                  # Python dependencies
├── .env                              # System settings & secret keys (git-ignored)
├── .gitignore
├── README.md                         # Project documentation
├── scripts/                          # Utility & Diagnostics Suite
│   ├── check_live_data.py            # Diagnostic script to print the latest DB entries
│   ├── clean_jsonl.py                # Local JSONL file cleanup helper
│   ├── cleanup_test_data.py          # Development cleanup utility
│   ├── inspect_payload.py            # Utility to pretty-print the latest live anomaly LLM payload from TimescaleDB
│   ├── run_migration.py              # Standalone migration script
│   ├── test_bitquery_usage.py        # Connection and query validator for Bitquery APIs
│   ├── test_integration.py           # Pipelines integration test pushing mock data
│   ├── train_anomaly_detector.py     # Unsupervised model training on TimescaleDB data
│   └── verify_db.py                  # Standalone verification script for db hypertables
└── src/                              # Core Application Codebase
    ├── __init__.py
    ├── core/
    │   ├── config/
    │   │   └── settings.py           # Dotenv configuration parser
    │   └── utils/
    │       ├── logging.py            # Color-coded logging configuration
    │       └── signals.py            # Graceful shutdown handler
    ├── data_sources/                 # Ingestion Drivers
    │   ├── binancewebsocket/
    │   │   ├── ws_trades_ingestion.py    # Binance Futures trades stream (aggTrade)
    │   │   └── ws_orderbook_ingestion.py # Binance orderbook snapshots poller
    │   ├── bitquery/                     # Bitquery integration module
    │   │   ├── cex_addresses.py          # Known CEX hot-wallets & smart contract keys
    │   │   ├── cex_flow_ingestion.py     # 5-min CEX inflows/outflows aggregation poller
    │   │   ├── http_polling.py           # Conserves limits by polling BSC & AVAX transfers
    │   │   ├── ws_evm_transfers.py       # ETH/EVM transfers WebSocket subscription
    │   │   ├── ws_solana_transfers.py    # Solana transfers WebSocket subscription
    │   │   └── ws_whale_trades.py        # DEX whale trades WebSocket subscription
    │   └── xquik/
    │       └── xquik_ingestion.py        # Keyword polling & sentiment orchestration
    ├── feature_engineering/          # Aggregation Engines
    │   ├── trade_aggregator.py       # Computes OHLCV, VWAP, buy/sell ratios
    │   ├── orderbook_aggregator.py   # Computes spread averages, depths, and imbalances
    │   ├── sentiment_aggregator.py   # Computes compound score distribution metrics
    │   └── sentiment_scorer.py       # FinBERT sentiment scoring implementation
    ├── models/                       # Deep Learning Engine
    │   ├── lstm_autoencoder.py       # PyTorch LSTM Autoencoder architecture
    │   ├── anomaly_pipeline.py       # Real-time anomaly inference pipeline
    │   └── saved_weights/            # Model parameters directory
    │       ├── lstm_autoencoder_*.pt # PyTorch model weights (git-ignored)
    │       └── scaler_params_*.json  # Scaler configuration files
    ├── db/                           # TimescaleDB Layer
    │   ├── db.py                     # Thread-safe PgPool connector
    │   └── db_schema.sql             # SQL migrations setup (Hypertables & indexes)
    └── sinks/                        # Sink Router Layer
        ├── base.py                   # Base interface
        ├── jsonl_sink.py             # Backup JSONL file sink
        └── timescale_sink.py         # Primary aggregator-routed DB sink
```

---

## 🚀 Setup & Execution

### 1. Install System Prerequisites
- **Python Version**: `Python 3.11+`
- **Database**: Active [TimescaleDB](https://www.timescale.com/) instance (e.g. Timescale Cloud).

### 2. Clone Repository & Setup Environment
```bash
git clone <repository-url>
cd CryptoSense

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configure `.env`
Create a `.env` file in the project root:

```env
# ── TimescaleDB Connection DSN ─────────────────────────
DB_URL=postgres://tsdb_user:tsdb_password@host:port/tsdb?sslmode=require

# ── XQuik API Credentials (Social Sentiment) ───────────
XQUIK_API=xq_your_xquik_key

# ── Bitquery API Keys (On-chain Fund Flows) ────────────
BITQUERY_API_KEY=your_bitquery_key

# ── Symbol & Network Configurations ────────────────────
BINANCE_SYMBOLS=btcusdt,ethusdt,solusdt,bnbusdt,avaxusdt
CEX_FLOW_NETWORKS=eth,bsc,solana

# ── Log Settings ───────────────────────────────────────
LOG_LEVEL=INFO
```

---

## 🏃 Operation Guide

### 1. Database Migrations
Run the schema setup script to initialize TimescaleDB hypertables, custom index structures, and the AI anomalies table. Safe to run multiple times:
```bash
python -m scripts.run_migration
```

### 2. Start the Pipeline Orchestrator
Start the main program. This connects to Binance futures, starts REST polls, initiates WebSocket whale subscriptions, registers XQuik keyword monitors, and runs the AI Anomaly Engine in the background:
```bash
python main.py
```

### 3. Model Training Sequence
To train or update the unsupervised LSTM Autoencoder on clean historical sequences extracted directly from your TimescaleDB instance:
```bash
python -m scripts.train_anomaly_detector
```
> **Note:** The script will automatically fit custom MinMax normalizers, save them as JSON parameters, optimize the LSTM network weights using an MSE criteria over 100 epochs, and export all assets directly to `src/models/saved_weights/` for instant live reload.

### 4. Diagnostics & Live Inspections
You can inspect the state of your database schemas, integration aggregates, and active records at any time:
- **Test Ingestion Flow (Mock Data)**:
  ```bash
  python -m scripts.test_integration
  ```
- **Database Hypertables Diagnostics**:
  ```bash
  python -m scripts.verify_db
  ```
- **Inspect Live Table Outputs**:
  ```bash
  python -m scripts.check_live_data
  ```

---

## 📊 Temporal Alignment & Cross-Table JOINs

Because all 5-minute ingestion buckets share identical boundaries, joining metrics for modeling or downstream analytics is simplified:

```sql
SELECT
    t.bucket,
    t.symbol,
    t.close AS last_price,
    t.volume AS trade_volume,
    o.avg_imbalance AS orderbook_imbalance,
    s.avg_score AS sentiment_score,
    s.tweet_count AS total_tweets,
    COALESCE(c.net_flow_usd, 0.0) AS net_wallet_flow,
    a.is_anomaly AS ai_anomaly_detected,
    a.mse_score AS autoencoder_loss
FROM trade_candles_5m t
LEFT JOIN orderbook_snapshots_5m o
    ON t.bucket = o.bucket AND t.symbol = o.symbol
LEFT JOIN tweet_sentiment_5m s
    ON t.bucket = s.bucket AND REPLACE(t.symbol, 'USDT', '') = s.symbol
LEFT JOIN (
    SELECT bucket, symbol, SUM(net_flow_usd) AS net_flow_usd
    FROM cex_flows_5m
    GROUP BY bucket, symbol
) c
    ON t.bucket = c.bucket AND REPLACE(t.symbol, 'USDT', '') = c.symbol
LEFT JOIN ai_anomalies_5m a
    ON t.bucket = a.bucket AND REPLACE(t.symbol, 'USDT', '') = a.symbol
WHERE t.symbol = 'BTCUSDT'
ORDER BY t.bucket DESC
LIMIT 10;
```

---

## 📈 Asset Matrix

| Symbol | Pair | Trades Stream | Orderbook | Sentiment | CEX Flow Network | AI Anomaly Monitored |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **BTC** | `BTCUSDT` | ✅ | ✅ | ✅ | `ethereum` | ✅ |
| **ETH** | `ETHUSDT` | ✅ | ✅ | ✅ | `ethereum`, `bsc` | ✅ |
| **SOL** | `SOLUSDT` | ✅ | ✅ | ✅ | `solana` | ✅ |
| **BNB** | `BNBUSDT` | ✅ | ✅ | ✅ | `bsc` | ✅ |
| **AVAX** | `AVAXUSDT` | ✅ | ✅ | ✅ | `bsc`, `ethereum` (wrapped) | ✅ |

---

## 💳 Credit & Ingestion Considerations

- **XQuik Keyword Billing**: Active monitors consume **21 credits/hour each** (105 credits/hour total across 5 tracked symbols). Event polling itself is free.
- **Bitquery Billing**: WebSockets and HTTP GraphQL requests consume credits according to your Bitquery Developer plan. Polling intervals for BSC/AVAX transfers are throttled to 5 minutes to keep credit usage efficient.
- **Binance WebSocket Ingestion**: Zero-cost, zero-API key required.

---

## 📄 License

This repository is maintained for research, analytical model development, and educational purposes. All deep learning and quantitative code is provided as-is.
