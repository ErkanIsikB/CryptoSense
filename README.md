# CryptoSense — Real-Time Crypto Data Pipeline & AI Anomaly Detection Engine

CryptoSense is a high-performance, production-grade data engineering pipeline that ingests, processes, and persists multi-source cryptocurrency market data, sentiment, and on-chain fund flows into TimescaleDB. 

In addition to temporal alignment and real-time ingestion, CryptoSense features a **state-of-the-art AI Anomaly Detection Engine** powered by unsupervised deep learning (LSTM Autoencoders) and an **automated LLM Decision Engine** (running structured Qwen 2.5 local models via Ollama) to output qualitative market diagnostics.

---

## 🏗️ Architecture Overview

```mermaid
graph TD
    %% Ingestion Layer
    subgraph Ingestion ["Ingestion Layer"]
        BWS[Binance WebSocket <br/> Trades & Orderbook]
        XQ[XQuik REST API <br/> Twitter Tweets]
        RSS[RSS Feeds <br/> Tier-1 Crypto News]
        B_CEX[Bitquery HTTP Poll <br/> 5m CEX Fund Flows]
        B_WS[Bitquery WebSockets <br/> whale_trades, evm, solana]
        B_HTTP[Bitquery HTTP Poll <br/> bnb & avax transfers]
    end

    %% Aggregation & Scoring
    subgraph Processing ["Feature Engineering & ML Layer"]
        TA[TradeAggregator <br/> 5-min OHLCV & Volume]
        OA[OrderbookAggregator <br/> 5-min Average Depth & Spread]
        SA[SentimentAggregator <br/> 5-min Tweet Sentiment]
        NA[News Ingestion Scorer <br/> 5-min RSS News Sentiment]
        CB[CryptoBERT Model <br/> Tweet Sentiment Scoring]
        FB[FinBERT Model <br/> News Sentiment Scoring]
    end

    %% Database Layer
    subgraph Storage ["TimescaleDB Database"]
        T_TC[(trade_candles_5m)]
        T_OS[(orderbook_snapshots_5m)]
        T_TS[(tweet_sentiment_5m)]
        T_NS[(news_sentiment_5m)]
        T_CF[(cex_flows_5m)]
        T_ANOM[(ai_anomalies_5m)]
        T_LLM[(llm_health_scores)]
    end

    %% AI Anomaly Detection & LLM Engines
    subgraph ML_Engine ["AI Anomaly & Decision Engines"]
        AP[Anomaly Pipeline <br/> Runs every 5 mins]
        LSTM[LSTM Autoencoder <br/> 18 Features, 10 Latent Dim]
        SCALER[Dynamic MinMax Normalizer]
        
        LLM[LLM Decision Stream <br/> Runs every 5 mins]
        QWEN[Qwen 2.5 7B Model <br/> Local Ollama Client]
        
        RS[Retraining Service <br/> train_symbol_model]
        SCHED[APScheduler Daemon <br/> every 14 days]
    end

    %% User Facing Layer
    subgraph UserInterface ["Presentation & API Layer"]
        API[FastAPI Backend REST API <br/> Port 8000]
        DASH[Streamlit Dashboard Grid <br/> Port 8501]
    end

    %% Flows
    BWS --> TA
    BWS --> OA
    XQ --> CB --> SA
    RSS --> FB --> NA
    B_CEX --> T_CF
    B_WS --> Storage
    B_HTTP --> Storage
    
    TA --> T_TC
    OA --> T_OS
    SA --> T_TS
    NA --> T_NS
    
    T_TC --> AP
    T_OS --> AP
    T_TS --> AP
    T_NS --> AP
    T_CF --> AP
    
    AP --> SCALER --> LSTM
    LSTM -->|MSE Error > Dynamic Youden J Threshold| AP
    AP -->|Upsert anomaly metrics & LLM payload| T_ANOM
    
    T_ANOM -->|Chronological 12-candle trigger| LLM
    LLM --> QWEN -->|Upsert market health scores| T_LLM
    
    SCHED -->|Trigger periodic training| RS
    RS -->|Optimize weights & Hot-swap| LSTM
    RS -->|Update mins/maxs| SCALER
    
    T_TC & T_OS & T_TS & T_NS & T_CF & T_ANOM & T_LLM --> API
    API --> DASH

    style ML_Engine fill:#2d1a47,stroke:#953df4,stroke-width:2px,color:#fff
    style Ingestion fill:#1b2a47,stroke:#3d8bf4,stroke-width:1px,color:#fff
    style Storage fill:#1a3c2a,stroke:#3df48b,stroke-width:1px,color:#fff
    style Processing fill:#3a3a1a,stroke:#f4d63d,stroke-width:1px,color:#fff
    style UserInterface fill:#1e3c3c,stroke:#20b2aa,stroke-width:1px,color:#fff
```

### Dynamic Ingestion & Machine Learning Loops
1. **Pipeline Execution & Thread Offloading**: Live trades (Binance), orderbook snapshot averages, sentiment models (XQuik/CryptoBERT for tweets, FinBERT for news), and exchange transfers are continuously aggregated into 5-minute buckets and committed directly to TimescaleDB. Heavy computations—like CryptoBERT/FinBERT text scoring, PyTorch forward-passes, and database batch flushes—are offloaded to background threads (`asyncio.to_thread` / `threading.Thread`), keeping the event loop 100% fluid.
2. **Dynamic DB CTE Barrier Sync & CEX Flow LEFT JOIN**: To eliminate timing drift and prevent "ghost" data entries from corrupting model normalization, the Anomaly Engine queries using a strict 3-table `INTERSECT` CTE barrier across high-frequency feeds (`trade_candles_5m`, `orderbook_snapshots_5m`, and `tweet_sentiment_5m`). The lower-frequency on-chain CEX flows (`cex_flows_5m`) are joined optionally via a `LEFT JOIN` and wrapped in `COALESCE` to default missing metrics to `0.0`.
3. **Thread-Safe Pooling & Teardown Security**: Database operations are protected by thread-safe PgPool async query wrappers guarded by an `asyncio.Semaphore(10)` matched to database pool limits. Teardown lifecycles implement a top-down closing hierarchy with a 1.0s settling grace period and synchronous shutdown flushes to prevent psycopg2 pool errors.
4. **Deduplicated Schemas & Shared ML Calculus**: The codebase enforces a single source of truth for SQL projections using a public `SQL_COLUMNS` constant in `anomaly_pipeline.py`. Model evaluation calculations and Youden statistics are centralized into `run_evaluation_inference` and `calculate_optimal_threshold` within `retraining_service.py` to prevent logic duplication.
5. **AI Anomaly Engine & Calibration**: Every 5 minutes, the engine queries the database, constructs a chronological 1-hour window (12 sequential 5-minute buckets) of 18 quantitative market and sentiment features (excluding the bucket timestamp). Features are scaled dynamically via trained MinMax matrices and analyzed using coin-specific LSTM Autoencoder models.
   * *Youden's J Optimization*: Model threshold calibrations utilize Youden's J statistic ($J = \text{TPR} - \text{FPR}$) under a tightened proxy outlier Z-score threshold of `3.0` (Three-Sigma). This achieves optimal sensitivity/recall (**71% to 91%**, averaging 80% to meet target requirements) while maintaining low False Alarm Rates (**18% to 28%**).
6. **LLM Decision Engine**: Every 5 minutes (offset by +35 seconds to allow models to write), the LLM engine polls `ai_anomalies_5m` for the latest candles. It reverse-sorts them into chronological order, strips anomaly tags from historical candles to preserve timeline purity, formats the raw sequence into a contextual prompt, and fires structured JSON queries to a local Qwen 2.5 instance via Ollama.
7. **APScheduler Retraining Loop**: If enabled (`RETRAIN_ENABLED=true`), an automated scheduler daemon starts on startup. Every 14 days, it triggers a background retraining service that pulls the past 14 days of clean records from TimescaleDB, isolates continuous 1-hour sequence blocks, trains the LSTM Autoencoder over 100 epochs, writes out new weights atomically, and hot-swaps the memory registry of the running pipeline without requiring a service reboot.

---

## 📡 Data Extraction & Ingestion

CryptoSense implements a dual-method data extraction pipeline to optimize both real-time ingestion fidelity and API credit conservation.

### 1. Market Data (Binance Futures & REST)
- **Aggregated Trades (`aggTrade`)**: Real-time trade events ingested over WebSockets from Binance Futures (`wss://fstream.binance.com/market/stream`).
  - *Tracked Pairs*: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `AVAXUSDT`.
- **Orderbook Depth**: REST polling of active order book snapshots (top 20 bid/ask levels) every ~2.0 seconds to track depth and compute imbalance indexes.

### 2. Social & News Sentiment (XQuik, RSS & Dual-Model Scoring)
- **X (Twitter) Monitoring**: Real-time keyword monitoring via the [XQuik](https://xquik.com) platform, checking keyword filters every 1 second server-side.
- **Institutional News Ingestion (RSS)**: Background polling of Tier-1 crypto news feeds (CoinDesk, CoinTelegraph, CryptoPanic, CryptoSlate, NewsBTC) every 5 minutes.
- **Dual-Model Sentiment Inference**: CryptoSense uses a dual-model architecture for optimal accuracy across different text domains:
  - **CryptoBERT** (`ElKulako/cryptobert`) — pre-trained on 3.2M crypto tweets, fine-tuned on ~2M StockTwits posts. Used for scoring informal, slang-heavy social-media text (XQuik tweets). Labels: *Bullish*, *Bearish*, *Neutral* (normalised to *positive*, *negative*, *neutral*).
  - **FinBERT** (`ProsusAI/finbert`) — trained on Reuters financial news. Used for scoring formal, well-structured text such as RSS news headlines and earnings reports. Labels: *positive*, *negative*, *neutral*.
  - Both models are loaded lazily on first use and produce a compound score in `[-1.0, +1.0]` via `score = p(positive) − p(negative)`. Batch inference is used to maximize GPU/CPU utilization.
- **English Language Filter**: Before scoring, tweets are passed through a `langdetect`-based English filter (`is_english()`). The filter first strips language-neutral noise (URLs, mentions, hashtags, cashtags) to improve detection accuracy on short/noisy social-media text. Non-English tweets are dropped to prevent garbage model predictions.
- **Off-Topic Tweet Filter**: Keyword monitors inevitably match generic "crypto news" tweets that append a wall of coin tags (e.g. `#BTC #ETH #SOL #DOGE #crypto`) to content unrelated to the tracked symbol. The `_is_offtopic_news_tweet()` filter strips tags/links/mentions from the text and checks whether the remaining prose genuinely mentions the tracked coin. Multi-coin tag blasts and pure tag-and-link spam are discarded.
- **Source-Weighted Aggregation**: Each scored tweet is resolved against `src/core/config/twitter_source_tiers.json`. Known institutional, protocol, founder, and economy-news accounts receive moderate source weights, while unknown accounts remain included with weight `1.0`. The database preserves the raw `avg_score` and stores `weighted_avg_score` separately for comparison and debugging.

#### Source Credibility Weighting
CryptoSense weights sentiment sources because a verified institutional news post should influence the 5-minute aggregate more than a random hype post, without erasing broader market/community sentiment. The resolver in `src/feature_engineering/source_credibility.py` normalizes handles case-insensitively and accepts values with or without `@`.

Default weights are intentionally moderate: `tier1=3.0`, `economy_news_sources=2.75`, `tier2=2.0`, `turkish_economy_sources=1.75`, `tier3=1.25`, and `unknown=1.0`. Unknown or missing handles are never filtered out. They are assigned `source_tier="unknown"` and remain part of both raw and weighted calculations.

The weighted score is calculated as:

```text
weighted_avg_score = sum(raw_sentiment_score * source_weight) / sum(source_weight)
```

Update source tiers by editing `src/core/config/twitter_source_tiers.json`; no sentiment model changes are required. Run `python3 scripts/validate_weighted_sentiment.py` to verify the sample calculation from the capstone prompt.

For a concise implementation write-up, see `docs/weighted_sentiment_implementation_report.md`.

### 3. On-Chain Exchange & Whale Flow (Bitquery GraphQL v2)
Bitquery integration is heavily optimized using both HTTP polling and subscription mechanisms:
- **CEX Flow Ingestion (`cex_flow_ingestion.py`)**: Runs every 5 minutes using HTTP GraphQL POSTs to compile aggregate exchange inflows and outflows for Ethereum, BSC, and Solana (`CEX_FLOW_NETWORKS=eth,bsc,solana`) using predefined CEX hot-wallet coordinates. Wrapped and pegged AVAX exchange flows (e.g. Binance-Peg AVAX on BSC, WAVAX on Ethereum) are tracked dynamically within the active BSC and Ethereum streams, conserving API limits while preserving model richness.
- **WebSocket Whale Trades (`ws_whale_trades.py`)**: Real-time subscription to DEX trades exceeding $100,000 in volume for our tracked assets.
- **WebSocket Transfer Streams (`ws_evm_transfers.py` & `ws_solana_transfers.py`)**: Real-time subscription to large transfers (> $100,000) for Ethereum/EVM and Solana networks.
- **Optimized Polling (`http_polling.py`)**: Periodic REST polling (5-minute interval) for BSC and Avalanche transfers to bypass WebSocket stream limits and conserve valuable API credits.

All raw streams from Whale WebSockets and polling are safely persisted in the background.

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
| `avg_score` | DOUBLE PRECISION | Average CryptoBERT score `[-1, +1]` |
| `tweet_count` | INTEGER | Total scored tweets matching keywords |
| `positive_count` | INTEGER | Tweets with compound score > `+0.1` |
| `negative_count` | INTEGER | Tweets with compound score < `-0.1` |
| `neutral_count` | INTEGER | Tweets scoring between `-0.1` and `+0.1` |
| `max_score` / `min_score` | DOUBLE PRECISION | Extremes of CryptoBERT scores observed in bucket |
| `sample_tweet` | TEXT | Text of the tweet with highest community engagement |
| `weighted_avg_score` | DOUBLE PRECISION | Source-credibility weighted average sentiment |
| `total_source_weight` | DOUBLE PRECISION | Sum of source weights used in the weighted average |
| `tier1_count` / `tier2_count` / `tier3_count` | INTEGER | Count of scored tweets matched to crypto source tiers |
| `economy_news_count` / `turkish_economy_count` | INTEGER | Count of scored tweets matched to economy/news source tiers |
| `unknown_count` | INTEGER | Count of included tweets with missing or non-listed source handles |

### 3.5. `news_sentiment_5m` — RSS News Sentiment Metrics
| Column | Type | Description |
| :--- | :--- | :--- |
| `bucket` 🔑 | TIMESTAMPTZ | 5-minute bucket start timestamp |
| `symbol` 🔑 | TEXT | Unified token symbol (e.g. `BTC`) |
| `avg_score` | DOUBLE PRECISION | Average FinBERT score `[-1, +1]` |
| `news_count` | INTEGER | Total scored news articles in bucket |
| `positive_count` | INTEGER | Articles with compound score > `+0.2` |
| `negative_count` | INTEGER | Articles with compound score < `-0.2` |
| `neutral_count` | INTEGER | Articles scoring between `-0.2` and `+0.2` |
| `sample_headline` | TEXT | Combined source, title, and description snippet |

### 4. `cex_flows_5m` — Exchange Fund Flow Metrics
| Column | Type | Description |
| :--- | :--- | :--- |
| `bucket` 🔑 | TIMESTAMPTZ | 5-minute bucket start timestamp |
| `symbol` 🔑 | TEXT | Unified token symbol (e.g. `ETH`) |
| `network` 🔑 | TEXT | Blockchain network (e.g. `ethereum`, `bsc`, `solana`) |
| `inflow_amount` | DOUBLE PRECISION | Cumulative volume of tokens moving into CEX wallets |
| `inflow_usd` | DOUBLE PRECISION | Cumulative USD value of CEX inflows |
| `outflow_amount` | DOUBLE PRECISION | Cumulative volume of tokens moving out of CEX wallets |
| `outflow_usd` | DOUBLE PRECISION | Cumulative USD value of CEX outflows |
| `net_flow_usd` | DOUBLE PRECISION | Net flow in USD (`inflow_usd - outflow_usd`) |
| `inflow_tx_count` / `outflow_tx_count` | INTEGER | Transaction counts per inflow/outflow direction |

### 5. `ai_anomalies_5m` — Deep Learning Engine Outputs
| Column | Type | Description |
| :--- | :--- | :--- |
| `bucket` 🔑 | TIMESTAMPTZ | 5-minute bucket start timestamp |
| `symbol` 🔑 | TEXT | Base asset symbol (e.g. `BTC`) |
| `mse_score` | DOUBLE PRECISION | Mean Squared Error (reconstruction loss) from Autoencoder |
| `is_anomaly` | BOOLEAN | `TRUE` if `mse_score` exceeds dynamically calibrated Youden's J threshold from scaler.json |
| `severity` | TEXT | Severity ranking (`HIGH` if `mse_score > threshold * 2`, else `NORMAL`) |
| `llm_payload` | JSONB | Complete JSON package ready for LLM consumption and reasoning |

### 6. `llm_health_scores` — Qualitative LLM Briefs
| Column | Type | Description |
| :--- | :--- | :--- |
| `bucket` 🔑 | TIMESTAMPTZ | 5-minute bucket start timestamp |
| `symbol` 🔑 | TEXT | Unified token symbol (e.g. `BTC`) |
| `health_score` | INTEGER | Qualitative health rating `[0, 100]` computed by local Qwen LLM |
| `reasoning` | TEXT | Primary driver metadata & trustworthiness classification header |
| `explanation` | TEXT | Structured 3-sentence quantitative summary of vectors shift |
| `model_name` | TEXT | Local LLM model identifier (e.g., `qwen2.5:7b`) |
| `latency_ms` | INTEGER | Time taken in milliseconds to run structured inference |
| `input_payload` | JSONB | The chronological 12-candle sequence payload used as LLM context |

---

## 🧠 AI Anomaly & LLM Decision Engines

### 1. PyTorch LSTM Autoencoder
* **Architecture**: Sequence length of `12` (exactly 1 hour of history) and a features dimension of `18` (covering price, depth, spread, volume, on-chain flows, and CryptoBERT/FinBERT sentiment, excluding the bucket timestamp). Hidden dimension bottleneck is `10` (`LATENT_DIM = 10`).
* **Unsupervised Anomaly Detection**: Minimizes reconstruction loss (MSE). A reconstruction error exceeding the dynamically calibrated Youden's J threshold (e.g., 0.003071 with a verified AUC of 0.77621 for AVAXUSDT, serialized in scaler_params_*.json) registers as a statistical anomaly.

### 2. Scheduled Retraining Daemon
* Scheduled retraining runs continuously in the background via `APScheduler` ID `retrain_job` if `RETRAIN_ENABLED=true` is set.
* Generates sliding continuous sequence windows from historical DB hypertables, trains a fresh model, writes parameters atomically, and dynamically updates the running in-memory model registry (`ModelRegistry`) using zero-downtime hot-swapping.

### 3. Deterministic Health Score & Local Ollama LLM Briefing
* **Deterministic Health Score (0-100)**: Calculated in Python before LLM inference to ensure absolute numerical reproducibility:
  * **Market Imbalance (30% weight, [-15, +15])**: Computed from the orderbook bid/ask depth ratio.
  * **Blended Sentiment (40% weight, [-20, +20])**: Combines **70% Institutional RSS news** and **30% Retail Twitter (XQuik)** to provide an anti-manipulation sentiment filter.
  * **On-Chain Flow (30% weight, [-15, +15])**: Hyperbolic tangent (`tanh`) scaling dynamically maps CEX net flow USD:
    $$\text{Flow Impact} = -15.0 \times \tanh\left(\frac{\text{Net CEX Flow USD}}{2,500,000.0}\right)$$
  * **Dynamic Anomaly Impact (range: -25.0 to +12.5)**: Triggered by the LSTM Autoencoder's reconstruction error (MSE) relative to the threshold:
    * *Bullish Anomaly*: A dynamic volatility boost of up to $+12.5$ points if price is above VWAP, net trade is positive, or sentiment is bullish.
    * *Bearish Anomaly*: A dynamic penalty of up to $-25.0$ points if indicators point downward.
* **Ollama structured output**: The calculated score and 12-candle sequence are passed to **Ollama** running `qwen2.5:7b`. It enforces strict, schema-locked token outputs mapped directly to the `CryptoSenseBrief` structure:
  ```python
  class CryptoSenseBrief(BaseModel):
      primary_metric_driver: Literal["volume_spike", "liquidity_flight", "sentiment_shift", "on_chain_whale_flow", "none"]
      market_trajectory_summary: str    # Strict factual 3-sentence quantitative summary explaining the health score
      trustworthiness_classification: Literal["HIGH_CONVICTION", "LOW_TRUST_SPECULATIVE", "LIQUIDITY_EXHAUSTION", "STABLE_BASELINE"]
  ```

---

## 🛠️ Project Structure

```
CryptoSense/
├── main.py                           # Unified system orchestrator (starts all pipelines)
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Multi-stage optimized application container definition
├── docker-compose.yml                # Composition layers (ingestion-pipeline, web-api, dashboard)
├── docker-compose.override.yml       # Local GPU Hardware Acceleration Override (git-ignored)
├── .env                              # User configurations & API keys
├── README.md                         # Project documentation
├── test_report.md                    # Generated automated testing report summary
├── scripts/                          # Utility & Diagnostics Suite
│   ├── run_all_tests.py              # Test runner executing unit & transaction-isolated tests
│   ├── check_live_data.py            # Diagnostic script to print the latest DB entries
│   ├── evaluate_roc_thresholds.py    # Offline ROC & AUC threshold calibrator (Phase 3)
│   ├── inspect_payload.py            # Pretty-prints latest anomaly LLM payload from TimescaleDB
│   ├── run_migration.py              # Standalone migration script
│   ├── simulate_anomaly.py           # Anomaly and LLM pipeline simulator (Phase 3)
│   ├── train_anomaly_detector.py     # Unsupervised model training on TimescaleDB data
│   ├── validate_weighted_sentiment.py # Small weighted sentiment validation example
│   └── verify_db.py                  # Standalone verification script for TimescaleDB tables
├── tests/                            # Automated Testing Suite
│   ├── test_integration.py           # Transaction-isolated (ROLLBACK) database integration test
│   ├── test_lstm_autoencoder.py      # LSTM Autoencoder architecture and gradient check
│   ├── test_data_processing.py       # Data sliding windows and label alignment check
│   ├── test_sentiment_scorer.py      # Dual-model (CryptoBERT + FinBERT) scorer and F1 check (>0.75)
│   ├── test_xquick.py                # XQuik live API tweet collection with English & off-topic filters
│   ├── test_xquik_filtering.py       # Off-topic tweet filter unit tests
│   ├── test_timescale_sink.py        # TimescaleSink routing adapter checks
│   ├── test_signals.py               # Unix/Windows graceful shutdown signal handler checks
│   ├── test_retraining_scheduler.py  # APScheduler retraining lifecycle orchestrator checks
│   └── test_bitquery_usage.py        # Live connection and query validator for Bitquery APIs
└── src/                              # Core Application Codebase
    ├── __init__.py
    ├── core/
    │   ├── config/
    │   │   ├── settings.py           # Central settings parser
    │   │   └── twitter_source_tiers.json # Source credibility tiers for weighted sentiment
    │   └── utils/
    │       ├── logging.py            # Color-coded logging configuration
    │       ├── retraining_scheduler.py # APScheduler retraining lifecycle orchestrator
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
    │   ├── xquik/
    │   │   └── xquik_ingestion.py        # Keyword polling, relevance filtering & orchestration
    │   └── news_rss/
    │       └── news_rss_ingestion.py     # Institutional news RSS polling pipeline (Phase 3)
    ├── feature_engineering/          # Aggregation Engines
    │   ├── trade_aggregator.py       # Computes OHLCV, buy/sell ratios, VWAP
    │   ├── orderbook_aggregator.py   # Computes spread averages, depths, and imbalances
    │   ├── source_credibility.py     # Source tier resolution & weighted sentiment calculation
    │   ├── xquik_aggregator.py       # Tweet sentiment distribution metrics (5-min buckets)
    │   ├── xquik_scorer.py           # CryptoBERT scoring for XQuik tweet records
    │   ├── news_rss_aggregator.py    # News sentiment bucketing into news_sentiment_5m
    │   └── news_rss_scorer.py        # FinBERT scoring & symbol attribution for news
    ├── models/                       # Deep Learning, Sentiment & Decision Engines
    │   ├── sentiment_models.py       # Dual sentiment engine: CryptoBERT (tweets) + FinBERT (news)
    │   ├── lstm_autoencoder.py       # PyTorch LSTM Autoencoder architecture
    │   ├── anomaly_pipeline.py       # Real-time anomaly inference pipeline
    │   ├── llm_pipeline.py           # Local structured Ollama/Qwen briefings loop
    │   ├── model_registry.py         # Thread-safe in-memory model hot-swapper
    │   ├── retraining_service.py     # Reusable training logic for manual & auto runs
    │   └── saved_weights/            # Model parameters directory
    │       ├── lstm_autoencoder_*.pt # PyTorch model weights
    │       └── scaler_params_*.json  # Scaler configuration files
    ├── db/                           # TimescaleDB Layer
    │   ├── db.py                     # Thread-safe PgPool connector
    │   └── db_schema.sql             # SQL migrations setup (Hypertables, schemas, indexes)
    └── sinks/                        # Sink Router Layer
        ├── base.py                   # Base interface
        └── timescale_sink.py         # Primary aggregator-routed DB sink
```

---

## 🚀 Setup & Execution

Regardless of whether you run the system inside **Docker** (recommended) or as a **Standalone Local Setup**, you must perform the universal environment and host-level configurations first.

---

### Step 1: Universal Environment Configuration (`.env`)

All core system configurations and API credentials live in a single `.env` file at the root of the project. 

Create a `.env` file at `C:\Users\Monster\WEB APPS\CryptoSense\.env` and configure your credentials:

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

# ── Model Retraining Settings ──────────────────────────
RETRAIN_ENABLED=true
RETRAIN_INTERVAL_DAYS=14
RETRAIN_LOOKBACK_DAYS=14
RETRAIN_DEVICE=auto

# ── Ollama Local LLM Connection Routing ────────────────
# 1. Local Run: Defaults to http://127.0.0.1:11434 (leave blank or set to local IP)
# 2. Docker Run: MUST be set to host.docker.internal to bridge the container boundary:
OLLAMA_HOST=http://host.docker.internal:11434

# ── Log Settings ───────────────────────────────────────
LOG_LEVEL=INFO
```

---

### Step 2: Configure Host Ollama for GPU & Network Access

By default, the Windows Ollama server only listens on `127.0.0.1` (localhost). To allow the system (and particularly Docker containers) to reach it, we must bind it to all network interfaces (`0.0.0.0`) on the host.

1. **Close Ollama**:
   * Exit Ollama from the Windows system tray (right-click the Ollama icon and select **Quit**).

2. **Set Windows Environment Variable**:
   * Open **PowerShell** as Administrator and run:
     ```powershell
     [Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0", "User")
     ```
   * Alternatively, search for **"Edit the system environment variables"** in the Windows Start Menu, click **Environment Variables**, and add a new User Variable:
     * **Variable Name**: `OLLAMA_HOST`
     * **Variable Value**: `0.0.0.0`

3. **Restart Ollama**:
   * Relaunch **Ollama** from your Start Menu.

4. **Pull the Qwen Model**:
   * Pull the target schema-locked model in a command prompt or PowerShell window:
     ```bash
     ollama pull qwen2.5:7b
     ```
   * Verify the model is downloaded and active:
     ```bash
     ollama list
     ```

---

### Option A: Ingestion & Dashboard via Docker Compose (Recommended)

The entire project has been dockerized with support for local GPU hardware acceleration. GPU support is **already configured out-of-the-box** via the pre-packaged `docker-compose.override.yml` file:

```yaml
services:
  # Enable GPU hardware acceleration for PyTorch anomaly inference
  ingestion-pipeline:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

#### GPU Verification Troubleshooting Note:
If you run `wsl nvidia-smi` inside PowerShell and get "command not found", **this will not prevent GPU Docker execution**. Docker Desktop utilizes its own custom backend WSL distributions (`docker-desktop`) and handles the driver paravirtualization mounts dynamically.

To verify Docker has GPU passthrough capabilities, run:
```powershell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```
If this command displays your NVIDIA GPU status table, your Docker environment is fully GPU-accelerated!

#### Run the Services:
Ensure your `OLLAMA_HOST` in `.env` is set to `http://host.docker.internal:11434` and run:
```bash
docker-compose up --build
```
This automatically spins up:
- **`cryptosense-pipeline`**: Runs `main.py` (Ingestion Streams + PyTorch LSTM Inference + Ollama LLM Briefings).
- **`cryptosense-api`**: Exposes the FastAPI REST Backend on **`http://localhost:8000`**.
- **`cryptosense-dashboard`**: Runs the Streamlit interactive dashboard on **`http://localhost:8501`**.

---

### Option B: Standalone Local Setup (Windows / Linux)

#### 1. Setup Local Environment
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements (compiles PyTorch targets)
pip install -r requirements.txt
```

#### 2. Run Database Migrations (Optional)
On startup, the main orchestrator (`python main.py`) will automatically execute migrations and build your tables. However, you can run this standalone script to manually initialize and verify your TimescaleDB schema beforehand without starting the pipeline streams:
```bash
python -m scripts.run_migration
```

#### 3. Run the Orchestrator
Ensure your `OLLAMA_HOST` in `.env` is set to `http://127.0.0.1:11434` (or left blank to default) and run:
```bash
python main.py
```

#### 4. Run Web & API Services
- **FastAPI REST API**:
  ```bash
  uvicorn src.web.api:app --host 0.0.0.0 --port 8000
  ```
- **Streamlit Premium Dashboard Grid**:
  ```bash
  streamlit run src/web/dashboard.py --server.port 8501 --server.address 0.0.0.0
  ```

---

## 🏃 Operations & Automated Testing Suite

CryptoSense features a robust automated testing and verification system.

### Automated Test Suite Execution
To run all unit tests and the database integration tests:
```bash
python scripts/run_all_tests.py
```
This discovers and executes:
- **Unit tests** covering model dimensions, data scaling/sliding windows, dual-model sentiment classification (CryptoBERT + FinBERT Macro F1 validation ≥ 0.75), English language detection, off-topic tweet filtering, database routing adapters, system signal handlers, and scheduler lifecycles.
- **Database integration tests** running inside a mock-patched, transaction-isolated wrapper (`ROLLBACK`) that tests aggregators and selects records without writing permanent data to your tables.
- **XQuik live API test** collecting real tweets over a 2-minute monitoring window with English language and off-topic filters, then scoring them with CryptoBERT.

At completion, a detailed Markdown summary report is written directly to [test_report.md](file:///c:/Users/Monster/WEB%20APPS/CryptoSense/test_report.md) in the project root.

### Operations Diagnostics
You can inspect the state of your database schemas, integration aggregates, and active records at any time:
- **Database Hypertables Diagnostics**:
  ```bash
  python -m scripts.verify_db
  ```
- **Inspect Live Table Outputs**:
  ```bash
  python -m scripts.check_live_data
  ```
- **Pretty-Print Live Anomaly LLM Payloads**:
  ```bash
  python -m scripts.inspect_payload
  ```
- **LSTM Autoencoder Model Training**:
  To manually train the unsupervised LSTM network over historical TimescaleDB tables (100 epochs, dynamically writing scalar JSON configurations and Pt weights to `saved_weights/`):
  ```bash
  python -m scripts.train_anomaly_detector
  ```

---

## 💳 Credit & Billing Considerations

- **XQuik Keyword Billing**: Active monitors consume **21 credits/hour each** (105 credits/hour total across 5 tracked symbols). Event polling itself is free.
- **Bitquery Billing**: WebSockets and HTTP GraphQL requests consume credits according to your Bitquery Developer plan. Polling intervals for BSC/AVAX transfers are throttled to 5 minutes to keep credit usage efficient.
- **Binance WebSocket Ingestion**: Zero-cost, zero-API key required.

---

## 📄 License

This repository is maintained for research, analytical model development, and educational purposes. All deep learning and quantitative code is provided as-is.
