-- ============================================================
-- CryptoSense TimescaleDB Schema Migration
-- Creates 4 hypertables for the 5-minute aggregated pipeline.
-- Run this once against the tsdb database.
-- ============================================================

-- Ensure TimescaleDB extension is enabled
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── Drop old test table (if any) ────────────────────────────
DROP TABLE IF EXISTS test_table CASCADE;

-- ============================================================
-- 1. Trade Candles (5-minute OHLCV + net trade)
-- ============================================================
CREATE TABLE IF NOT EXISTS trade_candles_5m (
    bucket          TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION,
    volume          DOUBLE PRECISION DEFAULT 0,
    quote_volume    DOUBLE PRECISION DEFAULT 0,
    trade_count     INTEGER         DEFAULT 0,
    buy_volume      DOUBLE PRECISION DEFAULT 0,
    sell_volume     DOUBLE PRECISION DEFAULT 0,
    net_trade       DOUBLE PRECISION DEFAULT 0,
    vwap            DOUBLE PRECISION,
    UNIQUE (bucket, symbol)
);

SELECT create_hypertable(
    'trade_candles_5m', 'bucket',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- ============================================================
-- 2. Orderbook Snapshots (5-minute averages)
-- ============================================================
CREATE TABLE IF NOT EXISTS orderbook_snapshots_5m (
    bucket          TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    avg_spread      DOUBLE PRECISION,
    avg_mid_price   DOUBLE PRECISION,
    avg_bid_depth   DOUBLE PRECISION,
    avg_ask_depth   DOUBLE PRECISION,
    avg_imbalance   DOUBLE PRECISION,
    snapshot_count  INTEGER         DEFAULT 0,
    UNIQUE (bucket, symbol)
);

SELECT create_hypertable(
    'orderbook_snapshots_5m', 'bucket',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- ── Drop legacy tables ──────────────────────────────────────
DROP TABLE IF EXISTS sentiment_scores CASCADE;

-- ============================================================
-- 3. Tweet Sentiment (5-minute aggregated from XQuik)
-- ============================================================
CREATE TABLE IF NOT EXISTS tweet_sentiment_5m (
    bucket          TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    avg_score       DOUBLE PRECISION DEFAULT 0,
    tweet_count     INTEGER         DEFAULT 0,
    positive_count  INTEGER         DEFAULT 0,
    negative_count  INTEGER         DEFAULT 0,
    neutral_count   INTEGER         DEFAULT 0,
    max_score       DOUBLE PRECISION,
    min_score       DOUBLE PRECISION,
    sample_tweet    TEXT,
    weighted_avg_score      DOUBLE PRECISION DEFAULT 0,
    total_source_weight     DOUBLE PRECISION DEFAULT 0,
    tier1_count             INTEGER DEFAULT 0,
    tier2_count             INTEGER DEFAULT 0,
    tier3_count             INTEGER DEFAULT 0,
    economy_news_count      INTEGER DEFAULT 0,
    turkish_economy_count   INTEGER DEFAULT 0,
    unknown_count           INTEGER DEFAULT 0,
    UNIQUE (bucket, symbol)
);

ALTER TABLE tweet_sentiment_5m
    ADD COLUMN IF NOT EXISTS weighted_avg_score DOUBLE PRECISION DEFAULT 0,
    ADD COLUMN IF NOT EXISTS total_source_weight DOUBLE PRECISION DEFAULT 0,
    ADD COLUMN IF NOT EXISTS tier1_count INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS tier2_count INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS tier3_count INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS economy_news_count INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS turkish_economy_count INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS unknown_count INTEGER DEFAULT 0;

SELECT create_hypertable(
    'tweet_sentiment_5m', 'bucket',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- ============================================================
-- 3.5 News Sentiment (5-minute aggregated from RSS)
-- ============================================================
CREATE TABLE IF NOT EXISTS news_sentiment_5m (
    bucket          TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    avg_score       DOUBLE PRECISION DEFAULT 0,
    news_count      INTEGER         DEFAULT 0,
    positive_count  INTEGER         DEFAULT 0,
    negative_count  INTEGER         DEFAULT 0,
    neutral_count   INTEGER         DEFAULT 0,
    sample_headline TEXT,
    UNIQUE (bucket, symbol)
);

SELECT create_hypertable(
    'news_sentiment_5m', 'bucket',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- ============================================================
-- 4. CEX Flows (5-minute aggregated inflow/outflow)
-- ============================================================
CREATE TABLE IF NOT EXISTS cex_flows_5m (
    bucket          TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    network         TEXT            NOT NULL,
    inflow_amount   DOUBLE PRECISION DEFAULT 0,
    inflow_usd      DOUBLE PRECISION DEFAULT 0,
    outflow_amount  DOUBLE PRECISION DEFAULT 0,
    outflow_usd     DOUBLE PRECISION DEFAULT 0,
    net_flow_usd    DOUBLE PRECISION DEFAULT 0,
    inflow_tx_count INTEGER         DEFAULT 0,
    outflow_tx_count INTEGER        DEFAULT 0,
    UNIQUE (bucket, symbol, network)
);

SELECT create_hypertable(
    'cex_flows_5m', 'bucket',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- ============================================================
-- 5. AI Anomaly Engine Outputs (5-minute intervals)
-- ============================================================
CREATE TABLE IF NOT EXISTS ai_anomalies_5m (
    bucket          TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    mse_score       DOUBLE PRECISION,
    is_anomaly      BOOLEAN         DEFAULT FALSE,
    severity        TEXT,
    llm_payload     JSONB,
    UNIQUE (bucket, symbol)
);

SELECT create_hypertable(
    'ai_anomalies_5m', 'bucket',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- ============================================================
-- 6. LLM Health Scores (5-minute intervals)
-- ============================================================
CREATE TABLE IF NOT EXISTS llm_health_scores (
    bucket          TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    health_score    INTEGER         NOT NULL,
    reasoning       TEXT,
    explanation     TEXT,
    model_name      TEXT,
    latency_ms      INTEGER,
    input_payload   JSONB,
    UNIQUE (bucket, symbol)
);

SELECT create_hypertable(
    'llm_health_scores', 'bucket',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- ── Indices for common query patterns ───────────────────────
CREATE INDEX IF NOT EXISTS idx_trade_candles_symbol   ON trade_candles_5m (symbol, bucket DESC);
CREATE INDEX IF NOT EXISTS idx_orderbook_symbol       ON orderbook_snapshots_5m (symbol, bucket DESC);
CREATE INDEX IF NOT EXISTS idx_tweet_sentiment_symbol ON tweet_sentiment_5m (symbol, bucket DESC);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol  ON news_sentiment_5m (symbol, bucket DESC);
CREATE INDEX IF NOT EXISTS idx_cex_flows_symbol       ON cex_flows_5m (symbol, bucket DESC);
CREATE INDEX IF NOT EXISTS idx_ai_anomalies_symbol    ON ai_anomalies_5m (symbol, bucket DESC);
CREATE INDEX IF NOT EXISTS idx_llm_health_scores_symbol ON llm_health_scores (symbol, bucket DESC);
