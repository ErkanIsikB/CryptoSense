"""CryptoSense Premium Multi-Coin Grid Dashboard — High-Performance Factual Decision Interface."""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import warnings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Suppress Altair selection parameter false-positive deduplication warnings in terminal
warnings.filterwarnings("ignore", category=UserWarning, message="Automatically deduplicated selection parameter")

# Configure Logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("dashboard")

# Try to import DB connection safely to prevent crashes if DB is completely missing
HAS_DB = False
try:
    from src.db.db import execute_query_fetch
    HAS_DB = True
except Exception as db_exc:
    LOGGER.warning("Could not initialize database connection pool. Running in offline-only mode: %s", db_exc)

# ── 1. Page & Layout Configurations ─────────────────────────────────

st.set_page_config(page_title="CryptoSense // Real-Time AI & LLM Analytics Grid", layout="wide", page_icon="🛡️")

import base64
import math
from pathlib import Path

def ensure_chime_exists():
    audio_path = Path("src/web/chime.wav")
    if not audio_path.exists():
        try:
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            sample_rate = 44100
            duration = 1.0
            num_samples = int(sample_rate * duration)
            import wave, struct
            with wave.open(str(audio_path), "w") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                for i in range(num_samples):
                    t = i / sample_rate
                    if t < 0.02:
                        envelope = t / 0.02
                    else:
                        envelope = math.exp(-6.0 * (t - 0.02))
                    val = 0.5 * math.sin(2 * math.pi * 1046.5 * t) + \
                          0.3 * math.sin(2 * math.pi * 1318.5 * t) + \
                          0.2 * math.sin(2 * math.pi * 1568.0 * t)
                    sample = int(val * envelope * 32767 * 0.4)
                    wav.writeframesraw(struct.pack('<h', sample))
            LOGGER.info("Successfully generated crystal chime audio asset at %s", audio_path)
        except Exception as e:
            LOGGER.error("Failed to generate chime audio: %s", e)

def get_base64_audio(file_path: str) -> str:
    try:
        p = Path(file_path)
        if p.exists():
            data = p.read_bytes()
            return base64.b64encode(data).decode()
    except Exception as e:
        LOGGER.error("Failed to load audio file: %s", e)
    return ""

ensure_chime_exists()
AUDIO_BASE64 = get_base64_audio("src/web/chime.wav")


# Custom Premium CSS Styling (Aurora mesh background, refined frosted glassmorphism, accent rails)
st.markdown(
    """
    <style>
        /* Import display + body type pairing */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');

        /* ── Aurora Glass theme tokens ── */
        :root, .stApp {
            --background-color: transparent !important;
            --secondary-background-color: rgba(255, 255, 255, 0.02) !important;
            --glass-stroke: rgba(255, 255, 255, 0.10);
            --accent-cyan: #38e8ff;
            --accent-violet: #a78bfa;
        }

        /* ── Aggressive CSS Override to Reveal Background through Streamlit ── */
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"], [data-testid="stHeader"], .main, .stMain, [data-testid="stMainTemplate"], .block-container, [data-testid="stVerticalBlock"], .element-container, [data-testid="element-container"] {
            background-color: transparent !important;
            background: transparent !important;
            font-family: 'Outfit', sans-serif;
            color: #e7ecf5;
        }

        /* Combined high-performance fixed background on html */
        html {
            background: 
                /* 1. Grain overlay */
                url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"),
                /* 2. Aurora Cyan blob (top left) */
                radial-gradient(circle at 15% 10%, rgba(34, 211, 238, 0.15) 0%, rgba(14, 116, 144, 0.05) 30%, transparent 60%),
                /* 3. Aurora Indigo blob (bottom right) */
                radial-gradient(circle at 85% 90%, rgba(129, 140, 248, 0.28) 0%, rgba(67, 56, 202, 0.08) 45%, transparent 75%),
                /* 4. Aurora Rose blob (center-left) */
                radial-gradient(circle at 45% 45%, rgba(244, 114, 182, 0.22) 0%, rgba(157, 23, 77, 0.06) 45%, transparent 75%),
                /* 5. Base Deep obsidian-slate gradient */
                radial-gradient(ellipse at 20% 0%, #07070a 0%, #030305 45%, #010101 100%) !important;
            
            background-attachment: fixed !important;
            background-repeat: repeat, no-repeat, no-repeat, no-repeat, no-repeat !important;
            background-size: auto, cover, cover, cover, cover !important;
            background-blend-mode: overlay, normal, normal, normal, normal !important;
        }

        html, body {
            margin: 0;
            padding: 0;
        }

        /* ── Prevent Streamlit's "stale" fade-out during reruns ── */
        div.element-container,
        [data-testid="stVerticalBlock"],
        [data-testid="stAppViewContainer"] {
            opacity: 1 !important;
            transition: none !important;
        }

        /* ── Refined Frosted Glass Panel (dark transparent shade, crisp blended borders) ── */
        div[class*="st-key-kpi_card_"],
        div[class*="st-key-snapshot_"],
        div[class*="st-key-ai_briefing_"],
        div[class*="st-key-chart_"] {
            position: relative;
            background: rgba(8, 8, 12, 0.67) !important; /* Exactly 67% opaque modern obsidian charcoal black */
            background-image:
                linear-gradient(160deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 38%, rgba(255, 255, 255, 0) 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 22px !important;
            padding: 18px !important;
            box-shadow:
                0 16px 40px -8px rgba(0, 0, 0, 0.75),
                inset 0 1px 0 0 rgba(255, 255, 255, 0.05),
                inset 0 0 0 1px rgba(255, 255, 255, 0.01) !important;

            /* High-Performance GPU Hardware Acceleration Overrides */
            transform: translate3d(0, 0, 0) !important;
            will-change: transform !important;

            animation: subtle-fade-in 0.4s ease-out !important;
            transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.3s ease !important;
        }

        /* Subtle top-edge specular highlight that reads as a glass bevel */
        div[class*="st-key-kpi_card_"]::before,
        div[class*="st-key-snapshot_"]::before,
        div[class*="st-key-ai_briefing_"]::before,
        div[class*="st-key-chart_"]::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 1px;
            border-radius: 22px 22px 0 0;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.25), transparent);
            opacity: 0.4;
            pointer-events: none;
        }

        @keyframes subtle-fade-in {
            from { opacity: 0; transform: translate3d(0, 5px, 0); }
            to { opacity: 1; transform: translate3d(0, 0, 0); }
        }

        /* ── Smooth aurora border glow & elevation on hover ── */
        div[class*="st-key-kpi_card_"]:hover,
        div[class*="st-key-snapshot_"]:hover,
        div[class*="st-key-ai_briefing_"]:hover,
        div[class*="st-key-chart_"]:hover {
            border-color: rgba(56, 232, 255, 0.40) !important; /* Aurora Cyan bezel glow */
            transform: translate3d(0, -3px, 0) !important;
            box-shadow:
                0 28px 56px -10px rgba(0, 0, 0, 0.85),
                0 0 28px -4px rgba(56, 232, 255, 0.22),
                inset 0 1px 0 0 rgba(255, 255, 255, 0.15) !important;
        }

        /* ── Glass pill Tab navigation ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: transparent !important;
            border-bottom: none !important;
            padding: 8px 0 16px 0 !important;
            overflow: visible !important;
        }
        .stTabs [data-baseweb="tab"],
        div[data-testid="stTabBar"] button,
        button[data-baseweb="tab"] {
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px;
            color: #9fb0c8 !important;
            background-color: rgba(8, 8, 12, 0.67) !important; /* Semi-transparent obsidian black matching containers */
            background: rgba(8, 8, 12, 0.67) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 999px !important;
            padding: 6px 20px !important; /* Reduced vertical padding to prevent cropping */
            transition: all 0.2s ease-in-out !important;
        }
        .stTabs [data-baseweb="tab"] p,
        div[data-testid="stTabBar"] button p {
            font-size: 15px !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }
        .stTabs [data-baseweb="tab"]:hover,
        div[data-testid="stTabBar"] button:hover {
            color: #e7ecf5 !important;
            background-color: rgba(12, 12, 18, 0.67) !important;
            background: rgba(12, 12, 18, 0.67) !important;
            border-color: rgba(255, 255, 255, 0.22) !important;
        }
        .stTabs [aria-selected="true"],
        div[data-testid="stTabBar"] button[aria-selected="true"],
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #38e8ff !important;
            background-color: rgba(16, 16, 24, 0.67) !important; /* Semi-transparent dark background for selected tab */
            background: rgba(16, 16, 24, 0.67) !important;
            border-color: rgba(56, 232, 255, 0.45) !important;
            box-shadow: 0 0 20px -6px rgba(56, 232, 255, 0.5) !important;
        }
        .stTabs [aria-selected="true"] p,
        div[data-testid="stTabBar"] button[aria-selected="true"] p {
            color: #38e8ff !important;
        }
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* ── Hide 'Show data' and Grid Table view buttons inside elements toolbar ── */
        div[data-testid="stVegaLiteChart"] [data-testid="stElementToolbar"] button:not(:last-child) {
            display: none !important;
        }

        /* ── Transparent child override inside glass containers ── */
        div[class*="st-key-kpi_card_"] div:not(.ai-card-score):not(.sev-chip):not(.indicator-pulse-green):not(.indicator-pulse-red):not(.indicator-pulse-amber),
        div[class*="st-key-snapshot_"] div:not(.ai-card-score):not(.sev-chip):not(.indicator-pulse-green):not(.indicator-pulse-red):not(.indicator-pulse-amber),
        div[class*="st-key-ai_briefing_"] div:not(.ai-card-score):not(.sev-chip):not(.indicator-pulse-green):not(.indicator-pulse-red):not(.indicator-pulse-amber),
        div[class*="st-key-chart_"] div:not(.ai-card-score):not(.sev-chip):not(.indicator-pulse-green):not(.indicator-pulse-red):not(.indicator-pulse-amber) {
            background: transparent !important;
            background-color: transparent !important;
        }

        /* ── Disable Streamlit's default grey-out/fade-out during script reruns ── */
        [data-stale="true"],
        [data-stale="true"] *,
        .stElementContainer[data-stale="true"],
        div[data-stale="true"],
        div[class*="stale"] {
            opacity: 1 !important;
            filter: none !important;
            transition: none !important;
        }

        /* ── KPI Overview Cards ── */
        .kpi-token {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 2.5px;
            text-transform: uppercase;
            color: #9fb0c8;
            display: flex;
            align-items: center;
        }
        .kpi-price {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(17px, 1.8vw, 27px); /* Scales down on narrow viewports instead of wrapping */
            font-weight: 700;
            color: #ffffff;
            margin: 6px 0 3px 0;
            line-height: 1.1;
            white-space: nowrap;
        }
        .kpi-delta-up { color: #4ade80; font-size: 13.5px; font-weight: 600; margin-bottom: 10px; }
        .kpi-delta-down { color: #f87171; font-size: 13.5px; font-weight: 600; margin-bottom: 10px; }

        /* ── Detail Snapshot Card ── */
        .snap-token {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 20px;
            font-weight: 700;
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        .snap-price {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 42px;
            font-weight: 700;
            color: #ffffff;
            line-height: 1.05;
            margin-bottom: 4px;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px 20px;
            margin: 18px 0 16px 0;
        }
        .stat-label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #8fa0b5;
            margin-bottom: 3px;
        }
        .stat-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 18px;
            font-weight: 600;
            color: #e7ecf5;
        }

        /* ── Severity chips ── */
        .sev-chip {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1.5px;
            padding: 5px 14px;
            border-radius: 999px;
            display: inline-block;
        }
        .sev-normal { background: rgba(148, 163, 184, 0.12); border: 1px solid rgba(148, 163, 184, 0.4); color: #cbd5e1; }
        .sev-high { background: rgba(245, 158, 11, 0.14); border: 1px solid rgba(251, 191, 36, 0.55); color: #fbbf24; box-shadow: 0 0 16px -4px rgba(245, 158, 11, 0.5); }
        .sev-critical { background: rgba(239, 68, 68, 0.14); border: 1px solid rgba(248, 113, 113, 0.55); color: #f87171; box-shadow: 0 0 16px -4px rgba(239, 68, 68, 0.5); }

        /* ── Chart Cards ── */
        .chart-card-head {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 8px;
        }
        .chart-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 12.5px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2.5px;
        }
        .chart-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 21px;
            font-weight: 700;
            color: #ffffff;
        }

        /* ── AI Analysis Card Components ── */
        .ai-card-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 20px;
            font-weight: 700;
            letter-spacing: 0.3px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
        }

        .ai-card-score {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-bottom: 14px;
            padding: 6px 16px;
            border-radius: 999px; /* Pill badge */
            display: inline-block;
            backdrop-filter: blur(6px);
        }

        .score-excellent { background: rgba(16, 185, 129, 0.14); border: 1px solid rgba(52, 211, 153, 0.55); color: #4ade80; box-shadow: 0 0 16px -4px rgba(16, 185, 129, 0.5); }
        .score-warning { background: rgba(245, 158, 11, 0.14); border: 1px solid rgba(251, 191, 36, 0.55); color: #fbbf24; box-shadow: 0 0 16px -4px rgba(245, 158, 11, 0.5); }
        .score-critical { background: rgba(239, 68, 68, 0.14); border: 1px solid rgba(248, 113, 113, 0.55); color: #f87171; box-shadow: 0 0 16px -4px rgba(239, 68, 68, 0.5); }

        .ai-reasoning {
            font-size: 12.5px;
            font-weight: 600;
            letter-spacing: 0.5px;
            color: #8fa0b5;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        .ai-card-explanation {
            font-size: 15px;
            line-height: 1.75;
            color: #d3dbe8;
            margin-bottom: 0px;
        }
        
        /* ── Glowing indicators next to token names ── */
        .indicator-pulse-red {
            width: 10px;
            height: 10px;
            background-color: #ef4444;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            box-shadow: 0 0 10px #ef4444;
            animation: pulse-red-anim 1.5s infinite;
        }
        
        @keyframes pulse-red-anim {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }

        .indicator-pulse-amber {
            width: 10px;
            height: 10px;
            background-color: #f59e0b;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            box-shadow: 0 0 10px #f59e0b;
            animation: pulse-amber-anim 1.5s infinite;
        }
        
        @keyframes pulse-amber-anim {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(245, 158, 11, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
        }

        .indicator-pulse-green {
            width: 10px;
            height: 10px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            box-shadow: 0 0 10px #10b981;
            animation: pulse-green-anim 1.5s infinite;
        }
        
        @keyframes pulse-green-anim {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── 2. Data Fetchers (Direct Database Integration Failsafe) ──────────


@st.cache_data(ttl=15)
def get_latest_metrics_db(symbol: str) -> pd.DataFrame:
    """Retrieve 24-candle timeseries data with complete joints from TimescaleDB."""
    if not HAS_DB:
        return pd.DataFrame()

    try:
        sql = """
        WITH finalized_buckets AS (
            SELECT bucket, symbol FROM trade_candles_5m
            INTERSECT
            SELECT bucket, symbol FROM orderbook_snapshots_5m
            INTERSECT
            SELECT bucket, symbol || 'USDT' AS symbol FROM tweet_sentiment_5m
        )
        SELECT 
            fb.bucket AS bucket,
            t.close AS close_price,
            t.volume AS volume_5m,
            t.vwap AS vwap,
            t.net_trade AS net_trade,
            o.avg_spread AS spread,
            o.avg_mid_price AS mid_price,
            o.avg_bid_depth AS bid_depth,
            o.avg_ask_depth AS ask_depth,
            o.avg_imbalance AS imbalance,
            s.avg_score AS sentiment_score,
            s.tweet_count AS tweet_count,
            COALESCE(c.net_flow_usd, 0.0) AS net_cex_flow_usd,
            COALESCE(a.mse_score, 0.0) AS mse_score,
            COALESCE(a.is_anomaly, FALSE) AS is_anomaly,
            COALESCE(a.severity, 'NORMAL') AS severity
        FROM finalized_buckets fb
        JOIN trade_candles_5m t ON t.bucket = fb.bucket AND t.symbol = fb.symbol
        JOIN orderbook_snapshots_5m o ON o.bucket = fb.bucket AND o.symbol = fb.symbol
        JOIN tweet_sentiment_5m s ON s.bucket = fb.bucket AND s.symbol = REPLACE(fb.symbol, 'USDT', '')
        LEFT JOIN (
            SELECT bucket, TRIM(symbol) as symbol, SUM(net_flow_usd) as net_flow_usd 
            FROM cex_flows_5m 
            GROUP BY bucket, symbol
        ) c ON c.bucket = fb.bucket AND c.symbol = TRIM(REPLACE(fb.symbol, 'USDT', ''))
        LEFT JOIN ai_anomalies_5m a ON a.bucket = fb.bucket AND a.symbol = REPLACE(fb.symbol, 'USDT', '')
        WHERE fb.symbol = %s
        ORDER BY bucket DESC
        LIMIT 24;
        """
        rows = execute_query_fetch(sql, (symbol,))
        if not rows:
            return pd.DataFrame()

        columns = [
            "bucket",
            "close_price",
            "volume_5m",
            "vwap",
            "net_trade",
            "spread",
            "mid_price",
            "bid_depth",
            "ask_depth",
            "imbalance",
            "sentiment_score",
            "tweet_count",
            "net_cex_flow_usd",
            "mse_score",
            "is_anomaly",
            "severity",
        ]
        df = pd.DataFrame(rows, columns=columns)
        df["bucket"] = pd.to_datetime(df["bucket"], utc=True)
        return df.sort_values("bucket").reset_index(drop=True)
    except Exception as exc:
        LOGGER.error("DB Query failed: %s", exc)
        return pd.DataFrame()


@st.cache_data(ttl=15)
def get_latest_health_score_db(symbol: str) -> dict | None:
    """Retrieve latest LLM decision health score for a target symbol."""
    if not HAS_DB:
        return None

    try:
        sql = """
        SELECT bucket, health_score, reasoning, explanation, latency_ms, input_payload
        FROM llm_health_scores
        WHERE symbol = %s
        ORDER BY bucket DESC
        LIMIT 1;
        """
        rows = execute_query_fetch(sql, (symbol,))
        if not rows:
            return None
        r = rows[0]
        return {
            "bucket": r[0],
            "health_score": int(r[1]) if r[1] is not None else 50,
            "reasoning": r[2],
            "explanation": r[3],
            "latency_ms": r[4],
            "input_payload": r[5],
        }
    except Exception as exc:
        LOGGER.error("DB Health Score fetch failed: %s", exc)
        return None


# ── 3. High-Fidelity Mock Data Engine (For Offline Demo Mode) ──────


def generate_mock_timeseries(symbol: str) -> pd.DataFrame:
    """Generate realistic 24-point 5-min bucket timeseries data for testing visuals."""
    now = datetime.now(timezone.utc)
    buckets = [now - timedelta(minutes=5 * i) for i in range(24)]
    buckets = sorted(buckets)

    base_prices = {"BTC": 77200.0, "ETH": 2110.0, "SOL": 85.5, "BNB": 663.0, "AVAX": 9.4}
    price = base_prices.get(symbol, 100.0)

    close_prices = []
    volumes = []
    vwaps = []
    net_trades = []
    spreads = []
    mid_prices = []
    bid_depths = []
    ask_depths = []
    imbalances = []
    sentiments = []
    tweets = []
    cex_flows = []
    mse_scores = []
    anomalies = []
    severities = []

    np.random.seed(sum(ord(c) for c in symbol))
    price_walk = price + np.cumsum(np.random.normal(0, price * 0.003, 24))

    for i in range(24):
        p = price_walk[i]
        vol = float(np.random.uniform(1000, 20000))
        net_t = float(np.random.uniform(-vol * 0.3, vol * 0.3))
        vw = float(p + np.random.normal(0, p * 0.0005))
        spr = float(np.random.uniform(0.001, 0.015))
        mid = float(p + spr / 2)
        b_d = float(np.random.uniform(5000, 25000))
        a_d = float(np.random.uniform(5000, 25000))
        imb = float((b_d - a_d) / (b_d + a_d))
        sent = float(np.random.uniform(-0.1, 0.2))
        tw = int(np.random.randint(5, 60))
        cex = float(np.random.uniform(-1000000, 1000000))
        mse = float(np.random.uniform(0.002, 0.006))
        anomaly = False
        sev = "NORMAL"

        if i >= 21:
            if symbol == "ETH":
                p -= 45.0
                vol *= 2.5
                net_t = -vol * 0.4
                b_d *= 0.35
                imb = -0.55
                cex = -2032901.03
                mse = 0.0125
                anomaly = True
                sev = "HIGH"
            elif symbol == "AVAX":
                p -= 1.8
                vol *= 3.0
                net_t = -vol * 0.5
                b_d *= 0.2
                imb = -0.72
                mse = 0.0602
                anomaly = True
                sev = "CRITICAL"
            elif symbol == "SOL":
                p += 4.5
                vol *= 4.0
                net_t = vol * 0.6
                a_d *= 0.25
                imb = 0.65
                cex = 6923688.54
                mse = 0.0082
                anomaly = True
                sev = "HIGH"

        close_prices.append(float(p))
        volumes.append(vol)
        vwaps.append(vw)
        net_trades.append(net_t)
        spreads.append(spr)
        mid_prices.append(mid)
        bid_depths.append(b_d)
        ask_depths.append(a_d)
        imbalances.append(imb)
        sentiments.append(sent)
        tweets.append(tw)
        cex_flows.append(cex)
        mse_scores.append(mse)
        anomalies.append(anomaly)
        severities.append(sev)

    df = pd.DataFrame(
        {
            "bucket": buckets,
            "close_price": close_prices,
            "volume_5m": volumes,
            "vwap": vwaps,
            "net_trade": net_trades,
            "spread": spreads,
            "mid_price": mid_prices,
            "bid_depth": bid_depths,
            "ask_depth": ask_depths,
            "imbalance": imbalances,
            "sentiment_score": sentiments,
            "tweet_count": tweets,
            "net_cex_flow_usd": cex_flows,
            "mse_score": mse_scores,
            "is_anomaly": anomalies,
            "severity": severities,
        }
    )
    return df


def generate_mock_health_score(symbol: str) -> dict:
    """Generate realistic, custom Qwen 2.5 structured response payload for each coin."""
    now = datetime.now(timezone.utc)

    mock_briefings = {
        "ETH": {
            "health_score": 42,
            "reasoning": "Driver: liquidity_flight | Trust: LIQUIDITY_EXHAUSTION",
            "explanation": "Ethereum experienced a sudden 2.4% price decline over the last 15 minutes, accompanied by a severe hollowing out of orderbook bid depth down to $85.5k. On-chain CEX flows recorded a massive net outflow of -$2.03M on Ethereum, indicating substantial whale exit activity. The AI Autoencoder registered a critical reconstruction error of 0.0125, confirming a severe structural shift in orderbook liquidity dynamics.",
        },
        "AVAX": {
            "health_score": 18,
            "reasoning": "Driver: liquidity_flight | Trust: LIQUIDITY_EXHAUSTION",
            "explanation": "Avalanche C-Chain registered a massive structural liquidity collapse as sell-side net trade volume spiked to 50% of total volume. Orderbook bid depth dropped by over 80% to critical lows, failing to absorb incoming sell pressure and causing rapid price depreciation. The AI Autoencoder registered a severe reconstruction error of 0.0602, triggering a critical anomaly state due to total imbalance exhaustion.",
        },
        "SOL": {
            "health_score": 91,
            "reasoning": "Driver: on_chain_whale_flow | Trust: HIGH_CONVICTION",
            "explanation": "Solana experienced a major positive breakout spike of +5.2% over the last hour, heavily backed by a massive net CEX flow inflow of +$6.92M. Orderbook ask depth was quickly exhausted as taker buyers dominated order fills, driving bid-ask imbalance to a highly bullish +0.65 skew. While the sudden volatility triggered a PyTorch anomaly warning (MSE: 0.0082), the underlying metrics reflect exceptionally strong bullish trajectory momentum.",
        },
        "BTC": {
            "health_score": 85,
            "reasoning": "Driver: none | Trust: STABLE_BASELINE",
            "explanation": "Bitcoin trading remains highly stable within a narrow $150 consolidation band with healthy bid-ask depth. Social sentiment scores remained flat with a minor positive shift, while net CEX flows and trading volumes are well within historical parameters. The PyTorch autoencoder reconstruction error remains at a baseline of 0.0054, confirming healthy symmetrical market mechanics.",
        },
        "BNB": {
            "health_score": 78,
            "reasoning": "Driver: sentiment_shift | Trust: STABLE_BASELINE",
            "explanation": "BNB shows solid baseline stability over the evaluated sequence window, maintaining robust liquidity profiles on both sides of the book. Positive social sentiment scores registered a healthy 0.12 average, supporting price consolidation. The PyTorch autoencoder reconstruction error remained well below the critical boundary, confirming baseline stability with zero active warning triggers.",
        },
    }

    brief = mock_briefings.get(symbol, mock_briefings["BTC"])
    return {
        "bucket": now,
        "health_score": brief["health_score"],
        "reasoning": brief["reasoning"],
        "explanation": brief["explanation"],
        "latency_ms": int(random.randint(180, 450)),
        "input_payload": [],
    }


# ── 4. Sidebar UI Controls & Token Selection ────────────────────────


st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style="font-family: 'Space Grotesk', sans-serif; background: linear-gradient(92deg, #38e8ff 0%, #a78bfa 100%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; letter-spacing: 2px; margin-bottom: 2px;">🛡️ CRYPTOSENSE</h2>
        <p style='color: #8fa0b5; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;'>AI Shield &amp; Decision System</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Standalone UI Test Mode ─────────────────────────────────────────
# Force demo (mock-data) mode without any DB / ingestion / model backends by setting
# the env var CRYPTOSENSE_DEMO=1  OR  appending  ?demo=1  to the dashboard URL.
# This lets you preview the full UI in isolation, even when a live DB is reachable.
import os

def _demo_forced() -> bool:
    if str(os.getenv("CRYPTOSENSE_DEMO", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    try:
        qp = st.query_params.get("demo", "")
        return str(qp).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        return False

FORCE_DEMO = _demo_forced()

# Sidebar Toggle for Offline Demo Mode
st.sidebar.markdown("### 🛠️ UI Mode Settings")
demo_mode = st.sidebar.toggle(
    "Enable Demo Offline Mode",
    value=(FORCE_DEMO or not HAS_DB),  # Forced via env/query param, or auto if no DB pool
    disabled=FORCE_DEMO,  # Lock the toggle on when test mode is explicitly requested
    help="Enable this to preview the dashboard visuals and glowing AI alerts with high-fidelity simulated data without querying any database or backend APIs. Tip: launch with CRYPTOSENSE_DEMO=1 or add ?demo=1 to the URL to force this on.",
)

# Sidebar System Health Metadata
st.sidebar.markdown("---")
st.sidebar.markdown("### 🖥️ Core Engine Status")

if demo_mode:
    st.sidebar.warning("🛠️ Running in: DEMO OFFLINE MODE")
    st.sidebar.info("🟢 Streamlit UI: STABLE\n\n⚪ Ingestion Streams: INACTIVE\n\n⚪ LSTM Autoencoder: SIMULATING\n\n⚪ LLM Decision: SIMULATING")
else:
    st.sidebar.success("📡 Running in: DATABASE LIVE MODE")
    st.sidebar.info("🟢 Ingestion Streams: ONLINE\n\n🟢 LSTM Autoencoder: ONLINE\n\n🟢 LLM Decision Engine: ONLINE")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Force Refresh Radar Array", width="stretch"):
    st.cache_data.clear()
    st.rerun()


# Initialize session state for demo simulation cache if not present
if "demo_timeseries" not in st.session_state:
    st.session_state["demo_timeseries"] = {
        token: generate_mock_timeseries(token) for token in ["BTC", "ETH", "SOL", "BNB", "AVAX"]
    }
if "demo_health_scores" not in st.session_state:
    st.session_state["demo_health_scores"] = {
        token: generate_mock_health_score(token) for token in ["BTC", "ETH", "SOL", "BNB", "AVAX"]
    }
if "fresh_anomaly_trigger" not in st.session_state:
    st.session_state["fresh_anomaly_trigger"] = False
if "fresh_anomaly_token" not in st.session_state:
    st.session_state["fresh_anomaly_token"] = None


def inject_next_candle():
    st.session_state["fresh_anomaly_trigger"] = False
    st.session_state["fresh_anomaly_token"] = None
    
    # Append a new candle to the end of the timeseries dataframe of all 5 tokens
    for token in ["BTC", "ETH", "SOL", "BNB", "AVAX"]:
        df: pd.DataFrame = st.session_state["demo_timeseries"][token]
        last_row = df.iloc[-1]
        
        # Calculate next bucket timestamp
        next_bucket = last_row["bucket"] + timedelta(minutes=5)
        
        # Generate random walk close price (shift between -1.5% and +1.5%)
        price_change_pct = np.random.uniform(-0.015, 0.015)
        next_price = float(last_row["close_price"] * (1.0 + price_change_pct))
        
        # Generate random values for volume, CEX flow, sentiment, and autoencoder MSE
        next_vol = float(np.random.uniform(5000, 30000))
        next_cex = float(np.random.uniform(-2000000, 2000000))
        next_sent = float(np.random.uniform(-0.15, 0.25))
        next_mse = float(np.random.uniform(0.002, 0.007))
        
        # Determine if it's a simulated anomaly (25% chance per candle)
        next_anomaly = False
        next_sev = "NORMAL"
        
        if np.random.random() < 0.25:
            next_anomaly = True
            next_mse = float(np.random.uniform(0.012, 0.05))
            next_sev = "HIGH" if np.random.random() < 0.8 else "CRITICAL"
            
            # Record the new anomaly trigger
            st.session_state["fresh_anomaly_trigger"] = True
            st.session_state["fresh_anomaly_token"] = token
            
        # Shift the health score slightly but keep the explanation consistent
        score_shift = random.randint(-4, 4)
        current_score = st.session_state["demo_health_scores"][token]["health_score"]
        new_score = max(0, min(100, current_score + score_shift))
        st.session_state["demo_health_scores"][token]["health_score"] = new_score
        
        # Append new row
        new_row = {
            "bucket": next_bucket,
            "close_price": next_price,
            "volume_5m": next_vol,
            "vwap": next_price + np.random.normal(0, next_price * 0.0005),
            "net_trade": float(np.random.uniform(-next_vol * 0.3, next_vol * 0.3)),
            "spread": float(np.random.uniform(0.001, 0.015)),
            "mid_price": next_price,
            "bid_depth": float(np.random.uniform(5000, 25000)),
            "ask_depth": float(np.random.uniform(5000, 25000)),
            "imbalance": float(np.random.uniform(-0.5, 0.5)),
            "sentiment_score": next_sent,
            "tweet_count": int(np.random.randint(5, 60)),
            "net_cex_flow_usd": next_cex,
            "mse_score": next_mse,
            "is_anomaly": next_anomaly,
            "severity": next_sev,
        }
        
        # Convert dictionary to DataFrame and append it
        new_df = pd.DataFrame([new_row])
        df_updated: pd.DataFrame = pd.concat([df, new_df], ignore_index=True)
        
        # Slice to keep only the latest 24 records to maintain Sparkline window size
        df_sliced = df_updated.iloc[-24:].reset_index(drop=True)
        st.session_state["demo_timeseries"][token] = df_sliced


# Create a static placeholder for audio playbacks to ensure element sequence is rigid and prevent layout shifts!
audio_placeholder = st.empty()

# If fresh anomaly is triggered by the injection button, play sound immediately!
if st.session_state.get("fresh_anomaly_trigger", False) and AUDIO_BASE64:
    audio_placeholder.markdown(
        f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{AUDIO_BASE64}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True
    )

# Render Widescreen Header Layout with Simulation inject button on the right
title_cols = st.columns([4, 1.2])
with title_cols[0]:
    st.markdown(
        """
        <div style='text-align: left; margin-bottom: 32px;'>
            <h1 style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 44px; letter-spacing: 1px; background: linear-gradient(92deg, #ffffff 0%, #a5f3fc 45%, #c4b5fd 100%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 28px rgba(56, 232, 255, 0.18); margin-bottom: 0px;">CRYPTOSENSE ANALYTICS DASHBOARD</h1>
            <p style='color: #8fa0b5; font-size: 14px; margin-top: 6px; letter-spacing: 2px; text-transform: uppercase;'>Real-time Token Analysis &nbsp;·&nbsp; LLM Insights</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with title_cols[1]:
    if demo_mode:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Inject Next Candle", key="inject_candle_btn", width="stretch"):
            inject_next_candle()
            st.rerun()


def create_glass_chart(df, y_column, line_color, value_title, value_format, token, height=270, axis_format=None):
    """Large-format glass chart: gradient area fill + smooth line + snapping hover dot on a real time axis."""
    # Unique selection name per token+metric to avoid Vega deduplication warnings;
    # clear="mouseout" kills the hover dot instantly when the cursor leaves.
    hover_selection = alt.selection_point(
        name=f"hover_{token.lower()}_{y_column}",
        on="mouseover",
        nearest=True,
        fields=["bucket"],
        encodings=["x"],  # Ignores Y-axis height and tracks strictly along the timeline!
        clear="mouseout",
        empty=False,
    )

    y_axis_kwargs = dict(
        title=None,
        grid=True,
        gridColor="rgba(255, 255, 255, 0.06)",
        ticks=False,
        domain=False,
        labelColor="#94a3b8",
        labelFontSize=12,
        labelPadding=8,
    )
    if axis_format:
        y_axis_kwargs["format"] = axis_format

    # Explicit padded Y domain computed from the data — Streamlit's Vega theme
    # otherwise forces the scale to include zero, flattening tight price ranges.
    y_lo = float(df[y_column].min())
    y_hi = float(df[y_column].max())
    y_pad = (y_hi - y_lo) * 0.10 or max(abs(y_hi), 1.0) * 0.01
    y_scale = alt.Scale(domain=[y_lo - y_pad, y_hi + y_pad], nice=False, zero=False)

    base = alt.Chart(df).encode(
        x=alt.X("bucket:T", axis=alt.Axis(
            title=None,
            format="%H:%M",
            grid=False,
            ticks=False,
            domain=False,
            labelColor="#94a3b8",
            labelFontSize=12,
            labelPadding=8,
            tickCount=6,
        )),
        y=alt.Y(f"{y_column}:Q", scale=y_scale, axis=alt.Axis(**y_axis_kwargs)),
    )

    # 1. Soft vertical gradient fill under the curve for depth.
    # The area's natural zero baseline falls outside the explicit Y domain for
    # high-value series (e.g. BTC price), so clip it to the plot frame.
    area = base.mark_area(
        interpolate="monotone",
        opacity=0.16,
        clip=True,
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="rgba(0, 0, 0, 0)", offset=0),
                alt.GradientStop(color=line_color, offset=1),
            ],
            x1=1, x2=1, y1=1, y2=0,
        ),
    )

    # 2. Main smooth line
    line = base.mark_line(color=line_color, strokeWidth=3, interpolate="monotone")

    # 3. Glowing hover dot that snaps onto the line based on horizontal mouse position
    points = base.mark_point(
        color=line_color, size=90, filled=True, stroke="white", strokeWidth=1.5
    ).encode(
        opacity=alt.condition(hover_selection, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("bucket:T", title="Time", format="%H:%M"),
            alt.Tooltip(f"{y_column}:Q", title=value_title, format=value_format),
        ],
    ).add_params(hover_selection)

    chart = alt.layer(area, line, points).properties(
        width="container",
        height=height,
        usermeta={"embedOptions": {"actions": False}},
    ).configure_view(strokeWidth=0).configure(background="transparent")

    return chart




# Initialize session state for anomaly seen tracker
if "last_seen_anomaly_time" not in st.session_state:
    st.session_state["last_seen_anomaly_time"] = None

# Preload all coin data to eliminate duplicate queries and optimize rendering performance
preloaded_data = {}
latest_anomaly_time = None
latest_anomaly_token = None

for t_token in ["BTC", "ETH", "SOL", "BNB", "AVAX"]:
    if demo_mode:
        df_temp = st.session_state["demo_timeseries"][t_token]
        brief_temp = st.session_state["demo_health_scores"][t_token]
    else:
        df_temp = get_latest_metrics_db(f"{t_token}USDT")
        brief_temp = get_latest_health_score_db(t_token)
        if df_temp.empty:
            df_temp = st.session_state["demo_timeseries"][t_token]
            brief_temp = st.session_state["demo_health_scores"][t_token]
            
    preloaded_data[t_token] = (df_temp, brief_temp)
    
    # Identify the maximum timestamp of anomalies
    if not df_temp.empty:
        latest_r = df_temp.iloc[-1]
        if latest_r["is_anomaly"]:
            b_time = latest_r["bucket"]
            b_time_str = b_time.isoformat() if hasattr(b_time, "isoformat") else str(b_time)
            
            if latest_anomaly_time is None or b_time_str > latest_anomaly_time:
                latest_anomaly_time = b_time_str
                latest_anomaly_token = t_token

# Play notifications sound if a new anomaly alert is triggered (timestamp > last seen)
trigger_sound_alert = False
if latest_anomaly_time is not None:
    last_seen = st.session_state["last_seen_anomaly_time"]
    if last_seen is None or latest_anomaly_time > last_seen:
        trigger_sound_alert = True
        st.session_state["last_seen_anomaly_time"] = latest_anomaly_time

if trigger_sound_alert and AUDIO_BASE64:
    audio_placeholder.markdown(
        f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{AUDIO_BASE64}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True
    )


TRACKED_TOKENS = ["BTC", "ETH", "SOL", "BNB", "AVAX"]
FULL_NAMES = {
    "BTC": "Bitcoin (BTC)",
    "ETH": "Ethereum (ETH)",
    "SOL": "Solana (SOL)",
    "BNB": "Binance Coin (BNB)",
    "AVAX": "Avalanche (AVAX)",
}


def get_pulse_indicator(is_anomaly_val: bool, score_val: int) -> str:
    """Glowing pulse dot: green for high-conviction breakouts, red for critical threats, amber otherwise."""
    if not is_anomaly_val:
        return ""
    if score_val >= 80:
        return '<span class="indicator-pulse-green" title="Positive Breakout / Volatility Alert"></span>'
    if score_val <= 40:
        return '<span class="indicator-pulse-red" title="Critical Structural Anomaly Warning"></span>'
    return '<span class="indicator-pulse-amber" title="Moderate Volatility Anomaly Warning"></span>'


def fmt_compact(v: float) -> str:
    """Human-compact number: 12.4K / 3.10M / 1.25B."""
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.2f}B"
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:,.0f}"


def fmt_usd_signed(v: float) -> str:
    sign = "-" if v < 0 else "+"
    return f"{sign}${fmt_compact(abs(v))}"


def score_css_class(score_val: int) -> str:
    return "score-excellent" if score_val >= 70 else "score-warning" if score_val >= 40 else "score-critical"


# ── 5a. Market Overview KPI Strip (all tokens at a glance) ──────────

kpi_cols = st.columns(5)
for tk_idx, tk in enumerate(TRACKED_TOKENS):
    df_k, brief_k = preloaded_data[tk]
    last_k = df_k.iloc[-1]
    first_k = df_k.iloc[0]
    delta_pct = ((last_k["close_price"] / first_k["close_price"]) - 1.0) * 100.0
    score_k = brief_k["health_score"] if brief_k else 50
    pulse_k = get_pulse_indicator(bool(last_k["is_anomaly"]), score_k)
    delta_cls = "kpi-delta-up" if delta_pct >= 0 else "kpi-delta-down"
    arrow = "▲" if delta_pct >= 0 else "▼"

    with kpi_cols[tk_idx]:
        with st.container(border=True, key=f"kpi_card_{tk.lower()}"):
            st.markdown(
                f"""
                <div class="kpi-token">{pulse_k}{tk}</div>
                <div class="kpi-price">${last_k['close_price']:,.2f}</div>
                <div class="{delta_cls}">{arrow} {delta_pct:+.2f}% / 2h</div>
                <div class="ai-card-score {score_css_class(score_k)}" style="margin-bottom: 12px;">{score_k}/100</div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)


# ── 5b. Per-Token Deep-Dive Tabs (large charts + full AI briefing) ──

tab_objs = st.tabs([FULL_NAMES[t] for t in TRACKED_TOKENS])

for tab, token in zip(tab_objs, TRACKED_TOKENS):
    with tab:
        token_df, token_brief = preloaded_data[token]
        latest_row = token_df.iloc[-1]
        token_is_anomaly = bool(latest_row["is_anomaly"])
        severity = str(latest_row["severity"])
        token_health_score = token_brief["health_score"] if token_brief else 50
        explanation = token_brief["explanation"] if token_brief else "No LLM analysis found."
        reasoning = (token_brief.get("reasoning") or "") if token_brief else ""
        pulse_indicator = get_pulse_indicator(token_is_anomaly, token_health_score)

        delta_pct = ((latest_row["close_price"] / token_df.iloc[0]["close_price"]) - 1.0) * 100.0
        delta_cls = "kpi-delta-up" if delta_pct >= 0 else "kpi-delta-down"
        arrow = "▲" if delta_pct >= 0 else "▼"
        sev_cls = {"CRITICAL": "sev-critical", "HIGH": "sev-high"}.get(severity, "sev-normal")

        # ── Row 1: Market Snapshot + AI / LLM Briefing ──
        top_cols = st.columns([1, 1.5])

        with top_cols[0]:
            with st.container(border=True, key=f"snapshot_{token.lower()}"):
                st.markdown(
                    f"""
                    <div class="snap-token">{pulse_indicator}{FULL_NAMES[token]}</div>
                    <div class="snap-price">${latest_row['close_price']:,.2f}</div>
                    <div class="{delta_cls}" style="font-size: 15px;">{arrow} {delta_pct:+.2f}% over window</div>
                    <div class="stat-grid">
                        <div><div class="stat-label">Volume 5m</div><div class="stat-value">{fmt_compact(latest_row['volume_5m'])}</div></div>
                        <div><div class="stat-label">CEX Net Flow</div><div class="stat-value">{fmt_usd_signed(latest_row['net_cex_flow_usd'])}</div></div>
                        <div><div class="stat-label">Sentiment</div><div class="stat-value">{latest_row['sentiment_score']:+.2f}</div></div>
                        <div><div class="stat-label">Tweets / 5m</div><div class="stat-value">{int(latest_row['tweet_count'])}</div></div>
                        <div><div class="stat-label">Book Imbalance</div><div class="stat-value">{latest_row['imbalance']:+.2f}</div></div>
                        <div><div class="stat-label">AE Error (MSE)</div><div class="stat-value">{latest_row['mse_score']:.4f}</div></div>
                    </div>
                    <div class="sev-chip {sev_cls}">{severity}</div>
                    """,
                    unsafe_allow_html=True,
                )

        with top_cols[1]:
            with st.container(border=True, key=f"ai_briefing_{token.lower()}"):
                st.markdown(
                    f"""
                    <div class="ai-card-title">🧠 AI &amp; LLM Analysis</div>
                    <div class="ai-card-score {score_css_class(token_health_score)}">LLM Health Score: {token_health_score}/100</div>
                    <div class="ai-reasoning">{reasoning}</div>
                    <p class="ai-card-explanation">{explanation}</p>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Row 2: Price + Volume large charts ──
        chart_row_1 = st.columns(2)

        with chart_row_1[0]:
            with st.container(border=True, key=f"chart_price_{token.lower()}"):
                st.markdown(
                    f'<div class="chart-card-head"><span class="chart-title" style="color: #67e8f9;">Price</span>'
                    f'<span class="chart-value">${latest_row["close_price"]:,.2f}</span></div>',
                    unsafe_allow_html=True,
                )
                st.altair_chart(
                    create_glass_chart(token_df, "close_price", "#22d3ee", "Price", "$,.2f", token, axis_format=",.0f"),
                    width="stretch",
                )

        with chart_row_1[1]:
            with st.container(border=True, key=f"chart_volume_{token.lower()}"):
                st.markdown(
                    f'<div class="chart-card-head"><span class="chart-title" style="color: #d8b4fe;">Volume (5m)</span>'
                    f'<span class="chart-value">{fmt_compact(latest_row["volume_5m"])}</span></div>',
                    unsafe_allow_html=True,
                )
                st.altair_chart(
                    create_glass_chart(token_df, "volume_5m", "#c084fc", "Volume (5m)", ",.0f", token, axis_format="~s"),
                    width="stretch",
                )

        # ── Row 3: CEX Net Flow + Sentiment large charts ──
        chart_row_2 = st.columns(2)

        with chart_row_2[0]:
            with st.container(border=True, key=f"chart_cex_{token.lower()}"):
                st.markdown(
                    f'<div class="chart-card-head"><span class="chart-title" style="color: #6ee7b7;">CEX Net Flow</span>'
                    f'<span class="chart-value">{fmt_usd_signed(latest_row["net_cex_flow_usd"])}</span></div>',
                    unsafe_allow_html=True,
                )
                st.altair_chart(
                    create_glass_chart(token_df, "net_cex_flow_usd", "#34d399", "CEX Net Flow", "$,.0f", token, axis_format="~s"),
                    width="stretch",
                )

        with chart_row_2[1]:
            with st.container(border=True, key=f"chart_sentiment_{token.lower()}"):
                st.markdown(
                    f'<div class="chart-card-head"><span class="chart-title" style="color: #fde68a;">Social Sentiment</span>'
                    f'<span class="chart-value">{latest_row["sentiment_score"]:+.2f}</span></div>',
                    unsafe_allow_html=True,
                )
                st.altair_chart(
                    create_glass_chart(token_df, "sentiment_score", "#fbbf24", "Sentiment Score", "+.2f", token, axis_format="+.2f"),
                    width="stretch",
                )

st.markdown("<br><br>", unsafe_allow_html=True)

# Simple footer
st.markdown(
    """
    <div style='text-align: center; padding: 20px; color: #506176; font-size: 13px; border-top: 1px solid rgba(255,255,255,0.05);'>
        🛡️ CryptoSense // Secure Multi-Coin Radar Engine & strategic AI Decision Array. Connected natively to host GPU Ollama.
    </div>
    """,
    unsafe_allow_html=True,
)

# Reset the sound/shake trigger at the very end of the page execution so it doesn't repeat on passive refreshes
if st.session_state.get("fresh_anomaly_trigger", False):
    st.session_state["fresh_anomaly_trigger"] = False
