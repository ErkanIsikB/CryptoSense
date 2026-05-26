"""CryptoSense Premium Multi-Coin Grid Dashboard — High-Performance Factual Decision Interface."""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import warnings

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
except Exception as e:
    LOGGER.warning("Could not initialize database connection pool. Running in offline-only mode: %s", e)

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


# Custom Premium CSS Styling (Dynamic 3-color wavy gradient background, glassmorphism, glowing micro-animations)
st.markdown(
    """
    <div class="liquid-bg-container">
        <div class="liquid-blob liquid-blob-1"></div>
        <div class="liquid-blob liquid-blob-2"></div>
        <div class="liquid-blob liquid-blob-3"></div>
    </div>
    <style>
        /* Import Outfit Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* ── Dynamic Organic Liquid Background (Pitch Black cosmic base with vibrant slow color splashes) ── */
        :root, .stApp {
            --background-color: transparent !important;
            --secondary-background-color: rgba(255, 255, 255, 0.02) !important;
        }

        /* ── Aggressive CSS Override to Reveal Liquid Background through Streamlit ── */
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"], [data-testid="stHeader"], .main, .stMain, [data-testid="stMainTemplate"], .block-container, [data-testid="stVerticalBlock"], .element-container, [data-testid="element-container"], div[class*="st-emotion-cache"] {
            background-color: transparent !important;
            background: transparent !important;
            font-family: 'Outfit', sans-serif;
            color: #ecf0f1;
        }

        html, body {
            margin: 0;
            padding: 0;
            background-color: #020205 !important; /* Base fallback */
        }

        .liquid-bg-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: -1;
            overflow: hidden;
            background-color: #020205 !important;
            background: #020205 !important;
            pointer-events: none;
        }
        
        .liquid-blob {
            position: absolute;
            border-radius: 50%;
            filter: blur(140px);
            opacity: 0.42; /* Highly intense floating splashes to drive gorgeous glass refraction */
            mix-blend-mode: screen;
            pointer-events: none;
            will-change: transform;
        }
        
        .liquid-blob-1 {
            width: 550px;
            height: 550px;
            background: radial-gradient(circle, #6366f1 0%, #312e81 40%, rgba(49, 46, 129, 0) 70%); /* Saturated Neon Indigo Core */
            top: -15%;
            left: 10%;
            animation: float-blob-1 25s ease-in-out infinite alternate;
        }
        
        .liquid-blob-2 {
            width: 480px;
            height: 480px;
            background: radial-gradient(circle, #14b8a6 0%, #0f766e 40%, rgba(15, 118, 110, 0) 70%); /* Saturated Neon Teal Core */
            bottom: -5%;
            right: 15%;
            animation: float-blob-2 28s ease-in-out infinite alternate;
        }
        
        .liquid-blob-3 {
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, #d946ef 0%, #581c87 40%, rgba(88, 28, 135, 0) 70%); /* Saturated Neon Violet/Magenta Core */
            top: 25%;
            left: 35%;
            animation: float-blob-3 26s ease-in-out infinite alternate;
        }
        
        @keyframes float-blob-1 {
            0% { transform: translate3d(-40px, -20px, 0) scale(0.95); }
            50% { transform: translate3d(60px, 40px, 0) scale(1.05); }
            100% { transform: translate3d(-40px, -20px, 0) scale(0.95); }
        }
        
        @keyframes float-blob-2 {
            0% { transform: translate3d(60px, 30px, 0) scale(1.05); }
            50% { transform: translate3d(-50px, -45px, 0) scale(0.95); }
            100% { transform: translate3d(60px, 30px, 0) scale(1.05); }
        }
        
        @keyframes float-blob-3 {
            0% { transform: translate3d(-30px, 40px, 0) scale(0.97); }
            50% { transform: translate3d(45px, -35px, 0) scale(1.03); }
            100% { transform: translate3d(-30px, 40px, 0) scale(0.97); }
        }

        /* ── Prevent Streamlit's "stale" fade-out during reruns ── */
        div.element-container, 
        [data-testid="stVerticalBlock"],
        [data-testid="stAppViewContainer"] {
            opacity: 1 !important;
            transition: none !important;
        }

        /* ── Premium iOS Dark-Tinted Liquid Glass Panel (Frosted Saturation, Gloss highlights, Bezel light-catch) ── */
        div[data-testid="stVerticalBlockBorderDiv"] {
            background: rgba(10, 8, 20, 0.55) !important;
            background-image: 
                linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.01) 100%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.03) 0%, transparent 40%) !important; /* Gloss reflection sheen! */
            backdrop-filter: blur(35px) saturate(220%) contrast(115%) !important;
            -webkit-backdrop-filter: blur(35px) saturate(220%) contrast(115%) !important;
            border: 1.5px solid rgba(255, 255, 255, 0.06) !important;
            border-left: 1.5px solid rgba(255, 255, 255, 0.16) !important; /* Bezel light catch! */
            border-top: 1.5px solid rgba(255, 255, 255, 0.16) !important;  /* Bezel light catch! */
            border-radius: 20px !important;
            padding: 16px !important;
            box-shadow: 
                0 20px 40px 0 rgba(0, 0, 0, 0.7), 
                inset 0 1px 1px 0 rgba(255, 255, 255, 0.16), /* Top-left inner catch light! */
                inset 0 -1px 1px 0 rgba(0, 0, 0, 0.4) !important;
            
            /* High-Performance GPU Hardware Acceleration Overrides */
            transform: translate3d(0, 0, 0) !important;
            will-change: transform, backdrop-filter !important;
            
            animation: subtle-fade-in 0.4s ease-out !important;
            transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.3s ease !important;
        }
        
        @keyframes subtle-fade-in {
            from { opacity: 0; transform: translate3d(0, 5px, 0); }
            to { opacity: 1; transform: translate3d(0, 0, 0); }
        }
        
        /* ── Premium smooth border glow & elevation transition ── */
        div[data-testid="stVerticalBlockBorderDiv"]:hover {
            border-color: rgba(99, 102, 241, 0.45) !important; /* Elegant Indigo Bezel Glow */
            transform: translate3d(0, -3px, 0) !important; /* Subtle premium elevation */
            box-shadow: 
                0 25px 50px 0 rgba(0, 0, 0, 0.8), 
                0 0 20px 0 rgba(99, 102, 241, 0.18), 
                inset 0 1px 1px 0 rgba(255, 255, 255, 0.25) !important;
        }

        /* ── Perfect Alignment styling override for Header containers containing .header-marker ── */
        div[data-testid="stVerticalBlockBorderDiv"]:has(.header-marker) {
            background: rgba(255, 255, 255, 0.02) !important;
            background-image: linear-gradient(180deg, rgba(255, 255, 255, 0.05) 0%, transparent 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-left: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-top: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 10px !important;
            padding: 3px 10px !important; /* Shrunk vertical padding to compress box height */
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
            transform: none !important;
            transition: none !important;
            min-height: auto !important; /* Forces layout to collapse tightly */
        }
        
        div[data-testid="stVerticalBlockBorderDiv"]:has(.header-marker):hover {
            border-color: rgba(255, 255, 255, 0.06) !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
            transform: none !important;
        }

        /* ── Hide 'Show data' and Grid Table view buttons inside elements toolbar ── */
        div[data-testid="stVegaLiteChart"] [data-testid="stElementToolbar"] button:not(:last-child) {
            display: none !important;
        }

        /* ── Transparent child override inside glass containers ── */
        div[data-testid="stVerticalBlockBorderDiv"] div {
            background-color: transparent !important;
        }

        /* ── Column Headers Visual styling ── */
        .col-header-box {
            text-align: center;
            font-weight: 800;
            text-transform: uppercase;
            font-size: 16.5px; /* Upscaled typography for bold premium visibility */
            letter-spacing: 2.2px;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.15 !important; /* Tight line height for ultra-compact layout */
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.25); /* soft clean premium glow */
        }

        .header-volume { color: #d8b4fe; }
        .header-price { color: #a5f3fc; }
        .header-inflow { color: #a7f3d0; }
        .header-sentiment { color: #fde68a; }
        .header-ai { color: #c7d2fe; }

        /* ── AI Analysis Card Components ── */
        .ai-card-title {
            font-size: 16px;
            font-weight: 800;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
        }
        
        .ai-card-score {
            font-size: 15px;
            font-weight: 800;
            margin-bottom: 10px;
            padding: 4px 10px;
            border-radius: 6px;
            display: inline-block;
        }
        
        .score-excellent { background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; }
        .score-warning { background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fbbf24; }
        .score-critical { background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #f87171; }

        .ai-card-explanation {
            font-size: 12px;
            line-height: 1.5;
            color: #cbd5e1;
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
        <h2 style='color: #6366f1; font-weight: 800; letter-spacing: 2px; margin-bottom: 0px;'>🛡️ CRYPTOSENSE</h2>
        <p style='color: #8fa0b5; font-size: 11px; text-transform: uppercase; font-weight: 600;'>AI Shield & Decision System</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Toggle for Offline Demo Mode
st.sidebar.markdown("### 🛠️ UI Mode Settings")
demo_mode = st.sidebar.toggle(
    "Enable Demo Offline Mode",
    value=(not HAS_DB),  # Auto-default to True if database pool isn't loaded!
    help="Enable this to preview the dashboard visuals and glowing AI alerts with high-fidelity simulated data without querying any database or backend APIs.",
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
        df = st.session_state["demo_timeseries"][token]
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
        df_updated = pd.concat([df, new_df], ignore_index=True)
        
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
            <h1 style='font-weight: 800; font-size: 44px; letter-spacing: 2px; color: #ffffff; text-shadow: 0 0 15px rgba(99, 102, 241, 0.4); margin-bottom: 0px;'>CRYPTO ANALYTICS DASHBOARD</h1>
            <p style='color: #8fa0b5; font-size: 15px; margin-top: 4px; letter-spacing: 1px; text-transform: uppercase;'>Real-time Token Analysis & LLM Insights</p>
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


# Generate Column Headers
head_cols = st.columns([1, 1, 1, 1, 1.8])

with head_cols[0]:
    with st.container(border=True):
        st.markdown('<span class="header-marker"></span><div class="col-header-box header-volume">Volume</div>', unsafe_allow_html=True)
with head_cols[1]:
    with st.container(border=True):
        st.markdown('<span class="header-marker"></span><div class="col-header-box header-price">Price</div>', unsafe_allow_html=True)
with head_cols[2]:
    with st.container(border=True):
        st.markdown('<span class="header-marker"></span><div class="col-header-box header-inflow">Net Inflow</div>', unsafe_allow_html=True)
with head_cols[3]:
    with st.container(border=True):
        st.markdown('<span class="header-marker"></span><div class="col-header-box header-sentiment">Sentiment</div>', unsafe_allow_html=True)
with head_cols[4]:
    with st.container(border=True):
        st.markdown('<span class="header-marker"></span><div class="col-header-box header-ai">AI Analysis</div>', unsafe_allow_html=True)


def create_premium_sparkline(df_reset, y_column, line_color, value_title, value_format, token):
    # Selection point tracking the nearest X coordinate strictly based on X horizontal distance
    # With a completely unique selection name mapping strictly per sparkline to resolve Vega deduplication warnings!
    # With clear="mouseout" to ensure the hover dot is killed instantly when cursor leaves
    hover_selection = alt.selection_point(
        name=f"hover_{token.lower()}_{y_column}",
        on="mouseover",
        nearest=True,
        fields=["index"],
        encodings=["x"],  # Ignores Y-axis height and tracks strictly along the timeline!
        clear="mouseout",
        empty=False
    )
    
    # 1. Base Line Chart
    line = alt.Chart(df_reset).mark_line(
        color=line_color, strokeWidth=2.5, interpolate="monotone"
    ).encode(
        x=alt.X("index:Q", axis=alt.Axis(
            title=None, 
            labels=True, 
            ticks=False, 
            grid=False,
            labelPadding=0,
            labelFontSize=9,
            labelColor="#8fa0b5"
        )),
        y=alt.Y(f"{y_column}:Q", scale=alt.Scale(zero=False), axis=alt.Axis(
            title=None,
            labels=True,
            ticks=False,
            grid=True,
            gridColor="rgba(255, 255, 255, 0.03)",
            labelColor="#8fa0b5",
            labelFontSize=9,
            labelPadding=4
        ))
    )
    
    # 2. Glowing hover dot that snaps exactly on the line based on horizontal mouse position
    # This also acts as the transparent hover grabber and carries the value-only tooltip.
    points = alt.Chart(df_reset).mark_point(
        color=line_color, size=65, filled=True, stroke="white", strokeWidth=1.5
    ).encode(
        x="index:Q",
        y=f"{y_column}:Q",
        opacity=alt.condition(hover_selection, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip(f"{y_column}:Q", title=value_title, format=value_format)
        ]
    ).add_params(
        hover_selection
    )
    
    # Combine layers together into a super fast 2-layer compiled plot
    chart = alt.layer(
        line, points
    ).properties(
        height=140,
        usermeta={"embedOptions": {"actions": False}}
    ).configure_view(
        strokeWidth=0
    )
    
    return chart


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

for token in TRACKED_TOKENS:
    # 1. Retrieve Preloaded Data
    df, llm_brief = preloaded_data[token]
    latest_row = df.iloc[-1]
    is_anomaly = latest_row["is_anomaly"]
    severity = latest_row["severity"]
    health_score = llm_brief["health_score"] if llm_brief else 50
    explanation = llm_brief["explanation"] if llm_brief else "No LLM analysis found."

    # 🚨 SMART UX PULSE INDICATOR SYSTEM:
    # Glow emerald green for high-conviction positive breakouts (>= 80),
    # neon amber for moderate/warning anomalies (between 41 and 79),
    # and pulsing red for critical threats (<= 40).
    if is_anomaly:
        if health_score >= 80:
            pulse_indicator = '<span class="indicator-pulse-green" title="Positive Breakout / Volatility Alert"></span>'
        elif health_score <= 40:
            pulse_indicator = '<span class="indicator-pulse-red" title="Critical Structural Anomaly Warning"></span>'
        else:
            pulse_indicator = '<span class="indicator-pulse-amber" title="Moderate Volatility Anomaly Warning"></span>'
    else:
        pulse_indicator = ""

    # Reset dataframe index to map cleanly as sequential sparkline coordinates
    df_reset = df.reset_index()

    # Render row columns
    row_cols = st.columns([1, 1, 1, 1, 1.8])

    # 📈 Column 1: Volume Sparkline (Altair-Optimized, Snapping Hover dot, Auto-scale Y, faint gridlines)
    with row_cols[0]:
        with st.container(border=True):
            st.markdown(f"<div style='font-size: 11.5px; font-weight: 800; text-transform: uppercase; color: #d8b4fe; margin-bottom: 2px;'>{token} Vol</div>", unsafe_allow_html=True)
            vol_chart = create_premium_sparkline(df_reset, "volume_5m", "#c084fc", "Volume (5m)", ",.0f", token)
            st.altair_chart(vol_chart, width="stretch")

    # 📈 Column 2: Price Sparkline (Altair-Optimized, Snapping Hover dot, Auto-scale Y, faint gridlines)
    with row_cols[1]:
        with st.container(border=True):
            st.markdown(f"<div style='font-size: 11.5px; font-weight: 800; text-transform: uppercase; color: #a5f3fc; margin-bottom: 2px;'>{token} Price: ${latest_row['close_price']:,.2f}</div>", unsafe_allow_html=True)
            price_chart = create_premium_sparkline(df_reset, "close_price", "#22d3ee", "Price", "$,.2f", token)
            st.altair_chart(price_chart, width="stretch")

    # 📈 Column 3: Net Inflow Sparkline (Altair-Optimized, Snapping Hover dot, Auto-scale Y, faint gridlines)
    with row_cols[2]:
        with st.container(border=True):
            st.markdown(f"<div style='font-size: 11.5px; font-weight: 800; text-transform: uppercase; color: #a7f3d0; margin-bottom: 2px;'>{token} CEX: ${latest_row['net_cex_flow_usd']:+,.0f}</div>", unsafe_allow_html=True)
            flow_chart = create_premium_sparkline(df_reset, "net_cex_flow_usd", "#34d399", "CEX Net Flow", "$,.0f", token)
            st.altair_chart(flow_chart, width="stretch")

    # 📈 Column 4: Sentiment Sparkline (Altair-Optimized, Snapping Hover dot, Auto-scale Y, faint gridlines)
    with row_cols[3]:
        with st.container(border=True):
            st.markdown(f"<div style='font-size: 11.5px; font-weight: 800; text-transform: uppercase; color: #fde68a; margin-bottom: 2px;'>{token} Sent: {latest_row['sentiment_score']:+.2f}</div>", unsafe_allow_html=True)
            sent_chart = create_premium_sparkline(df_reset, "sentiment_score", "#fbbf24", "Sentiment Score", "+.2f", token)
            st.altair_chart(sent_chart, width="stretch")

    # 🧠 Column 5: AI & LLM Analysis Card
    with row_cols[4]:
        with st.container(border=True):
            # Determine color-coded severity class
            score_class = "score-excellent" if health_score >= 70 else "score-warning" if health_score >= 40 else "score-critical"

            st.markdown(
                f"""
                <div class="ai-card-title">{pulse_indicator}{FULL_NAMES[token]}</div>
                <div class="ai-card-score {score_class}">LLM Score: {health_score}/100</div>
                <p class="ai-card-explanation">"{explanation}"</p>
                """,
                unsafe_allow_html=True,
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
