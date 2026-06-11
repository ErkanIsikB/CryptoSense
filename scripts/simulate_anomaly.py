import asyncio
import json
import time
from datetime import datetime, timezone
from pprint import pprint

import ollama

from src.db.db import execute_query_fetch_async
from src.models.llm_pipeline import build_user_prompt, CryptoSenseBrief, calculate_deterministic_health_score

async def simulate_anomaly():
    print("==========================================================")
    print("🚨 CRYPTOSENSE V2.1: ANOMALY SIMULATION MODULE 🚨")
    print("==========================================================")
    print("Senaryo: BTC için son 1 saatin verilerini veritabanından çekip,")
    print("son muma (şu an) Trump'ın 'İran'a yaptırım' açıklaması kaynaklı")
    print("devasa bir panik satış anomalisini enjekte edeceğiz.")
    print("Ardından sistemi çalıştırıp LLM'in bu krizi nasıl yorumladığına bakacağız.\n")

    symbol = "BTC"
    
    # 1. Fetch real recent data (last 11 buckets)
    sql = """
        SELECT bucket, symbol, llm_payload 
        FROM ai_anomalies_5m 
        WHERE symbol = %s 
        ORDER BY bucket DESC 
        LIMIT 11;
    """
    rows = await execute_query_fetch_async(sql, (symbol,))
    if not rows or len(rows) < 11:
        print("Yeterli veri yok. Simülasyon için veritabanında en az 11 kayıt olmalıdır.")
        return

    # Reverse rows to chronological
    rows = rows[::-1]
    
    clean_sequence = []
    for r in rows:
        payload = r[2] if isinstance(r[2], dict) else json.loads(r[2])
        if "AI_ENGINE" in payload:
            del payload["AI_ENGINE"]
        clean_sequence.append(payload)

    # 2. Inject FAKE ANOMALOUS 12th BUCKET (The Crisis)
    fake_bucket_time = datetime.now(timezone.utc)
    fake_payload = {
        "timestamp": fake_bucket_time.isoformat(),
        "symbol": "BTC",
        "market_data": {
            "close_price": 58000.00,  # Sudden drop
            "volume_5m": 9500.0,      # MASSIVE volume spike
            "vwap": 59000.00,
            "net_trade": -4000.0
        },
        "orderbook": {
            "avg_spread": 15.5,           # Wide spread (liquidity drained)
            "avg_imbalance": -0.85        # Huge seller dominance
        },
        "sentiment": {
            "retail_avg_score": -0.92,       # Extreme panic sentiment
            "institutional_avg_score": -0.85, # Extreme institutional panic sentiment
            "tweet_count": 4500,
            "positive_count": 100,
            "negative_count": 4000,
            "neutral_count": 400
        },
        "on_chain": {
            "net_cex_flow_usd": 250_000_000.0  # $250M entered exchanges to dump
        }
    }
    
    fake_macro_alert = {
        "is_anomaly": True,
        "mse_score": 0.045,           # Way above 0.008 threshold
        "threshold": 0.008,
        "severity": "CRITICAL"
    }

    clean_sequence.append(fake_payload)
    
    ctx = {
        "latest_bucket": fake_bucket_time,
        "macro_alert": fake_macro_alert,
        "clean_sequence": clean_sequence,
        "raw_payload": fake_payload
    }

    print("📊 ADIM 1: Veri Paketi (Payload) Oluşturuldu.")
    print("Enjekte Edilen Kriz Verileri:")
    print(f" - Emir Defteri İhtilafı (Imbalance): {fake_payload['orderbook']['avg_imbalance']}")
    print(f" - Duygu Puanı (Sentiment): {fake_payload['sentiment']['retail_avg_score']}")
    print(f" - Borsalara Giren Para (On-Chain): ${fake_payload['on_chain']['net_cex_flow_usd']:,.2f}")
    print(f" - PyTorch Anomali Kararı: {fake_macro_alert['is_anomaly']} (MSE: {fake_macro_alert['mse_score']})\n")

    # 3. Calculate Deterministic Score
    deterministic_score = calculate_deterministic_health_score(ctx)
    print("🧠 ADIM 2: Kod Tarafından Deterministik Sağlık Skoru Hesaplanıyor...")
    print(f"✅ HESAPLANAN SKOR: {deterministic_score} / 100\n")
    
    # 4. Ask Qwen to interpret the deterministic score
    print("🤖 ADIM 3: Qwen 2.5 Büyük Dil Modeli Uyandırılıyor...")
    print("LLM'den bu deterministik skoru finansal prensiplere göre açıklaması isteniyor...\n")
    
    user_prompt = build_user_prompt(symbol, ctx, deterministic_score)
    
    start_time = time.time()
    response = await asyncio.to_thread(
        ollama.chat,
        model="qwen2.5:7b",
        messages=[
            {"role": "system", "content": "You are an Expert Portfolio Manager and Macro Analyst for CryptoSense. Your single job is to interpret a 12-candle data sequence and a deterministically calculated Health Score (0-100). CRITICAL INSTRUCTIONS: You DO NOT calculate the health score. You explain WHY it is what it is using traditional trading principles. You must respond exclusively using the structured JSON keys provided."},
            {"role": "user", "content": user_prompt},
        ],
        format=CryptoSenseBrief.model_json_schema(),
        options={"temperature": 0.0, "top_p": 0.1},
    )
    
    raw_json = response["message"]["content"]
    parsed_result = json.loads(raw_json)
    latency = int((time.time() - start_time) * 1000)

    print("==========================================================")
    print(f"🎯 LLM ÇIKTISI (Süre: {latency} ms)")
    print("==========================================================")
    print(f"Ana İtici Güç (Primary Driver): {parsed_result['primary_metric_driver']}")
    print(f"Güvenilirlik Sınıfı (Trust): {parsed_result['trustworthiness_classification']}")
    print(f"\n📝 Uzman Yönetici Yorumu:")
    print(parsed_result['market_trajectory_summary'])
    print("==========================================================")

if __name__ == "__main__":
    asyncio.run(simulate_anomaly())
