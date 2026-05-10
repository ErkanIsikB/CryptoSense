"""Periodic crypto sentiment tracker using the Tavily search API."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests
import schedule

from src.core.config import settings
from src.sinks.base import BaseSink
from src.sinks.jsonl_sink import JsonlFileSink
from src.feature_engineering.sentiment_scorer import score_and_store as _score_and_store_to_db

LOGGER = logging.getLogger("sentiment_tracker")

TOKENS: list[dict[str, str]] = [
    {"name": "Bitcoin", "symbol": "BTC"},
    {"name": "Ethereum", "symbol": "ETH"},
    {"name": "Solana", "symbol": "SOL"},
    {"name": "BNB", "symbol": "BNB"},
    {"name": "Avalanche", "symbol": "AVAX"},
]

OUTPUT_DIR = settings.DATA_DIR / "sentiment"
STATE_FILE = OUTPUT_DIR / ".last_run.json"
URL_DEDUP_WINDOW_SECONDS = 86400


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    candidate = value.strip()
    if not candidate:
        return None

    normalized = candidate.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        pass

    for pattern in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
    ):
        try:
            parsed = datetime.strptime(candidate, pattern)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            continue
    return None


def _load_state() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        return {
            "last_successful_run_timestamp": {},
            "seen_urls": {},
        }

    try:
        with STATE_FILE.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {
            "last_successful_run_timestamp": {},
            "seen_urls": {},
        }

    return {
        "last_successful_run_timestamp": raw.get("last_successful_run_timestamp", {}),
        "seen_urls": raw.get("seen_urls", {}),
    }


def _save_state(state: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, separators=(",", ":"))


def _normalize_url(url: str) -> str:
    candidate = url.strip()
    if not candidate:
        return ""

    parsed = urlparse(candidate)
    if not parsed.scheme and not parsed.netloc and candidate.startswith("x.com"):
        parsed = urlparse(f"https://{candidate}")

    if not parsed.netloc:
        return ""

    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    normalized = urlunparse(
        (
            parsed.scheme or "https",
            netloc,
            parsed.path.rstrip("/"),
            "",
            "",
            "",
        )
    )
    return normalized


def _is_authentic_status_url(url: str) -> bool:
    normalized = _normalize_url(url)
    if not normalized:
        return False
    parsed = urlparse(normalized)
    
    # Hem x.com hem de twitter.com domainlerine izin verildi
    is_valid_domain = parsed.netloc.endswith("x.com") or parsed.netloc.endswith("twitter.com")
    
    return is_valid_domain and "/status/" in parsed.path


def _is_meaningful_content(content: str) -> bool:
    text = content.strip()
    if not text:
        return False

    if len(text) < 20:
        return False

    alpha_count = sum(1 for char in text if char.isalpha())
    if alpha_count < 10:
        return False

    lowered = text.lower()
    metadata_markers = (
        "lang=",
        "src=",
        "ref_src",
        "x.com/search",
        " q=",
        "utm_",
    )
    if any(marker in lowered for marker in metadata_markers) and len(text) < 140:
        return False

    return True


def _result_matches_token(item: dict[str, Any], token: dict[str, str]) -> bool:
    combined = " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("content") or ""),
            str(item.get("url") or ""),
            str(item.get("raw_content") or ""),
        ]
    ).lower()
    return token["name"].lower() in combined or token["symbol"].lower() in combined


def build_query(token: dict[str, str]) -> str:
    """
    Daha fazla hacim elde etmek için otorite (hesap) filtresi kaldırılmış,
    sadece token'ın kendisi ve piyasa etkisine odaklanan optimize query.
    """
    name = token["name"]
    symbol = token["symbol"]
    
    if name.lower() == symbol.lower():
        token_identifier = f'"{symbol}"'
    else:
        token_identifier = f'("{name}" OR "{symbol}")'

    # Fiyatı etkileyen anahtar kelimeleri biraz daha genişlettik
    impact_keywords = "(breaking news OR market alert OR sentiment OR price OR crypto OR update)"

    return f"{token_identifier} {impact_keywords}"


def _request_with_retry(payload: dict[str, Any]) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(settings.SENTIMENT_MAX_RETRIES + 1):
        try:
            response = requests.post(
                settings.TAVILY_API_URL, json=payload, timeout=settings.SENTIMENT_TIMEOUT_S
            )
            if response.status_code == 429:
                wait_seconds = min(30, 2**attempt)
                LOGGER.warning("tavily_rate_limited wait_s=%s attempt=%s", wait_seconds, attempt + 1)
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= settings.SENTIMENT_MAX_RETRIES:
                break
            wait_seconds = min(20, 2**attempt)
            LOGGER.warning("tavily_retry wait_s=%s attempt=%s error=%s", wait_seconds, attempt + 1, exc)
            time.sleep(wait_seconds)

    raise RuntimeError(f"Tavily request failed after retries: {last_error}")


def fetch_token_sentiment(
    token_group: list[dict[str, str]],
    last_successful_run_by_token: dict[str, float | None],
    seen_urls_by_token: dict[str, dict[str, Any]],
    scan_started_ts: float,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, dict[str, float]]]:
    
    current_token = token_group[0] 

    payload = {
        "api_key": settings.TAVILY_API_KEY,
        "query": build_query(current_token), 
        "topic": "news",
        "search_depth": settings.SENTIMENT_SEARCH_DEPTH,
        "max_results": 5,           
        "include_answer": False, 
        "include_images": False, 
        "include_raw_content": False,
        "include_domains": ["x.com", "twitter.com"], 
        "time_range": "day", 
    }
    
    try:
        response = _request_with_retry(payload)
        data = response.json()
    except Exception as exc:
        symbols = [token["symbol"] for token in token_group]
        LOGGER.error("tavily_fetch_failed group=%s error=%s", symbols, exc)
        return [], {}, {}

    raw_results = data.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    cutoff_ts = scan_started_ts - URL_DEDUP_WINDOW_SECONDS
    normalized_seen_by_token: dict[str, dict[str, float]] = {}
    for token in token_group:
        symbol = token["symbol"]
        seen_urls_for_token = seen_urls_by_token.get(symbol, {})
        normalized_seen: dict[str, float] = {}
        for url, ts in seen_urls_for_token.items():
            parsed_ts = _parse_timestamp(ts)
            normalized_url = _normalize_url(str(url))
            if parsed_ts is None or not normalized_url:
                continue
            if parsed_ts >= cutoff_ts:
                normalized_seen[normalized_url] = parsed_ts
        normalized_seen_by_token[symbol] = normalized_seen

    filtered_results_by_token: dict[str, list[dict[str, Any]]] = {
        token["symbol"]: [] for token in token_group
    }
    max_processed_by_token: dict[str, float] = {}

    for item in raw_results:
            if not isinstance(item, dict):
                continue

            score = item.get("score")
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                score_value = 0.0

            if score_value < 0.3:
                continue

            raw_url = str(item.get("url") or item.get("link") or "")
            normalized_url = _normalize_url(raw_url)
            if not _is_authentic_status_url(normalized_url):
                continue

            content_text = str(item.get("content") or item.get("raw_content") or "")
            if not _is_meaningful_content(content_text):
                continue

            published_raw = item.get("published_date")
            published_ts = _parse_timestamp(published_raw)
            
            # B PLANI (Dürüst Ingestion):
            # Tavily tarih bulamazsa verinin sistemimiz tarafından 'şu an' işlendiğini dürüstçe kabul ediyoruz.
            # DİKKAT: continue kaldırıldı!
            if published_ts is None:
                published_ts = scan_started_ts

            # 24 SAATLİK KESİN FİLTRE (86400 saniye)
            if published_ts < (scan_started_ts - 86400):
                continue

            item_copy = dict(item)
            
            # Çıktıyı şişiren ham HTML/Metadata verisini JSON'dan atıyoruz
            item_copy.pop("raw_content", None) 
            
            item_copy["url"] = normalized_url
            item_copy["content"] = content_text.strip()
            
            # KRİTİK DÜZELTME: Hesaplanan veya çekilen tarihi kesin olarak JSON'a string olarak yazıyoruz!
            item_copy["published_date"] = datetime.fromtimestamp(
                published_ts, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z")

            for token in token_group:
                symbol = token["symbol"]

                if not _result_matches_token(item_copy, token):
                    continue

                token_last_successful_run_ts = last_successful_run_by_token.get(symbol)
                if token_last_successful_run_ts is not None and published_ts <= token_last_successful_run_ts:
                    continue

                token_seen_urls = normalized_seen_by_token[symbol]
                if normalized_url in token_seen_urls:
                    continue

                filtered_results_by_token[symbol].append(item_copy)
                token_seen_urls[normalized_url] = scan_started_ts
                max_processed_by_token[symbol] = max(max_processed_by_token.get(symbol, 0.0), published_ts)

    records: list[dict[str, Any]] = []
    for token in token_group:
        symbol = token["symbol"]
        token_results = filtered_results_by_token[symbol]
        if not token_results:
            continue

        records.append(
            {
                "event_type": "sentiment",
                "timestamp": _now_iso_utc(),
                "token": symbol,
                "query": payload["query"],
                "answer": data.get("answer"),
                "results": token_results,
                "response_time": data.get("response_time"),
            }
        )

    if not records:
        return [], {}, normalized_seen_by_token

    return records, max_processed_by_token, normalized_seen_by_token


def _append_sentiment(sink: BaseSink, record: dict[str, Any]) -> None:
    asyncio.run(sink.write("sentiment", record))
    # Also score and write to TimescaleDB
    try:
        _score_and_store_to_db(record)
    except Exception as exc:
        LOGGER.warning("sentiment_db_write_failed error=%s", exc)


def _fetch_crypto_sentiment_cycle(sink: BaseSink) -> None:
    if not settings.TAVILY_API_KEY:
        LOGGER.error("missing_env_var name=TAVILY_API_KEY")
        return

    selected_tokens = TOKENS[: settings.SENTIMENT_MAX_TOKENS_PER_CYCLE]

    LOGGER.info(
        "sentiment_cycle_start tokens=%s",
        [token["symbol"] for token in selected_tokens],
    )

    state = _load_state()
    last_run_by_token = state.get("last_successful_run_timestamp", {})
    seen_urls_by_token = state.get("seen_urls", {})

    state_dirty = False

    for token in selected_tokens:
        symbol = token["symbol"]
        scan_started_ts = time.time()

        last_successful_run_by_token: dict[str, float | None] = {
            symbol: _parse_timestamp(last_run_by_token.get(symbol))
        }
        seen_urls_for_group: dict[str, dict[str, Any]] = {
            symbol: seen_urls_by_token.get(symbol, {})
        }

        records, max_processed_by_token, updated_seen_urls_by_token = fetch_token_sentiment(
            token_group=[token], 
            last_successful_run_by_token=last_successful_run_by_token,
            seen_urls_by_token=seen_urls_for_group,
            scan_started_ts=scan_started_ts,
        )

        for sym, urls in updated_seen_urls_by_token.items():
            seen_urls_by_token[sym] = urls
        state_dirty = True

        if not records:
            LOGGER.info("sentiment_no_new_results token=%s", symbol)
            continue

        for record in records:
            try:
                _append_sentiment(sink, record)
            except Exception as exc:
                LOGGER.error("sentiment_write_failed token=%s error=%s", symbol, exc)
                continue

            effective_ts = max_processed_by_token.get(symbol, scan_started_ts)
            last_run_by_token[symbol] = datetime.fromtimestamp(effective_ts, tz=timezone.utc).isoformat().replace(
                "+00:00", "Z"
            )
            LOGGER.info("sentiment_written token=%s new_items=%s", symbol, len(record.get("results", [])))
            state_dirty = True

    if state_dirty:
        _save_state(
            {
                "last_successful_run_timestamp": last_run_by_token,
                "seen_urls": seen_urls_by_token,
            }
        )

def start_sentiment_stream(stop: asyncio.Event, sink: BaseSink | None = None) -> None:
    """Public entry point — blocks until *stop* is set.

    Parameters
    ----------
    stop : asyncio.Event
        Set to signal shutdown.
    sink : BaseSink | None
        If provided, records are also written to this sink.
        A JSONL file sink is always used for state persistence.
    """
    jsonl_sink = JsonlFileSink(OUTPUT_DIR)

    # Uygulama başlarken ilk döngüyü çalıştır
    _fetch_crypto_sentiment_cycle(jsonl_sink)
    
    # Config dosyasındaki dakikaya göre çalıştır
    schedule.every(settings.SENTIMENT_INTERVAL_MINUTES).minutes.do(_fetch_crypto_sentiment_cycle, jsonl_sink)

    try:
        while not stop.is_set():
            schedule.run_pending()
            time.sleep(30)
    finally:
        asyncio.run(jsonl_sink.close())

if __name__ == "__main__":
    import sys

    # 1. Logları zorla terminale (stdout) basacak şekilde yapılandırıyoruz
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # 2. Ortam Değişkeni (API Key) var mı diye kontrol edip ekrana uyarı basıyoruz
    print("=== TEST BAŞLATILIYOR ===")
    if not settings.TAVILY_API_KEY:
        print("KRİTİK HATA: TAVILY_API_KEY bulunamadı! Çalıştırma ortamında (terminal/env) key tanımlı değil.")
        sys.exit(1)
    else:
        print(f"API Key bulundu (İlk 5 karakter): {settings.TAVILY_API_KEY[:5]}***")

    # 3. Döngüyü tek seferlik manuel tetikliyoruz
    try:
        test_sink = JsonlFileSink(OUTPUT_DIR)
        _fetch_crypto_sentiment_cycle(test_sink)
    finally:
        asyncio.run(test_sink.close())
        
    print("=== TEST TAMAMLANDI ===")