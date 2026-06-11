import asyncio
import logging
import sys
import time
import httpx
from src.core.config import settings
from src.models.sentiment_models import is_english
from src.data_sources.xquik.xquik_ingestion import _is_offtopic_news_tweet, KEYWORD_QUERIES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOGGER = logging.getLogger("xquik_live_test")

XQUIK_BASE = "https://xquik.com/api/v1"
TEST_DURATION_S = 120
POLL_INTERVAL_S = 2

def get_headers() -> dict[str, str]:
    return {
        "x-api-key": settings.XQUIK_API,
        "Content-Type": "application/json",
    }

async def list_monitors(client: httpx.AsyncClient) -> list[dict]:
    resp = await client.get(f"{XQUIK_BASE}/monitors/keywords", headers=get_headers())
    resp.raise_for_status()
    return resp.json().get("monitors", [])

async def set_monitor_active(client: httpx.AsyncClient, monitor_id: str, active: bool):
    resp = await client.patch(
        f"{XQUIK_BASE}/monitors/keywords/{monitor_id}",
        headers=get_headers(),
        json={"isActive": active}
    )
    resp.raise_for_status()

async def test_live_tweets():
    """Run a live XQuik monitoring window and return collected tweet texts."""
    LOGGER.info("⏳ Starting self-contained XQuik Live Tweet Test (no local module dependencies)...")
    api_key = settings.XQUIK_API
    if not api_key:
        raise ValueError("XQUIK_API key is not configured in settings (.env)")

    collected_tweets = []  # collect all tweet texts for downstream use

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Step 1: List existing monitors
        try:
            monitors = await list_monitors(client)
            if not monitors:
                raise RuntimeError("No existing keyword monitors found in your account!")
            
            LOGGER.info(f"📋 Found {len(monitors)} existing monitors:")
            for m in monitors:
                LOGGER.info(f"   - ID: {m.get('id')} | Query: '{m.get('query')}' | Active: {m.get('isActive')}")
        except Exception as e:
            LOGGER.exception(f"❌ Failed to list monitors: {e}")
            raise

        activated_monitors = []
        
        try:
            # Step 2: Unpause all existing monitors to start receiving live tweets
            LOGGER.info("\n--- Unpausing all existing monitors ---")
            for m in monitors:
                m_id = m.get('id')
                if m_id:
                    LOGGER.info(f"Activating monitor ID {m_id} ('{m.get('query')[:30]}')...")
                    await set_monitor_active(client, m_id, active=True)
                    activated_monitors.append(m_id)
                    await asyncio.sleep(0.12)  # rate limiting protection
            
            LOGGER.info("✅ All monitors unpaused. Waiting to collect live tweets...")
            
            # Step 3: Fast-forward/initialize baseline event IDs
            last_event_ids = {}
            for m_id in activated_monitors:
                try:
                    resp = await client.get(
                        f"{XQUIK_BASE}/events",
                        headers=get_headers(),
                        params={"keywordMonitorId": m_id, "limit": 1},
                    )
                    resp.raise_for_status()
                    events = resp.json().get("events", [])
                    if events:
                        last_event_ids[m_id] = str(events[0].get("id", ""))
                except Exception:
                    pass

            # Step 4: Wait and poll periodically
            start_time = time.time()
            elapsed = 0
            
            LOGGER.info(f"\n--- Starting {TEST_DURATION_S} seconds monitoring window (polling every {POLL_INTERVAL_S}s) ---")
            
            while elapsed < TEST_DURATION_S:
                await asyncio.sleep(POLL_INTERVAL_S)
                elapsed = int(time.time() - start_time)
                LOGGER.info(f"⏱️ Elapsed time: {elapsed}/{TEST_DURATION_S}s. Polling for new events...")
                
                for m_id in activated_monitors:
                    # Find query/symbol for printing
                    query = next((m.get('query', '') for m in monitors if m.get('id') == m_id), m_id)
                    
                    try:
                        # Fetch events newer than our baseline
                        resp = await client.get(
                            f"{XQUIK_BASE}/events",
                            headers=get_headers(),
                            params={"keywordMonitorId": m_id, "limit": 50},
                        )
                        resp.raise_for_status()
                        events = resp.json().get("events", [])
                        
                        # Count new events since the baseline
                        baseline_id = last_event_ids.get(m_id)
                        new_events = []
                        for evt in events:
                            evt_id = str(evt.get("id", ""))
                            if evt_id == baseline_id:
                                break
                            new_events.append(evt)
                        
                        if new_events:
                            LOGGER.info(f"   🔥 [New Tweets!] Monitor {m_id} ('{query[:30]}...'): +{len(new_events)} new tweets")
                            for idx, evt in enumerate(new_events[:3]):  # show up to 3 samples
                                tweet_text = evt.get("data", {}).get("text", "").replace("\n", " ")
                                LOGGER.info(f"      - {tweet_text[:100]}...")

                            # Find symbol for off-topic filtering
                            symbol = None
                            for sym, q in KEYWORD_QUERIES.items():
                                if q == query:
                                    symbol = sym
                                    break

                            # Collect English-only and on-topic tweet texts for sentiment analysis
                            for evt in new_events:
                                tweet_text = evt.get("data", {}).get("text", "").strip()
                                if not tweet_text:
                                    continue
                                if not is_english(tweet_text):
                                    continue
                                if symbol and _is_offtopic_news_tweet(symbol, tweet_text):
                                    LOGGER.info(f"      🚫 [Filtered Off-Topic] for {symbol}: {tweet_text[:80]}...")
                                    continue
                                collected_tweets.append(tweet_text)

                            # Update the baseline to the newest seen event
                            last_event_ids[m_id] = str(events[0].get("id", ""))
                        else:
                            LOGGER.info(f"   ℹ️ Monitor {m_id} ('{query[:30]}...'): No new tweets in this interval.")
                    except Exception as e:
                        LOGGER.error(f"   ❌ Error polling monitor {m_id}: {e}")
                    
                    await asyncio.sleep(0.12)  # rate limit protection

        finally:
            # Step 5: Always pause the monitors again to save credits!
            LOGGER.info("\n--- Pausing all activated monitors to save credits ---")
            for m_id in activated_monitors:
                try:
                    LOGGER.info(f"Pausing monitor ID {m_id}...")
                    await set_monitor_active(client, m_id, active=False)
                except Exception as e:
                    LOGGER.error(f"❌ Failed to pause monitor {m_id}: {e}")
                await asyncio.sleep(0.12)
            LOGGER.info("✅ Finished cleaning up. Monitors have been paused.")

    LOGGER.info(f"\n📊 Total tweets collected for sentiment analysis: {len(collected_tweets)}")
    return collected_tweets

if __name__ == "__main__":
    try:
        asyncio.run(test_live_tweets())
    except KeyboardInterrupt:
        LOGGER.info("🛑 Test manually interrupted.")
