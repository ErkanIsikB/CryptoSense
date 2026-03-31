import os
import time
import requests
import schedule

# The 5 tokens we are tracking for market sentiment
TOKENS = [
    {"name": "Bitcoin", "symbol": "BTC"},
    {"name": "Ethereum", "symbol": "ETH"},
    {"name": "Solana", "symbol": "SOL"},
    {"name": "BNB", "symbol": "BNB"},
    {"name": "Avalanche", "symbol": "AVAX"}
]

TAVILY_API_URL = "https://api.tavily.com/search"

def fetch_crypto_sentiment():
    """
    Constructs and sends a focused query to the Tavily API for each token.
    Runs for all 5 tokens per cycle (Total cost: 10 credits).
    """
    # Fetch the API key from the environment variables securely
    api_key = "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"
    if not api_key:
        print("Error: TAVILY_API_KEY environment variable is missing.")
        return

    print(f"\n--- Starting Sentiment Fetch Cycle at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    for token in TOKENS:
        # Constructing the dynamic query with powerful crypto-specific keywords
        query_text = (
            f"{token['name']} {token['symbol']} latest news OR market sentiment OR "
            f"breaking OR price moving events OR FUD OR FOMO"
        )
        
        # Packing the API parameters per your exact requirements 
        payload = {
            "api_key": api_key,
            "query": query_text,
            "topic": "news",               # Forces search to focus on fresh news articles
            "search_depth": "advanced",    # Deeper scan for higher quality (Costs 2 credits)
            "include_answer": True,        # Returns an AI-generated summary across the results
            "include_images": False        # Disabled to save bandwidth/unnecessary cost
        }
        
        try:
            print(f"Fetching data for {token['name']} ({token['symbol']})...")
            response = requests.post(TAVILY_API_URL, json=payload)
            response.raise_for_status() # Raise an exception for HTTP errors
            
            data = response.json()
            
            # Here you would typically pipe this data into your DB or sentiment analyzer.
            # For demonstration, we'll print the AI summarized answer snippet:
            answer = data.get("answer", "No summary provided by API.")
            print(f"Result for {token['symbol']}: {answer[:200]}...\n")
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {token['name']}: {e}")

# -------------------------------------------------------------
# Credit Management Scheduler
# -------------------------------------------------------------
# Budget limits: 4000 / month
# Cost per cycle: 5 tokens * 2 credits = 10 credits
# Max cycles per month: 400 
# Max cycles per day: ~13 
# We schedule this to run every 2 hours, resulting in 12 cycles 
# per day (120 credits/day -> ~3600 credits/month). Very safe!

schedule.every(2).hours.do(fetch_crypto_sentiment)

if __name__ == "__main__":
    print("Initializing Crypto Sentiment Tracker...")
    
    # Fire off an immediate cycle on boot so we don't have to wait 2 hours initially
    fetch_crypto_sentiment()
    
    # Enter the infinite loop waiting for the next 2-hour window
    while True:
        schedule.run_pending()
        time.sleep(60) # Only wake up once a minute to check the schedule
