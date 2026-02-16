import massive as ms
import os
from dotenv import load_dotenv
import json

load_dotenv()
client = ms.RESTClient(api_key=os.getenv("MASSIVE_API_KEY"))

# Test fetching a few quotes for an option contract on Feb 13
# SPY   260213P00680000
symbol = "O:SPY260213P00680000"
date = "2026-02-13"

try:
    print(f"Fetching quotes for {symbol} on {date}...")
    # list_quotes(ticker, timestamp=None, order=None, limit=None, sort=None, ...)
    quotes = client.list_quotes(
        ticker=symbol,
        limit=5,
        timestamp_gte=f"{date}T14:30:00Z" # 9:30 AM ET
    )
    
    for i, q in enumerate(quotes):
        print(f"\nQuote {i+1}:")
        # Print attributes to see if greeks are there
        print(f" Timestamp: {q.participant_timestamp}")
        print(f" Bid: {q.bid_price}, Ask: {q.ask_price}")
        if hasattr(q, 'greeks'):
            print(f" Greeks: {q.greeks}")
        else:
            print(" No 'greeks' attribute found.")
            # Check all attributes
            print(f" All Attributes: {[a for a in dir(q) if not a.startswith('_')]}")

except Exception as e:
    print(f"Error fetching quotes: {e}")
