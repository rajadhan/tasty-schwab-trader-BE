import massive as ms
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
client = ms.RESTClient(api_key=os.getenv("MASSIVE_API_KEY"))

# Test fetching aggregates for an option contract
symbol = "O:SPY260213P00680000"
date = "2026-02-13"

try:
    print(f"Fetching 1m aggs for {symbol} on {date}...")
    aggs = client.list_aggs(
        ticker=symbol,
        multiplier=1,
        timespan="minute",
        from_=date,
        to=date,
        limit=50000
    )
    
    count = 0
    for agg in aggs:
        if count == 0:
            print(f"First Agg Sample: Close: {agg.close}, Volume: {agg.volume}, Time: {agg.timestamp}")
        count += 1
    
    print(f"Total bars fetched: {count}")

except Exception as e:
    print(f"Error fetching option aggs: {e}")
