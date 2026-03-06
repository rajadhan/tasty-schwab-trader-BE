import os
import requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
api_key = os.getenv("PUBLIC_API_KEY")

def test_endpoint(name, url, method="GET", payload=None):
    print(f"\n--- Testing {name} ---")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    }
    try:
        if method == "GET":
            res = requests.get(url, headers=headers, timeout=10)
        else:
            res = requests.post(url, headers=headers, json=payload, timeout=10)
        
        print(f"Status: {res.status_code}")
        print(f"Headers: {dict(res.headers)}")
        if res.status_code == 200:
            print(f"Success! Data: {res.json()}")
        else:
            print(f"Failed Body: {res.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")

# 1. Trading Account
test_endpoint("Trading Account", "https://api.public.com/userapigateway/trading/account")

# 2. Market Data Quotes
test_endpoint("Market Data Quotes", "https://api.public.com/userapigateway/marketdata/quotes?symbols=AAPL")
