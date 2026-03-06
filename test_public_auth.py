import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("PUBLIC_API_KEY")

def test_auth():
    print(f"Testing with API KEY: {api_key[:5]}...")
    
    # 1. Try Bearer Token (User's suggestion)
    url = "https://api.public.com/userapigateway/trading/account"
    headers_bearer = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    print("\nTrial 1: Authorization: Bearer {API_KEY}")
    res1 = requests.get(url, headers=headers_bearer)
    print(f"Status: {res1.status_code}")
    print(f"Body: {res1.text[:100]}")

    # 2. Try X-API-KEY header
    headers_x = {
        "X-API-KEY": api_key,
        "Accept": "application/json"
    }
    print("\nTrial 2: X-API-KEY: {API_KEY}")
    res2 = requests.get(url, headers=headers_x)
    print(f"Status: {res2.status_code}")
    print(f"Body: {res2.text[:100]}")

    # 3. Try Token Exchange Flow
    auth_url = "https://api.public.com/userapiauthservice/personal/access-tokens"
    payload = {"secretKey": api_key}
    print(f"\nTrial 3: Token Exchange at {auth_url}")
    try:
        res3 = requests.post(auth_url, json=payload)
        print(f"Status: {res3.status_code}")
        print(f"Body: {res3.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

    # 4. Try Market Data Smoke Test with Bearer
    smoke_url = "https://api.public.com/userapigateway/marketdata/quotes?symbols=AAPL"
    print("\nTrial 4: Market Data Smoke Test with Bearer")
    res4 = requests.get(smoke_url, headers=headers_bearer)
    print(f"Status: {res4.status_code}")
    print(f"Body: {res4.text[:100]}")

if __name__ == "__main__":
    if api_key:
        test_auth()
    else:
        print("No API Key found.")
