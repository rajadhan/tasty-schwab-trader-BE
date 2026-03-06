import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ALPHAVANTAGE_API_KEY")

def fetch_av_options(symbol: str):
    """Fetch options data using alphavantage API"""
    # Try the realtime endpoint for the option chain (we don't specify a date, so it should fetch the current/next chain)
    url = f"https://www.alphavantage.co/query?function=REALTIME_OPTIONS&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Data for {symbol}:")
        if "data" in data and len(data["data"]) > 0:
             print(f"Total records found {len(data['data'])}")
             print("Sample of first 2 records:")
             print(data["data"][:2])
        else:
            print("No 'data' or unexpected format.")
            print(data)
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    fetch_av_options("SPY")
