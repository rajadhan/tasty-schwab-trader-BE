import os
import requests
import json
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Optional

# Force reload to get the new keys
load_dotenv(find_dotenv(), override=True)
# The user's PUBLIC_API_KEY is the Secret Key required for exchange
secret_key = os.getenv("PUBLIC_API_KEY")

class PublicAPIClient:
    GATEWAY_URL = "https://api.public.com/userapigateway"
    AUTH_URL = "https://api.public.com/userapiauthservice"

    def __init__(self, secret: str):
        self.secret = secret
        self.access_token = None
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        }

    def refresh_access_token(self) -> bool:
        """Exchanges the Secret Key for an Access Token as per documentation."""
        url = f"{self.AUTH_URL}/personal/access-tokens"
        payload = {
            "secret": self.secret,
            "validityInMinutes": 1440 # 24 hours
        }
        try:
            print(f"Exchanging secret key for access token at {url}...")
            res = requests.post(url, headers=self.headers, json=payload, timeout=10)
            print(f"Auth Status: {res.status_code}")
            
            if res.status_code in [200, 201]:
                data = res.json()
                self.access_token = data.get("accessToken")
                if self.access_token:
                    print("Successfully acquired Access Token.")
                    self.headers["Authorization"] = f"Bearer {self.access_token}"
                    return True
            print(f"Token exchange failed: {res.text}")
        except Exception as e:
            print(f"Error during token exchange: {e}")
        return False

    def get_account(self) -> Optional[str]:
        """Fetches the primary BROKERAGE account ID."""
        url = f"{self.GATEWAY_URL}/trading/account"
        try:
            print(f"Fetching account from {url}...")
            res = requests.get(url, headers=self.headers, timeout=10)
            print(f"Account Status: {res.status_code}")
            
            if res.status_code == 200:
                data = res.json()
                accounts = data.get("accounts", [])
                # Filter for BROKERAGE account
                brokerage_accounts = [a for a in accounts if a.get("accountType") == "BROKERAGE"]
                target = brokerage_accounts[0] if brokerage_accounts else (accounts[0] if accounts else None)
                
                if target:
                    account_id = target.get("accountId")
                    print(f"Selected Account ID: {account_id}")
                    return account_id
        except Exception as e:
            print(f"Error fetching account: {e}")
        return None

    def get_expirations(self, symbol: str) -> List[str]:
        """Fetches available expiration dates for a symbol (POST as per doc)."""
        url = f"{self.GATEWAY_URL}/marketdata/option-expirations"
        payload = {
            "instrument": {
                "symbol": symbol,
                "type": "EQUITY"
            }
        }
        try:
            print(f"Fetching expirations for {symbol} (POST)...")
            res = requests.post(url, headers=self.headers, json=payload, timeout=10)
            print(f"Expirations Status: {res.status_code}")
            if res.status_code == 200:
                data = res.json()
                expirations = data.get("expirationDates") or []
                print(f"Found {len(expirations)} expirations.")
                return sorted(expirations)
            else:
                print(f"Error: {res.text}")
        except Exception as e:
            print(f"Error: {e}")
        return []

    def get_option_chain(self, account_id: str, symbol: str, expiration_date: str) -> Dict:
        """Fetches the option chain for a specific expiration (POST)."""
        url = f"{self.GATEWAY_URL}/marketdata/{account_id}/option-chain"
        payload = {
            "instrument": {
                "symbol": symbol,
                "type": "EQUITY"
            },
            "expirationDate": expiration_date
        }
        try:
            print(f"Fetching option chain for {symbol} {expiration_date} (POST)...")
            res = requests.post(url, headers=self.headers, json=payload, timeout=15)
            print(f"Chain Status: {res.status_code}")
            if res.status_code == 200:
                return res.json()
            else:
                print(f"Error: {res.text}")
        except Exception as e:
            print(f"Error: {e}")
        return {}

def run_diagnostic(symbol="SPY"):
    if not secret_key:
        print("Error: PUBLIC_API_KEY (Secret) not found in .env")
        return

    client = PublicAPIClient(secret_key)
    
    # 1. Auth
    if not client.refresh_access_token():
        return

    # 2. Get Account
    account_id = client.get_account()
    if not account_id:
        return

    # 3. Get Expirations
    expirations = client.get_expirations(symbol)
    if not expirations:
        return

    # 4. Get first chain
    first_exp = expirations[0]
    chain = client.get_option_chain(account_id, symbol, first_exp)
    if chain:
        print(f"Success! Retrieved chain with {len(chain.get('contracts', []))} contracts.")
        with open("tmp_public_chain_sample.json", "w") as f:
            json.dump(chain, f, indent=2)
    else:
        print("Failed to retrieve chain.")

if __name__ == "__main__":
    run_diagnostic()
