import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Optional

class PublicChainService:
    """Service to interact with Public.com API for options data."""
    GATEWAY_URL = "https://api.public.com/userapigateway"
    AUTH_URL = "https://api.public.com/userapiauthservice"

    def __init__(self, secret_key: Optional[str] = None):
        load_dotenv(find_dotenv(), override=True)
        self.secret_key = secret_key or os.getenv("PUBLIC_API_KEY")
        self.access_token = None
        self.account_id = None
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "PublicTradingService/1.0"
        }

    def authenticate(self) -> bool:
        """Exchanges secret key for access token."""
        if not self.secret_key:
            return False
            
        url = f"{self.AUTH_URL}/personal/access-tokens"
        payload = {"secret": self.secret_key, "validityInMinutes": 1440}
        
        try:
            res = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if res.status_code in [200, 201]:
                self.access_token = res.json().get("accessToken")
                self.headers["Authorization"] = f"Bearer {self.access_token}"
                return True
        except Exception:
            pass
        return False

    def get_brokerage_account_id(self) -> Optional[str]:
        """Retrieves the primary brokerage account ID."""
        if not self.access_token:
            if not self.authenticate(): return None
            
        url = f"{self.GATEWAY_URL}/trading/account"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                accounts = res.json().get("accounts", [])
                brokerage = [a for a in accounts if a.get("accountType") == "BROKERAGE"]
                if brokerage:
                    self.account_id = brokerage[0].get("accountId")
                    return self.account_id
        except Exception:
            pass
        return None

    def get_instrument(self, symbol: str) -> Optional[Dict]:
        """Fetches detailed instrument information."""
        if not self.access_token:
            if not self.authenticate(): return None
            
        url = f"{self.GATEWAY_URL}/marketdata/instruments/{symbol}"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass
        return None

    def get_expirations(self, symbol: str) -> List[str]:
        """Fetches available expiration dates for a symbol."""
        if not self.account_id:
            if not self.get_brokerage_account_id(): return []
            
        url = f"{self.GATEWAY_URL}/marketdata/{self.account_id}/option-expirations"
        payload = {"instrument": {"symbol": symbol, "type": "EQUITY"}}
        
        try:
            # Note: Browser subagent confirmed URL is /{accountId}/option-expirations
            res = requests.post(url, headers=self.headers, json=payload, timeout=10)
            if res.status_code == 200:
                # The response structure might have 'expirations' or 'expirationDates'
                data = res.json()
                return sorted(data.get("expirations") or data.get("expirationDates") or [])
        except Exception:
            pass
        return []

    def get_option_chain(self, symbol: str, expiration_date: str) -> Dict:
        """Fetches the full option chain for a specific expiration."""
        if not self.account_id:
            if not self.get_brokerage_account_id(): return {}
            
        url = f"{self.GATEWAY_URL}/marketdata/{self.account_id}/option-chain"
        payload = {
            "instrument": {"symbol": symbol, "type": "EQUITY"},
            "expirationDate": expiration_date
        }
        
        try:
            res = requests.post(url, headers=self.headers, json=payload, timeout=20)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass
        return {}

    def get_formatted_chain(self, symbol: str, expiration_date: str) -> List[Dict]:
        """
        Fetches the chain and formats it for StrategyPermutator.
        Format: [{'strike': float, 'call_bid': float, 'call_ask': float, 'put_bid': float, 'put_ask': float}]
        """
        raw_chain = self.get_option_chain(symbol, expiration_date)
        # raw_chain has 'calls' and 'puts' lists
        calls = raw_chain.get("calls", [])
        puts = raw_chain.get("puts", [])
        
        # Organize by strike
        by_strike = {}
        
        def extract_strike_from_symbol(opt_symbol):
            # Format usually: AAPL  260320C00200000 -> 200.00
            # Last 8 digits are strike * 1000
            try:
                strike_str = opt_symbol[-8:]
                return float(strike_str) / 1000.0
            except:
                return 0.0

        for call in calls:
            # Some APIs include 'strike' in instrument, others need parsing
            inst = call.get("instrument", {})
            strike = inst.get("strike") or extract_strike_from_symbol(inst.get("symbol", ""))
            if strike not in by_strike:
                by_strike[strike] = {"strike": strike, "call_bid": 0.0, "call_ask": 0.0, "put_bid": 0.0, "put_ask": 0.0}
            by_strike[strike]["call_bid"] = float(call.get("bid", 0.0))
            by_strike[strike]["call_ask"] = float(call.get("ask", 0.0))
            
        for put in puts:
            inst = put.get("instrument", {})
            strike = inst.get("strike") or extract_strike_from_symbol(inst.get("symbol", ""))
            if strike not in by_strike:
                by_strike[strike] = {"strike": strike, "call_bid": 0.0, "call_ask": 0.0, "put_bid": 0.0, "put_ask": 0.0}
            by_strike[strike]["put_bid"] = float(put.get("bid", 0.0))
            by_strike[strike]["put_ask"] = float(put.get("ask", 0.0))
            
        return sorted(by_strike.values(), key=lambda x: x["strike"])

if __name__ == "__main__":
    # Quick self-test
    svc = PublicChainService()
    print("Authenticating...")
    if svc.authenticate():
        print("Success. Fetching SPY expirations...")
        exps = svc.get_expirations("SPY")
        if exps:
            print(f"Nearest monthly: {exps[0]}")
            chain = svc.get_option_chain("SPY", exps[0])
            print(f"Retrieved chain with {len(chain.get('contracts', []))} contracts.")
        else:
            print("No expirations found.")
    else:
        print("Auth failed.")
