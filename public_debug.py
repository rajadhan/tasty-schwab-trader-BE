import os
import requests
import json
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
secret_key = os.getenv("PUBLIC_API_KEY")

def run_debug():
    if not secret_key:
        print("No secret key found.")
        return

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    # 1. Auth
    auth_url = "https://api.public.com/userapiauthservice/personal/access-tokens"
    print(f"\n--- 1. Auth: {auth_url} ---")
    res = requests.post(auth_url, headers=headers, json={"secret": secret_key, "validityInMinutes": 60})
    print(f"Status: {res.status_code}")
    if res.status_code not in [200, 201]:
        print(f"Auth failed: {res.text}")
        return
    
    token = res.json().get("accessToken")
    headers["Authorization"] = f"Bearer {token}"
    print("Token acquired.")

    # 2. List Accounts
    acc_url = "https://api.public.com/userapigateway/trading/account"
    print(f"\n--- 2. List Accounts: {acc_url} ---")
    res = requests.get(acc_url, headers=headers)
    print(f"Status: {res.status_code}")
    if res.status_code == 200:
        data = res.json()
        print(f"Data: {json.dumps(data, indent=2)}")
        accounts = data.get("accounts", [])
        brokerage = [a for a in accounts if a.get("accountType") == "BROKERAGE"]
        target_acc = brokerage[0].get("accountId") if brokerage else (accounts[0].get("accountId") if accounts else None)
        print(f"Target Account: {target_acc}")
    else:
        print(f"Failed: {res.text}")
        return

    # 3. Expirations (POST)
    exp_url = "https://api.public.com/userapigateway/marketdata/option-expirations"
    print(f"\n--- 3. Expirations: {exp_url} (POST) ---")
    payload = {"instrument": {"symbol": "SPY", "type": "EQUITY"}}
    res = requests.post(exp_url, headers=headers, json=payload)
    print(f"Status: {res.status_code}")
    print(f"Response: {res.text}")

    # 4. Option Chain (POST)
    if target_acc:
        chain_url = f"https://api.public.com/userapigateway/marketdata/{target_acc}/option-chain"
        print(f"\n--- 4. Option Chain: {chain_url} ---")
        payload = {
            "instrument": {"symbol": "SPY", "type": "EQUITY"},
            "expirationDate": "2026-03-20"  # Dummy date for test
        }
        res = requests.post(chain_url, headers=headers, json=payload)
        print(f"Status: {res.status_code}")
        print(f"Response: {res.text}")

if __name__ == "__main__":
    run_debug()
