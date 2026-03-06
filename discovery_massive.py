import massive as ms
import os
from dotenv import load_dotenv

load_dotenv()
client = ms.RESTClient(api_key=os.getenv("MASSIVE_API_KEY"))

try:
    print("Fetching SPY options chain for next expiry...")
    # list_snapshot_options_chain(underlying_asset="SPY", ...)
    chain = client.list_snapshot_options_chain("SPY")
    
    count = 0
    for contract in chain:
        if count < 5:
            print(contract)
        count += 1
        
    print(f"Total contracts: {count}")
    
except Exception as e:
    print(f"Error: {e}")
