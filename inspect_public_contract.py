import os
import json
from public_chain_service import PublicChainService

def inspect_chain():
    svc = PublicChainService()
    if svc.authenticate():
        exps = svc.get_expirations("SPY")
        if exps:
            chain = svc.get_option_chain("SPY", exps[0])
            contracts = chain.get("contracts", [])
            if contracts:
                print(json.dumps(contracts[0], indent=2))
            else:
                print("No contracts found in chain.")
        else:
            print("No expirations found.")
    else:
        print("Auth failed.")

if __name__ == "__main__":
    inspect_chain()
