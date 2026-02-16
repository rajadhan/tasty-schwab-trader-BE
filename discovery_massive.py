import massive as ms
import os
from dotenv import load_dotenv

load_dotenv()
client = ms.RESTClient(api_key=os.getenv("MASSIVE_API_KEY"))

methods = sorted([m for m in dir(client) if not m.startswith("_")])
print("ALL RESTClient Methods:")
for m in methods:
    print(f" - {m}")
