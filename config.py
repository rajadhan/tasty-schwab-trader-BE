import os

api_key = "1iSr8ykD9qh2M2HoQv56wM2R1kWgQYZI"
api_secret = "cVbjaA4euOF923RL"

account_username = "tim30mcke"
account_password = "Sweeny07!"
account_id = "13241612"

tastytrade_link = "https://api.tastyworks.com"

username = "tmckenna@boardwalkag.com"    # tastytrade
password = "Boardwalk2024!"              # tastytrade

account_id = "5WX28756"
# MSTR, TSLA, NVDA, /MES, /ES, /MNQ, /NQ, /MRTY, /RTY
symbols = [
    "MSTR",
    "TSLA",
    "NVDA",
    "/MES",
    "/ES",
    "/MNQ",
    "/NQ",
    "/M2K",
    "/RTY",
]  #  removed

# TOKEN PATHS
access_token_path = "tokens/access_token.txt"
refresh_token_path = "tokens/refresh_token.txt"
refresh_token_link_path = "jsons/refresh_token_link.json"
tastytrade_access_token_path = "tokens/tastytrade_access_token.txt"
tickers_path = os.path.join("settings", "ticker_data.json")
gs_json_path = "jsons/creds.json"

api_callback_url = "https://127.0.0.1"

base_api_url = "https://api.schwabapi.com"
authurl = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={api_key}&redirect_uri={api_callback_url}"
authtoken_link = "https://api.schwabapi.com/v1/oauth/token"
schwab_market_data_link = "https://api.schwabapi.com/marketdata/v1"
schwab_trader_link = "https://api.schwabapi.com/trader/v1"

Google_sheet_name = "tim_McKenna_algo"
parameter_sheet = "Equities"
link_sheet = "Token_Link"


time_zone = "US/Eastern"