import os

api_key = "1iSr8ykD9qh2M2HoQv56wM2R1kWgQYZI"
api_secret = "cVbjaA4euOF923RL"

account_username = "tim30mcke"
account_password = "Sweeny07!"
schwab_account_id = "13241612"

#------------------------------#
# Tasty Trade credentials
TASTY_API = "https://api.tastyworks.com"

TASTY_USERNAME = "tmckenna@boardwalkag.com"    # tastytrade
TASTY_PASSWORD = "Boardwalk2024!"              # tastytrade

TASTY_CLIENT_SECRET = "94cf0194a470e5f376284dcc0baa444b37b534d9"
TASTY_CLIENT_ID = "f1ff7542-2fa9-446c-a57e-22f95108e02c"

TASTY_MY_APP_URL = "https://my.tastytrade.com/auth.html"
TASTY_REDIRECT_URI = "https://api.tastyworks.com"

TASTY_ACCESS_TOKEN_PATH = os.path.join("tokens", "tastytrade_tokens.txt")

TASTY_ACCOUNT_ID = "5WW38442"
#------------------------------#


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
]   # removed

# TOKEN PATHS
access_token_path = "tokens/access_token.txt"
refresh_token_path = "tokens/refresh_token.txt"
refresh_token_link_path = "jsons/refresh_token_link.json"
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

EMA_TICKER_DATA_PATH = os.path.join('settings', 'ema_ticker_data.json')
SUPER_TICKER_DATA_PATH = os.path.join('settings', 'super_ticker_data.json')
ZERODAY_TICKER_DATA_PATH = os.path.join('settings', 'zeroday_ticker_data.json')