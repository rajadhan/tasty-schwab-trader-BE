from config import TIME_ZONE
from datetime import datetime, timedelta
from flask import jsonify
from pytz import timezone as pytz_timezone
from time import sleep
from urllib.parse import unquote
from utils import (
    params_parser, time_frame_config, configure_logger, get_trade_file_path, load_json, save_json, get_strategy_prarams
)
import base64
import os
import pandas as pd
import requests
import json

SCHWAB_API_KEY = os.getenv("SCHWAB_API_KEY")
SCHWAB_API_SECRET = os.getenv("SCHWAB_API_SECRET")
SCHWAB_ACCOUNT_ID = os.getenv("SCHWAB_ACCOUNT_ID")
SCHWAB_API = "https://api.schwabapi.com"
SCHWAB_CALLBACK_URL = "https://127.0.0.1"
SCHWAB_ACCESS_TOKEN_PATH = os.path.join("tokens", "schwab_tokens.txt")


def authorize_url():
    params = {"client_id": SCHWAB_API_KEY, "redirect_uri": SCHWAB_CALLBACK_URL}
    request_url = (
        requests.Request("GET", f"{SCHWAB_API}/v1/oauth/authorize", params=params)
        .prepare()
        .url
    )
    return request_url


def create_api_header(access_token):
    # Ensure token is properly formatted
    if not access_token.startswith("Bearer "):
        token = f"Bearer {access_token}"
    return {
        "Authorization": token,
    }


def create_auth_header():
    credentials = f"{SCHWAB_API_KEY}:{SCHWAB_API_SECRET}"
    encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    return {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded",
    }


def get_refresh_token(redirect_link):
    # Fix the code extraction logic
    if "code=" not in redirect_link:
        print("No authorization code found in link")
        return False

    # Extract the authorization code properly
    code_start = redirect_link.index("code=") + 5
    code_end = redirect_link.find("&", code_start)
    if code_end == -1:
        code_end = len(redirect_link)
    code = redirect_link[code_start:code_end]
    code = unquote(code)

    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": SCHWAB_CALLBACK_URL,
    }
    authtoken_link = f"{SCHWAB_API}/v1/oauth/token"

    response = requests.post(authtoken_link, data=payload, headers=create_auth_header())
    # Check HTTP status code first
    if response.status_code != 200:
        print(f"API request failed with status {response.status_code}: {response.text}")
        return False

    response_data = response.json()
    refresh_token = response_data.get("refresh_token")
    access_token = response_data.get("access_token")

    if access_token and refresh_token:
        tokens_to_save = {"access_token": access_token, "refresh_token": refresh_token}
        with open(SCHWAB_ACCESS_TOKEN_PATH, "w") as f:
            json.dump(tokens_to_save, f)
        return True
    else:
        return False


def refresh_access_token():
    """Refresh Charles Schwab access token using existing refresh token"""
    try:
        with open(SCHWAB_ACCESS_TOKEN_PATH, "r") as f:
            tokens = json.load(f)
            refresh_token = tokens.get("refresh_token", "")

        if not refresh_token:
            return None

        schwab_refresh_access_token_url = f"{SCHWAB_API}/v1/oauth/token"
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        response = requests.post(
            schwab_refresh_access_token_url, headers=create_auth_header(), data=payload
        )
        if response.status_code != 200:
            return (
                None
            )
        response_data = response.json()
        new_access_token = response_data.get("access_token")
        new_refresh_token = response_data.get("refresh_token", refresh_token)

        if not new_access_token:
            return jsonify({"success": False, "error": "No access token received"}), 400

        tokens_to_save = {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
        }

        with open(SCHWAB_ACCESS_TOKEN_PATH, "w") as f:
            json.dump(tokens_to_save, f)

        return (
           new_access_token
        )

    except Exception as e:
        return None


def schwab_api_request(method, url, **kwargs):
    with open(SCHWAB_ACCESS_TOKEN_PATH, "r") as f:
        tokens = json.load(f)
    access_token = tokens.get("access_token")
    headers = kwargs.pop("headers", {})
    headers.update(create_api_header(access_token))
    response = requests.request(method, url, headers=headers, **kwargs)
    if response.status_code == 401 or "invalid_token" in response.text or "expired_token" in response.text:
        access_token = refresh_access_token()
        headers = create_api_header(access_token)
        response = requests.request(method, url, headers=headers, **kwargs)
    return response


def get_quotes(symbol):
    url = f"{SCHWAB_API}/marketdata/v1/quotes"
    params = {"symbols": symbol, 'fields': 'quote,fundamental'}
    response = schwab_api_request("GET", url, params=params)
    data = response.json()
    quote = {}
    if isinstance(data, dict):
        for sym, payload in data.items():
            last = (
                payload.get("quote", {}).get("lastPrice")
                or payload.get("lastPrice")
                or payload.get("mark")
            )
            ts = (
                payload.get("quote", {}).get("quoteTime")
                or payload.get("quoteTime")
                or payload.get("tradeTime")
            )
            quote = {"last": last, "quoteTime": ts}
    return quote


def get_accounts():
    url = f"{SCHWAB_API}/trader/v1/accounts"
    response = schwab_api_request("get", url)
    print("response", response.json())


def _time_convert(dt=None, form="8601"):
    """
    Convert time to the correct format, passthrough if a string, preserve None if None for params parser
    :param dt: datetime.pyi object to convert
    :type dt: datetime.pyi | str | None
    :param form: what to convert input to
    :type form: str
    :return: converted time or passthrough
    :rtype: str | None
    """
    if dt is None or not isinstance(dt, datetime):
        return dt
    elif form == "8601":  # assume datetime object from here on
        return f"{dt.isoformat().split('+')[0][:-3]}Z"
    elif form == "epoch":
        return int(dt.timestamp())
    elif form == "epoch_ms":
        return int(dt.timestamp() * 1000)
    elif form == "YYYY-MM-DD":
        return dt.strftime("%Y-%m-%d")
    else:
        return dt


def historical_data(symbol, time_frame, logger):
    try:
        if symbol in ["SPX"]:
            symbol = f"${symbol}"
        current_datetime = datetime.now(tz=pytz_timezone(TIME_ZONE))
        endtime = current_datetime + timedelta(seconds=85)
        if isinstance(time_frame, str):
            if time_frame.isdigit():
                time_frame = int(time_frame)
        get_historical_data_url = f"{SCHWAB_API}/marketdata/v1/pricehistory"
        config = time_frame_config(time_frame)
        periodType, period, frequencyType, frequency = (
            config["periodType"],
            config["period"],
            config["frequencyType"],
            config["frequency"],
        )
        params = {
            "symbol": symbol,
            "periodType": periodType,
            "period": period,
            "frequencyType": frequencyType,
            "frequency": frequency,
            "needExtendedHoursData": True if symbol[0] == "/" else False,
            # "endDate": _time_convert(endtime, "epoch_ms"),
        }
        response = schwab_api_request("GET", get_historical_data_url, params=params)
        response_data = response.json()
        if "error" in response_data:
            logger.error(f"API returned error: {response_data['error']}")
            raise ValueError(f"API error: {response_data['error']}")

        data = response_data.get("candles", [])
        if not data:
            raise ValueError("No data returned from API.")

        # Data transformation
        df = pd.DataFrame(data)
        df["symbol"] = symbol
        df["datetime"] = (
            pd.to_datetime(df["datetime"], unit="ms")
            .dt.tz_localize("UTC")
            .dt.tz_convert(TIME_ZONE)
        )
        df = df[["datetime", "symbol", "open", "high", "low", "close"]]
        # Resampling if applicable
        if "resample" in config:
            resample_freq = config["resample"]
            if time_frame == "1h":
                # Adjust resampling for 30-minute offset
                df = (
                    df.set_index("datetime")
                    .resample(resample_freq)  # 30-minute offset
                    .agg(
                        {"open": "first", "high": "max", "low": "min", "close": "last"}
                    )
                    .dropna()
                    .reset_index()
                )
            elif time_frame == "4h":
                # Adjust resampling for 90-minute offset
                df = (
                    df.set_index("datetime")
                    .resample(resample_freq)  # 90-minute offset
                    .agg(
                        {"open": "first", "high": "max", "low": "min", "close": "last"}
                    )
                    .dropna()
                    .reset_index()
                )
            else:
                # Standard resampling for other time frames
                df = (
                    df.set_index("datetime")
                    .resample(resample_freq)
                    .agg(
                        {"open": "first", "high": "max", "low": "min", "close": "last"}
                    )
                    .dropna()
                    .reset_index()
                )
        if df.iloc[-1]["datetime"].strftime("%H:%M") == datetime.now(
            tz=pytz_timezone(TIME_ZONE)
        ).strftime("%H:%M"):
            df = df[:-1]
        logger.info(f"Historical data for {symbol} fetched successfully")
        return df
    except Exception as e:
        logger.error(f"Error in getting historical data for {symbol}: {str(e)}")
        return None


def get_encrypted_account_id(schwab_account_id, logger):
    try:
        get_encrypted_account_id_url = f"{SCHWAB_API}/trader/v1/accounts/accountNumbers"
        response = schwab_api_request("get", get_encrypted_account_id_url)
        encrypted_account_id = response.json()[0]["hashValue"]
        return encrypted_account_id
    except Exception as e:
        logger.error(
            f"Error in getting encrypted account ID for {schwab_account_id}: {str(e)}"
        )
        return None


def place_order(symbol, quantity, action, logger):
    try:
        logger.info(
            f"Placing order for {symbol}, Action: {action}, Quantity: {quantity}"
        )
        # Schwab Orders API supports EQUITY and OPTION instruments for this endpoint
        if symbol.startswith("/"):
            logger.error(
                f"Unsupported instrument for Schwab Orders API: '{symbol}'. Valid assetType is EQUITY or OPTION."
            )
            return None

        if symbol.startswith("SPX"):
            asset_type = "OPTION"
        else:
            asset_type = "EQUITY"

        params = {
            "session": "NORMAL",
            "duration": "DAY",
            "orderType": "MARKET",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": action,
                    "quantity": quantity,
                    "instrument": {
                        "assetType": asset_type,
                        "symbol": symbol,
                    },
                }
            ],
        }
        encrypted_account_id = get_encrypted_account_id(SCHWAB_ACCOUNT_ID, logger)
        place_order_url = f"{SCHWAB_API}/trader/v1/accounts/{encrypted_account_id}/orders"
        response = schwab_api_request(
            "post",
            place_order_url,
            json=params,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code < 200 or response.status_code >= 300:
            logger.error(
                f"Order placement failed ({response.status_code}): {getattr(response, 'text', '')}"
            )
            return None

        location_header = dict(response.headers).get("Location")
        if not location_header:
            logger.error("Missing Location header in order response")
            return None
        order_id = location_header.split("/")[-1]
        return order_id

    except Exception as e:
        logger.error(f"Error in placing order for {symbol}: {str(e)}")
        return None


def place_option_order(symbol, quantity, action, logger, contract_type):
    if symbol == "SPX":
        symbol = "$SPX"
    today = datetime.now().strftime("%Y-%m-%d")
    chains_url = f"{SCHWAB_API}/marketdata/v1/chains"
    params = {
        "symbol": symbol,
        "contractType": contract_type,
        "includeQuotes": True,
        "strikeCount": 5,
        "strategy": "SINGLE",
        "toDate": today
    }
    response = schwab_api_request("GET", chains_url, params=params)
    data = response.json()
    underlying_price = data["underlyingPrice"]
    closest_symbol = None
    closest_diff = float("inf")
    if contract_type == "CALL":
        data = data["callExpDateMap"]
    else:
        data = data["putExpDateMap"]
    for exp, strikes in data.items():
        for strike, options in strikes.items():
            opt = options[0]
            option_symbol = opt["symbol"]
            strike_price = float(opt["strikePrice"])
            diff= abs(strike_price - underlying_price)

            if diff < closest_diff:
                closest_diff = diff
                closest_symbol = option_symbol

    print("Selected option:", closest_symbol)

    resp = place_order(closest_symbol, quantity, action, logger)
    return [resp, closest_symbol]


def manual_trigger_action(action, logger, ticker="SPX"):
    """Handle manual actions for Schwab zeroday options on SPX.

    Actions supported:
      - "call": open long via Buy to Open (auto-closes PUT or CALL if present)
      - "put": open long via Buy to Open (auto-closes CALL or PUT if present)
      - "close": close any existing position

    Stores and reads state from trades/zeroday/<ticker>.json
    """
    try:
        trade_file = get_trade_file_path(ticker, "zeroday")
        trades = load_json(trade_file)
        [
            timeframe,
            schwab_qty,
            trade_enabled,
            tasty_qty,
            trend_line_1,
            period_1,
            trend_line_2,
            period_2,
        ] = get_strategy_prarams("zeroday", ticker, logger)
        # Determine quantity for Schwab from settings if available, default 1
        qty = int(schwab_qty)
        # Attempt to pull quantity from zeroday settings if present
        # We avoid importing get_strategy_prarams here to prevent cycles; quantity of 1 is safe default

        existing = trades.get(ticker, {})
        existing_symbol = existing.get("schwab_option_symbol")
        existing_action = existing.get("action")  # LONG or SHORT
        existing_type = existing.get("contract")  # CALL or PUT

        # Helper to place close order if we have an existing specific option symbol
        def close_existing():
            nonlocal trades
            if not existing_symbol:
                logger.info("No existing position to close")
                return None
            instruction = "SELL_TO_CLOSE"
            logger.info(f"Closing existing {existing_type} position: {existing_symbol}")
            resp = place_order(existing_symbol, qty, instruction, logger)
            
            return None  # Close operations don't return order details

        if action == "close":
            close_existing()
            return None

        if action == "call":
            # Close any existing position first
            close_existing()
            [order_id, selected_symbol] = place_option_order("SPX", qty, "BUY_TO_OPEN", logger, "CALL")
            
            return [order_id, selected_symbol]
        elif action == "put":
            # Close any existing position first
            close_existing()
            [order_id, selected_symbol] = place_option_order("SPX", qty, "BUY_TO_OPEN", logger, "PUT")
            
            return [order_id, selected_symbol]
        else:
            logger.error(f"Unsupported manual action for Schwab: {action}")
            return None
    except Exception as e:
        logger.error(f"Error in Schwab manual trigger: {str(e)}")
        return None


if __name__  == "__main__":
    logger = configure_logger("SPX", "zeroday")
    encrypted_account_id = get_encrypted_account_id(SCHWAB_ACCOUNT_ID, logger)
    print("encrypted_account_id", encrypted_account_id)
    url = f"{SCHWAB_API}/trader/v1/accounts/{encrypted_account_id}/orders/1004238708712"
    response = schwab_api_request("get", url)
    data = response.json()
    print("data", data)