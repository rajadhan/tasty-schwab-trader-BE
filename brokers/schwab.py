from config import TIME_ZONE
from datetime import datetime, timedelta
from flask import jsonify
from pytz import timezone as pytz_timezone
from time import sleep
from urllib.parse import unquote
from utils import (
    params_parser, time_frame_config, configure_logger
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


def get_quotes(symbols):
    """
    Fetch real-time quotes for one or more symbols using Schwab marketdata v1 quotes endpoint.
    Symbols can include futures (e.g., "/MES"), indices (e.g., "$SPX"), or equities (e.g., "AAPL").
    Returns a dict of symbol -> quote payload, with useful fields normalized.
    """
    if not symbols:
        return {}
    # Schwab expects comma-separated list; also handle SPX mapping if needed upstream
    symbols_param = ",".join(symbols)
    url = f"{SCHWAB_API}/marketdata/v1/quotes"
    params = {"symbols": symbols_param}
    resp = schwab_api_request("GET", url, params=params)
    try:
        data = resp.json() if hasattr(resp, "json") else {}
    except Exception:
        data = {}
    # Normalize into a simple map: symbol -> {last, bid, ask, quoteTime}
    quotes = {}
    if isinstance(data, dict):
        for sym, payload in data.items():
            # Schwab returns different payloads by asset type; try common fields
            last = (
                payload.get("quote", {}).get("lastPrice")
                or payload.get("lastPrice")
                or payload.get("mark")
            )
            bid = (
                payload.get("quote", {}).get("bidPrice")
                or payload.get("bidPrice")
            )
            ask = (
                payload.get("quote", {}).get("askPrice")
                or payload.get("askPrice")
            )
            ts = (
                payload.get("quote", {}).get("quoteTime")
                or payload.get("quoteTime")
                or payload.get("tradeTime")
            )
            quotes[sym] = {"last": last, "bid": bid, "ask": ask, "quoteTime": ts}
    return quotes


def get_accounts():
    url = f"{SCHWAB_API}/trader/v1/accounts"
    response = schwab_api_request("get", url)
    print("response", response)


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


def place_order(symbol, quantity, action, logger, position_effect):
    try:
        logger.info(
            f"Placing order for {symbol}, Action: {action}, Quantity: {quantity}"
        )
        asset_type = "FUTURE" if symbol.startswith("/") else "EQUITY"
        order_payload = {
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": action,
                    "quantity": quantity,
                    "instrument": {"symbol": symbol, "assetType": asset_type},
                }
            ],
        }

        encrypted_account_id = get_encrypted_account_id(SCHWAB_ACCOUNT_ID, logger)
        place_order_url = f"{SCHWAB_API}/trader/v1/accounts/{encrypted_account_id}/orders"
        response = schwab_api_request("post", place_order_url, order_payload)

        order_id = dict(response.headers)["Location"].split("/")[-1]
        # is_filled, traded_qty = check_order_status(order_id, logger)
        # if is_filled:
        #     logger.info(f"Order placed successfully for {symbol}. Order ID: {order_id}")
        #     return order_id
        # else:
        #     logger.warning(f"Order not filled for {symbol}. Order ID: {order_id}")
        #     logger.warning(f"Placing order again for {symbol}. Order ID: {order_id}")
        #     order_id = place_order(
        #         symbol, quantity, action, logger, position_effect
        #     )
        #     return order_id
        return order_id

    except Exception as e:
        logger.error(f"Error in placing order for {symbol}: {str(e)}")
        return None


def get_encrypted_account_id(schwab_account_id, logger):
    try:
        get_encrypted_account_id_url = f"{SCHWAB_API}/trader/v1/accounts/accountNumbers"
        print("get_encrypted_account_id_url", get_encrypted_account_id_url)
        response = schwab_api_request("get", get_encrypted_account_id_url)
        
        encrypted_account_id = response.json()[0]["hashValue"]
        return encrypted_account_id
    except Exception as e:
        logger.error(
            f"Error in getting encrypted account ID for {schwab_account_id}: {str(e)}"
        )
        return None


def check_position_status(symbol, account_id, logger):
    try:
        logger.info(f"Checking position status for {symbol}")
        position_url = f"{schwab_trader_link}/accounts"
        response = requests.get(
            url=position_url,
            params={"fields": "positions"},
            headers=create_api_header("Bearer", logger),
        )
        positions = response.json()[0]["securitiesAccount"]["positions"]
        for position in positions:
            if position["instrument"]["symbol"] == symbol:
                logger.info(f"Position found for {symbol}")
                return True

        logger.info(f"No position found for {symbol}")
        return False
    except Exception as e:
        logger.error(f"Error in checking position status for {symbol}: {str(e)}")
        return False


def check_order_status(order_id, logger):
    try:
        logger.info(f"Checking order status for Order ID: {order_id}")
        encrypted_account_id = get_encrypted_account_id(account_id, logger)
        check_order_status_url = (
            f"{schwab_trader_link}/accounts/{encrypted_account_id}/orders/{order_id}"
        )
        response = requests.get(
            url=check_order_status_url, headers=create_api_header("Bearer", logger)
        )
        order_history = response.json()
        order_status = order_history["status"]
        traded_qty = int(order_history["quantity"])

        if order_status == "FILLED":
            logger.info(f"Order ID {order_id} is filled")
            return True, traded_qty
        else:
            if order_status == "REJECTED":
                logger.warning(f"Order ID {order_id} is rejected")
                return False, 0
            else:
                logger.warning(f"Order ID {order_id} is not filled")
                sleep(1)
                is_filled, traded_qty = check_order_status(order_id, logger)
                return is_filled, traded_qty

    except Exception as e:
        logger.error(
            f"Error in checking order status for Order ID {order_id}: {str(e)}"
        )
        is_filled, traded_qty = check_order_status(order_id, logger)
        return is_filled, traded_qty


def cancel_order(order_id, account_id, schwab_account_id, logger):
    try:
        logger.info(f"Cancelling order with Order ID: {order_id}")
        cancel_order_url = (
            f"{schwab_trader_link}/accounts/{account_id}/orders/{order_id}"
        )
        response = requests.delete(
            url=cancel_order_url, headers=create_api_header("Bearer", logger)
        )
        logger.info(f"Order ID {order_id} cancelled successfully")
    except Exception as e:
        logger.error(f"Error in cancelling order with Order ID {order_id}: {str(e)}")


def validate_refresh_link(link, REFRESH_TOKEN_LINK):
    """
    Validate the provided Schwab refresh‐token link.
    If valid, save it to session_state and file.
    Returns (bool, str) for success status and message.
    """
    try:
        # Validate link format
        if not link or not isinstance(link, str):
            return False, "Invalid link format"

        if "code=" not in link:
            return False, "No authorization code found in link"

        is_valid = get_refresh_token(link)
        if is_valid:
            # Save the link as JSON with proper structure
            link_data = {"refresh_token_link": link}
            with open(REFRESH_TOKEN_LINK, "w") as file:
                json.dump(link_data, file, indent=2)
            return True, "Token refreshed successfully"
        else:
            return False, "Link is expired or invalid"
    except Exception as e:
        return False, f"Error validating link: {str(e)}"

if __name__  == "__main__":
    logger = configure_logger("SPX", "zeroday")
    get_accounts()