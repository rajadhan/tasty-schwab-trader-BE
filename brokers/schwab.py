from config import *
from datetime import datetime, timedelta
from flask import jsonify
from pytz import timezone as pytz_timezone
from time import sleep
from urllib.parse import unquote
from utils import *
import base64
import os
import pandas as pd
import requests
import json

SCHWAB_API_KEY = os.getenv("SCHWAB_API_KEY")
SCHWAB_API_SECRET = os.getenv("SCHWAB_API_SECRET")
SCHWAB_API = "https://api.schwabapi.com"
SCHWAB_CALLBACK_URL = "https://127.0.0.1"
logger = logging.getLogger(__name__)

def authorize_url():
    params = {
        "client_id": SCHWAB_API_KEY,
        "redirect_uri": SCHWAB_CALLBACK_URL
    }
    request_url = (
        requests.Request("GET", f"{SCHWAB_API}/v1/oauth/authorize", params=params).prepare().url
    )
    return request_url

def create_api_header(access_token):
    # Ensure token is properly formatted
    if not access_token.startswith("Bearer "):
        token = f"Bearer {access_token}"
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
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
    authtoken_link = f"{SCHWAB_API}/v1/oauth/authorize"

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
        # Load existing refresh token
        try:
            with open(SCHWAB_ACCESS_TOKEN_PATH, "r") as f:
                tokens = json.load(f)
                refresh_token = tokens.get("refresh_token", "")
        except Exception:
            return jsonify({"success": False, "error": "No refresh token found"}), 400

        if not refresh_token:
            return (
                jsonify({"success": False, "error": "No refresh token available"}),
                400,
            )

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
                jsonify(
                    {
                        "success": False,
                        "error": f"Failed to refresh token: {response.status_code}",
                        "details": response.text,
                    }
                ),
                400,
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
            jsonify(
                {
                    "success": True,
                    "message": "Charles Schwab tokens refreshed successfully",
                    "access_token": new_access_token,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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
        logger.info(f"Fetching historical data for {symbol}")
        current_datetime = datetime.now(tz=pytz_timezone(time_zone))
        endtime = current_datetime + timedelta(seconds=85)
        if isinstance(time_frame, str):
            if time_frame.isdigit():
                time_frame = int(time_frame)
        # Mapping of time frames to configurations
        time_frame_config = {
            "1h": {
                "periodType": "day",
                "period": 10,
                "frequencyType": "minute",
                "frequency": 30,
                "resample": "1H",
            },
            "4h": {
                "periodType": "day",
                "period": 10,
                "frequencyType": "minute",
                "frequency": 30,
                "resample": "4H",
            },
            "1d": {
                "periodType": "month",
                "period": 2,
                "frequencyType": "daily",
                "frequency": 1,
            },
            1: {
                "periodType": "day",
                "period": 2,
                "frequencyType": "minute",
                "frequency": 1,
            },
            2: {
                "periodType": "day",
                "period": 2,
                "frequencyType": "minute",
                "frequency": 1,
                "resample": "2min",
            },
            5: {
                "periodType": "day",
                "period": 2,
                "frequencyType": "minute",
                "frequency": 5,
            },
            15: {
                "periodType": "day",
                "period": 5,
                "frequencyType": "minute",
                "frequency": 15,
            },
            30: {
                "periodType": "day",
                "period": 5,
                "frequencyType": "minute",
                "frequency": 30,
            },
        }

        if time_frame not in time_frame_config:
            raise ValueError(f"Unsupported time frame: {time_frame}")

        # Extract configuration for the given time_frame
        config = time_frame_config[time_frame]
        periodType, period, frequencyType, frequency = (
            config["periodType"],
            config["period"],
            config["frequencyType"],
            config["frequency"],
        )

        # Prepare request parameters
        params = {
            "symbol": symbol,
            "periodType": periodType,
            "endDate": _time_convert(endtime, "epoch_ms"),
            "period": period,
            "frequencyType": frequencyType,
            "frequency": frequency,
            "needExtendedHoursData": True if symbol[0] == "/" else False,
        }

        # API Request
        response = requests.get(
            f"{base_api_url}/marketdata/v1/pricehistory",
            headers=create_api_header("Bearer", logger),
            params=params_parser(params),
        )

        # Check if response is successful
        if response.status_code == 401:
            logger.warning("Access token expired, attempting to refresh...")
            if refresh_access_token(logger):
                # Retry the request with new token
                response = requests.get(
                    f"{base_api_url}/marketdata/v1/pricehistory",
                    headers=create_api_header("Bearer", logger),
                    params=params_parser(params),
                )
                if response.status_code != 200:
                    logger.error(
                        f"API request failed after token refresh: {response.status_code}: {response.text}"
                    )
                    raise ValueError(
                        f"API request failed with status code {response.status_code}"
                    )
            else:
                logger.error("Failed to refresh access token")
                raise ValueError("Authentication failed and token refresh failed")
        elif response.status_code != 200:
            logger.error(
                f"API request failed with status code {response.status_code}: {response.text}"
            )
            raise ValueError(
                f"API request failed with status code {response.status_code}"
            )

        # Try to parse JSON response
        try:
            response_data = response.json()
        except ValueError as json_error:
            logger.error(f"Failed to parse JSON response: {response.text[:200]}...")
            raise ValueError(f"Invalid JSON response from API: {str(json_error)}")

        # Check for error in response
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
            .dt.tz_convert(time_zone)
        )
        df = df[["datetime", "symbol", "open", "high", "low", "close"]]
        # Resampling if applicable
        if "resample" in config:
            resample_freq = config["resample"]
            if time_frame == "1h":
                # Adjust resampling for 30-minute offset
                df = (
                    df.set_index("datetime")
                    .resample(resample_freq, offset="30min")  # 30-minute offset
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
                    .resample(resample_freq, offset="90min")  # 90-minute offset
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
            tz=pytz_timezone(time_zone)
        ).strftime("%H:%M"):
            df = df[:-1]
        logger.info(f"Historical data for {symbol} fetched successfully")
        return df
    except Exception as e:
        logger.error(f"Error in getting historical data for {symbol}: {str(e)}")
        # Add a small delay before retry to avoid overwhelming the API
        sleep(5)
        # Limit retries to prevent infinite loops
        try:
            df = historical_data(symbol, time_frame, logger)
            return df
        except Exception as retry_error:
            logger.error(f"Retry failed for {symbol}: {str(retry_error)}")
            # Return empty DataFrame as fallback
            return pd.DataFrame(
                columns=["datetime", "symbol", "open", "high", "low", "close"]
            )


def place_order(symbol, quantity, action, account_id, logger, position_effect):
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

        encrypted_account_id = get_encrypted_account_id(account_id, logger)
        place_order_url = f"{schwab_trader_link}/accounts/{encrypted_account_id}/orders"
        response = requests.post(
            url=place_order_url,
            json=order_payload,
            headers=create_api_header("Bearer", logger),
        )

        order_id = dict(response.headers)["Location"].split("/")[-1]
        is_filled, traded_qty = check_order_status(order_id, logger)
        if is_filled:
            logger.info(f"Order placed successfully for {symbol}. Order ID: {order_id}")
            return order_id
        else:
            logger.warning(f"Order not filled for {symbol}. Order ID: {order_id}")
            logger.warning(f"Placing order again for {symbol}. Order ID: {order_id}")
            order_id = place_order(
                symbol, quantity, action, account_id, logger, position_effect
            )
            return order_id

    except Exception as e:
        logger.error(f"Error in placing order for {symbol}: {str(e)}")
        return None


def get_encrypted_account_id(schwab_account_id, logger):
    try:
        get_encrypted_account_id_url = f"{schwab_trader_link}/accounts/accountNumbers"
        response = requests.get(
            url=get_encrypted_account_id_url, headers=create_api_header("Bearer", logger)
        )
        encrypted_account_id = response.json()[0]["hashValue"]
        return encrypted_account_id
    except Exception as e:
        logger.error(
            f"Error in getting encrypted account ID for {schwab_account_id}: {str(e)}"
        )
        sleep(10)
        encrypted_account_id = get_encrypted_account_id(schwab_account_id, logger)
        return encrypted_account_id


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
