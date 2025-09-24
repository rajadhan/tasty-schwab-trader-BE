import json
import logging
import os
import pandas as pd
import pytz
import redis
import requests
import threading
import time
from datetime import datetime, timedelta
from flask import jsonify
from time import sleep
from utils import get_strategy_prarams, save_json, get_active_exchange_symbol, get_trade_file_path, load_json

TASTY_API = "https://api.tastyworks.com"
# TASTY_API = "https://api.cert.tastyworks.com"  # TEST
TASTY_REDIRECT_URI = "https://api.tastyworks.com"
TASTY_MY_APP_URL = "https://my.tastytrade.com/auth.html"
TASTY_CLIENT_SECRET = os.getenv("TASTY_CLIENT_SECRET")
TASTY_CLIENT_ID = os.getenv("TASTY_CLIENT_ID")
TASTY_ACCOUNT_ID = os.getenv("TASTY_ACCOUNT_ID")
TASTY_ACCESS_TOKEN_PATH = os.path.join("tokens", "tastytrade_tokens.txt")
logger = logging.getLogger(__name__)

# Redis client
REDIS_CLIENT = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0))
)


def authorize_url():
    params = {
        "response_type": "code",
        "client_id": TASTY_CLIENT_ID,
        "redirect_uri": TASTY_REDIRECT_URI,
    }
    request_url = requests.Request("GET", TASTY_MY_APP_URL, params=params).prepare().url
    print("Visit this URL in your browser to authorize:")
    print(request_url)
    return request_url


def access_token(authorization_code):
    access_token_url = f"{TASTY_API}/oauth/token"
    payload = {
        "grant_type": "authorization_code",
        "code": authorization_code,
        "client_id": TASTY_CLIENT_ID,
        "client_secret": TASTY_CLIENT_SECRET,
        "redirect_uri": TASTY_REDIRECT_URI,
    }
    response = requests.post(access_token_url, data=payload)
    tokens = response.json()
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if access_token and refresh_token:
        tokens_to_save = {"access_token": access_token, "refresh_token": refresh_token}
        with open(TASTY_ACCESS_TOKEN_PATH, "w") as f:
            json.dump(tokens_to_save, f)
        return jsonify(
            {
                "success": True,
                "message": "Tastytrade connection established successfully",
            }
        )
    else:
        return jsonify(
            {"success": False, "message": "Failed to establish Tastytrade connection"}
        )


def create_header(token):
    # Ensure token is properly formatted
    if not token.startswith("Bearer "):
        token = f"Bearer {token}"
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": token,
    }


def refresh_tastytrade_token():
    """Refresh TastyTrade access token using existing refresh token"""
    try:
        # Load existing refresh token
        try:
            with open(TASTY_ACCESS_TOKEN_PATH, "r") as f:
                tokens = json.load(f)
                refresh_token = tokens.get("refresh_token", "")
        except Exception:
            return jsonify({
                "success": False,
                "error": "No refresh token found"
            }), 400
        
        if not refresh_token:
            return jsonify({
                "success": False,
                "error": "No refresh token available"
            }), 400
        
        refresh_url = f"{TASTY_API}/oauth/token"
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": TASTY_CLIENT_ID,
            "client_secret": TASTY_CLIENT_SECRET,
            "redirect_uri": TASTY_REDIRECT_URI,
        }
        response = requests.post(refresh_url, data=payload)
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
        new_tokens = response.json()
        new_access_token = new_tokens.get("access_token")
        new_refresh_token = new_tokens.get(
            "refresh_token", refresh_token
        )
        if not new_access_token:
            return jsonify({"success": False, "error": "No access token received"}), 400

        # Save new tokens
        tokens_to_save = {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
        }
        save_json(TASTY_ACCESS_TOKEN_PATH, tokens_to_save)
        return jsonify({
            "success": True,
            "message": "TastyTrade tokens refreshed successfully",
            "access_token": new_access_token,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def tastytrade_api_request(method, url, **kwargs):
    # Load token
    with open(TASTY_ACCESS_TOKEN_PATH, "r") as f:
        tokens = json.load(f)
    access_token = tokens.get("access_token")
    headers = kwargs.pop("headers", {})
    headers.update(create_header(access_token))
    response = requests.request(method, url, headers=headers, **kwargs)
    if response.status_code == 401 or "invalid_token" in response.text or "expired_token" in response.text:
        # Try refresh
        access_token = refresh_tastytrade_token()
        headers = create_header(access_token)
        response = requests.request(method, url, headers=headers, **kwargs)
    return response


def update_tastytrade_instruments_csv(output_file="tastytrade_instruments.csv"):
    url = f"{TASTY_API}/instruments/futures"
    try:
        resp = tastytrade_api_request("GET", url)
        resp.raise_for_status()
        data = resp.json()
        print("data", data)

        if "data" not in data or "items" not in data["data"]:
            raise ValueError("Unexpected API response format.")

        items = data["data"]["items"]
        print("itmes", items)
        df = pd.DataFrame(items)
        print("df", df)

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"[{datetime.now()}] Saved {len(df)} rows to {output_file}")

    except Exception as e:
        print(f"Error updating Tastytrade instruments: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check that your Tastytrade tokens are valid in tokens/tastytrade_tokens.txt")
        print("2. You may need to re-authenticate with Tastytrade through the main application")
        print("3. Verify that the TASTY_API endpoint in config.py is correct")
        print("4. Ensure you have network connectivity to the Tastytrade API servers")


def create_order_payload(symbol, qty, action, account_id, logger):
    instrument_type = "Future" if symbol[0] == "/" else "Equity"
    order_payload = {
        "order-type": "Market",
        "time-in-force": "Day",
        "legs": [
            {
                "instrument-type": instrument_type,
                "symbol": symbol,
                "action": action,
                "quantity": str(qty),
            }
        ],
    }

    logger.info(f"{account_id}=> Credit order payload: {order_payload}")
    return order_payload


def place_order(order_payload, account_id, logger):
    place_order_url = f"{TASTY_API}/accounts/{account_id}/orders"
    print("place order url", place_order_url)
    try:
        response = tastytrade_api_request("POST", place_order_url, json=order_payload)
        order = response.json()
        logger.info(f"{account_id}=> order response = {order}")
        order_id = order.get("data", {}).get("order", {}).get("id", "")
        return order_id
    except Exception as e:
        logger.error(
            f"{account_id}=> ERROR! in placing order = {e}. Response text = {getattr(response, 'text', '')}"
        )
        return ""


def check_order_status(order_id, account_id, logger):
    try:
        check_order_status_url = f"{TASTY_API}/accounts/{account_id}/orders/{order_id}"
        response = tastytrade_api_request("GET", check_order_status_url)
        order_history = response.json()
        
        logger.info(f"order_history for user {account_id} = {order_history}")
        order_status = order_history["data"]["items"][0]["status"]
        logger.info(f"{account_id}=> order status = {order_status}")
        traded_qty = int(order_history["data"]["items"][0]["legs"][0]["quantity"])
        logger.info(f"{account_id}=> traded qty = {traded_qty}")
        if order_status == "Rejected" or order_status == "Cancelled":
            return False, traded_qty
        else:
            if order_status == "Filled":
                return True, traded_qty
            else:
                sleep(1)
                is_filled, traded_qty = check_order_status(order_id, account_id, logger)
                return is_filled, traded_qty

    except Exception as e:
        logger.error(
            f"{account_id}=> ERROR! in checking order status = {str(e)}. Response text = {getattr(response, 'text', '')}"
        )
        return False, 0


def place_tastytrade_order(symbol, qty, action, logger):
    print("place tastytrade order", symbol, qty, action, TASTY_ACCOUNT_ID, logger)
    try:
        symbol = "/" + get_active_exchange_symbol(symbol)
        if symbol[-1].isdigit():
            symbol = f"/{symbol.lstrip('/')}"
        order_payload = create_order_payload(symbol, qty, action, TASTY_ACCOUNT_ID, logger)
        print("order payload", order_payload)
        order_id = place_order(order_payload, TASTY_ACCOUNT_ID, logger)
        is_filled, _ = check_order_status(order_id, TASTY_ACCOUNT_ID, logger)
        print("is filled", is_filled)
        if is_filled:
            return order_id
        else:
            logger.warning(f"Order not filled for {symbol}. Order ID: {order_id}")
            logger.warning(f"Placing order again for {symbol}. Order ID: {order_id}")
            order_id = place_order(order_payload, TASTY_ACCOUNT_ID, logger)
            return order_id
    except Exception as e:
        print(f"{TASTY_ACCOUNT_ID}=> ERROR! in placing order = {e}.")
        return ""


def manual_trigger_action(ticker, action, logger):
    trade_file = get_trade_file_path(ticker, "zeroday")
    trades = load_json(trade_file)
    [tasty_qty] = get_strategy_prarams("zeroday", ticker, logger)[3:4]
    if ticker not in trades:
        if action == "long":
            order_id_tastytrade = (
                place_option_trade(
                    ticker, "CALL", "Buy to Open", tasty_qty, TASTY_ACCOUNT_ID, logger
                )
                if int(tasty_qty) > 0
                else 0
            )
            trades[ticker] = {
                "action": "LONG",
                "option_type": "CALL",
                "order_id_tastytrade": order_id_tastytrade,
                "entry_time": datetime.now(pytz.utc).isoformat(),
            }
        elif action == "short":
            order_id_tastytrade = (
                place_option_trade(
                    ticker, "PUT", "Buy to Open", tasty_qty, TASTY_ACCOUNT_ID, logger
                )
                if int(tasty_qty) > 0
                else 0
            )
            trades[ticker] = {
                "action": "SHORT",
                "option_type": "PUT",
                "order_id_tastytrade": order_id_tastytrade,
                "entry_time": datetime.now(pytz.utc).isoformat(),
            }
    else:
        if action == "long":
            if trades.get(ticker, {}).get("action") == "SHORT":
                order_id_tastytrade = (
                    place_option_trade(
                        ticker, "PUT", "Sell to Close", tasty_qty, TASTY_ACCOUNT_ID, logger
                    )
                    if int(tasty_qty) > 0
                    else 0
                )
            order_id_tastytrade = (
                place_option_trade(
                    ticker, "CALL", "Buy to Open", tasty_qty,
                    TASTY_ACCOUNT_ID, logger
                )
                if int(tasty_qty) > 0
                else 0
            )
            trades[ticker] = {
                "action": "LONG",
                "option_type": "CALL",
                "order_id_tastytrade": order_id_tastytrade,
                "entry_time": datetime.now(pytz.utc).isoformat(),
            }
        elif action == "short":
            if trades.get(ticker, {}).get("action") == "LONG":
                order_id_tastytrade = (
                    place_option_trade(
                        ticker, "CALL", "Sell to Close", tasty_qty, TASTY_ACCOUNT_ID, logger
                    )
                    if int(tasty_qty) > 0
                    else 0
                )
            order_id_tastytrade = (
                place_option_trade(
                    ticker, "PUT", "Buy to Open", tasty_qty, TASTY_ACCOUNT_ID, logger
                )
                if int(tasty_qty) > 0
                else 0
            )
            trades[ticker] = {
                "action": "SHORT",
                "option_type": "PUT",
                "order_id_tastytrade": order_id_tastytrade,
                "entry_time": datetime.now(pytz.utc).isoformat(),
            }

    with open(trade_file, "w") as file:
        json.dump(trades, file)

    logger.info(f"Manual trigger executed: {action} for {ticker}")


def manual_ema_trigger_action(ticker, action, logger):
    trade_file = get_trade_file_path(ticker, "ema")
    trades = load_json(trade_file)
    [tasty_qty] = get_strategy_prarams("ema", ticker, logger)[3:4]

    if ticker not in trades:
        if action == "LONG":
            order_id_tastytrade = (
                place_tastytrade_order(
                    ticker, tasty_qty, "Buy to Open", TASTY_ACCOUNT_ID, logger
                )
                if int(tasty_qty) > 0
                else 0
            )
            trades[ticker] = {
                "action": "LONG",
                "order_id_tastytrade": order_id_tastytrade,
            }
        elif action == "SHORT":
            order_id_tastytrade = (
                place_tastytrade_order(
                    ticker, tasty_qty, "Sell to Open", TASTY_ACCOUNT_ID, logger
                )
                if int(tasty_qty) > 0
                else 0
            )
            trades[ticker] = {
                "action": "SHORT",
                "order_id_tastytrade": order_id_tastytrade,
            }
    else:
        # If switching positions, place a single order in the opposite direction
        # This automatically closes the existing position and opens the new one
        if trades[ticker]["action"] == "LONG" and action == "SHORT":
            order_id_tastytrade = (
                place_tastytrade_order(
                    ticker, tasty_qty, "Sell to Close", TASTY_ACCOUNT_ID, logger
                )
                if int(tasty_qty) > 0
                else 0
            )
            order_id_tastytrade = (
                place_tastytrade_order(
                    ticker, tasty_qty, "Sell to Open", TASTY_ACCOUNT_ID, logger
                )
                if int(tasty_qty) > 0
                else 0
            )
            trades[ticker] = {
                "action": "SHORT",
                "order_id_tastytrade": order_id_tastytrade,
            }

        elif trades[ticker]["action"] == "SHORT" and action == "LONG":
            order_id_tastytrade = (
                place_tastytrade_order(
                    ticker, tasty_qty, "Buy to Close", TASTY_ACCOUNT_ID, logger
                )
                if int(tasty_qty) > 0
                else 0
            )
            order_id_tastytrade = (
                place_tastytrade_order(
                    ticker, tasty_qty, "Buy to Open", TASTY_ACCOUNT_ID, logger
                )
                if int(tasty_qty) > 0
                else 0
            )
            trades[ticker] = {
                "action": "LONG",
                "order_id_tastytrade": order_id_tastytrade,
            }

    with open(
        f"trades/ema/{ticker[1:] if '/' == ticker[0] else ticker}.json", "w"
    ) as file:
        json.dump(trades.copy(), file)

    logger.info(f"Strategy for {ticker} completed.")


if __name__ == "__main__":
    print("tastu_data", get_account_info())