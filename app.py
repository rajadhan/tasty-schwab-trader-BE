import json
import os
import jwt
import requests
from datetime import timedelta, datetime, timezone
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from functools import wraps
import redis
from main_equities import run_every_week
from utils import ticker_data_path_for_strategy
from config import *

app = Flask(__name__, static_folder="static")
CORS(app)

# ========= Secret Key (keep this secret and safe!) =========
JWT_SECRET = "secret-trading"  # Use a secure random string in production
JWT_EXPIRATION_MINUTES = 60 * 24 * 30

# ================== File Paths ===================
ADMIN_CREDENTIALS_PATH = os.path.join("credentials", "admin_credentials.json")
SYMBOL_DATA_PATH = os.path.join("consts", "symbol.json")
TREND_DATA_PATH = os.path.join("consts", "trend_line.json")
REFRESH_TOKEN_LINK = os.path.join("jsons", "refresh_token_link.json")
REDIS_CLIENT = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", 6379)), db=int(os.getenv("REDIS_DB", 0)))


# ================== JSON Utilities ===============
def load_json(filepath):
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except Exception:
        return {}


def save_json(filepath, data):
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)


# ================== Load Initial Data ===============
admin_credentials = load_json(ADMIN_CREDENTIALS_PATH)
symbol_data = load_json(SYMBOL_DATA_PATH)
trend_data = load_json(TREND_DATA_PATH)

# ================== JWT Auth Utilities ===================


def generate_token(email, password):
    payload = {
        "email": email,
        "password": password,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRATION_MINUTES),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return token


def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = None
        # Expecting Authorization: Bearer <token>
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
        if not token:
            return jsonify({"success": False, "error": "Token is missing"}), 401

        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"success": False, "error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"success": False, "error": "Invalid token"}), 401

        return f(*args, **kwargs)

    return decorator


# ================== Routes ==================

@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        email = data.get("email")
        password = data.get("password")

        if email == admin_credentials.get(
            "email"
        ) and password == admin_credentials.get("password"):
            token = generate_token(email, password)

            try:
                with open(refresh_token_path, "r") as file:
                    refresh_token = json.load(file)
            except Exception:
                refresh_token = ""

            try:
                with open(TASTY_ACCESS_TOKEN_PATH, "r") as file:
                    tasty_token = json.load(file)
                    access_token = tasty_token.get("access_token", "")
            except Exception:
                access_token = ""

            return (
                jsonify(
                    {
                        "success": True,
                        "token": token,
                        "refreshToken": refresh_token,
                        "tastyToken": access_token,
                    }
                ),
                200,
            )
        else:
            return (
                jsonify({"success": False, "message": "Invalid email or password"}),
                401,
            )
    except Exception as e:
        print("error", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/tasty/authorize-url", methods=["GET"])
@token_required
def tasty_authorize_url():
    params = {
        "response_type": "code",
        "client_id": TASTY_CLIENT_ID,
        "redirect_uri": TASTY_REDIRECT_URI,
    }
    request_url = requests.Request("GET", TASTY_MY_APP_URL, params=params).prepare().url
    print("Visit this URL in your browser to authorize:")
    print(request_url)
    return request_url


@app.route("/api/tasty/access-token", methods=["POST"])
@token_required
def tasty_access_token():
    data = request.json
    authorization_code = data.get("authorizationCode")
    access_token_url = f"{TASTY_API}/oauth/token"
    payload = {
        "grant_type": "authorization_code",
        "code": authorization_code,
        "client_id": TASTY_CLIENT_ID,
        "client_secret": TASTY_CLIENT_SECRET,
        "redirect_uri": TASTY_REDIRECT_URI,
    }
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    response = requests.post(access_token_url, data=payload, headers=headers)
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


@app.route("/api/tasty/refresh-token", methods=["POST"])
@token_required
def tasty_refresh_token():
    """Refresh TastyTrade access token using existing refresh token"""
    try:
        # Load existing refresh token
        try:
            with open(TASTY_ACCESS_TOKEN_PATH, "r") as f:
                tokens = json.load(f)
                refresh_token = tokens.get("refresh_token", "")
        except Exception:
            return jsonify({"success": False, "error": "No refresh token found"}), 400

        if not refresh_token:
            return jsonify({"success": False, "error": "No refresh token available"}), 400

        # Exchange refresh token for new access token
        refresh_url = f"{TASTY_API}/oauth/token"
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": TASTY_CLIENT_ID,
            "client_secret": TASTY_CLIENT_SECRET,
        }
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        
        response = requests.post(refresh_url, data=payload, headers=headers)
        
        if response.status_code != 200:
            return jsonify({
                "success": False, 
                "error": f"Failed to refresh token: {response.status_code}",
                "details": response.text
            }), 400

        new_tokens = response.json()
        new_access_token = new_tokens.get("access_token")
        new_refresh_token = new_tokens.get("refresh_token", refresh_token)  # Keep old if not provided

        if not new_access_token:
            return jsonify({"success": False, "error": "No access token received"}), 400

        # Save new tokens
        tokens_to_save = {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token
        }
        
        with open(TASTY_ACCESS_TOKEN_PATH, "w") as f:
            json.dump(tokens_to_save, f)

        return jsonify({
            "success": True,
            "message": "TastyTrade tokens refreshed successfully",
            "access_token": new_access_token
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/update-credentials", methods=["POST"])
@token_required
def update_credentials():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        current_password = data.get("currentPassword")
        new_email = data.get("newEmail")
        new_password = data.get("newPassword")

        # Basic validation
        if not current_password:
            return jsonify({"success": False, "error": "Current password is required"}), 400

        # Verify current password
        if current_password != admin_credentials.get("password"):
            return jsonify({"success": False, "error": "Current password is incorrect"}), 401

        # Apply updates in-memory
        updated = False
        if new_email and isinstance(new_email, str):
            admin_credentials["email"] = new_email
            updated = True
        if new_password and isinstance(new_password, str):
            admin_credentials["password"] = new_password
            updated = True

        if not updated:
            return jsonify({"success": False, "error": "Nothing to update"}), 400

        # Persist to disk
        save_json(ADMIN_CREDENTIALS_PATH, admin_credentials)

        # Issue a fresh token with potentially updated email/password
        fresh_email = admin_credentials.get("email")
        fresh_password = admin_credentials.get("password")
        token = generate_token(fresh_email, fresh_password)

        return jsonify({
            "success": True,
            "message": "Credentials updated successfully",
            "token": token,
            "email": fresh_email,
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/add-ticker", methods=["POST"])
@token_required
def add_ticker():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        if data.get("strategy") == "ema" or data.get("strategy") == "zeroday":
            if (
                data.get("symbol") in symbol_data.get("symbol")
                and data.get("trend_line_1") in trend_data.get("trend")
                and data.get("trend_line_2") in trend_data.get("trend")
                and data.get("period_1")
                and data.get("period_2")
                and data.get("timeframe")
                and
                # data.get('schwab_quantity') and
                # data.get('tastytrade_quantity') and
                isinstance(data.get("trade_enabled"), bool)
            ):

                # Convert timeframe to desired format
                raw_timeframe = str(data.get("timeframe"))
                if raw_timeframe.endswith("Min"):
                    formatted_timeframe = raw_timeframe.replace("Min", "")
                elif raw_timeframe.endswith("Hour"):
                    formatted_timeframe = raw_timeframe.replace("Hour", "h")
                elif raw_timeframe.endswith("Day"):
                    formatted_timeframe = raw_timeframe.replace("Day", "d")
                else:
                    formatted_timeframe = raw_timeframe  # e.g., 100t remains unchanged

                symbol_key = f"{data.get('symbol')}"
                formatted = [
                    formatted_timeframe,
                    str(data.get("schwab_quantity")),
                    str(data.get("trade_enabled")).upper(),
                    str(data.get("tastytrade_quantity")),
                    str(data.get("trend_line_1")),
                    str(data.get("period_1")),
                    str(data.get("trend_line_2")),
                    str(data.get("period_2")),
                ]

                # Add call_enabled and put_enabled for zeroday strategy
                if data.get("strategy") == "zeroday":
                    formatted.append(str(data.get("call_enabled", True)).upper())
                    formatted.append(str(data.get("put_enabled", True)).upper())

            else:
                return jsonify({"success": False, "error": "Invalid input data"}), 400
        elif data.get("strategy") == "supertrend":
            # Convert timeframe to desired format
            raw_timeframe = str(data.get("timeframe"))
            if raw_timeframe.endswith("Min"):
                formatted_timeframe = raw_timeframe.replace("Min", "")
            elif raw_timeframe.endswith("Hour"):
                formatted_timeframe = raw_timeframe.replace("Hour", "h")
            elif raw_timeframe.endswith("Day"):
                formatted_timeframe = raw_timeframe.replace("Day", "d")
            else:
                formatted_timeframe = raw_timeframe  # e.g., 100t remains unchanged
            symbol_key = f"{data.get('symbol')}"
            formatted = [
                formatted_timeframe,
                str(data.get("schwab_quantity")),
                str(data.get("trade_enabled")).upper(),
                str(data.get("tastytrade_quantity")),
                str(data.get("short_ma_length")),
                str(data.get("short_ma_type")),
                str(data.get("mid_ma_length")),
                str(data.get("mid_ma_type")),
                str(data.get("long_ma_length")),
                str(data.get("long_ma_type")),
                str(data.get("zigzag_percent_reversal")),
                str(data.get("atr_length")),
                str(data.get("zigzag_atr_multiple")),
                str(data.get("fibonacci_enabled")),
                str(data.get("support_demand_enabled")),
            ]
        # Load current data
        strategy = data.get("strategy")
        TICKER_DATA_PATH = ticker_data_path_for_strategy(strategy)
        saved_data = load_json(TICKER_DATA_PATH)
        saved_data[symbol_key] = formatted
        save_json(TICKER_DATA_PATH, saved_data)

        return jsonify({"success": True, "data": saved_data}), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/get-ticker", methods=["GET"])
@token_required
def get_ticker():
    try:
        strategy = request.args.get("strategy")  # Get query from request

        # Load current data
        TICKER_DATA_PATH = ticker_data_path_for_strategy(strategy)
        data = load_json(TICKER_DATA_PATH)

        # In case of no data
        if not data:
            return jsonify({"success": True, "data": []}), 200

        return jsonify({"success": True, "data": data}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/delete-ticker", methods=["DELETE"])
@token_required
def delete_ticker():
    try:
        data = request.get_json()
        strategy = data.get("strategy")
        symbol_key = data.get("symbol")

        # Load current data
        TICKER_DATA_PATH = ticker_data_path_for_strategy(strategy)
        saved_data = load_json(TICKER_DATA_PATH)

        # in case of empty data
        if not saved_data:
            return jsonify({"success": True, "data": []}), 200

        # Check if symbol exists in data
        if symbol_key not in saved_data:
            return jsonify({"success": False, "error": "Symbol not found"}), 404

        # Delete the symbol from data
        del saved_data[symbol_key]

        # Save the updated data back to file
        save_json(TICKER_DATA_PATH, saved_data)

        return jsonify({"success": True, "data": saved_data}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/refresh-token-link", methods=["POST"])
@token_required
def refresh_link():
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400
    link = data.get("refresh_token_link")
    success, message = validate_refresh_link(link, REFRESH_TOKEN_LINK)
    return jsonify({"success": success, "message": message})


@app.route("/api/start-trading", methods=["GET"])
@token_required
def start_trading():
    try:
        strategy = request.args.get("strategy")  # Get query from request
        # Load current data
        TICKER_DATA_PATH = ticker_data_path_for_strategy(strategy)
        data = load_json(TICKER_DATA_PATH)
        if not data:
            return jsonify({"success": False, "error": "No ticker data found"}), 404

        trade_enabled_symbols = []
        for symbol, values in data.items():
            # values[2] is trade_enabled ("TRUE" or "FALSE")
            if len(values) >= 3 and values[2] == "TRUE":
                trade_enabled_symbols.append(symbol)
        results = {}
        if trade_enabled_symbols:
            # Clear stop flag for this strategy
            try:
                REDIS_CLIENT.delete(f"trading:stop:{strategy}")
            except Exception:
                pass
            print("Loading trading parameters ... ")
            run_every_week(strategy)
            # result = requests.get('https://api.schwabapi.com/v1/oauth/authorize?response_type=code&client_id=1iSr8ykD9qh2M2HoQv56wM2R1kWgQYZI&redirect_uri=https://127.0.0.1', allow_redirects=True)
            print("Trading started!")
            # After running the strategy, collect trade and signal/chart data for each enabled ticker
            import os
            import json

            for symbol in trade_enabled_symbols:
                trade_file = os.path.join(f"trades/{strategy}", f"{symbol}.json")
                trades = None
                if os.path.exists(trade_file):
                    try:
                        with open(trade_file, "r") as f:
                            trades = json.load(f)
                    except Exception as e:
                        trades = {"error": str(e)}
                # Optionally, add OHLCV/signals/chart data here if available
                # For now, just return trades
                results[symbol] = {"trades": trades}
            return (
                jsonify(
                    {
                        "success": True,
                        "message": "Trading has started",
                        "enabled_symbols": trade_enabled_symbols,
                        "results": results,
                    }
                ),
                200,
            )
        else:
            return (
                jsonify({"success": False, "error": "No enabled symbols for trading"}),
                400,
            )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stop-trading", methods=["GET"])
@token_required
def stop_trading():
    try:
        strategy = request.args.get("strategy")
        if not strategy:
            return jsonify({"success": False, "error": "strategy is required"}), 400
        REDIS_CLIENT.set(f"trading:stop:{strategy}", "1")
        return jsonify({"success": True, "message": f"Stop signal sent for {strategy}"}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/manual-trade")
def manual_trade_page():
    return send_from_directory(app.static_folder, "manual_trade.html")


@app.route("/trades/<strategy>/<ticker>.json")
@token_required
def get_trade_file(strategy, ticker):
    try:
        trade_file = f"trades/{strategy}/{ticker}.json"
        if os.path.exists(trade_file):
            with open(trade_file, "r") as file:
                trades = json.load(file)
            return jsonify(trades), 200
        else:
            return jsonify({}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/manual-trade", methods=["POST"])
@token_required
def manual_trade():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        ticker = data.get("ticker")
        action = data.get("action")  # "BUY" or "SELL"

        if not ticker or not action or action not in ["BUY", "SELL"]:
            return jsonify({"success": False, "error": "Invalid ticker or action"}), 400

        # Configure logger
        from utils import configure_logger

        logger = configure_logger(ticker, "zeroday")

        # Get strategy parameters
        from utils import get_strategy_prarams

        params = get_strategy_prarams("zeroday", ticker, logger)

        if not params:
            return (
                jsonify({"success": False, "error": "Strategy parameters not found"}),
                404,
            )

        # Extract parameters
        [
            timeframe,
            schwab_qty,
            trade_enabled,
            tasty_qty,
            trend_line_1,
            period_1,
            trend_line_2,
            period_2,
            call_enabled,
            put_enabled,
        ] = params

        # Default to True if call_enabled or put_enabled are not present
        call_enabled = (
            call_enabled if isinstance(call_enabled, bool) else call_enabled == "TRUE"
        )
        put_enabled = (
            put_enabled if isinstance(put_enabled, bool) else put_enabled == "TRUE"
        )

        # Convert quantities to integers
        schwab_qty = int(schwab_qty)
        tasty_qty = int(tasty_qty)

        # Check if trading is enabled
        if trade_enabled != "TRUE":
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Trading is not enabled for this ticker",
                    }
                ),
                400,
            )

        # Create trades directory if it doesn't exist
        import os

        os.makedirs(f"trades/zeroday", exist_ok=True)
        trade_file = f"trades/zeroday/{ticker}.json"

        # Load existing trades
        import json

        try:
            with open(trade_file, "r") as file:
                trades = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            trades = {}

        # Get current position
        current_position = (
            trades.get(ticker, {}).get("action") if ticker in trades else None
        )

        # Import necessary functions
        from schwab import place_order
        from tastytrade import place_tastytrade_order
        from config import schwab_account_id, TASTY_ACCOUNT_ID
        import pytz
        from datetime import datetime

        # Execute trade based on action and current position
        if action == "BUY":
            if current_position == "LONG":
                return (
                    jsonify({"success": False, "error": "Already in LONG position"}),
                    400,
                )

            # Close SHORT position if exists
            if current_position == "SHORT":
                logger.info(f"Closing SHORT position for {ticker} (manual trigger)")
                short_order_id_schwab = (
                    place_order(
                        ticker,
                        schwab_qty,
                        "BUY_TO_COVER",
                        schwab_account_id,
                        logger,
                        "CLOSING",
                    )
                    if schwab_qty > 0
                    else 0
                )
                short_order_id_tastytrade = (
                    place_tastytrade_order(
                        ticker, tasty_qty, "Buy to Close", TASTY_ACCOUNT_ID, logger
                    )
                    if tasty_qty > 0
                    else 0
                )

            # Open LONG position
            logger.info(f"Opening LONG position for {ticker} (manual trigger)")
            order_id_schwab = (
                place_order(
                    ticker, schwab_qty, "BUY", schwab_account_id, logger, "OPENING"
                )
                if schwab_qty > 0
                else 0
            )
            order_id_tastytrade = (
                place_tastytrade_order(
                    ticker, tasty_qty, "Buy to Open", TASTY_ACCOUNT_ID, logger
                )
                if tasty_qty > 0
                else 0
            )
            trades[ticker] = {
                "action": "LONG",
                "order_id_schwab": order_id_schwab,
                "order_id_tastytrade": order_id_tastytrade,
                "entry_time": datetime.now(pytz.utc).isoformat(),
                "entry_type": "manual",
            }

        elif action == "SELL":
            if current_position == "SHORT":
                return (
                    jsonify({"success": False, "error": "Already in SHORT position"}),
                    400,
                )

            # Close LONG position if exists
            if current_position == "LONG":
                logger.info(f"Closing LONG position for {ticker} (manual trigger)")
                long_order_id_schwab = (
                    place_order(
                        ticker, schwab_qty, "SELL", schwab_account_id, logger, "CLOSING"
                    )
                    if schwab_qty > 0
                    else 0
                )
                long_order_id_tastytrade = (
                    place_tastytrade_order(
                        ticker, tasty_qty, "Sell to Close", TASTY_ACCOUNT_ID, logger
                    )
                    if tasty_qty > 0
                    else 0
                )

            # Open SHORT position
            logger.info(f"Opening SHORT position for {ticker} (manual trigger)")
            order_id_schwab = (
                place_order(
                    ticker,
                    schwab_qty,
                    "SELL_SHORT",
                    schwab_account_id,
                    logger,
                    "OPENING",
                )
                if schwab_qty > 0
                else 0
            )
            order_id_tastytrade = (
                place_tastytrade_order(
                    ticker, tasty_qty, "Sell to Open", TASTY_ACCOUNT_ID, logger
                )
                if tasty_qty > 0
                else 0
            )
            trades[ticker] = {
                "action": "SHORT",
                "order_id_schwab": order_id_schwab,
                "order_id_tastytrade": order_id_tastytrade,
                "entry_time": datetime.now(pytz.utc).isoformat(),
                "entry_type": "manual",
            }

        # Save trade data
        with open(trade_file, "w") as file:
            json.dump(trades, file)

        return (
            jsonify(
                {
                    "success": True,
                    "message": f"Manual {action} trade executed for {ticker}",
                    "trade": trades[ticker],
                }
            ),
            200,
        )

    except Exception as e:
        import traceback

        traceback_str = traceback.format_exc()
        return (
            jsonify({"success": False, "error": str(e), "traceback": traceback_str}),
            500,
        )


@app.route("/api/manual-spx-trigger", methods=["POST"])
@token_required
def manual_spx_trigger():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        action = data.get("action")  # "buy_call", "buy_put", or "close_position"
        symbol = data.get("symbol")  # Should be "SPX"

        if not symbol or symbol != "SPX":
            return (
                jsonify({"success": False, "error": "Only SPX symbol is supported"}),
                400,
            )

        if not action or action not in ["buy_call", "buy_put", "close_position"]:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid action. Must be 'buy_call', 'buy_put', or 'close_position'",
                    }
                ),
                400,
            )

        # Configure logger
        from utils import configure_logger

        logger = configure_logger(symbol, "zeroday")

        # Get strategy parameters
        from utils import get_strategy_prarams

        params = get_strategy_prarams("zeroday", symbol, logger)

        if not params:
            return (
                jsonify({"success": False, "error": "Strategy parameters not found"}),
                404,
            )

        # Extract parameters
        timeframe = params[0]
        schwab_qty = params[1]
        trade_enabled = params[2]
        tasty_qty = params[3]
        call_enabled = params[8] if len(params) > 8 else True
        put_enabled = params[9] if len(params) > 9 else True

        # Convert quantities to integers
        schwab_qty = int(schwab_qty)
        tasty_qty = int(tasty_qty)

        # Check if trading is enabled
        if trade_enabled != "TRUE":
            return (
                jsonify({"success": False, "error": "Trading is not enabled for SPX"}),
                400,
            )

        # Create trades directory if it doesn't exist
        import os

        os.makedirs(f"trades/zeroday", exist_ok=True)
        trade_file = f"trades/zeroday/{symbol}.json"

        # Load existing trades
        import json

        try:
            with open(trade_file, "r") as file:
                trades = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            trades = {}

        # Get current position
        current_position = (
            trades.get(symbol, {}).get("action") if symbol in trades else None
        )

        # Import necessary functions
        from schwab import place_order
        from tastytrade import (
            place_tastytrade_order,
            get_option_chain,
            find_atm_option_symbol,
            place_tastytrade_option_order,
            get_spx_current_price,
            calculate_atm_strike,
        )
        from config import schwab_account_id, TASTY_ACCOUNT_ID
        import pytz
        from datetime import datetime

        # Optional overrides for strike/expiration
        strike_override = data.get("strike")
        expiration_override = data.get("expiration")

        # Execute trade based on action
        if action == "buy_call":
            # Check if call options are enabled
            if not call_enabled:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Call options are disabled for this ticker",
                        }
                    ),
                    400,
                )
            
            # Close any existing position first
            if current_position == "LONG_CALL":
                return (
                    jsonify(
                        {"success": False, "error": "Already in LONG CALL position"}
                    ),
                    400,
                )

            if current_position:
                logger.info(
                    f"Closing existing {current_position} position for {symbol}"
                )

                # Determine the appropriate closing action based on current position
                if current_position == "LONG_PUT":
                    schwab_action = "SELL"
                    tasty_action = "Sell to Close"
                elif current_position == "SHORT":
                    schwab_action = "BUY_TO_COVER"
                    tasty_action = "Buy to Close"
                else:  # Generic close for any other position type
                    schwab_action = "SELL"
                    tasty_action = "Sell to Close"

                # Execute closing orders
                close_order_id_schwab = (
                    place_order(
                        symbol,
                        schwab_qty,
                        schwab_action,
                        schwab_account_id,
                        logger,
                        "CLOSING",
                    )
                    if schwab_qty > 0
                    else 0
                )
                close_order_id_tastytrade = (
                    place_tastytrade_order(
                        symbol, tasty_qty, tasty_action, TASTY_ACCOUNT_ID, logger
                    )
                    if tasty_qty > 0
                    else 0
                )

            # Get current SPX price and calculate ATM strike if not provided
            if not strike_override:
                spx_price = get_spx_current_price(logger)
                if spx_price:
                    strike_override = calculate_atm_strike(spx_price, logger)
                    logger.info(f"Auto-calculated ATM strike: {strike_override}")

            # Resolve ATM 0DTE option symbol (SPXW) and open LONG CALL
            logger.info(f"Opening LONG CALL position for {symbol}")
            option_symbol = find_atm_option_symbol(
                symbol, 
                option_type="CALL", 
                strike=strike_override, 
                expiration=expiration_override,
                logger=logger
            )
            
            if not option_symbol:
                return jsonify({
                    "success": False, 
                    "error": f"Could not find suitable CALL option for {symbol}"
                }), 400

            order_id_schwab = 0  # Schwab option routing not implemented in this endpoint
            order_id_tastytrade = (
                place_tastytrade_option_order(
                    option_symbol, tasty_qty, "Buy to Open", TASTY_ACCOUNT_ID, logger
                ) if tasty_qty > 0 and option_symbol else 0
            )
            
            trades[symbol] = {
                "action": "LONG_CALL",
                "order_id_schwab": order_id_schwab,
                "order_id_tastytrade": order_id_tastytrade,
                "option_symbol": option_symbol,  # Store the actual option symbol
                "strike": strike_override,
                "expiration": expiration_override or "0DTE",
                "entry_time": datetime.now(pytz.utc).isoformat(),
                "entry_type": "manual",
            }

        elif action == "buy_put":
            # Check if put options are enabled
            if not put_enabled:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Put options are disabled for this ticker",
                        }
                    ),
                    400,
                )
            
            # Close any existing position first
            if current_position == "LONG_PUT":
                return (
                    jsonify(
                        {"success": False, "error": "Already in LONG PUT position"}
                    ),
                    400,
                )

            if current_position:
                logger.info(
                    f"Closing existing {current_position} position for {symbol}"
                )

                # Determine the appropriate closing action based on current position
                if current_position == "LONG_CALL":
                    schwab_action = "SELL"
                    tasty_action = "Sell to Close"
                elif current_position == "SHORT":
                    schwab_action = "BUY_TO_COVER"
                    tasty_action = "Buy to Close"
                else:  # Generic close for any other position type
                    schwab_action = "SELL"
                    tasty_action = "Sell to Close"

                # Execute closing orders
                close_order_id_schwab = (
                    place_order(
                        symbol,
                        schwab_qty,
                        schwab_action,
                        schwab_account_id,
                        logger,
                        "CLOSING",
                    )
                    if schwab_qty > 0
                    else 0
                )
                close_order_id_tastytrade = (
                    place_tastytrade_order(
                        symbol, tasty_qty, tasty_action, TASTY_ACCOUNT_ID, logger
                    )
                    if tasty_qty > 0
                    else 0
                )

            # Get current SPX price and calculate ATM strike if not provided
            if not strike_override:
                spx_price = get_spx_current_price(logger)
                if spx_price:
                    strike_override = calculate_atm_strike(spx_price, logger)
                    logger.info(f"Auto-calculated ATM strike: {strike_override}")

            # Resolve ATM 0DTE option symbol (SPXW) and open LONG PUT
            logger.info(f"Opening LONG PUT position for {symbol}")
            option_symbol = find_atm_option_symbol(
                symbol, 
                option_type="PUT", 
                strike=strike_override, 
                expiration=expiration_override,
                logger=logger
            )
            
            if not option_symbol:
                return jsonify({
                    "success": False, 
                    "error": f"Could not find suitable PUT option for {symbol}"
                }), 400

            order_id_schwab = 0
            order_id_tastytrade = (
                place_tastytrade_option_order(
                    option_symbol, tasty_qty, "Buy to Open", TASTY_ACCOUNT_ID, logger
                ) if tasty_qty > 0 and option_symbol else 0
            )
            
            trades[symbol] = {
                "action": "LONG_PUT",
                "order_id_schwab": order_id_schwab,
                "order_id_tastytrade": order_id_tastytrade,
                "option_symbol": option_symbol,  # Store the actual option symbol
                "strike": strike_override,
                "expiration": expiration_override or "0DTE",
                "entry_time": datetime.now(pytz.utc).isoformat(),
                "entry_type": "manual",
            }

        elif action == "close_position":
            if not current_position:
                return (
                    jsonify({"success": False, "error": "No open position to close"}),
                    400,
                )

            logger.info(f"Closing {current_position} position for {symbol}")

            # Determine the appropriate closing action based on current position
            if current_position in ["LONG_CALL", "LONG_PUT", "LONG"]:
                schwab_action = "SELL"
                tasty_action = "Sell to Close"
            else:  # SHORT or any other position
                schwab_action = "BUY_TO_COVER"
                tasty_action = "Buy to Close"

            # Get the actual option symbol if stored
            actual_symbol = trades.get(symbol, {}).get("option_symbol") if symbol in trades else None
            if not actual_symbol:
                # Fallback to generic symbol
                actual_symbol = symbol
                if current_position == "LONG_CALL":
                    actual_symbol = f"{symbol}_CALL"
                elif current_position == "LONG_PUT":
                    actual_symbol = f"{symbol}_PUT"

            # Execute closing orders
            close_order_id_schwab = (
                place_order(
                    actual_symbol,
                    schwab_qty,
                    schwab_action,
                    schwab_account_id,
                    logger,
                    "CLOSING",
                )
                if schwab_qty > 0
                else 0
            )
            close_order_id_tastytrade = (
                place_tastytrade_order(
                    actual_symbol, tasty_qty, tasty_action, TASTY_ACCOUNT_ID, logger
                )
                if tasty_qty > 0
                else 0
            )

            # Clear the position
            if symbol in trades:
                del trades[symbol]

        # Save trade data
        with open(trade_file, "w") as file:
            json.dump(trades, file)

        return (
            jsonify(
                {
                    "success": True,
                    "message": f"SPX {action.replace('_', ' ')} executed successfully",
                    "trade": trades.get(symbol, {}),
                }
            ),
            200,
        )

    except Exception as e:
        import traceback

        traceback_str = traceback.format_exc()
        return (
            jsonify({"success": False, "error": str(e), "traceback": traceback_str}),
            500,
        )


@app.route("/api/spx-price", methods=["GET"])
@token_required
def get_spx_price():
    """Get current SPX price for frontend display"""
    try:
        from tastytrade import get_spx_current_price
        from utils import configure_logger
        
        logger = configure_logger("SPX", "zeroday")
        price = get_spx_current_price(logger)
        
        if price:
            return jsonify({
                "success": True,
                "price": price,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Unable to fetch SPX price"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
