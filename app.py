from config import *
from datetime import timedelta, datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
from main_equities import run_every_week
from brokers.schwab import (
    refresh_access_token as schwab_refresh_access_token,
    get_refresh_token as schwab_get_refresh_token,
    authorize_url as schwab_authorization_url,
)
from brokers.tastytrade import (
    manual_trigger_action,
    manual_ema_trigger_action,
    refresh_tastytrade_token,
    authorize_url as tastytrade_authorize_url,
    access_token as tastytrade_access_token,
    update_tastytrade_instruments_csv,
)
from utils import ticker_data_path_for_strategy, configure_logger
import json
import jwt
import os
import requests
import redis
import threading

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
REDIS_CLIENT = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
)


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


def _start_tastytrade_update_instruments_background():
    """Start updating Tastytrade instruments in a background thread with app context."""
    def _task():
        try:
            with app.app_context():
                update_tastytrade_instruments_csv()
        except Exception as e:
            # Avoid failing silently if background thread errors
            print(f"Error in background Tastytrade instruments update: {e}")

    threading.Thread(target=_task, daemon=True).start()


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

            # Update Tastytrade instruments CSV in background after successful login
            try:
                _start_tastytrade_update_instruments_background()
            except Exception as e:
                print(f"Error starting background CSV update on login: {e}")

            return (
                jsonify(
                    {
                        "success": True,
                        "token": token,
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
    return tastytrade_authorize_url()


@app.route("/api/tasty/access-token", methods=["POST"])
@token_required
def tasty_access_token():
    data = request.json
    authorization_code = data.get("authorizationCode")
    return tastytrade_access_token(authorization_code)


@app.route("/api/tasty/refresh-token", methods=["POST"])
@token_required
def tasty_refresh_token():
    return refresh_tastytrade_token()


@app.route("/api/schwab/authorize-url", methods=["GET"])
@token_required
def schwab_authorize_url():
    request_url = schwab_authorization_url()
    return request_url


@app.route("/api/schwab/access-token", methods=["POST"])
@token_required
def schwab_access_token():
    data = request.json
    authorization_link = data.get("authorizationLink")
    response = schwab_get_refresh_token(authorization_link)
    if response:
        return jsonify(
            {
                "success": True,
                "message": "Charles Schwab connection established successfully",
            }
        )
    else:
        return jsonify(
            {
                "success": False,
                "message": "Failed to establish Charles Schwab connection",
            }
        )


@app.route("/api/schwab/refresh-token", methods=["POST"])
@token_required
def refresh_schwab_token():
    access_token = schwab_refresh_access_token()
    if access_token:
        return jsonify(
            {
                "success": True,
                "message": "Charles Schwab token refreshed successfully",
                "access_token": access_token,
            }
        )
    else:
        return jsonify(
            {
                "success": False,
                "message": "Failed to refresh Charles Schwab token",
            }
        )


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
            return (
                jsonify({"success": False, "error": "Current password is required"}),
                400,
            )

        # Verify current password
        if current_password != admin_credentials.get("password"):
            return (
                jsonify({"success": False, "error": "Current password is incorrect"}),
                401,
            )

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

        return (
            jsonify(
                {
                    "success": True,
                    "message": "Credentials updated successfully",
                    "token": token,
                    "email": fresh_email,
                }
            ),
            200,
        )
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
        # results = {}
        if trade_enabled_symbols:
            # for symbol in trade_enabled_symbols:
            #     trade_file = os.path.join(f"trades/{strategy}", f"{symbol}.json")
            #     trades = None
            #     if os.path.exists(trade_file):
            #         try:
            #             with open(trade_file, "r") as f:
            #                 trades = json.load(f)
            #         except Exception as e:
            #             trades = {"error": str(e)}
            #     # Optionally, add OHLCV/signals/chart data here if available
            #     # For now, just return trades
            #     results[symbol] = {"trades": trades}
            # Clear stop flag for this strategy
            try:
                REDIS_CLIENT.delete(f"trading:stop:{strategy}")
            except Exception:
                pass
            print("Loading trading parameters ... ")
            run_every_week(strategy)
            print("Trading started!")

            return (
                jsonify(
                    {
                        "success": True,
                        "message": "Trading has started",
                        # "enabled_symbols": trade_enabled_symbols,
                        # "results": results,
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
        return (
            jsonify({"success": True, "message": f"Stop signal sent for {strategy}"}),
            200,
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/manual-trigger", methods=["POST"])
@token_required
def manual_trigger():
    data = request.json
    ticker = data.get("ticker", "SPX")
    action = data["action"]

    logger = configure_logger(ticker, "zeroday")
    logger.info(f"Manual trigger received for {ticker} with action {action}")
    manual_trigger_action(ticker, action, logger)
    return {"status": "success", "action": action, "ticker": ticker}


@app.route('/api/ema-manual-trigger', methods = ["POST"])
@token_required
def ema_manual_trigger():
    
    data = request.json
    ticker = data.get("ticker")
    action = data["action"]

    logger = configure_logger(ticker, "ema")
    logger.info(f"Manual trigger received for {ticker} with action {action}")
    manual_ema_trigger_action(ticker, action, logger)
    return {"status": "success", "action": action, "ticker": ticker}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
