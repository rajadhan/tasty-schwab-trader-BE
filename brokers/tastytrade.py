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
TASTY_CLIENT_SECRET = "8283d5bbbb61c1e58b5e1b5b913ffc775e12d46a"
TASTY_CLIENT_ID = "f1ff7542-2fa9-446c-a57e-22f95108e02c"
# TASTY_CLIENT_SECRET = "9fb238518d48e77f966cf87d7474876b0e6f760a" # TEST
# TASTY_CLIENT_ID = "011d2533-2c97-402c-a360-6aba494cc8c9" # TEST
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
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        response = requests.post(refresh_url, data=payload, headers=headers)
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
        order_status = order_history["data"]["status"]
        traded_qty = int(order_history["data"]["size"])
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


def place_tastytrade_order(symbol, qty, action, account_id, logger):
    print("place tastytrade order", symbol, qty, action, account_id, logger)
    try:
        symbol = "/" + get_active_exchange_symbol(symbol)
        if symbol[-1].isdigit():
            symbol = f"/{symbol.lstrip('/')}"
        order_payload = create_order_payload(symbol, qty, action, account_id, logger)
        print("order payload", order_payload)
        order_id = place_order(order_payload, account_id, logger)
        is_filled, _ = check_order_status(order_id, account_id, logger)
        print("is filled", is_filled)
        if is_filled:
            return order_id
        else:
            logger.warning(f"Order not filled for {symbol}. Order ID: {order_id}")
            logger.warning(f"Placing order again for {symbol}. Order ID: {order_id}")
            order_id = place_order(order_payload, account_id, logger)
            return order_id
    except Exception as e:
        print(f"{account_id}=> ERROR! in placing order = {e}.")
        return ""


def get_atm_option_symbol(ticker, option_type, logger):
    """Get at-the-money 0-day option symbol for the given ticker"""
    try:
        # For SPX, use SPX as the underlying
        underlying = "SPX" if ticker.upper() in ["SPX", "SPXW"] else ticker
        
        # Get current price to calculate ATM strike
        if underlying == "SPX":
            current_price = get_spx_current_price(logger)
        else:
            # For other tickers, you might need to implement a different price fetch
            current_price = get_spx_current_price(logger)  # Placeholder
        
        if not current_price:
            logger.error(f"Could not get current price for {underlying}")
            return None
        
        # Calculate ATM strike
        atm_strike = calculate_atm_strike(current_price, logger)
        
        # Find the option symbol
        option_symbol = find_atm_option_symbol(
            underlying, 
            option_type, 
            strike=atm_strike, 
            same_day=True, 
            logger=logger
        )
        
        if option_symbol:
            logger.info(f"Found {option_type} option symbol for {ticker}: {option_symbol} at strike {atm_strike}")
        else:
            logger.warning(f"Could not find {option_type} option symbol for {ticker}")
        
        return option_symbol
        
    except Exception as e:
        logger.error(f"Error getting ATM option symbol for {ticker}: {e}")
        return None


def place_option_trade(ticker, option_type, action, qty, account_id, logger):
    """Place an option trade (PUT or CALL)"""
    try:
        # Get the option symbol
        option_symbol = get_atm_option_symbol(ticker, option_type, logger)
        
        if not option_symbol:
            logger.error(f"Could not get option symbol for {ticker} {option_type}")
            return ""
        
        # Place the option order
        order_id = place_tastytrade_option_order(
            option_symbol, 
            qty, 
            action, 
            account_id, 
            logger
        )
        
        if order_id:
            logger.info(f"Successfully placed {action} order for {option_symbol}: {order_id}")
        else:
            logger.error(f"Failed to place {action} order for {option_symbol}")
        
        return order_id
        
    except Exception as e:
        logger.error(f"Error placing option trade for {ticker}: {e}")
        return ""
        

def get_spxw_historical_data(start, end):
    try:
        print("start", start)
        print("end", end)
        url = f"{TASTY_API}/market-metrics/history/SPXW"
        end_time = datetime.now(pytz.utc)
        start_time = end_time - timedelta(days=7)  # Get last 7 days of data
        
        # Format dates for Tastytrade API
        start_date = start_time.strftime("%Y-%m-%d")
        end_date = end_time.strftime("%Y-%m-%d")
        params = {
            "start-date": start_date,
            "end-date": end_date,
            "interval": "1m"
        }
        response = tastytrade_api_request("GET", url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        print(f"ERROR! in getting SPXW historical data = {e}")
        return None



def get_account_info():
    url = f"{TASTY_API}/customers/me/accounts"
    response = tastytrade_api_request("GET", url)
    return response.json()




def get_spx_current_price(logger=None):
    """Get current SPX price for ATM strike calculation"""
    try:
        # Try to get SPX price from TastyTrade
        url = f"{TASTY_API}/quotes/SPX"
        response = tastytrade_api_request("GET", url)
        response.raise_for_status()
        data = response.json()
        
        if "data" in data and "last" in data["data"]:
            price = float(data["data"]["last"])
            if logger:
                logger.info(f"Current SPX price: {price}")
            return price
        else:
            if logger:
                logger.warning("No price data found in SPX quote response")
            return None
            
    except Exception as e:
        if logger:
            logger.error(f"Error getting SPX price: {e}")
        return None


class SPXTickDataFetcher:
    """Fetches SPX tick data from TastyTrade and integrates with Redis buffer system"""
    
    def __init__(self, redis_client=None, logger=None):
        self.redis_client = redis_client or REDIS_CLIENT
        self.logger = logger or logging.getLogger(__name__)
        self.is_running = False
        self.tick_thread = None
        
    def get_spx_tick_data(self):
        """Get current SPX tick data from TastyTrade"""
        try:
            url = f"{TASTY_API}/quotes/SPX"
            response = tastytrade_api_request("GET", url)
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data:
                return None
                
            tick_data = data["data"]
            current_time = datetime.now(pytz.utc)
            
            # Create tick data structure compatible with your buffer system
            tick = {
                'timestamp': current_time,
                'price': float(tick_data.get('last', 0)),
                'volume': int(tick_data.get('last-size', 0)),
                'symbol': 'SPXW',  # Use SPXW as the symbol for compatibility
                'bid': float(tick_data.get('bid', 0)),
                'ask': float(tick_data.get('ask', 0)),
                'bid_size': int(tick_data.get('bid-size', 0)),
                'ask_size': int(tick_data.get('ask-size', 0))
            }
            
            return tick
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error fetching SPX tick data: {e}")
            return None
    
    def store_tick_in_redis(self, tick_data):
        """Store tick data in Redis for buffer processing"""
        try:
            if not tick_data:
                return
                
            symbol = tick_data['symbol']
            timestamp = tick_data['timestamp']
            
            # Store raw tick
            tick_key = f"tick:{symbol}:{timestamp.isoformat()}"
            self.redis_client.setex(tick_key, 3600, json.dumps({
                **tick_data,
                'timestamp': timestamp.isoformat()
            }))
            
            # Add to tick list for bar conversion
            tick_list_key = f"ticks:{symbol}"
            self.redis_client.lpush(tick_list_key, json.dumps({
                **tick_data,
                'timestamp': timestamp.isoformat()
            }))
            self.redis_client.ltrim(tick_list_key, 0, 9999)  # Keep last 10k ticks
            
            # Publish to Redis pub/sub for real-time updates
            channel = f"spx_ticks:{symbol}"
            self.redis_client.publish(channel, json.dumps({
                **tick_data,
                'timestamp': timestamp.isoformat()
            }))
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error storing SPX tick in Redis: {e}")
    
    def start_spx_tick_streaming(self, interval=1):
        """Start SPX tick data streaming"""
        if self.is_running:
            if self.logger:
                self.logger.warning("SPX tick streaming already running")
            return
            
        self.is_running = True
        
        def stream_worker():
            if self.logger:
                self.logger.info(f"Starting SPX tick data streaming at {interval}s intervals")
            
            while self.is_running:
                try:
                    # Get current tick data
                    tick_data = self.get_spx_tick_data()
                    
                    if tick_data:
                        # Store in Redis
                        self.store_tick_in_redis(tick_data)
                        
                        if self.logger:
                            self.logger.debug(f"SPX tick: {tick_data['price']} @ {tick_data['timestamp']}")
                    
                    # Wait for next update
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    if self.logger:
                        self.logger.info("SPX tick data stream stopped by user")
                    break
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in SPX tick data stream: {e}")
                    time.sleep(interval)  # Wait before retrying
        
        # Start streaming in a separate thread
        self.tick_thread = threading.Thread(target=stream_worker, daemon=True)
        self.tick_thread.start()
        
        if self.logger:
            self.logger.info("SPX tick data streaming started")
    
    def stop_spx_tick_streaming(self):
        """Stop SPX tick data streaming"""
        if self.logger:
            self.logger.info("Stopping SPX tick data streaming")
        self.is_running = False
        
        if self.tick_thread and self.tick_thread.is_alive():
            self.tick_thread.join(timeout=5)
    
    def get_latest_spx_bars(self, count=100):
        """Get latest SPX bars from Redis"""
        try:
            tick_list_key = "ticks:SPXW"
            ticks_data = self.redis_client.lrange(tick_list_key, 0, count-1)
            
            if not ticks_data:
                return []
            
            ticks = []
            for tick_json in ticks_data:
                try:
                    tick = json.loads(tick_json)
                    tick["timestamp"] = datetime.fromisoformat(tick["timestamp"].replace("Z", "+00:00"))
                    ticks.append(tick)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error parsing SPX tick: {e}")
                    continue
            
            # Sort by timestamp
            ticks.sort(key=lambda x: x["timestamp"])
            return ticks
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting latest SPX bars: {e}")
            return []


def calculate_atm_strike(price: float, logger=None):
    """Calculate at-the-money strike price for SPX options"""
    # SPX options typically have 5-point strike intervals
    strike = round(price / 5) * 5
    
    if logger:
        logger.info(f"Calculated ATM strike: {strike} from price {price}")
    
    return strike


def find_atm_option_symbol(symbol: str, option_type: str, strike: float = None, expiration: str = None, same_day: bool = True, logger=None):
    """Find at-the-money option symbol for SPX 0-day options"""
    try:
        if logger:
            logger.info(f"Searching for {option_type} option for {symbol}, strike: {strike}, expiration: {expiration}")
        
        # Get option chain
        option_chain = get_option_chain(symbol, logger)
        if not option_chain or "data" not in option_chain or "items" not in option_chain["data"]:
            if logger:
                logger.error(f"No option chain data found for {symbol}")
            return None
        
        items = option_chain["data"]["items"]
        
        # Filter by option type
        candidates = [i for i in items if i.get("option-type", "").upper() == option_type.upper()]
        if not candidates:
            if logger:
                logger.warning(f"No {option_type} options found for {symbol}")
            return None
        
        # Filter by expiration
        if expiration:
            candidates = [i for i in candidates if i.get("expiration-date") == expiration]
        elif same_day:
            # Default to same-day expiration (0 DTE)
            candidates = [i for i in candidates if i.get("days-to-expiration") == 0]
        
        if not candidates:
            if logger:
                logger.warning(f"No options found with specified expiration criteria for {symbol}")
            return None
        
        # If explicit strike provided, find exact match
        if strike is not None:
            try:
                strike_float = float(strike)
                for item in candidates:
                    if abs(float(item.get("strike-price", 0)) - strike_float) < 0.01:
                        option_symbol = item.get("symbol")
                        if logger:
                            logger.info(f"Found exact strike match for {symbol} {option_type} at {strike}: {option_symbol}")
                        return option_symbol
            except (ValueError, TypeError) as e:
                if logger:
                    logger.error(f"Error processing strike price {strike}: {e}")
                return None
        
        # Find ATM option (closest to current price)
        if same_day:
            # For 0-day options, find the middle strike
            strikes = sorted({float(i.get("strike-price", 0)) for i in candidates if i.get("strike-price")})
            if not strikes:
                if logger:
                    logger.warning(f"No valid strikes found for {symbol} {option_type}")
                return None
            
            # Find middle strike (closest to ATM)
            mid_strike = strikes[len(strikes)//2]
            
            for item in candidates:
                if abs(float(item.get("strike-price", 0)) - mid_strike) < 0.01:
                    option_symbol = item.get("symbol")
                    if logger:
                        logger.info(f"Found ATM option for {symbol} {option_type} at strike {mid_strike}: {option_symbol}")
                    return option_symbol
        
        if logger:
            logger.warning(f"No suitable option found for {symbol} {option_type}")
        return None
        
    except Exception as e:
        if logger:
            logger.error(f"Error finding ATM option symbol for {symbol} {option_type}: {e}")
        return None


def place_tastytrade_option_order(option_symbol: str, qty: int, action: str, account_id: str, logger=None):
    """Place a market order for a single-option leg (SPXW)"""
    if not option_symbol:
        if logger:
            logger.error("No option symbol provided for order")
        return ""
    
    if logger:
        logger.info(f"Placing {action} order for {option_symbol}, quantity: {qty}")
    
    order_payload = {
        "order-type": "Market",
        "time-in-force": "Day",
        "legs": [
            {
                "instrument-type": "Equity Option",
                "symbol": option_symbol,
                "action": action,
                "quantity": str(qty),
            }
        ],
    }
    
    try:
        result = place_order(order_payload, account_id, logger)
        if logger:
            logger.info(f"Option order placed successfully: {result}")
        return result
    except Exception as e:
        if logger:
            logger.error(f"Error placing option order: {e}")
        return ""


def manual_trigger_action(ticker, action, logger):
    trade_file = get_trade_file_path(ticker, "zeroday")
    trades = load_json(trade_file)
    [tasty_qty] = get_strategy_prarams("zeroday", ticker, logger)[2:3]

    if ticker not in trades:
        if action == "long":
            order_id_tastytrade = (
                place_option_trade(
                    ticker, "CALL", "Buy to Open", tasty_qty, TASTY_ACCOUNT_ID, logger
                )
                if tasty_qty > 0
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
                if tasty_qty > 0
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
                    if tasty_qty > 0
                    else 0
                )
            order_id_tastytrade = (
                place_option_trade(
                    ticker, "CALL", "Buy to Open", tasty_qty,
                    TASTY_ACCOUNT_ID, logger
                )
                if tasty_qty > 0
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
                    if tasty_qty > 0
                    else 0
                )
            order_id_tastytrade = (
                place_option_trade(
                    ticker, "PUT", "Buy to Open", tasty_qty, TASTY_ACCOUNT_ID, logger
                )
                if tasty_qty > 0
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


if __name__ == "__main__":
    print("tastu_data", get_account_info())