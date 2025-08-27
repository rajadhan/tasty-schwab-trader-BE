import requests
from time import sleep
from config import *
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os
import json
import logging
import threading
import redis

from utils import save_json, get_active_exchange_symbol

# Configure logging
logger = logging.getLogger(__name__)

# Redis client
REDIS_CLIENT = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0))
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
    with open(TASTY_ACCESS_TOKEN_PATH, "r") as f:
        tokens = json.load(f)
    tasty_refresh_access_token_url = f"{TASTY_API}/oauth/token"
    refresh_token = tokens.get("refresh_token")
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": TASTY_CLIENT_ID,
        "client_secret": TASTY_CLIENT_SECRET,
        "redirect_uri": TASTY_REDIRECT_URI,
    }
    response = requests.post(tasty_refresh_access_token_url, data=payload)
    new_tokens = response.json()
    if "access_token" in new_tokens:
        tokens = {
            "access_token": new_tokens.get('access_token'),
            "refresh_token": refresh_token
        }
        save_json(TASTY_ACCESS_TOKEN_PATH, tokens)
        return new_tokens["access_token"]
    else:
        raise Exception(f"Failed to refresh token: {new_tokens}")


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


# ====== Simple helpers for SPX options routing ======

def get_option_chain(symbol: str, logger=None):
    """Get option chain for a symbol (e.g., SPX)"""
    try:
        url = f"{TASTY_API}/option-chains/{symbol}"
        response = tastytrade_api_request("GET", url)
        response.raise_for_status()
        data = response.json()
        if logger:
            logger.info(f"Retrieved option chain for {symbol}: {len(data.get('data', {}).get('items', []))} options")
        return data
    except Exception as e:
        if logger:
            logger.error(f"Error getting option chain for {symbol}: {e}")
        return None

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


def calculate_atm_strike(price: float, logger=None):
    """Calculate at-the-money strike price for SPX options"""
    # SPX options typically have 5-point strike intervals
    strike = round(price / 5) * 5
    
    if logger:
        logger.info(f"Calculated ATM strike: {strike} from price {price}")
    
    return strike


def get_instruments():
    try:
        url = f"{TASTY_API}/instruments/futures"
        response = tastytrade_api_request("GET", url)
        instruments = response.json()["data"]["items"]
        df = pd.DataFrame(instruments)
        df.to_csv("tastytrade_instruments.csv")
    except Exception as e:
        print(f"ERROR! in getting instruments = {e}")


def get_account_info():
    url = f"{TASTY_API}/customers/me/accounts"
    response = tastytrade_api_request("GET", url)
    return response.json()



def get_spx_option_tick_data(option_symbol: str, logger=None):
    """Get real-time tick data for SPX options from TastyTrade feed"""
    try:
        if logger:
            logger.info(f"Fetching tick data for SPX option: {option_symbol}")
        
        # Get real-time quote for the specific option
        url = f"{TASTY_API}/quotes/{option_symbol}"
        response = tastytrade_api_request("GET", url)
        response.raise_for_status()
        data = response.json()
        
        if "data" not in data:
            if logger:
                logger.warning(f"No data found in response for {option_symbol}")
            return None
        
        tick_data = data["data"]
        
        # Extract relevant tick information
        tick_info = {
            "symbol": option_symbol,
            "timestamp": datetime.now().isoformat(),
            "last_price": tick_data.get("last"),
            "bid": tick_data.get("bid"),
            "ask": tick_data.get("ask"),
            "bid_size": tick_data.get("bid-size"),
            "ask_size": tick_data.get("ask-size"),
            "volume": tick_data.get("volume"),
            "open_interest": tick_data.get("open-interest"),
            "implied_volatility": tick_data.get("implied-volatility"),
            "delta": tick_data.get("delta"),
            "gamma": tick_data.get("gamma"),
            "theta": tick_data.get("theta"),
            "vega": tick_data.get("vega"),
            "underlying_price": tick_data.get("underlying-price"),
            "strike_price": tick_data.get("strike-price"),
            "expiration_date": tick_data.get("expiration-date"),
            "days_to_expiration": tick_data.get("days-to-expiration"),
            "option_type": tick_data.get("option-type")
        }
        
        if logger:
            logger.info(f"Retrieved tick data for {option_symbol}: Last={tick_info['last_price']}, Bid={tick_info['bid']}, Ask={tick_info['ask']}")
        
        return tick_info
        
    except Exception as e:
        if logger:
            logger.error(f"Error getting SPX option tick data for {option_symbol}: {e}")
        return None

def get_spx_option_chain_tick_data(symbol: str = "SPX", logger=None):
    """Get real-time tick data for entire SPX option chain"""
    try:
        if logger:
            logger.info(f"Fetching tick data for SPX option chain")
        
        # Get the option chain first
        option_chain = get_option_chain(symbol, logger)
        if not option_chain or "data" not in option_chain or "items" not in option_chain["data"]:
            if logger:
                logger.error("No option chain data available")
            return None
        
        items = option_chain["data"]["items"]
        tick_data_list = []
        
        # Get tick data for each option (limit to avoid overwhelming the API)
        max_options = 50  # Limit to prevent API rate limiting
        for i, item in enumerate(items[:max_options]):
            option_symbol = item.get("symbol")
            if option_symbol:
                tick_data = get_spx_option_tick_data(option_symbol, logger)
                if tick_data:
                    tick_data_list.append(tick_data)
                
                # Small delay to avoid rate limiting
                if i < max_options - 1:
                    import time
                    time.sleep(0.1)
        
        if logger:
            logger.info(f"Retrieved tick data for {len(tick_data_list)} SPX options")
        
        return tick_data_list
        
    except Exception as e:
        if logger:
            logger.error(f"Error getting SPX option chain tick data: {e}")
        return None

def get_spx_atm_options_tick_data(symbol: str = "SPX", logger=None):
    """Get real-time tick data for at-the-money SPX options only"""
    try:
        if logger:
            logger.info(f"Fetching tick data for ATM SPX options")
        
        # Get current SPX price
        spx_price = get_spx_current_price(logger)
        if not spx_price:
            if logger:
                logger.error("Could not get current SPX price")
            return None
        
        # Calculate ATM strike
        atm_strike = calculate_atm_strike(spx_price, logger)
        
        # Get option chain
        option_chain = get_option_chain(symbol, logger)
        if not option_chain or "data" not in option_chain or "items" not in option_chain["data"]:
            if logger:
                logger.error("No option chain data available")
            return None
        
        items = option_chain["data"]["items"]
        atm_options = []
        
        # Find options closest to ATM strike
        for item in items:
            try:
                strike = float(item.get("strike-price", 0))
                # Consider options within 2 strikes of ATM
                if abs(strike - atm_strike) <= 10:
                    option_symbol = item.get("symbol")
                    if option_symbol:
                        tick_data = get_spx_option_tick_data(option_symbol, logger)
                        if tick_data:
                            atm_options.append(tick_data)
            except (ValueError, TypeError):
                continue
        
        if logger:
            logger.info(f"Retrieved tick data for {len(atm_options)} ATM SPX options around strike {atm_strike}")
        
        return atm_options
        
    except Exception as e:
        if logger:
            logger.error(f"Error getting ATM SPX options tick data: {e}")
        return None

def stream_spx_option_tick_data(option_symbol: str, callback_func, interval: int = 5, logger=None):
    """
    Stream real-time SPX option tick data at specified intervals
    
    Args:
        option_symbol: The option symbol to stream (e.g., "SPXW240315C5000")
        callback_func: Function to call with tick data
        interval: Update interval in seconds
        logger: Logger instance
    """
    import time
    import threading
    
    def stream_worker():
        if logger:
            logger.info(f"Starting tick data stream for {option_symbol} at {interval}s intervals")
        
        while True:
            try:
                # Get current tick data
                tick_data = get_spx_option_tick_data(option_symbol, logger)
                
                if tick_data:
                    # Call the callback function with tick data
                    callback_func(tick_data)
                
                # Wait for next update
                time.sleep(interval)
                
            except KeyboardInterrupt:
                if logger:
                    logger.info(f"Tick data stream stopped for {option_symbol}")
                break
            except Exception as e:
                if logger:
                    logger.error(f"Error in tick data stream for {option_symbol}: {e}")
                time.sleep(interval)  # Wait before retrying
    
    # Start streaming in a separate thread
    stream_thread = threading.Thread(target=stream_worker, daemon=True)
    stream_thread.start()
    
    if logger:
        logger.info(f"Tick data stream started for {option_symbol}")
    
    return stream_thread

def get_spx_option_historical_data(option_symbol: str, start_date: str, end_date: str, logger=None):
    """Get historical data for SPX options from TastyTrade"""
    try:
        if logger:
            logger.info(f"Fetching historical data for {option_symbol} from {start_date} to {end_date}")
        
        # Format dates for TastyTrade API
        url = f"{TASTY_API}/market-metrics/history/{option_symbol}"
        params = {
            "start-date": start_date,
            "end-date": end_date,
            "interval": "1m"  # 1-minute intervals
        }
        
        response = tastytrade_api_request("GET", url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "data" not in data or "items" not in data["data"]:
            if logger:
                logger.warning(f"No historical data found for {option_symbol}")
            return None
        
        historical_data = data["data"]["items"]
        
        if logger:
            logger.info(f"Retrieved {len(historical_data)} historical data points for {option_symbol}")
        
        return historical_data
        
    except Exception as e:
        if logger:
            logger.error(f"Error getting historical data for {option_symbol}: {e}")
        return None

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

def test_spx_option_tick_data():
    """Test function to demonstrate SPX option tick data functionality"""
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("SPX_Test")
    
    print("=== Testing SPX Option Tick Data Functions ===\n")
    
    try:
        # Test 1: Get current SPX price
        print("1. Getting current SPX price...")
        spx_price = get_spx_current_price(logger)
        if spx_price:
            print(f"   Current SPX price: {spx_price}")
        else:
            print("   Failed to get SPX price")
            return
        
        # Test 2: Calculate ATM strike
        print("\n2. Calculating ATM strike...")
        atm_strike = calculate_atm_strike(spx_price, logger)
        print(f"   ATM strike: {atm_strike}")
        
        # Test 3: Find ATM option symbols
        print("\n3. Finding ATM option symbols...")
        call_symbol = find_atm_option_symbol("SPX", "CALL", strike=atm_strike, same_day=True, logger=logger)
        put_symbol = find_atm_option_symbol("SPX", "PUT", strike=atm_strike, same_day=True, logger=logger)
        
        print(f"   ATM CALL symbol: {call_symbol}")
        print(f"   ATM PUT symbol: {put_symbol}")
        
        # Test 4: Get tick data for ATM call option
        if call_symbol:
            print(f"\n4. Getting tick data for {call_symbol}...")
            call_tick_data = get_spx_option_tick_data(call_symbol, logger)
            if call_tick_data:
                print(f"   Last price: {call_tick_data.get('last_price')}")
                print(f"   Bid: {call_tick_data.get('bid')}")
                print(f"   Ask: {call_tick_data.get('ask')}")
                print(f"   Volume: {call_tick_data.get('volume')}")
                print(f"   Implied Vol: {call_tick_data.get('implied_volatility')}")
            else:
                print("   Failed to get tick data")
        
        # Test 5: Get ATM options tick data
        print("\n5. Getting ATM options tick data...")
        atm_options_data = get_spx_atm_options_tick_data("SPX", logger)
        if atm_options_data:
            print(f"   Retrieved {len(atm_options_data)} ATM options")
            for i, option in enumerate(atm_options_data[:3]):  # Show first 3
                print(f"   Option {i+1}: {option.get('symbol')} - Last: {option.get('last_price')}")
        else:
            print("   Failed to get ATM options data")
        
        print("\n=== Test completed ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        logger.error(f"Test failed: {e}")

# ====== SPXW Tick Data Streaming for Zeroday Strategy ======

def stream_spxw_tick_data(symbol: str = "SPXW", callback_func=None, interval: int = 1, logger=None):
    """
    Stream real-time SPXW tick data at specified intervals
    
    Args:
        symbol: The symbol to stream (default: "SPXW")
        callback_func: Function to call with tick data
        interval: Interval in seconds between data updates
        logger: Logger instance
    """
    import threading
    import time
    
    def stream_worker():
        try:
            if logger:
                logger.info(f"Starting SPXW tick data stream for {symbol} at {interval}s intervals")
            
            while True:
                try:
                    # Get current SPXW price from Tastytrade
                    url = f"{TASTY_API}/quotes/{symbol}"
                    response = tastytrade_api_request("GET", url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "data" in data and "items" in data["data"] and data["data"]["items"]:
                            quote = data["data"]["items"][0]
                            
                            tick_data = {
                                "symbol": symbol,
                                "timestamp": datetime.now(pytz.utc).isoformat(),
                                "price": float(quote.get("last-price", 0)),
                                "bid": float(quote.get("bid-price", 0)),
                                "ask": float(quote.get("ask-price", 0)),
                                "volume": int(quote.get("last-size", 0)),
                                "bid_size": int(quote.get("bid-size", 0)),
                                "ask_size": int(quote.get("ask-size", 0))
                            }
                            
                            # Store tick in Redis
                            store_tick_in_redis(tick_data, logger)
                            
                            # Call callback if provided
                            if callback_func:
                                callback_func(tick_data)
                            
                            if logger:
                                logger.debug(f"SPXW tick: {tick_data['price']} @ {tick_data['timestamp']}")
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    if logger:
                        logger.error(f"Error in SPXW tick stream: {e}")
                    time.sleep(interval)
                    
        except Exception as e:
            if logger:
                logger.error(f"Error in SPXW tick data stream for {symbol}: {e}")
    
    # Start streaming in a separate thread
    stream_thread = threading.Thread(target=stream_worker, daemon=True)
    stream_thread.start()
    
    if logger:
        logger.info(f"SPXW tick data stream started for {symbol}")
    
    return stream_thread


def store_tick_in_redis(tick_data, logger=None):
    """Store tick data in Redis for bar conversion"""
    try:
        symbol = tick_data["symbol"]
        timestamp = tick_data["timestamp"]
        price = tick_data["price"]
        volume = tick_data["volume"]
        
        # Store raw tick
        tick_key = f"tick:{symbol}:{timestamp}"
        REDIS_CLIENT.setex(tick_key, 3600, json.dumps(tick_data))  # Expire in 1 hour
        
        # Add to tick list for bar conversion
        tick_list_key = f"ticks:{symbol}"
        REDIS_CLIENT.lpush(tick_list_key, json.dumps(tick_data))
        REDIS_CLIENT.ltrim(tick_list_key, 0, 9999)  # Keep last 10k ticks
        
        if logger:
            logger.debug(f"Stored tick in Redis: {tick_key}")
            
    except Exception as e:
        if logger:
            logger.error(f"Error storing tick in Redis: {e}")


def convert_ticks_to_bars(symbol: str, timeframe: str, logger=None):
    """
    Convert accumulated ticks to bars based on timeframe
    
    Args:
        symbol: Symbol to convert (e.g., "SPXW")
        timeframe: Timeframe for bars (e.g., "1Min", "5Min", "1Hour")
        logger: Logger instance
    """
    try:
        tick_list_key = f"ticks:{symbol}"
        ticks_data = REDIS_CLIENT.lrange(tick_list_key, 0, -1)
        
        if not ticks_data:
            if logger:
                logger.warning(f"No ticks found for {symbol}")
            return None
        
        # Parse ticks
        ticks = []
        for tick_json in ticks_data:
            try:
                tick = json.loads(tick_json)
                tick["timestamp"] = datetime.fromisoformat(tick["timestamp"].replace("Z", "+00:00"))
                ticks.append(tick)
            except Exception as e:
                if logger:
                    logger.error(f"Error parsing tick: {e}")
                continue
        
        if not ticks:
            return None
        
        # Sort by timestamp
        ticks.sort(key=lambda x: x["timestamp"])
        
        # Convert timeframe to seconds
        timeframe_seconds = parse_timeframe_to_seconds(timeframe)
        
        # Group ticks into bars
        bars = []
        current_bar_start = None
        current_bar_ticks = []
        
        for tick in ticks:
            tick_time = tick["timestamp"]
            
            if current_bar_start is None:
                current_bar_start = tick_time
                current_bar_ticks = [tick]
            elif (tick_time - current_bar_start).total_seconds() >= timeframe_seconds:
                # Create bar from accumulated ticks
                bar = create_bar_from_ticks(current_bar_ticks, current_bar_start, timeframe)
                bars.append(bar)
                
                # Start new bar
                current_bar_start = tick_time
                current_bar_ticks = [tick]
            else:
                current_bar_ticks.append(tick)
        
        # Create final bar if there are remaining ticks
        if current_bar_ticks:
            bar = create_bar_from_ticks(current_bar_ticks, current_bar_start, timeframe)
            bars.append(bar)
        
        # Store bars in Redis
        if bars:
            store_bars_in_redis(symbol, timeframe, bars, logger)
            
        if logger:
            logger.info(f"Converted {len(ticks)} ticks to {len(bars)} bars for {symbol} {timeframe}")
        
        return bars
        
    except Exception as e:
        if logger:
            logger.error(f"Error converting ticks to bars: {e}")
        return None


def parse_timeframe_to_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds"""
    timeframe = timeframe.lower()
    
    if "min" in timeframe:
        minutes = int(timeframe.replace("min", ""))
        return minutes * 60
    elif "hour" in timeframe:
        hours = int(timeframe.replace("hour", ""))
        return hours * 3600
    elif "day" in timeframe:
        days = int(timeframe.replace("day", ""))
        return days * 86400
    else:
        return 60  # Default to 1 minute


def create_bar_from_ticks(ticks, bar_start_time, timeframe):
    """Create a bar from accumulated ticks"""
    if not ticks:
        return None
    
    prices = [tick["price"] for tick in ticks]
    volumes = [tick["volume"] for tick in ticks]
    
    bar = {
        "timestamp": bar_start_time.isoformat(),
        "open": prices[0],
        "high": max(prices),
        "low": min(prices),
        "close": prices[-1],
        "volume": sum(volumes),
        "tick_count": len(ticks),
        "timeframe": timeframe
    }
    
    return bar


def store_bars_in_redis(symbol: str, timeframe: str, bars, logger=None):
    """Store bars in Redis"""
    try:
        bars_key = f"bars:{symbol}:{timeframe}"
        
        # Store latest bars
        for bar in bars:
            bar_key = f"{bars_key}:{bar['timestamp']}"
            REDIS_CLIENT.setex(bar_key, 86400, json.dumps(bar))  # Expire in 24 hours
        
        # Store bar list
        bar_list_key = f"bar_list:{symbol}:{timeframe}"
        bar_timestamps = [bar["timestamp"] for bar in bars]
        REDIS_CLIENT.delete(bar_list_key)
        if bar_timestamps:
            REDIS_CLIENT.rpush(bar_list_key, *bar_timestamps)
            REDIS_CLIENT.expire(bar_list_key, 86400)
        
        if logger:
            logger.debug(f"Stored {len(bars)} bars in Redis for {symbol} {timeframe}")
            
    except Exception as e:
        if logger:
            logger.error(f"Error storing bars in Redis: {e}")


def get_latest_bars_from_redis(symbol: str, timeframe: str, count: int = 100, logger=None):
    """Get latest bars from Redis"""
    try:
        bar_list_key = f"bar_list:{symbol}:{timeframe}"
        bar_timestamps = REDIS_CLIENT.lrange(bar_list_key, -count, -1)
        
        bars = []
        for timestamp in bar_timestamps:
            bar_key = f"bars:{symbol}:{timeframe}:{timestamp.decode()}"
            bar_data = REDIS_CLIENT.get(bar_key)
            if bar_data:
                bars.append(json.loads(bar_data))
        
        # Sort by timestamp
        bars.sort(key=lambda x: x["timestamp"])
        
        if logger:
            logger.debug(f"Retrieved {len(bars)} bars from Redis for {symbol} {timeframe}")
        
        return bars
        
    except Exception as e:
        if logger:
            logger.error(f"Error getting bars from Redis: {e}")
        return []


def start_spxw_tick_streaming(symbol: str = "SPXW", interval: int = 1, logger=None):
    """Start SPXW tick streaming and bar conversion"""
    try:
        # Start tick streaming
        stream_thread = stream_spxw_tick_data(symbol, None, interval, logger)
        
        # Start bar conversion thread
        def bar_conversion_worker():
            while True:
                try:
                    # Convert ticks to bars for different timeframes
                    timeframes = ["1Min", "5Min", "15Min", "1Hour"]
                    for tf in timeframes:
                        convert_ticks_to_bars(symbol, tf, logger)
                    
                    sleep(60)  # Convert every minute
                    
                except Exception as e:
                    if logger:
                        logger.error(f"Error in bar conversion worker: {e}")
                    sleep(60)
        
        bar_thread = threading.Thread(target=bar_conversion_worker, daemon=True)
        bar_thread.start()
        
        if logger:
            logger.info(f"Started SPXW tick streaming and bar conversion for {symbol}")
        
        return stream_thread, bar_thread
        
    except Exception as e:
        if logger:
            logger.error(f"Error starting SPXW tick streaming: {e}")
        return None, None

if __name__ == "__main__":
    print("tastu_data", get_account_info())