import json
import logging
import os
import pandas as pd
import pandas_market_calendars as mcal
import pytz
import time
from time import sleep
from config import *
from pytz import timezone
from datetime import datetime, timedelta


# Function to check if the market is closed due to a holiday
def is_holiday(date):
    nyse = mcal.get_calendar("NYSE")
    holidays = nyse.holidays().holidays
    holidays = pd.DatetimeIndex(holidays)
    final_holidays = [holiday.date() for holiday in holidays.to_pydatetime()]
    # if date.date() in final_holidays:
    if date in final_holidays:
        return True
    return False


def is_within_time_range():
    current_datetime = datetime.now(tz=timezone(TIME_ZONE))
    # Find most recent Sunday 6:00 PM
    days_since_sunday = (current_datetime.weekday() + 1) % 7
    last_sunday = current_datetime - timedelta(days=days_since_sunday)
    start_time = last_sunday.replace(hour=18, minute=0, second=0, microsecond=0)
    if current_datetime < start_time:
        # If before this week's Sunday 6pm, go to previous Sunday
        start_time -= timedelta(days=7)
    end_time = start_time + timedelta(days=4, hours=23)  # Friday 5:00 PM

    return start_time <= current_datetime <= end_time


def get_strategy_prarams(strategy, ticker, logger):
    try:
        TICKER_DATA_PATH = ticker_data_path_for_strategy(strategy)
        with open(TICKER_DATA_PATH, "r") as file:
            strategy_params = json.load(file)
        return strategy_params[ticker]
    except Exception as e:
        logger.error(f"Error in getting strategy params: {str(e)}")
        sleep(10)
        strategy_params = get_strategy_prarams(strategy, ticker, logger)
        return strategy_params


def ticker_data_path_for_strategy(strategy):
    if strategy == "ema":
        TICKER_DATA_PATH = EMA_TICKER_DATA_PATH
    elif strategy == "supertrend":
        TICKER_DATA_PATH = SUPER_TICKER_DATA_PATH
    elif strategy == "zeroday":
        TICKER_DATA_PATH = ZERODAY_TICKER_DATA_PATH
    return TICKER_DATA_PATH


# Function to get the market hours for a specific date
def get_market_hours(date):
    if is_weekend(date):
        return None, "Market closed on weekends"

    if is_holiday(date):
        return None, "Market closed due to holiday"

    nyse = mcal.get_calendar("NYSE")
    market_date_time = nyse.schedule(date, date, tz=TIME_ZONE)
    start_time = market_date_time.iloc[0]["market_open"].time()
    end_time = (market_date_time.iloc[0]["market_close"] - timedelta(minutes=1)).time()

    return (start_time, end_time), "Regular trading day"


# Function to check if a given day is a weekend
def is_weekend(date):
    return date.weekday() >= 5  # Saturday = 5, Sunday = 6


def configure_logger(ticker, strategy):
    """Configure a logger specific to each thread (ticker)."""
    logger = logging.getLogger(ticker)
    logger.setLevel(logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler(f"logs/{strategy}/{ticker}.log")
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger


def get_current_datetime():
    current_dt = datetime.now(tz=timezone(TIME_ZONE))
    current_time = current_dt.time()
    current_date = current_dt.date()
    return current_time, current_date


def store_logs(ticker, strategy):

    ticker_name = ticker[1:] if "/" == ticker[0] else ticker
    filename = f"previous_logs/{strategy}/{ticker_name}.txt"

    os.system(f"cat logs/{strategy}/{ticker_name}.log > {filename}")
    # clear log file
    with open(f"logs/{strategy}/{ticker_name}.log", "w") as file:
        file.write("")


def sleep_base_on_timeframe(interval_minutes):
    if interval_minutes.isdigit():
        interval_minutes = int(interval_minutes)
    if interval_minutes in [1, 2, 5]:
        next_interval = 10  # Check every 10 seconds for 1, 2, and 5-minute settings
    elif interval_minutes in [15, 30] or interval_minutes in ['1h', '4h', '1d']:
        next_interval = 60  # Check every minute for 15, 30-minute, 1-hour, 4-hour, and daily settings
    else:
        interval_str = str(interval_minutes)
        if interval_str.endswith('t'):
            try:
                num = int(interval_str[:-1])  # Everything except last char 't'
                if num <= 1000:
                    next_interval = 5  # Check every 5 seconds for 1-1000 tick settings
                elif num <= 1600:
                    next_interval = 15  # Check every 15 seconds for 1001-1600 tick settings
                else:  # num > 1600
                    next_interval = 30  # Check every 30 seconds for 1601+ tick settings
            except ValueError:
                raise ValueError("Invalid interval")
        else:
            raise ValueError("Invalid interval")
    sleep(next_interval)


def get_strategy_for_ticker(ticker):
    """Determine which strategy configuration to use for a ticker."""
    ema_data, super_data, zeroday_data = load_strategy_configs()

    if ticker in ema_data:
        return 'ema', ema_data[ticker]
    elif ticker in super_data:
        return 'supertrend', super_data[ticker]
    elif ticker in zeroday_data:
        return 'zeroday', zeroday_data[ticker]
    else:
        return None, None


def load_strategy_configs():
    """Load all strategy configuration files."""
    try:
        ema_data = load_json(os.path.join('settings', 'ema_ticker_data.json'))
        super_data = load_json(os.path.join('settings', 'super_ticker_data.json'))
        zeroday_data = load_json(os.path.join('settings', 'zeroday_ticker_data.json'))
        return ema_data, super_data, zeroday_data
    except Exception as e:
        print(f"Error loading strategy configs: {e}")
        return {}, {}, {}


# Strategy Router Functions
def load_json(filepath):
    """Load JSON file safely."""
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except Exception:
        return {}


def save_json(filepath, data):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


def get_trade_file_path(ticker, strategy_type):
    """Get strategy-specific trade file path."""
    base_name = ticker[1:] if ticker.startswith('/') else ticker
    return f"trades/{strategy_type}/{base_name}.json"


def time_convert(dt=None, form="8601"):
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


def wilders_smoothing(df, length=14):
    initial_mean = pd.Series(
        data=[df['close'].iloc[:length].mean()],
        index=[df['close'].index[length-1]],
    )
    remaining_data = df['close'].iloc[length:]

    smoothed_values = pd.concat([initial_mean, remaining_data]).ewm(
        alpha=1.0 / length,
        adjust=False,
    ).mean()

    return smoothed_values


def params_parser(params: dict):
    """
    Removes None (null) values
    :param params: params to remove None values from
    :type params: dict
    :return: params without None values
    :rtype: dict
    """
    for key in list(params.keys()):
        if params[key] is None:
            del params[key]
    return params


def sleep_until_next_interval(ticker, interval_minutes):
    """
    Sleeps until the next specified interval in minutes or hours.

    Parameters:
    - interval_minutes: int or str
        The interval to sleep for:
        - Acceptable intervals: 1, 2, 5, 15, 30 (in minutes)
        - '1h', '4h' (in hours)
        - '1d' for daily
    """
    now = datetime.now(tz=timezone(TIME_ZONE))
    if interval_minutes.isdigit():
        interval_minutes = int(interval_minutes)
    if "/" == ticker[0]:
        if isinstance(interval_minutes, str):
            # Handle special cases for hours and daily
            if interval_minutes == "1h":
                next_interval = now.replace(
                    minute=0, second=0, microsecond=0
                ) + timedelta(hours=1)
            elif interval_minutes == "4h":
                # Calculate the next 4-hour block (e.g., 00:00, 04:00, 08:00, etc.)
                hour_block = ["02:00", "06:00", "10:00", "14:00", "18:00", "22:00"]
                next_interval = None
                for i in hour_block:
                    if now.time() < time.fromisoformat(i):
                        next_interval = now.replace(
                            hour=int(i.split(":")[0]),
                            minute=int(i.split(":")[1]),
                            second=0,
                            microsecond=0,
                        )
                        break
                if next_interval is None:
                    next_interval = now.replace(
                        hour=int(hour_block[0].split(":")[0]),
                        minute=int(hour_block[0].split(":")[1]),
                        second=0,
                        microsecond=0,
                    ) + timedelta(days=1)

            elif interval_minutes == "1d":
                # Sleep until the start of the next day
                next_interval = now.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ) + timedelta(days=1)
            else:
                raise ValueError("Invalid interval string. Use '1h', '4h', or '1d'.")
        else:
            # Handle minute intervals
            if interval_minutes not in [1, 2, 5, 15, 30]:
                raise ValueError("Invalid interval in minutes. Use 1, 2, 5, 15, or 30.")

            # Calculate the next interval in minutes
            minutes = (now.minute // interval_minutes + 1) * interval_minutes
            next_interval = now.replace(minute=minutes % 60, second=0, microsecond=0)

            # Handle hour overflow
            if minutes >= 60:
                next_interval = next_interval + timedelta(hours=1)
    else:
        if isinstance(interval_minutes, str):
            # Handle special cases for hours and daily
            if interval_minutes == "1h":
                hourly_interval = [
                    "10:30",
                    "11:30",
                    "12:30",
                    "13:30",
                    "14:30",
                    "15:30",
                    "15:59",
                ]
                # sleep until next hour and minute from the list above
                next_interval = None
                for i in hourly_interval:
                    if now.time() < time.fromisoformat(i):
                        next_interval = now.replace(
                            hour=int(i.split(":")[0]),
                            minute=int(i.split(":")[1]),
                            second=0,
                            microsecond=0,
                        )
                        break
                if next_interval is None:
                    next_interval = now.replace(
                        hour=int(hourly_interval[0].split(":")[0]),
                        minute=int(hourly_interval[0].split(":")[1]),
                        second=0,
                        microsecond=0,
                    ) + timedelta(days=1)

            elif interval_minutes == "4h":
                # Calculate the next 4-hour block
                four_hour_intervals = ["13:30", "15:59"]
                next_interval = None
                for i in four_hour_intervals:
                    if now.time() < time.fromisoformat(i):
                        next_interval = now.replace(
                            hour=int(i.split(":")[0]),
                            minute=int(i.split(":")[1]),
                            second=0,
                            microsecond=0,
                        )
                        break
                if next_interval is None:
                    next_interval = now.replace(
                        hour=int(four_hour_intervals[0].split(":")[0]),
                        minute=int(four_hour_intervals[0].split(":")[1]),
                        second=0,
                        microsecond=0,
                    ) + timedelta(days=1)

            elif interval_minutes == "1d":
                # Sleep until the start of the next day
                next_interval = now.replace(hour=15, minute=59, second=0, microsecond=0)
                if now >= next_interval:
                    next_interval = next_interval + timedelta(days=1)
            else:
                raise ValueError("Invalid interval string. Use '1h', '4h', or '1d'.")
        else:
            # Handle minute intervals
            if interval_minutes not in [1, 2, 5, 15, 30]:
                raise ValueError("Invalid interval in minutes. Use 1, 2, 5, 15, or 30.")

            # Calculate the next interval in minutes
            minutes = (now.minute // interval_minutes + 1) * interval_minutes
            next_interval = now.replace(minute=minutes % 60, second=0, microsecond=0)

            # Handle hour overflow
            if minutes >= 60:
                next_interval = next_interval + timedelta(hours=1)

    # Calculate the difference in seconds
    seconds_until_next_interval = (next_interval - now).total_seconds()

    sleep(seconds_until_next_interval)


# Utility functions for tick data
def is_tick_timeframe(timeframe):
    """Check if timeframe is tick-based (ends with 't')."""
    return isinstance(timeframe, str) and timeframe.lower().endswith('t')


def extract_tick_count(timeframe):
    """Extract number of ticks from timeframe string."""
    if is_tick_timeframe(timeframe):
        return int(timeframe[:-1])
    return None


def get_tick_data(ticker, timeframe, tick_buffers, logger):
    """Get tick-based data for the specified ticker."""
    if not is_tick_timeframe(timeframe):
        return None

    # Get DataFrame from tick buffer
    df = tick_buffers[ticker].get_dataframe()
    if df is None:
        logger.warning(f"Insufficient tick data for {ticker}. Need more bars.")
        return None

    logger.info(f"Retrieved {len(df)} tick bars for {ticker}")
    return df


def get_active_exchange_symbol(symbol):
    if symbol[0] != '/':
        return symbol  # Not a futures symbol, return as is

    instrument_df = pd.read_csv("tastytrade_instruments.csv")
    df = instrument_df[
        (instrument_df["product-code"] == symbol[1:]) & (instrument_df["active-month"] == True)
    ][["exchange-symbol", "expires-at"]]
    expiry = pd.to_datetime(df["expires-at"].values[0])

    # Convert expiry to US/Eastern timezone
    eastern = pytz.timezone('US/Eastern')
    expiry_eastern = expiry.astimezone(eastern)

    # Convert datetime.now() to US/Eastern timezone
    now_eastern = datetime.now(pytz.utc).astimezone(eastern)
    if expiry_eastern <= now_eastern:
        # Use next active month
        next_symbol = instrument_df[
            (instrument_df["product-code"] == symbol[1:]) & (instrument_df["next-active-month"] == True)
        ]["exchange-symbol"].tolist()[0]
        return next_symbol
    else:
        return df["exchange-symbol"].values[0]


async def on_tick_received(tick_data, tick_buffers):
    """Callback function when new tick data is received."""
    symbol = tick_data['symbol']
    if symbol in tick_buffers:
        tick_buffers[symbol].add_tick(tick_data)


def parse_strategy_params(config, strategy_type):
    """Parse configuration parameters for any strategy."""
    try:
        if strategy_type == 'ema':
            return {
                'timeframe': config[0],
                'schwab_qty': int(config[1]) if config[1].isdigit() else 0,
                'trade_enabled': config[2] == "TRUE",
                'tasty_qty': int(config[3]) if config[3].isdigit() else 0,
                'trend_line_1': config[4],
                'period_1': int(config[5]),
                'trend_line_2': config[6],
                'period_2': int(config[7])
            }
        elif strategy_type == 'supertrend':
            return {
                'timeframe': config[0],
                'schwab_qty': int(config[1]) if config[1].isdigit() else 0,
                'trade_enabled': config[2] == "TRUE",
                'tasty_qty': int(config[3]) if config[3].isdigit() else 0,
                'short_ma_len': int(config[4]),
                'short_ma_type': config[5],
                'mid_ma_len': int(config[6]),
                'mid_ma_type': config[7],
                'long_ma_len': int(config[8]),
                'long_ma_type': config[9],
                'atr_length': int(config[10]),
                'zigzag_percent': float(config[11]),
                'atr_multiple': float(config[12]),
                'fibonacci_enabled': config[13] == "True",
                'support_demand_enabled': config[14] == "True"
            }
        elif strategy_type == 'zeroday':
            # Handle case where config might not have call/put enabled fields yet
            call_enabled = config[8] == "TRUE" if len(config) > 8 else True
            put_enabled = config[9] == "TRUE" if len(config) > 9 else True

            return {
                'timeframe': config[0],
                'schwab_qty': int(config[1]) if config[1].isdigit() else 0,
                'trade_enabled': config[2] == "TRUE",
                'tasty_qty': int(config[3]) if config[3].isdigit() else 0,
                'trend_line_1': config[4],
                'period_1': int(config[5]),
                'trend_line_2': config[6],
                'period_2': int(config[7]),
                'call_enabled': call_enabled,
                'put_enabled': put_enabled
            }
        else:
            return None
    except Exception as e:
        print(f"Error parsing strategy params for {strategy_type}: {e}")
        return None


def get_all_active_tickers():
    """Get all tickers that are configured and enabled for trading across all strategies."""
    ema_data, super_data, zeroday_data = load_strategy_configs()

    active_tickers = []

    # Check EMA strategy tickers
    for ticker, config in ema_data.items():
        if config[2] == "TRUE":  # trade_enabled
            active_tickers.append((ticker, 'ema'))

    # Check Supertrend strategy tickers
    for ticker, config in super_data.items():
        if config[2] == "TRUE":  # trade_enabled
            active_tickers.append((ticker, 'supertrend'))

    # Note: Zero-day tickers are manual, not automatic
    # But we can include them for reference
    for ticker, config in zeroday_data.items():
        if config[2] == "TRUE":  # trade_enabled
            active_tickers.append((ticker, 'zeroday'))

    return active_tickers


def validate_strategy_config(ticker, strategy_type, config):
    """Validate that a strategy configuration is complete and valid."""
    try:
        if strategy_type == 'ema':
            required_length = 8
            if len(config) != required_length:
                return False, f"EMA strategy requires {required_length} parameters, got {len(config)}"
        elif strategy_type == 'supertrend':
            required_length = 15
            if len(config) != required_length:
                return False, f"Supertrend strategy requires {required_length} parameters, got {len(config)}"
        elif strategy_type == 'zeroday':
            required_length = 8
            if len(config) != required_length:
                return False, f"Zero-day strategy requires {required_length} parameters, got {len(config)}"
        else:
            return False, f"Unknown strategy type: {strategy_type}"

        # Check if trade_enabled is valid
        if config[2] not in ["TRUE", "FALSE"]:
            return False, f"trade_enabled must be 'TRUE' or 'FALSE', got {config[2]}"

        return True, "Configuration is valid"
    except Exception as e:
        return False, f"Error validating config: {str(e)}"


def get_dataset(ticker):
    if ticker.startswith("/"):  # for CME futures
        return "GLBX.MDP3"
    else:  # for stocks
        return "XNAS.ITCH"


def get_symbol_for_data(ticker):
    if ticker.startswith("/"):
        return get_active_exchange_symbol(ticker)
    else:
        return ticker