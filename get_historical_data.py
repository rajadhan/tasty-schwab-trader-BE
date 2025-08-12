from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from utils import extract_tick_count, get_active_exchange_symbol, is_tick_timeframe
import os
import redis
import databento as db
import pandas as pd
import asyncio


load_dotenv()
DB_API_KEY = os.getenv("DATABENTO_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_DB = os.getenv("REDIS_DB")

REDIS_CLIENT = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
DATABENTO_CLIENT = db.Historical(DB_API_KEY)


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


# Get metadata to determine safe historical end time
def get_historical_end_time(ticker, dataset, logger):
    if ticker.startswith("/"):
        try:
            metadata = DATABENTO_CLIENT.metadata.get_dataset_range(dataset=dataset)
            schema_range = metadata.get("schema", {}).get("trades", {})
            if "end" in schema_range:
                safe_historical_end = pd.to_datetime(schema_range["end"]).tz_convert(
                    "UTC"
                )
            else:
                safe_historical_end = datetime.now(timezone.utc) - timedelta(minutes=10)
            return safe_historical_end
        except Exception as e:
            logger.error(
                f"Error fetching historical end time for dataset {dataset}: {e}"
            )
            safe_historical_end = datetime.now(timezone.utc) - timedelta(minutes=10)
            return safe_historical_end
    else:
        safe_historical_end = datetime.now(timezone.utc) - timedelta(days=1)
        return safe_historical_end


# Get historical start time
def get_historical_start_time(ticker, timeframe, end_time):

    if ticker.startswith("/"):
        if timeframe.isdigit():
            start_time = end_time - timedelta(days=5)
        elif timeframe == "1h":
            start_time = end_time - timedelta(days=10)
        elif timeframe == "4h":
            start_time = end_time - timedelta(days=30)
        elif timeframe == "1d":
            start_time = end_time - timedelta(days=60)
        elif timeframe.endswith("t"):
            start_time = end_time - timedelta(days=1)
        return start_time
    else:
        start_time = end_time - timedelta(days=3)
        return start_time


def get_tick_based_data(dataset, start, end, symbol, timeframe, ticker):
    if timeframe.endswith("t"):
        schema = "trades"
        stype_in = "raw_symbol"
        stype_out = "instrument_id"
        try:
            data = DATABENTO_CLIENT.timeseries.get_range(
                dataset=dataset,
                symbols=[symbol],
                schema=schema,
                stype_in=stype_in,
                stype_out=stype_out,
                start=start,
                end=end,
            ).to_df()
            return data
        except Exception as e:
            print(f"Error in get_tick_based_data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    else:
        schema = "ohlcv-1s"
        stype_in = "raw_symbol"

        try:
            data = DATABENTO_CLIENT.timeseries.get_range(
                dataset=dataset,
                symbols=[symbol],
                schema=schema,
                stype_in=stype_in,
                start=start,
                end=end,
            ).to_df()
            data.index = pd.to_datetime(data.index, utc=True)
            data.index.name = "ts_event"
            if timeframe.endswith('d'):
                prefix = 'd'
                timeframe = int(timeframe[:-1])
            elif timeframe.endswith('h'):
                prefix = 'h'
                timeframe = int(timeframe[:-1])
            else:
                timeframe = int(timeframe)
                prefix = 'min'
            data = data.resample(f"{timeframe}{prefix}").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna()
            return data
        except Exception as e:
            print(f"Error in get_tick_based_data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error


def get_historical_data(ticker, timeframe, logger):
    dataset = get_dataset(ticker)
    symbol = get_symbol_for_data(ticker)

    end_time = get_historical_end_time(ticker, dataset, logger)
    start_time = get_historical_start_time(ticker, timeframe, end_time)
    # Also keep ISO format for timeseries API
    end_time = end_time.isoformat()
    start_time = start_time.isoformat()
    df = get_tick_based_data(dataset, start_time, end_time, symbol, timeframe, ticker)
    df.to_csv(f"{symbol}_{timeframe}.csv", index=True)

    print(f"[{datetime.now()}] Saved {len(df)} rows to {symbol}_{timeframe}.csv")
    print("df", ticker, df.head())
    return df


def get_historical_and_live_data(ticker, timeframe, logger, strategy):
    return get_historical_data(ticker, timeframe, logger)
