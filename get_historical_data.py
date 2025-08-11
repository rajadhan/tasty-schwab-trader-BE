from datetime import timedelta, timezone
from dotenv import load_dotenv
from utils import (
    extract_tick_count,
    get_active_exchange_symbol,
    is_tick_timeframe
)
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
    if ticker.startswith('/'):  # for CME futures
        return 'GLBX.MDP3'
    else:                       # for stocks
        return 'XNAS.ITCH'


def get_symbol_for_data(ticker):
    if ticker.startswith('/'):
        return get_active_exchange_symbol(ticker)
    else:
        return ticker


# Get metadata to determine safe historical end time
def get_historical_end_time(dataset, logger):
    try:
        metadata = DATABENTO_CLIENT.metadata.get_dataset_range(dataset=dataset)
        schema_range = metadata.get("schema", {}).get("trades", {})
        if "end" in schema_range:
            safe_historical_end = pd.to_datetime(schema_range["end"]).tz_convert("UTC")
        else:
            safe_historical_end = datetime.now(timezone.utc) - timedelta(minutes=10)
        return safe_historical_end
    except Exception as e:
        logger.error(f"Error fetching historical end time for dataset {dataset}: {e}")
        safe_historical_end = datetime.now(timezone.utc) - timedelta(minutes=10)
        return safe_historical_end


# Get historical start time
def get_historical_start_time(timeframe, end_time):
    if timeframe.isdigit():
        start_time = (end_time - timedelta(days=5))
    elif timeframe == '1h':
        start_time = (end_time - timedelta(days=10))
    elif timeframe == '4h':
        start_time = (end_time - timedelta(days=30))
    elif timeframe == '1d':
        start_time = (end_time - timedelta(days=60))
    elif timeframe.endswith('t'):
        start_time = (end_time - timedelta(days=1))
    return start_time


def symbology_resolve(dataset, symbol, start, end):
    try:
        result = DATABENTO_CLIENT.symbology.resolve(
            dataset=dataset,
            symbols=[symbol],
            stype_in="raw_symbol",
            stype_out="instrument_id",
            start_date=start,
            end_date=end
        )
        df = result.to_df()
        # instrument_id = df['result']
        print("Resolved instrument_id: ", df)
        return df
    except db.BentoClientError as e:
        print("e", e)



def get_tick_based_data(tick_size, dataset, start, end, symbol):
    print("symbol", symbol, tick_size, dataset)
    try:
        data = DATABENTO_CLIENT.timeseries.get_range(
            dataset=dataset,
            symbols=[symbol],
            stype_in="raw_symbol",
            stype_out="instrument_id",
            start=start,
            end=end,
        ).to_df()
        print("data", tick_size, data)
        return data
    except db.BentoClientError as e:
        print("e", e)


def get_historical_data(ticker, timeframe, logger):
    dataset = get_dataset(ticker)
    symbol = get_symbol_for_data(ticker)
    end_time = get_historical_end_time(dataset, logger)
    start_time = get_historical_start_time(timeframe, end_time)
    end_time = end_time.isoformat()
    start_time = start_time.isoformat()
    if is_tick_timeframe(timeframe):
        tick_size = extract_tick_count(timeframe)
        instrument_id = symbology_resolve(dataset, symbol, start_time, end_time)
        df = get_tick_based_data(tick_size, dataset, start_time, end_time, instrument_id)
        print("df===============", df)


def get_historical_and_live_data(ticker, timeframe, logger, strategy):
    get_historical_data(ticker, timeframe, logger)
