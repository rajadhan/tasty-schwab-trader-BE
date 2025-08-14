import asyncio
import databento as db
import json
import logging
import os
import pandas as pd
import redis
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from tick_buffer import DatabentoLiveManager, TickDataBuffer
from utils import (
    extract_tick_count,
    get_active_exchange_symbol,
    get_dataset,
    get_symbol_for_data,
    is_tick_timeframe,
    load_json,
    ticker_data_path_for_strategy,
)

load_dotenv()  # Load environment variables from .env

# Global clients - created once and reused
DB_API_KEY = os.getenv("DATABENTO_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Global Redis client
REDIS_CLIENT = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# Global Databento clients
HISTORICAL_CLIENT = db.Historical(DB_API_KEY)
LIVE_CLIENT = db.Live(key=DB_API_KEY)


class TickProducer:
    def __init__(self):
        # Import here to avoid circular imports
        self.TickDataBuffer = TickDataBuffer  # Store the class, not an instance
        self.tick_buffers = {}
        self.live_manager = DatabentoLiveManager()
        self.logger = logging.getLogger("TickProducer")
        logging.basicConfig(level=logging.INFO)

    def setup_tick_buffers(self, tickers_config):
        """Setup tick buffers for all tick-based symbols"""
        max_period = 3
        live_symbols_config = {}

        for ticker, config in tickers_config.items():
            time_frame = config[0]
            period1 = config[5]
            period2 = config[7]
            
            max_period = max(int(period1), int(period2))
            dataset = get_dataset(ticker)
            safe_historical_end_time = get_historical_end_time(ticker, dataset, self.logger)
            safe_historical_start_time = get_historical_start_time(ticker,time_frame, safe_historical_end_time)
            historical_end_time = safe_historical_end_time.isoformat()
            historical_start_time = safe_historical_start_time.isoformat()
            ticker_for_data = get_active_exchange_symbol(ticker) if ticker.startswith('/') else ticker
                
            if ticker_for_data not in self.tick_buffers:
                buffer = TickDataBuffer(
                    ticker=ticker_for_data,
                    time_frame=time_frame,
                    redis_client=REDIS_CLIENT,
                    max_period=max_period,
                    logger=self.logger
                )
                self.tick_buffers[ticker_for_data] = buffer
                print(
                    f"Initialized TickDataBuffer for {ticker_for_data} with time frame {time_frame}"
                )
            # Warmup with historical data
            self.tick_buffers[ticker_for_data].warmup_with_historical_ticks(
                symbol=ticker_for_data,
                dataset=dataset,
                start=historical_end_time,
                end=historical_start_time,
                schema="trades",
            )

            bars = self.tick_buffers[ticker_for_data].processed_bars
            if bars:
                last_ts = pd.Timestamp(bars[-1]["timestamp"])
                replay_start_time = last_ts.value + 1  # nanoseconds
            else:
                replay_start_time = 0

            print(f"LIVE SYMBOL CONFIG {live_symbols_config}")
            live_symbols_config[ticker_for_data] = {
                "dataset": dataset,
                "schema": "trades",
                "start_time": replay_start_time,  # int nanoseconds
            }

        return live_symbols_config

    async def start_live_feeds(self, live_symbols_config):
        """Start live data feeds"""
        if live_symbols_config:
            self.live_manager = DatabentoLiveManager(
                db_api_key=os.getenv("DATABENTO_API_KEY")
            )
            self.logger.info(
                f"Starting live feeds for: {list(live_symbols_config.keys())}"
            )
            await self.live_manager.start_live_feeds(
                live_symbols_config, self.tick_buffers
            )

    def run(self, tickers_config):
        """Main run method"""
        print("tickers config", tickers_config)  # TODO
        self.logger.info("Starting Tick Producer...")
        live_symbols_config = self.setup_tick_buffers(tickers_config)
        asyncio.run(self.start_live_feeds(live_symbols_config))


def get_historical_end_time(ticker, dataset, logger):
    if ticker.startswith("/"):
        try:
            metadata = HISTORICAL_CLIENT.metadata.get_dataset_range(dataset=dataset)
            schema_range = metadata.get("schema", {}).get("trades", {})
            logger.info(f"{ticker}'s schema range: {schema_range}")
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


def get_historical_and_live_data(ticker, logger, strategy):
    """Get historical data for a specific ticker and return as DataFrame"""
    try:
        # Load ticker configuration
        ticker_data_path = ticker_data_path_for_strategy(strategy)
        tickers_config = load_json(ticker_data_path)

        # Create producer
        producer = TickProducer()

        # Run fetching historical and live data
        producer.run(tickers_config)

        # Find the specific ticker we want in case of ticker is future
        if ticker.startswith("/"):
            ticker_for_data = get_active_exchange_symbol(ticker)
        else:
            ticker_for_data = ticker

        # Get the buffer for this specific ticker
        if ticker_for_data in producer.tick_buffers:
            tick_buffer = producer.tick_buffers[ticker_for_data]

            if tick_buffer.historical_loaded:
                # Return the DataFrame from the tick buffer
                df = tick_buffer.get_dataframe(min_bars=50)  # Get at least 50 bars
                logger.info(f"Successfully retrieved {len(df)} bars for {ticker}")
                return df
            else:
                logger.error(f"Historical data not loaded for {ticker}")
                return None
        else:
            logger.error(f"Ticker {ticker} not found in buffers")
            return None

    except Exception as e:
        logger.error(f"Error getting data for {ticker}: {e}")
        return None


