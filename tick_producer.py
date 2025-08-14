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
    is_tick_timeframe,
    load_json,
    ticker_data_path_for_strategy,
)


load_dotenv()  # Load environment variables from .env



DB_API_KEY = os.getenv("DATABENTO_API_KEY")


class TickProducer:
    def __init__(
        self, redis_host="localhost", redis_port=6379, redis_db=0, db_api_key=None
    ):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.tick_buffers = {}
        self.live_manager = None
        self.logger = logging.getLogger("TickProducer")
        self.db_api_key = db_api_key
        logging.basicConfig(level=logging.INFO)
        # Initialize Databento client for metadata queries
        if db_api_key:
            self.db_client = db.Historical(key=db_api_key)
        else:
            self.db_client = db.Historical()

    def get_available_schema_end_time(self, dataset: str, schema: str) -> datetime:
        """Get the available end time for a given dataset and schema using Databento metadata API."""
        try:
            metadata = self.db_client.metadata.get_dataset_range(dataset=dataset)
            schema_range = metadata.get("schema", {}).get(schema, {})
            if "end" in schema_range:
                return pd.to_datetime(schema_range["end"]).tz_convert("UTC")
        except Exception as e:
            self.logger.error(f"Failed to get dataset range from metadata: {e}")
        return datetime.now(timezone.utc) - timedelta(minutes=10)  # fallback

    def setup_tick_buffers(self, tickers_config):
        """Setup tick buffers for all tick-based symbols"""
        max_period = 3  # can be inferred from indicators in config

        # Use live metadata to determine safe historical end
        safe_historical_end = self.get_available_schema_end_time("GLBX.MDP3", "trades")

        live_symbols_config = {}
        for ticker, config in tickers_config.items():
            time_frame = config[0]
            period1 = config[3]
            period2 = config[5]
            if not is_tick_timeframe(time_frame):
                continue

            ticker_for_data = (
                get_active_exchange_symbol(ticker) if ticker.startswith("/") else ticker
            )
            print(f"Ticker for data {ticker_for_data}")
            max_period = max(int(period1), int(period2))
            historical_end_date = safe_historical_end.isoformat()
            historical_start_date = (
                safe_historical_end - timedelta(days=1)
            ).isoformat()  # changing things as per new period based ema
            print(f"Processing ticker: {ticker_for_data} with time frame: {time_frame}")
            if ticker_for_data not in self.tick_buffers:
                tick_size = extract_tick_count(time_frame)

                buffer = TickDataBufferWithRedis(
                    ticker=ticker_for_data,
                    tick_size=tick_size,
                    redis_client=self.redis_client,
                    db_api_key=os.getenv("DATABENTO_API_KEY"),
                    max_period=max_period,
                )
                self.tick_buffers[ticker_for_data] = buffer

                print(
                    f"Initialized TickDataBuffer for {ticker_for_data} with tick size {tick_size}"
                )

            dataset = "GLBX.MDP3"

            # Warmup with historical data
            self.tick_buffers[ticker_for_data].warmup_with_historical_ticks(
                symbol=ticker_for_data,
                dataset=dataset,
                start=historical_start_date,
                end=historical_end_date,
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
        self.logger.info("Starting Tick Producer...")
        live_symbols_config = self.setup_tick_buffers(tickers_config)
        asyncio.run(self.start_live_feeds(live_symbols_config))


class TickDataBufferWithRedis(TickDataBuffer):
    """Extended TickDataBuffer that publishes bars to Redis"""

    def __init__(self, ticker, tick_size, redis_client, db_api_key=None, max_period=3):
        super().__init__(ticker, tick_size, db_api_key)
        self.redis_client = redis_client
        self.max_period = max_period

    def _create_bar_from_ticks(self):
        if not self.buffer:
            return None

        bar = super()._create_bar_from_ticks()

        if bar:
            bar_data = {
                "symbol": self.ticker,
                "timestamp": bar["timestamp"].isoformat(),
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            }

            # ---- Publish to Redis Pub/Sub channel ----
            channel = f"tick_bars:{self.ticker}"
            self.redis_client.publish(channel, json.dumps(bar_data))

            # ---- Store in Redis ZSET for count-based access ----
            zset_key = f"bars_history:{self.ticker}"
            timestamp_score = int(
                pd.Timestamp(bar["timestamp"]).timestamp()
            )  # seconds since epoch
            self.redis_client.zadd(zset_key, {json.dumps(bar_data): timestamp_score})

            # ---- Trim ZSET to only keep the latest max_period items ----
            # Remove all but the latest max_period items (highest scores)
            self.redis_client.zremrangebyrank(zset_key, 0, -(self.max_period + 2))

            self.logger.info(
                f"ZSET updated for {self.ticker} with timestamp {timestamp_score}"
            )

        return bar


if __name__ == "__main__":
    with open("jsons/tickers.json", "r") as file:
        tickers_config = json.load(file)

    db_api_key = os.getenv("DATABENTO_API_KEY")
    producer = TickProducer(db_api_key=db_api_key)
    producer.run(tickers_config)


def get_historical_and_live_data(ticker, timeframe, logger, strategy):
    """Get historical data for a specific ticker and return as DataFrame"""
    try:
        # Load ticker configuration
        ticker_data_path = ticker_data_path_for_strategy(strategy)
        print("ticker_data_path", ticker_data_path)
        ticker_configs = load_json(ticker_data_path)
        print("ticker_configs", ticker_configs)

        # Create producer
        producer = TickProducer(db_api_key=DB_API_KEY)

        # Setup tick buffers (this fetches historical data)
        live_symbols_config = producer.setup_tick_buffers(tickers_config)

        # Find the specific ticker we want
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
