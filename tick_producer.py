import asyncio
import databento as db
import json
import logging
import os
import pandas as pd
import redis
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from tick_buffer import DatabentoLiveManager, TickDataBuffer, TimeBasedBarBufferWithRedis
from utils import (
    extract_tick_count,
    get_active_exchange_symbol,
    get_dataset,
    get_symbol_for_data,
    is_tick_timeframe,
    load_json,
    ticker_data_path_for_strategy,
    get_schema,
    parse_strategy_params,
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
        self.strategy = None
        self.tick_buffers = {}
        self.live_manager = DatabentoLiveManager()
        self.logger = logging.getLogger("TickProducer")
        logging.basicConfig(level=logging.INFO)

    def setup_tick_buffers(self, tickers_config):
        """Setup buffers for all symbols (tick- or time-based)."""
        live_symbols_config = {}

        for ticker, config in tickers_config.items():
            # Parse the config safely based on the active strategy to avoid index/type errors
            parsed = parse_strategy_params(config, self.strategy)
            if not parsed:
                self.logger.error(f"Failed to parse config for {ticker} with strategy {self.strategy}")
                continue

            time_frame = parsed.get('timeframe')
            # Determine a conservative warmup window (max_period)
            if self.strategy in ['ema', 'zeroday']:
                try:
                    max_period = max(int(parsed.get('period_1', 1)), int(parsed.get('period_2', 1)))
                except Exception:
                    self.logger.error(f"Invalid periods for {ticker}: {parsed.get('period_1')}, {parsed.get('period_2')}")
                    max_period = 1
            elif self.strategy == 'supertrend':
                max_period = 50
            else:
                max_period = 1
            dataset = get_dataset(ticker)
            schema = get_schema(time_frame)
            safe_historical_end_time = get_historical_end_time(
                ticker, dataset, self.logger, schema
            )
            safe_historical_start_time = get_historical_start_time(
                ticker, time_frame, safe_historical_end_time
            )
            historical_end_time = safe_historical_end_time.isoformat()
            historical_start_time = safe_historical_start_time.isoformat()
            ticker_for_data = get_symbol_for_data(ticker)

            self._setup_symbol_buffer(
                ticker_for_data,
                time_frame,
                max_period,
                historical_end_time,
                historical_start_time,
                live_symbols_config,
                dataset,
                schema,
            )
        return live_symbols_config

    def _setup_symbol_buffer(
        self,
        ticker_for_data,
        time_frame,
        max_period,
        historical_end_time,
        historical_start_time,
        live_symbols_config,
        dataset,
        schema,
    ):
        if ticker_for_data not in self.tick_buffers:
            if is_tick_timeframe(time_frame):
                buffer = TickDataBufferWithRedis(
                    ticker=ticker_for_data,
                    strategy=self.strategy,
                    time_frame=time_frame,
                    max_period=max_period,
                    logger=self.logger,
                )
            else:
                buffer = TimeBasedBarBufferWithRedis(
                    ticker=ticker_for_data,
                    strategy=self.strategy,
                    time_frame=time_frame,
                    max_period=max_period,
                    logger=self.logger,
                )
            self.tick_buffers[ticker_for_data] = buffer
            print(
                f"Initialized buffer for {ticker_for_data} of {self.strategy} with timeframe {time_frame}"
            )

        if is_tick_timeframe(time_frame):
            self.tick_buffers[ticker_for_data].warmup_with_historical_ticks(
                symbol=ticker_for_data,
                dataset=dataset,
                start=historical_start_time,
                end=historical_end_time,
                schema=schema,
            )
        else:
            self.tick_buffers[ticker_for_data].warmup_with_historical_timebars(
                symbol=ticker_for_data,
                dataset=dataset,
                start=historical_start_time,
                end=historical_end_time,
                base_schema=schema,
            )

        # Setup live feed
        bars = self.tick_buffers[ticker_for_data].processed_bars
        if bars:
            last_ts = pd.Timestamp(bars[-1]["timestamp"])
            replay_start_time = last_ts.value + 1
        else:
            replay_start_time = 0

        live_symbols_config[ticker_for_data] = {
            "dataset": dataset,
            "schema": get_schema(time_frame) if not is_tick_timeframe(time_frame) else 'trades',
            "start_time": replay_start_time,
        }

    
    async def start_live_feeds(self, live_symbols_config):
        """Start live data feeds"""
        if live_symbols_config:
            # Handle SPXW separately with Tastytrade
            spxw_symbols = {k: v for k, v in live_symbols_config.items() if k == "SPXW"}
            other_symbols = {k: v for k, v in live_symbols_config.items() if k != "SPXW"}
            
            # Start Tastytrade feeds for SPXW
            if spxw_symbols:
                self.logger.info(f"Starting Tastytrade feeds for SPXW symbols")
                for symbol, config in spxw_symbols.items():
                    if symbol in self.tick_buffers:
                        self.tick_buffers[symbol].start_live_subscription(
                            symbol, config.get("dataset"), config.get("schema")
                        )
            
            # Start Databento feeds for other symbols
            if other_symbols:
                self.live_manager = DatabentoLiveManager()
                self.logger.info(
                    f"Starting Databento live feeds for: {list(other_symbols.keys())}"
                )
                await self.live_manager.start_live_feeds(
                    other_symbols, self.tick_buffers
                )

    def run(self, tickers_config, strategy):
        """Main run method"""
        self.strategy = strategy
        self.logger.info("Starting Tick Producer...")
        live_symbols_config = self.setup_tick_buffers(tickers_config)
        asyncio.run(self.start_live_feeds(live_symbols_config))


class TickDataBufferWithRedis(TickDataBuffer):
    """Extended TickDataBuffer that publishes bars to Redis"""

    def __init__(self, ticker, strategy, time_frame, max_period, logger):
        super().__init__(ticker, strategy, time_frame, max_period, logger)
        self.redis_client = REDIS_CLIENT
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
            zset_key_strategy = f"bars_history:{self.strategy}{self.ticker}"
            zset_key_plain = f"bars_history:{self.ticker}"
            timestamp_score = int(
                pd.Timestamp(bar["timestamp"]).timestamp()
            )  # seconds since epoch
            payload = json.dumps(bar_data)
            self.redis_client.zadd(zset_key_strategy, {payload: timestamp_score})
            self.redis_client.zadd(zset_key_plain, {payload: timestamp_score})

            # ---- Trim ZSET to only keep the latest max_period items ----
            # Remove all but the latest max_period items (highest scores)
            self.redis_client.zremrangebyrank(zset_key_strategy, 0, -1000)
            self.redis_client.zremrangebyrank(zset_key_plain, 0, -1000)

            # self.logger.info(f"ZSET updated for {self.ticker} with timestamp {timestamp_score}")

        return bar


def get_historical_end_time(ticker, dataset, logger, schema):
    try:
        metadata = HISTORICAL_CLIENT.metadata.get_dataset_range(dataset=dataset)
        schema_range = metadata.get("schema", {}).get(schema, {})
        logger.info(f"{ticker}'s schema range for {schema}: {schema_range}")
        if "end" in schema_range:
            safe_historical_end = pd.to_datetime(schema_range["end"]).tz_convert(
                "UTC"
            )
        else:
            safe_historical_end = datetime.now(timezone.utc) - timedelta(minutes=10)
        return safe_historical_end
    except Exception as e:
        logger.error(
            f"Error fetching historical end time for dataset {dataset} and schema {schema}: {e}"
        )
        safe_historical_end = datetime.now(timezone.utc) - timedelta(minutes=10)
        return safe_historical_end


def get_historical_start_time(ticker, timeframe, end_time):
    start_time = end_time
    timeframe = str(timeframe)
    if is_tick_timeframe(timeframe):
        tick_timeframe = int(timeframe[:-1])
        if tick_timeframe <= 1000:
            start_time = end_time - timedelta(days=1)
        elif tick_timeframe <= 1600:
            start_time = end_time - timedelta(days=2)
        else:
            start_time = end_time - timedelta(days=4)
    else:
        if timeframe == "1":
            start_time = end_time - timedelta(days=1)
        elif timeframe == "2":
            start_time = end_time - timedelta(days=1)
        elif timeframe == "5":
            start_time = end_time - timedelta(days=2)
        elif timeframe == "15":
            start_time = end_time - timedelta(days=3)
        elif timeframe == "30":
            start_time = end_time - timedelta(days=5)
        elif timeframe == "1h":
            start_time = end_time - timedelta(days=10)
        elif timeframe == "4h":
            start_time = end_time - timedelta(days=30)
        elif timeframe == "1d":
            start_time = end_time - timedelta(days=60)
    
    # Ensure minimum time difference to avoid Databento errors
    min_diff = timedelta(days=1)
    if (end_time - start_time) < min_diff:
        start_time = end_time - min_diff
    
    return start_time
        