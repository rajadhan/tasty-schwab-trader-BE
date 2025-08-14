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
            ticker_for_data = get_symbol_for_data(ticker)
            
            if is_tick_timeframe(time_frame):
                self._setup_tick_based_buffer(
                    ticker_for_data, time_frame, max_period, historical_end_time, historical_start_time, live_symbols_config, dataset
                )
            else:
                self._setup_time_based_buffer(
                    ticker_for_data, time_frame, max_period, historical_end_time, historical_start_time, live_symbols_config, dataset
                )

        return live_symbols_config


    def _setup_tick_based_buffer(self, ticker_for_data, time_frame, max_period, historical_end_time, historical_start_time, live_symbols_config, dataset):
        """ Setup buffer for tick-based timeframes (e.g., 512t) """
        print(f"Setting up TICK_BASED buffer for {ticker_for_data}: {time_frame}")
        if ticker_for_data not in self.tick_buffers:
            tick_size = extract_tick_count(time_frame)
            buffer = TickDataBufferWithRedis(
                ticker=ticker_for_data,
                tick_size=tick_size,
                max_period=max_period,
                logger=self.logger
            )
            self.tick_buffers[ticker_for_data] = buffer
            print(f"Initialized TICK-BASED buffer for {ticker_for_data} with tick size {tick_size}")
        
        # Warmup with historical data
        self.tick_buffers[ticker_for_data].warmup_with_historical_ticks(
            symbol=ticker_for_data,
            dataset=dataset,
            start=historical_start_time,
            end=historical_end_time,
            schema="trades"
        )
        print("tick buffers", self.tick_buffers[ticker_for_data])

        # Setup live feed
        bars = self.tick_buffers[ticker_for_data].processed_bars
        if bars:
            last_ts = pd.Timestamp(bars[-1]['timestamp'])
            replay_start_time = last_ts.value + 1
        else:
            replay_start_time = 0

        live_symbols_config[ticker_for_data] = {
            'dataset': dataset,
            'schema': 'trades',
            'start_time': replay_start_time
        }


    def _setup_time_based_buffer(self, ticker_for_data, time_frame, max_period, historical_end_time, historical_start_time, live_symbols_config, dataset):
        """ Setup buffer for time-based timeframes (e.g., 30min, 4h) """
        print(f"Setting up TIME-BASED buffer for {ticker_for_data}: {time_frame}")
        if ticker_for_data not in self.tick_buffers:
            # For time-based, we'll use a small tick size and resample
            tick_size = 100  # Use 100 ticks as base, then resample
            
            buffer = TimeBasedDataBuffer(  # New class for time-based
                ticker=ticker_for_data,
                tick_size=tick_size,
                target_timeframe=time_frame,  # e.g., "30m", "4h"
                redis_client=self.redis_client,
                db_api_key=os.getenv("DATABENTO_API_KEY"),
                max_period=max_period,
            )
            self.tick_buffers[ticker_for_data] = buffer
            
            print(f"Initialized TIME-BASED buffer for {ticker_for_data} targeting {time_frame}")

        dataset = "GLBX.MDP3"
        
        # Warmup with historical data (will be resampled to target timeframe)
        self.tick_buffers[ticker_for_data].warmup_with_historical_ticks(
            symbol=ticker_for_data,
            dataset=dataset,
            start=historical_start_time,
            end=historical_end_time,
            schema="trades",
        )
        
        # Setup live feed
        bars = self.tick_buffers[ticker_for_data].processed_bars
        if bars:
            last_ts = pd.Timestamp(bars[-1]['timestamp'])
            replay_start_time = last_ts.value + 1
        else:
            replay_start_time = 0
        
        live_symbols_config[ticker_for_data] = {
            'dataset': dataset,
            'schema': 'trades',
            'start_time': replay_start_time
        }


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
        print("tickers config", tickers_config)
        self.logger.info("Starting Tick Producer...")
        live_symbols_config = self.setup_tick_buffers(tickers_config)
        asyncio.run(self.start_live_feeds(live_symbols_config)) # TODO


class TickDataBufferWithRedis(TickDataBuffer):
    """Extended TickDataBuffer that publishes bars to Redis"""

    def __init__(self, ticker, tick_size, max_period, logger):
        super().__init__(ticker, tick_size, max_period, logger)
        self.redis_client = REDIS_CLIENT
        self.max_period = max_period

    def _create_bar_from_ticks(self):
        if not self.buffer:
            return None

        bar = super()._create_bar_from_ticks()

        if bar:
            bar_data = {
                'symbol': self.ticker,
                'timestamp': bar['timestamp'].isoformat(),
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar['volume']
            }

            # ---- Publish to Redis Pub/Sub channel ----
            channel = f"tick_bars:{self.ticker}"
            self.redis_client.publish(channel, json.dumps(bar_data))

            # ---- Store in Redis ZSET for count-based access ----
            zset_key = f"bars_history:{self.ticker}"
            timestamp_score = int(pd.Timestamp(bar['timestamp']).timestamp())  # seconds since epoch
            self.redis_client.zadd(zset_key, {json.dumps(bar_data): timestamp_score})

            # ---- Trim ZSET to only keep the latest max_period items ----
            # Remove all but the latest max_period items (highest scores)
            self.redis_client.zremrangebyrank(zset_key, 0, -(self.max_period + 2))

            self.logger.info(f"ZSET updated for {self.ticker} with timestamp {timestamp_score}")

        return bar


class TimeBasedDataBuffer(TickDataBuffer):
    """Buffer that resamples tick data to time-based bars"""
    
    def __init__(self, ticker, tick_size, target_timeframe, **kwargs):
        super().__init__(ticker, tick_size, **kwargs)
        self.target_timeframe = target_timeframe
        self.raw_ticks = []  # Store raw ticks for resampling
    
    def warmup_with_historical_ticks(self, symbol, dataset, start, end, schema='trades'):
        """Fetch historical data and resample to target timeframe"""
        try:
            self.logger.info(f"Fetching historical data for {symbol} targeting {self.target_timeframe}")
            
            # Fetch raw tick data
            data = self.db_client.timeseries.get_range(
                dataset=dataset,
                symbols=[symbol],
                schema=schema,
                start=start,
                end=end
            )
            
            df = data.to_df()
            if df.empty:
                self.logger.warning(f"No historical data returned for {symbol}")
                return
            
            # Process and store raw ticks
            df['price'] = df['price'] / 1e9
            df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns')
            df['volume'] = df['size']
            
            # Resample to target timeframe
            resampled_bars = self._resample_to_timeframe(df)
            
            # Store resampled bars
            self.processed_bars = resampled_bars
            self.historical_loaded = True
            
            self.logger.info(f"Created {len(self.processed_bars)} {self.target_timeframe} bars for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Historical warmup failed for {symbol}: {e}")
    
    def _resample_to_timeframe(self, df):
        """Resample tick data to target timeframe"""
        # Convert timeframe to pandas offset
        timeframe_offset = self._convert_to_pandas_offset(self.target_timeframe)
        
        # Resample to target timeframe
        ohlcv = df['price'].resample(timeframe_offset).ohlc()
        volume = df['volume'].resample(timeframe_offset).sum()
        
        result = pd.concat([ohlcv, volume], axis=1)
        result.columns = ['open', 'high', 'low', 'close', 'volume']
        result.dropna(inplace=True)
        
        # Convert to list of dictionaries
        bars = []
        for timestamp, row in result.iterrows():
            bar = {
                'timestamp': timestamp,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }
            bars.append(bar)
        
        return bars
    
    def _convert_to_pandas_offset(self, timeframe):
        """Convert timeframe string to pandas offset"""
        if timeframe == "1m": return "1T"
        elif timeframe == "5m": return "5T"
        elif timeframe == "15m": return "15T"
        elif timeframe == "30m": return "30T"
        elif timeframe == "1h": return "1H"
        elif timeframe == "4h": return "4H"
        elif timeframe == "1d": return "1D"
        else: return "1T"  # Default to 1 minute


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
        if timeframe == '1':
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


