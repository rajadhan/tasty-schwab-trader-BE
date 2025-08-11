import os
import json
import redis
import pandas as pd
import asyncio
import logging
import databento as db
from datetime import datetime, timezone, timedelta
from databento import Historical
from tick_buffer import TickDataBuffer, DatabentoLiveManager
from utils import is_tick_timeframe, extract_tick_count, get_active_exchange_symbol, configure_logger
from dotenv import load_dotenv

load_dotenv()

class TickDataBufferWithRedis(TickDataBuffer):
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


def get_historical_data(ticker: str, timeframe: str, logger):
    db_api_key = os.getenv("DATABENTO_API_KEY")
    if not db_api_key:
        logger.error("DATABENTO_API_KEY not set.")
        raise EnvironmentError("DATABENTO_API_KEY not set.")
    
    # Check for both naming conventions for Redis environment variables
    redis_host = os.getenv("RDIS_HOST")
    redis_port = int(os.getenv("RDIS_PORT"))
    redis_db = int(os.getenv("RDIS_DB"))

    redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

    # Determine dataset based on ticker
    if ticker.startswith('/'):
        dataset = "GLBX.MDP3"
    elif ticker == "SPY":
        dataset = "ARCX.ITCH"
    elif ticker == "SPX":
        dataset = None
    else:
        dataset = "XNAS.ITCH"
    
    # Check if this is a tick-based timeframe
    if is_tick_timeframe(timeframe):
        return get_tick_based_historical_data(ticker, timeframe, logger, redis_client, db_api_key, dataset)
    else:
        return get_time_based_historical_data(ticker, timeframe, logger, db_api_key, dataset)


def get_tick_based_historical_data(ticker, timeframe, logger, redis_client, db_api_key, dataset):
    ticker_for_data = get_active_exchange_symbol(ticker) if ticker.startswith("/") else ticker
    # Extract tick count from timeframe
    tick_size = extract_tick_count(timeframe)
    if not tick_size:
        logger.error(f"Invalid tick timeframe: {timeframe}")
        return None
    
    # Determine max period for historical data (can be adjusted based on strategy needs)
    max_period = 100  # Default value
    
    # Initialize tick buffer
    tick_buffer = TickDataBufferWithRedis(
        ticker=ticker_for_data,
        tick_size=tick_size,
        redis_client=redis_client,
        db_api_key=db_api_key,
        max_period=max_period
    )
    
    # Get metadata to determine safe historical end time
    db_client = db.Historical(key=db_api_key)
    try:
        metadata = db_client.metadata.get_dataset_range(dataset=dataset)
        schema_range = metadata.get("schema", {}).get("trades", {})
        if "end" in schema_range:
            safe_historical_end = pd.to_datetime(schema_range["end"]).tz_convert("UTC")
        else:
            safe_historical_end = datetime.now(timezone.utc) - timedelta(minutes=10)  # fallback
    except Exception as e:
        logger.error(f"Failed to get dataset range from metadata: {e}")
        safe_historical_end = datetime.now(timezone.utc) - timedelta(minutes=10)  # fallback
    
    # Set historical date range
    historical_end_date = safe_historical_end.isoformat()
    historical_start_date = (safe_historical_end - timedelta(days=1)).isoformat()
    
    # Warmup with historical data
    tick_buffer.warmup_with_historical_ticks(
        symbol=ticker_for_data,
        dataset=dataset,
        start=historical_start_date,
        end=historical_end_date,
        schema='trades'
    )
    
    # Get dataframe from tick buffer
    df = tick_buffer.get_dataframe()
    if df is None or len(df) < 5:  # Ensure we have enough data
        logger.warning(f"Insufficient tick data for {ticker}. Need more bars.")
        return None
    
    logger.info(f"Retrieved {len(df)} tick bars for {ticker}")
    return df


def get_time_based_historical_data(ticker, timeframe, logger, db_api_key, dataset):
    """Get historical data for time-based timeframes using Databento Historical API."""
    # Get the active exchange symbol if it's a futures contract
    ticker_for_data = get_active_exchange_symbol(ticker) if ticker.startswith("/") else ticker
    
    # Initialize Databento client
    db_client = Historical(key=db_api_key)
    
    # Determine time period based on timeframe
    if timeframe.isdigit():
        # Minutes timeframe
        minutes = int(timeframe)
        start_date = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    elif timeframe == '1h':
        start_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    elif timeframe == '4h':
        start_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    elif timeframe == '1d':
        start_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    else:
        logger.error(f"Unsupported timeframe: {timeframe}")
        return None
    
    end_date = datetime.now(timezone.utc).isoformat()
    
    try:
        # Resolve symbol to instrument ID
        result = db_client.symbology.resolve(
            dataset=dataset,
            symbols=[ticker_for_data],
            stype_in="raw_symbol",
            stype_out="instrument_id",
            start_date=start_date,
            end_date=end_date,
        )
        logger.info(f"Resolved symbol: {result}")
        
        # Get historical data
        data = db_client.timeseries.get_range(
            dataset=dataset,
            symbols=[ticker_for_data],
            schema='trades',
            start=start_date,
            end=end_date
        )
        
        if data is None:
            logger.warning(f"No data returned for {ticker}")
            return None
        
        # Convert to DataFrame
        df = data.to_df()
        if df.empty:
            logger.warning(f"Empty DataFrame returned for {ticker}")
            return None
        
        # Process the DataFrame
        df.sort_values("ts_event", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Normalize price
        df['price'] = df['price'] / 1e9
        df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns')
        df['volume'] = df['size']
        
        # Create OHLC bars based on timeframe
        if timeframe.isdigit():
            # Minutes-based resampling
            rule = f"{timeframe}T"
        elif timeframe == '1h':
            rule = '1H'
        elif timeframe == '4h':
            rule = '4H'
        elif timeframe == '1d':
            rule = '1D'
        
        # Resample to create OHLC bars
        ohlc_df = df.set_index('timestamp').resample(rule).agg({
            'price': 'ohlc',
            'volume': 'sum'
        })
        
        # Flatten the multi-index columns
        ohlc_df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Remove rows with NaN values
        ohlc_df = ohlc_df.dropna()
        
        logger.info(f"Created {len(ohlc_df)} {timeframe} bars for {ticker}")
        return ohlc_df
        
    except Exception as e:
        logger.error(f"Error getting historical data for {ticker}: {e}", exc_info=True)
        return None


class TickProducer:
    """Class to manage tick data production for multiple tickers."""
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0, db_api_key=None):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.tick_buffers = {}
        self.live_manager = None
        self.logger = logging.getLogger('TickProducer')
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

            ticker_for_data = get_active_exchange_symbol(ticker) if ticker.startswith("/") else ticker
            self.logger.info(f"Processing ticker: {ticker_for_data} with time frame: {time_frame}")
            max_period = max(int(period1), int(period2))
            historical_end_date = safe_historical_end.isoformat()
            historical_start_date = (safe_historical_end - timedelta(days=1)).isoformat()
            
            if ticker_for_data not in self.tick_buffers:
                tick_size = extract_tick_count(time_frame)

                buffer = TickDataBufferWithRedis(
                    ticker=ticker_for_data,
                    tick_size=tick_size,
                    redis_client=self.redis_client,
                    db_api_key=self.db_api_key,
                    max_period=max_period
                )
                self.tick_buffers[ticker_for_data] = buffer

                self.logger.info(f"Initialized TickDataBuffer for {ticker_for_data} with tick size {tick_size}")

            dataset = "GLBX.MDP3"

            # Warmup with historical data
            self.tick_buffers[ticker_for_data].warmup_with_historical_ticks(
                symbol=ticker_for_data,
                dataset=dataset,
                start=historical_start_date,
                end=historical_end_date,
                schema='trades'
            )

            bars = self.tick_buffers[ticker_for_data].processed_bars
            if bars:
                last_ts = pd.Timestamp(bars[-1]['timestamp'])
                replay_start_time = last_ts.value + 1  # nanoseconds
            else:
                replay_start_time = 0

            live_symbols_config[ticker_for_data] = {
                'dataset': dataset,
                'schema': 'trades',
                'start_time': replay_start_time  # int nanoseconds
            }

        return live_symbols_config

    async def start_live_feeds(self, live_symbols_config):
        """Start live data feeds"""
        if live_symbols_config:
            self.live_manager = DatabentoLiveManager(db_api_key=self.db_api_key)
            self.logger.info(f"Starting live feeds for: {list(live_symbols_config.keys())}")
            await self.live_manager.start_live_feeds(live_symbols_config, self.tick_buffers)

    def run(self, tickers_config):
        """Main run method"""
        self.logger.info("Starting Tick Producer...")
        live_symbols_config = self.setup_tick_buffers(tickers_config)
        asyncio.run(self.start_live_feeds(live_symbols_config))


def setup_and_run_tick_producer(tickers_config):
    """Setup and run the tick producer for the given tickers configuration.
    
    Args:
        tickers_config (dict): Dictionary of ticker configurations
    """
    db_api_key = os.getenv("DATABENTO_API_KEY")
    if not db_api_key:
        raise EnvironmentError("DATABENTO_API_KEY not set.")
    
    # Check for both naming conventions for Redis environment variables
    redis_host = os.getenv("REDIS_HOST", os.getenv("RDIS_HOST", "localhost"))
    redis_port = int(os.getenv("REDIS_PORT", os.getenv("RDIS_PORT", 6379)))
    redis_db = int(os.getenv("REDIS_DB", os.getenv("RDIS_DB", 0)))
    
    producer = TickProducer(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        db_api_key=db_api_key
    )
    
    producer.run(tickers_config)


def get_historical_and_live_data(tickers_config, logger=None):
    """Get historical data and start live data feeds for the given tickers configuration.
    
    This function combines the functionality of get_historical_data and TickProducer.
    It first gets historical data for all tickers, then starts live data feeds for
    tick-based timeframes.
    
    Args:
        tickers_config (dict): Dictionary of ticker configurations
        logger: Logger instance for logging
        
    Returns:
        dict: Dictionary of ticker -> DataFrame with historical data
    """
    if logger is None:
        logger = logging.getLogger('get_historical_and_live_data')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # Get historical data for all tickers
    historical_data = {}
    tick_based_tickers = []
    
    for ticker, config in tickers_config.items():
        timeframe = config[0]
        
        # Check if this is a tick-based timeframe
        if is_tick_timeframe(timeframe):
            tick_based_tickers.append(ticker)
            continue
        
        # Get historical data for time-based timeframes
        df = get_historical_data(ticker, timeframe, logger)
        if df is not None:
            historical_data[ticker] = df
    
    # Start live data feeds for tick-based timeframes in a separate thread
    if tick_based_tickers:
        import threading
        
        # Filter tickers_config to only include tick-based tickers
        tick_config = {ticker: config for ticker, config in tickers_config.items() 
                      if ticker in tick_based_tickers}
        
        # Start tick producer in a separate thread
        tick_thread = threading.Thread(
            target=setup_and_run_tick_producer,
            args=(tick_config,),
            daemon=True  # Make thread a daemon so it exits when main program exits
        )
        tick_thread.start()
        logger.info(f"Started live data feeds for tick-based tickers: {tick_based_tickers}")
    
    return historical_data


if __name__ == "__main__":
    # Example usage
    import sys
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        stream=sys.stdout)
    logger = logging.getLogger('main')
    
    # Example ticker configuration
    tickers_config = {
        # Ticker with time-based timeframe
        "SPY": ["5", "1", "TRUE", "1", "ema", "9", "ema", "21"],
        # Ticker with tick-based timeframe
        "/ES": ["100t", "1", "TRUE", "1", "ema", "9", "ema", "21"],
    }
    
    # Get historical data and start live feeds
    try:
        historical_data = get_historical_and_live_data(tickers_config, logger)
        
        # Print historical data summary
        for ticker, df in historical_data.items():
            logger.info(f"Historical data for {ticker}:")
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")
            logger.info(f"  First row: {df.iloc[0]}")
            logger.info(f"  Last row: {df.iloc[-1]}")
        
        # Keep the main thread running to allow the tick producer thread to continue
        logger.info("Press Ctrl+C to exit...")
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

    