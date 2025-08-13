from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from utils import (
    extract_tick_count,
    get_active_exchange_symbol,
    is_tick_timeframe,
    load_json,
    ticker_data_path_for_strategy,
)
import databento as db
import os
import pandas as pd
import redis
import threading


load_dotenv()
DB_API_KEY = os.getenv("DATABENTO_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_DB = os.getenv("REDIS_DB")


class TickProducer:
    def __init__(self, logger):
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        self.historical_client = db.Historical(DB_API_KEY)
        self.live_client = db.Live(key=DB_API_KEY)
        self.logger = logger
        self.tick_buffers = {}

    def setup_tick_buffers(self, tickers_config):
        for ticker, config in tickers_config.items():
            dataset = get_dataset(ticker)
            safe_historical_end = get_historical_end_time(ticker, dataset, self.logger)
            safe_historical_start = get_historical_start_time(
                ticker, time_frame, safe_historical_end
            )
            time_frame = config[0]
            if is_tick_timeframe(time_frame):
                """tick_based timeframe"""
                if ticker.startwith("/"):
                    """Future"""
                    ticker_for_data = get_active_exchange_symbol(ticker)
                    print(
                        f"Processing ticker: {ticker_for_data} with time frame: {time_frame}"
                    )
                    if ticker_for_data not in self.tick_buffers:
                        tick_size = extract_tick_count(time_frame)
                        print("tick_size", ticker_for_data, tick_size, DB_API_KEY)
                        buffer = TickDataBufferWithRedis(
                            ticker=ticker_for_data,
                            tick_size=tick_size,
                            redis_client=self.redis_client,
                            historical_client=self.historical_client,
                            live_client=self.live_client,
                            logger=self.logger,
                        )
                        print(
                            f"Initialized TickDataBuffer for {ticker_for_data} with tick size {tick_size}"
                        )

                    self.tick_buffers[ticker_for_data].warmup_with_historical_ticks(
                        symbol=ticker_for_data,
                        dataset=dataset,
                        start=safe_historical_start,
                        end=safe_historical_end,
                        schema="trades",
                    )
                else:
                    """Stock"""
                    ticker_for_data = ticker
                    print(
                        f"Processing ticker: {ticker_for_data} with time frame: {time_frame}"
                    )
                    start_time = get_historical_start_time(
                        ticker, time_frame, safe_historical_end
                    )
                    self.get_tick_based_data(
                        dataset,
                        start_time,
                        safe_historical_end,
                        ticker_for_data,
                        time_frame,
                        ticker,
                    )

    def run(self, tickers_config):
        self.logger.info("Starting Tick Producer...")
        live_symbols_config = self.setup_tick_buffers(tickers_config)
        # asyncio.run(self.start_live_feeds(live_symbols_config))


class TickDataBuffer:
    def __init__(
        self, ticker, tick_size, logger, historical_client, live_client, redis_client
    ):
        self.ticker = ticker
        self.tick_size = tick_size
        self.buffer = []
        self.processed_bars = []
        self.lock = threading.Lock()
        self.logger = logger
        self.new_bar_event = threading.Event()
        self.historical_loaded = False
        self.historical_client = historical_client
        self.live_client = live_client
        self.live_session = None
        self.redis_client = redis_clinet

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


    def warmup_with_historical_ticks(
        self, symbol, dataset, start, end, schema="trades"
    ):
        try:
            self.logger.info(
                f"Fetching historical tick data for warmup: {symbol} [{start} to {end}]"
            )
            print(
                f"Fetching historical tick data for warmup: {symbol} [{start} to {end}]"
            )
            # Fetch raw trade ticks
            result = self.db_client.symbology.resolve(
                dataset=dataset,
                symbols=[symbol],
                stype_in="raw_symbol",
                stype_out="instrument_id",
                start_date=start,
                end_date=end,
            )
            print(f"Resolved symbol: {result}")
            data = self.db_client.timeseries.get_range(
                dataset=dataset, symbols=[symbol], schema=schema, start=start, end=end
            )

            print(f"Data fetched for {symbol}: {data}")
            df = data.to_df()
            if df.empty:
                self.logger.warning(f"No historical data returned for {symbol}")
                return

            # Only print the first row if DataFrame is not empty
            print(df.iloc[0].to_json(indent=4))
            if df.empty:
                self.logger.warning(f"No historical data returned for {symbol}")
                return

            df.sort_values("ts_event", inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Normalize price and construct tick dicts
            df["price"] = df["price"] / 1e9
            df["timestamp"] = pd.to_datetime(df["ts_event"], unit="ns")
            df["volume"] = df["size"]

            ticks = df[["timestamp", "price", "volume"]].to_dict(orient="records")

            self.logger.info(f"Fetched and parsed {len(ticks)} ticks for {symbol}")

            # Build bars from tick chunks
            bar_ticks = []
            for tick in ticks:
                bar_ticks.append(tick)
                if len(bar_ticks) >= self.tick_size:
                    self.buffer = bar_ticks
                    bar = self._create_bar_from_ticks()
                    self.processed_bars.append(bar)
                    bar_ticks = []

            self.buffer = []  # clear residuals
            self.historical_loaded = True
            self.logger.info(
                f"Historical warmup completed for {symbol}: {len(self.processed_bars)} bars created."
            )

        except Exception as e:
            self.logger.error(
                f"Databento historical warmup failed for {symbol}: {e}", exc_info=True
            )


    async def start_live_subscription(self, symbol, dataset, schema='trades', start_time=0):
        """Start live tick data subscription using Databento Live API with optional replay"""
        try:
            self.logger.info(f"Starting live tick subscription for {symbol} on dataset {dataset}")
            if start_time:
                self.logger.info(f"Using intraday replay starting from {start_time}")
            
            # Create live session
            self.live_session = self.live_client
            
            # Subscribe to the symbol with optional start time for intraday replay
            self.live_session.subscribe(
                dataset=dataset,
                schema=schema,
                symbols=[symbol],
                stype_in="raw_symbol",
                start=0  # This enables intraday replay from specified time
            )
            
            self.logger.info(f"Successfully subscribed to live data for {symbol}")
            
            # Start the session to begin receiving data
            # self.live_session.start()
            
            # Start consuming live data
            async for record in self.live_session:
                try:
                    # Only process trade messages (rtype == "Trade" or check type)
                    if hasattr(record, "price") and hasattr(record, "size"):
                        tick_data = {
                            'timestamp': pd.to_datetime(record.ts_event, unit='ns'),
                            'price': record.price / 1e9,
                            'volume': record.size,
                            'symbol': symbol
                        }
                        self.add_tick(tick_data)
                    else:
                        # Optionally log or skip non-trade messages
                        self.logger.debug(f"Ignored non-trade message for {symbol}: {record}")
                except Exception as e:
                    self.logger.error(f"Error processing live tick for {symbol}: {e}")
                    
            self.logger.info(f"Live subscription ended for {symbol}")      
        except Exception as e:
            self.logger.error(f"Error in live subscription for {symbol}: {e}", exc_info=True)
        finally:
            if self.live_session:
                await self.live_session.stop()

    def stop_live_subscription(self):
        """Stop the live subscription"""
        if self.live_session:
            try:
                asyncio.create_task(self.live_session.stop())
                self.logger.info(f"Live subscription stopped for {self.ticker}")
            except Exception as e:
                self.logger.error(f"Error stopping live subscription for {self.ticker}: {e}")

    def add_tick(self, tick_data):
        with self.lock:
            self.buffer.append(tick_data)
            if len(self.buffer) >= self.tick_size:
                bar = self._create_bar_from_ticks()
                self.processed_bars.append(bar)
                self.logger.info(
                    f"Created new tick‐bar for {self.ticker}: "
                    f"open={bar['open']}, high={bar['high']}, low={bar['low']}, close={bar['close']}, volume={bar['volume']}"
                )
                self.buffer = []
                self.new_bar_event.set()

    def _create_bar_from_ticks(self):
        if not self.buffer:
            return None
        prices = [tick['price'] for tick in self.buffer]
        volumes = [tick['volume'] for tick in self.buffer]
        timestamps = [tick['timestamp'] for tick in self.buffer]
        return {
            'timestamp': timestamps[-1],
            'open': prices[0],
            'high': max(prices),
            'low': min(prices),
            'close': prices[-1],
            'volume': sum(volumes)
        }

    def get_dataframe(self, min_bars=5):
        with self.lock:
            if len(self.processed_bars) < min_bars:
                return None
            df = pd.DataFrame(self.processed_bars[-min_bars:])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df

    def wait_for_new_bar(self, timeout=None):
        return self.new_bar_event.wait(timeout)

    def consume_new_bar_signal(self):
        self.new_bar_event.clear()


class TickDataBufferWithRedis(TickDataBuffer):
    """Extended TickDataBuffer that publishes bars to Redis"""
    def __init__(
        self, ticker, tick_size, redis_client, historical_client, live_client, logger
    ):
        super().__init__(
            ticker, tick_size, logger, historical_client, live_client, redis_client
        )


class DatabentoLiveManager:
    """Manages live Databento subscriptions for multiple symbols"""
    
    def __init__(self, db_api_key=None):
        self.db_api_key = db_api_key
        self.live_tasks = {}
        self.logger = logging.getLogger('DatabentoLiveManager')
        
    async def start_live_feeds(self, symbols_config, tick_buffers):
        """
        Start live feeds for multiple symbols
        symbols_config: dict like {'ESM2': {'dataset': 'GLBX.MDP3', 'schema': 'trades', 'start_time': timestamp}}
        tick_buffers: dict of ticker -> TickDataBuffer instances
        """
        tasks = []
        
        for symbol, config in symbols_config.items():
            if symbol in tick_buffers:
                dataset = config.get('dataset', 'GLBX.MDP3')
                schema = config.get('schema', 'trades')
                start_time = config.get('start_time', 0) # Default to 0 for full replay
                
                # Create task for this symbol's live feed
                task = asyncio.create_task(
                    tick_buffers[symbol].start_live_subscription(symbol, dataset, schema, start_time)
                )
                self.live_tasks[symbol] = task
                tasks.append(task)
                
                self.logger.info(f"Started live feed task for {symbol} with start_time: {start_time}")
        
        if tasks:
            # Run all live feeds concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def stop_all_feeds(self, tick_buffers):
        """Stop all live feeds"""
        for symbol, task in self.live_tasks.items():
            if not task.done():
                task.cancel()
                self.logger.info(f"Cancelled live feed for {symbol}")
        
        # Also stop individual buffer subscriptions
        for symbol, buffer in tick_buffers.items():
            buffer.stop_live_subscription()




def get_dataset(ticker):
    if ticker.startswith("/"):  # for CME futures
        return "GLBX.MDP3"
    else:  # for stocks
        return "XNAS.ITCH"


def get_historical_end_time(ticker, dataset, logger) -> datetime:
    """Get the available end time for a given dataset and schema using Databento metadata API."""
    if ticker.startswith("/"):
        try:
            metadata = self.historical_client.metadata.get_dataset_range(
                dataset=dataset
            )
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


def get_historical_and_live_data(ticker, timeframe, logger, strategy):
    ticker_data_path = ticker_data_path_for_strategy(strategy)
    tickers_config = load_json(ticker_data_path)
    producer = TickProducer(logger)
    producer.run(tickers_config)
