import asyncio
import databento as db
import json
import logging
import os
import pandas as pd
import pytz
import redis
import threading
from datetime import datetime, timedelta, timezone
from utils import configure_logger, extract_tick_count


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


class TickDataBuffer:
    def __init__(self, ticker, tick_size, max_period, logger):
        self.ticker = ticker
        self.tick_size = tick_size
        self.redis_client = REDIS_CLIENT
        self.historical_client = HISTORICAL_CLIENT
        self.live_client = LIVE_CLIENT
        self.max_period = max_period
        self.logger = logger
        self.buffer = []
        self.processed_bars = []
        self.lock = threading.Lock()
        self.new_bar_event = threading.Event()
        self.historical_loaded = False
        self.live_session = None


    # Get historical data - ticks and then convert ticks into bars, then save bars into processed_bars
    def warmup_with_historical_ticks(self, symbol, dataset, start, end, schema='trades'):
        try:
            self.logger.info(f"Fetching historical tick data for warmup: {symbol} [{start} to {end}]")
            print(f"Fetching historical tick data for warmup: {symbol} [{start} to {end}]")
            # Fetch raw trade ticks
            result = self.historical_client.symbology.resolve(
                dataset=dataset,
                symbols=[symbol],
                stype_in="raw_symbol",
                stype_out="instrument_id",
                start_date=start,
                end_date=end,
            )
            print(f"Resolved symbol: {result}")

            try:
                data = self.historical_client.timeseries.get_range(
                    dataset=dataset,
                    symbols=[symbol],
                    schema=schema,
                    start=start,
                    end=end
                )
            except Exception as e:
                print(f"Error in get historical data {symbol}: {e}")
                data = pd.DataFrame()
            df = data.to_df()
            print(f"Data fetched for {symbol}: {df}")
            if df.empty:
                self.logger.warning(f"No historical data returned for {symbol}")
                return

            df.sort_values("ts_event", inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Normalize price and construct tick dicts
            df['price'] = df['price']
            df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns')
            df['volume'] = df['size']

            ticks = df[['timestamp', 'price', 'volume']].to_dict(orient='records')

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
            # self.logger.info(f"Historical warmup completed for {symbol}: {len(self.processed_bars)} bars created.")

        except Exception as e:
            self.logger.error(f"Databento historical warmup failed for {symbol}: {e}", exc_info=True)


    async def start_live_subscription(self, symbol, dataset, schema, start_time=0):
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
                start=start_time
            )
            
            self.logger.info(f"Successfully subscribed to live data for {symbol}")
            # Start the session to begin receiving data
            await self.live_session.start()
            
            # Start consuming live data
            async for record in self.live_session:
                print("real time record", record)
                try:
                    # Only process trade messages (rtype == "Trade" or check type)
                    if hasattr(record, "price") and hasattr(record, "size"):
                        tick_data = {
                            'timestamp': pd.to_datetime(record.ts_event, unit='ns'),
                            'price': record.price,
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
        

    def get_dataframe(self, min_bars):
        with self.lock:
            if len(self.processed_bars) < min_bars:
                return None
            df = pd.DataFrame(self.processed_bars[-min_bars:])
            print("df - 50", df)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print("df timestamp", df['timestamp'])
            df.set_index('timestamp', inplace=True)
            print("df", df)
            return df

    def wait_for_new_bar(self, timeout=None):
        return self.new_bar_event.wait(timeout)

    def consume_new_bar_signal(self):
        self.new_bar_event.clear()


class DatabentoLiveManager:
    """Manages live Databento subscriptions for multiple symbols"""
    
    def __init__(self):
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


