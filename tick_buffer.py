import asyncio
import databento as db
import json
import logging
import os
import pandas as pd
import pytz
import redis
import threading
from brokers.schwab import historical_data
from datetime import timedelta, datetime
from utils import extract_tick_count


# Global clients - created once and reused
DB_API_KEY = os.getenv("DATABENTO_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Global Redis client
REDIS_CLIENT = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# Global Databento clients
HISTORICAL_CLIENT = db.Historical(DB_API_KEY)


class TickDataBuffer:
    def __init__(self, ticker, strategy, time_frame, max_period, logger):
        self.ticker = ticker
        self.strategy = strategy
        self.time_frame = time_frame
        self.redis_client = REDIS_CLIENT
        self.historical_client = HISTORICAL_CLIENT
        self.max_period = max_period
        self.logger = logger
        self.buffer = []
        self.processed_bars = []
        self.lock = threading.Lock()
        self.new_bar_event = threading.Event()
        self.historical_loaded = False
        self.live_session = None
        self.tick_size = extract_tick_count(self.time_frame)


    # Get historical data - ticks and then convert ticks into bars, then save bars into processed_bars
    def warmup_with_historical_ticks(self, symbol, dataset, start, end, schema):
        try:
            self.logger.info(f"Fetching historical tick data for warmup: {symbol} [{start} to {end}]")
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
                    stype_in="raw_symbol",
                    schema=schema,
                    start=start,
                    end=end
                )
            except Exception as e:
                print(f"Error in get historical data {symbol}: {e}")
                data = pd.DataFrame()
            df = data.to_df()
            df.to_csv(f"logs/{self.strategy}/schwab_mes.csv")
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

            self.logger.info(f"Fetched and parsed {len(ticks)} ticks for {symbol} of {self.strategy}")

            bar_ticks = []
            for tick in ticks:
                bar_ticks.append(tick)
                tick_size = self.tick_size
                if tick_size > 0 and len(bar_ticks) >= tick_size:
                    self.buffer = bar_ticks
                    bar = self._create_bar_from_ticks()
                    self.processed_bars.append(bar)
                    bar_ticks = []

            self.buffer = []  # clear residuals
            self.historical_loaded = True
            self.logger.info(f"Historical warmup completed for {symbol}: {len(self.processed_bars)} bars created.")

        except Exception as e:
            self.logger.error(f"Databento historical warmup failed for {symbol}: {e}", exc_info=True)


    async def start_live_subscription(self, symbol, dataset, schema, start_time=0):
        """Start live tick data subscription using Databento Live API with optional replay"""
        try:
            self.logger.info(f"Starting live tick subscription for {symbol} on dataset {dataset}")
            if start_time:
                self.logger.info(f"Using intraday replay starting from {start_time}")
            # Create live session
            live_client = db.Live(key=DB_API_KEY)
            self.live_session = live_client
            # Subscribe to the symbol with optional start time for intraday replay
            self.live_session.subscribe(
                dataset=dataset,
                schema=schema,
                symbols=[symbol],
                stype_in="raw_symbol",
                start=start_time
            )
            
            self.logger.info(f"Successfully subscribed to live data for {symbol}")
            
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


class TimeBasedBarBufferWithRedis:
    """Aggregate trades into time-based OHLCV bars with optional resampling and persist to Redis."""

    def __init__(self, ticker, strategy, time_frame, max_period, logger):
        self.ticker = ticker
        self.strategy = strategy
        self.time_frame = time_frame
        self.redis_client = REDIS_CLIENT
        self.historical_client = HISTORICAL_CLIENT
        self.live_client = db.Live(key=DB_API_KEY)
        self.max_period = max_period
        self.logger = logger
        self.processed_bars = []
        self.lock = threading.Lock()
        self.new_bar_event = threading.Event()
        self.historical_loaded = False
        self.live_session = None
        self.resample_rule = self._determine_resample_rule(time_frame)
        self._agg_bucket_start = None
        self._agg_bar = None

    def _determine_resample_rule(self, timeframe):
        mapping = {
            "2": "2min",
            "5": "5min",
            "15": "15min",
            "30": "30min",
            "4h": "4h",
        }
        # Base resolutions (1m, 1h, 1d) -> no resample
        return mapping.get(timeframe, None)

    def _bars_to_df(self, data):
        df = data.to_df()
        print("df=================", df)
        if df.empty:
            return df
        # Expect OHLCV schema: ts_event, open, high, low, close, volume
        if 'ts_event' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns', utc=True)
        else:
            df['timestamp'] = pd.to_datetime(df.index, utc=True)
        if 'volume' not in df.columns and 'size' in df.columns:
            df['volume'] = df['size']
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            self.logger.warning(f"Missing columns in OHLCV data: {missing}")
            return pd.DataFrame()
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        # Normalize to display units (Databento OHLCV prices are in nanounits)
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        return df

    def _resample_bars(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        if bars_df.empty:
            return bars_df
        if self.resample_rule is None:
            return bars_df.set_index('timestamp').sort_index()
        s = bars_df.set_index('timestamp').sort_index()
        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }
        return s.resample(self.resample_rule, label='right', closed='right').agg(agg).dropna()

    def _save_bar_to_redis(self, bar_ts: pd.Timestamp, bar_row: pd.Series):
        # Normalize timestamp from row or fallback to bar_ts
        row_dt = bar_row.get("datetime") if isinstance(bar_row, (pd.Series, dict)) else None
        try:
            ts = pd.to_datetime(row_dt) if row_dt is not None else pd.to_datetime(bar_ts)
        except Exception:
            ts = pd.to_datetime(bar_ts)
        # Ensure Python datetime for JSON serialization
        ts_py = pd.Timestamp(ts).to_pydatetime()
        payload = {
            "symbol": self.ticker,
            "timestamp": ts_py.isoformat(),
            "open": float(bar_row['open']),
            "high": float(bar_row['high']),
            "low": float(bar_row['low']),
            "close": float(bar_row['close']),
        }
        with self.lock:
            self.processed_bars.append({
                'timestamp': ts_py,
                'open': payload['open'],
                'high': payload['high'],
                'low': payload['low'],
                'close': payload['close'],
            })
        score = int(pd.Timestamp(ts_py).timestamp())
        data_str = json.dumps(payload)
        zset_key_strategy = f"bars_history:{self.strategy}{self.ticker}"
        zset_key_plain = f"bars_history:{self.ticker}"

        try:
            self.redis_client.zadd(zset_key_strategy, {data_str: score})
            self.redis_client.zadd(zset_key_plain, {data_str: score})
            self.redis_client.zremrangebyrank(zset_key_strategy, 0, -1000)
            self.redis_client.zremrangebyrank(zset_key_plain, 0, -1000)
        except Exception as e:
            self.logger.error(f"Failed to write OHLCV bar for {self.ticker}: {e}")

    def warmup_with_historical_timebars(self, symbol, dataset, start, end, base_schema):
        try:
            self.logger.info(f"Fetching historical OHLCV base for warmup: {symbol} [{start} to {end}] schema={base_schema}")
            # data = self.historical_client.timeseries.get_range(
            #     dataset=dataset,
            #     symbols=[symbol],
            #     schema=base_schema,
            #     start=start,
            #     end=end,
            # )
            data = historical_data(symbol, self.time_frame, self.logger)
            # bars_df = self._bars_to_df(data)
            # if bars_df.empty:
            #     self.logger.warning(f"No historical OHLCV returned for {symbol}")
            #     return
            # resampled = self._resample_bars(bars_df)
            # self.logger.info(f"Resampled bars for {symbol}: {resampled}")
            for ts, row in data.iterrows():
                self._save_bar_to_redis(ts, row)
            self.historical_loaded = True
            self.logger.info(f"Historical time-based warmup completed for {symbol}: {len(self.processed_bars)} bars created.")
        except Exception as e:
            self.logger.error(f"Historical time-based warmup failed for {symbol}: {e}", exc_info=True)

    def _finalize_and_emit_current_agg(self):
        if self._agg_bucket_start is None or self._agg_bar is None:
            return
        ts = self._agg_bucket_start
        row = pd.Series(self._agg_bar)
        self._save_bar_to_redis(ts, row)
        self.logger.info(
            f"Created new aggregated live bar for {self.ticker} ({self.time_frame}): "
            f"open={self._agg_bar['open']}, high={self._agg_bar['high']}, low={self._agg_bar['low']}, close={self._agg_bar['close']}, volume={self._agg_bar['volume']}"
        )
        self._agg_bucket_start = None
        self._agg_bar = None
        self.new_bar_event.set()

    def _update_aggregator_with_base_bar(self, base_ts: pd.Timestamp, base_bar: dict):
        bucket_start = base_ts.tz_convert('UTC') if base_ts.tzinfo else base_ts.tz_localize('UTC')
        if self.resample_rule:
            bucket_start = bucket_start.floor(self.resample_rule)
        if self._agg_bucket_start is None:
            self._agg_bucket_start = bucket_start
            self._agg_bar = dict(base_bar)
            return
        if bucket_start != self._agg_bucket_start:
            self._finalize_and_emit_current_agg()
            self._agg_bucket_start = bucket_start
            self._agg_bar = dict(base_bar)
            return
        self._agg_bar['high'] = max(self._agg_bar['high'], base_bar['high'])
        self._agg_bar['low'] = min(self._agg_bar['low'], base_bar['low'])
        self._agg_bar['close'] = base_bar['close']
        self._agg_bar['volume'] = (self._agg_bar.get('volume') or 0) + (base_bar.get('volume') or 0)

    async def start_live_subscription(self, symbol, dataset, schema, start_time=0):
        try:
            self.logger.info(f"Starting live OHLCV base subscription for {symbol} on {dataset} schema={schema}")
            live_client = db.Live(key=DB_API_KEY)
            self.live_session = live_client
            self.live_session.subscribe(
                dataset=dataset,
                schema=schema,
                symbols=[symbol],
                stype_in="raw_symbol",
                start=start_time,
            )
            async for record in self.live_session:
                try:
                    # Expect OHLCV base fields
                    if all(hasattr(record, attr) for attr in ["open", "high", "low", "close"]) and hasattr(record, "ts_event"):
                        ts = pd.to_datetime(record.ts_event, unit='ns', utc=True)
                        base_bar = {
                            'open': float(record.open) / 1e9,
                            'high': float(record.high) / 1e9,
                            'low': float(record.low) / 1e9,
                            'close': float(record.close) / 1e9,
                            'volume': float(getattr(record, 'volume', 0)),
                        }
                        if self.resample_rule is None:
                            self._save_bar_to_redis(ts, pd.Series(base_bar))
                            self.new_bar_event.set()
                            self.logger.info(
                                f"Created new live bar for {self.ticker}: "
                                f"record={record.ts_event}, {ts} open={base_bar['open']}, high={base_bar['high']}, low={base_bar['low']}, close={base_bar['close']}, volume={base_bar['volume']}"
                            )
                        else:
                            self._update_aggregator_with_base_bar(ts, base_bar)
                    else:
                        self.logger.debug(f"Ignored non-OHLCV message for {symbol}: {record}")
                except Exception as e:
                    self.logger.error(f"Error processing OHLCV record for {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"Error in time-based live subscription for {symbol}: {e}", exc_info=True)
        finally:
            self._finalize_and_emit_current_agg()
            if self.live_session:
                await self.live_session.stop()
