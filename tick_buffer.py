import asyncio
import massive as ms
import json
import logging
import os
import pandas as pd
import pytz
import redis
import threading
from brokers.schwab import historical_data, get_quotes
from datetime import timedelta, datetime
from utils import extract_tick_count


# Global clients - created once and reused
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Global Redis client
REDIS_CLIENT = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# Global Massive REST client
REST_CLIENT = ms.RESTClient(api_key=MASSIVE_API_KEY) if MASSIVE_API_KEY else None


class TickDataBuffer:
    def __init__(self, ticker, strategy, time_frame, max_period, logger):
        self.ticker = ticker
        self.strategy = strategy
        self.time_frame = time_frame
        self.redis_client = REDIS_CLIENT
        self.max_period = max_period
        self.logger = logger
        self.buffer = []
        self.processed_bars = []
        self.lock = threading.Lock()
        self.new_bar_event = threading.Event()
        self.historical_loaded = False
        self.tick_size = extract_tick_count(self.time_frame) if self.time_frame else 1


    def warmup_with_historical_ticks(self, symbol, dataset, start, end, schema):
        """Warmup using Massive Aggregates"""
        try:
            self.logger.info(f"Fetching historical tick data for warmup: {symbol} [{start} to {end}]")
            
            # Use 1-minute aggregates for warmup
            aggs = REST_CLIENT.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="minute",
                from_=start,
                to=end,
                limit=50000
            )

            processed_count = 0
            for agg in aggs:
                bar = {
                    'timestamp': pd.to_datetime(agg.timestamp, unit='ms', utc=True),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                }
                self.processed_bars.append(bar)
                processed_count += 1

            self.historical_loaded = True
            self.logger.info(f"Historical warmup completed for {symbol}: {processed_count} bars created.")

        except Exception as e:
            self.logger.error(f"Massive historical warmup failed for {symbol}: {e}", exc_info=True)


    def stop_live_subscription(self):
        """Placeholder for backward compatibility"""
        pass

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
        
        # If greeks are present in the last tick, include them in the bar for the engine to use
        last_tick = self.buffer[-1]
        
        bar = {
            'timestamp': timestamps[-1],
            'open': prices[0],
            'high': max(prices),
            'low': min(prices),
            'close': prices[-1],
            'volume': sum(volumes)
        }
        
        if 'delta' in last_tick:
            bar.update({
                'delta': last_tick['delta'],
                'gamma': last_tick['gamma'],
                'theta': last_tick['theta']
            })
            
        return bar


    def get_dataframe(self, min_bars):
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


class MassiveLiveManager:
    """Manages live Massive.com (Polygon.io) subscriptions for multiple symbols"""
    
    def __init__(self):
        self.ws_client = None
        self.logger = logging.getLogger('MassiveLiveManager')
        self.api_key = MASSIVE_API_KEY
        self.running = False
        
    async def start_live_feeds(self, symbols_config, tick_buffers):
        """
        Start live feeds for multiple symbols using Massive WebSocket & Polling
        """
        self.running = True
        
        # 1. Polling Task for Options (Greeks)
        async def poll_snapshots():
            while self.running:
                try:
                    for ticker, buffer in tick_buffers.items():
                        if ticker.startswith("O:"):
                            # Get snapshot for Greeks
                            snapshot = REST_CLIENT.get_option_contract_snapshot(ticker)
                            if snapshot and hasattr(snapshot, "greeks"):
                                g = snapshot.greeks
                                tick_data = {
                                    'timestamp': datetime.now(pytz.UTC),
                                    'price': snapshot.day.last_price if snapshot.day else 0,
                                    'volume': snapshot.day.volume if snapshot.day else 0,
                                    'delta': getattr(g, 'delta', 0),
                                    'gamma': getattr(g, 'gamma', 0),
                                    'theta': getattr(g, 'theta', 0),
                                    'symbol': ticker
                                }
                                buffer.add_tick(tick_data)
                except Exception as e:
                    self.logger.error(f"Snapshot polling error: {e}")
                await asyncio.sleep(60)

        # 2. WebSocket Task for Trades/Quotes (Equities/Indices if needed)
        async def ws_consumer():
            # Future: Implement WebSocket trades for 0DTE underlyings
            pass

        await asyncio.gather(poll_snapshots(), ws_consumer())

    def stop_all_feeds(self, tick_buffers):
        self.running = False


class TimeBasedBarBufferWithRedis:
    """Aggregate trades into time-based OHLCV bars using Massive data."""

    def __init__(self, ticker, strategy, time_frame, max_period, logger):
        self.ticker = ticker
        self.strategy = strategy
        self.time_frame = time_frame
        self.redis_client = REDIS_CLIENT
        self.max_period = max_period
        self.logger = logger
        self.processed_bars = []
        self.buffer = []
        self.lock = threading.Lock()
        self.new_bar_event = threading.Event()
        self.historical_loaded = False
        self._agg_bucket_start = None
        self._agg_bar = {}

    def _timeframe_freq(self):
        tf = (self.time_frame or "1min").lower()
        mapping = {
            "1": "1min", "2": "2min", "5": "5min", "15": "15min",
            "30": "30min", "1h": "1h", "4h": "4h", "1d": "1d",
        }
        return mapping.get(tf, "1min")

    def _save_bar_to_redis(self, bar_ts: pd.Timestamp, bar_row: pd.Series):
        ts_py = pd.Timestamp(bar_ts).to_pydatetime()
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
            self.redis_client.zremrangebyscore(zset_key_strategy, score, score)
            self.redis_client.zremrangebyscore(zset_key_plain, score, score)
            self.redis_client.zadd(zset_key_strategy, {data_str: score})
            self.redis_client.zadd(zset_key_plain, {data_str: score})
            self.redis_client.zremrangebyrank(zset_key_strategy, 0, -1000)
            self.redis_client.zremrangebyrank(zset_key_plain, 0, -1000)
        except Exception as e:
            self.logger.error(f"Failed to write OHLCV bar for {self.ticker}: {e}")

    def warmup_with_historical_timebars(self, symbol, dataset, start, end, base_schema):
        try:
            self.logger.info(f"Fetching historical OHLCV via Schwab fall-back for {symbol}")
            data = historical_data(symbol, self.time_frame, self.logger)
            for ts, row in data.iterrows():
                self._save_bar_to_redis(ts, row)
            self.historical_loaded = True
        except Exception as e:
            self.logger.error(f"Historical warmup failed for {symbol}: {e}")
