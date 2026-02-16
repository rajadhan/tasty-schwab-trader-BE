import asyncio
import massive as ms
import json
import logging
import os
import pandas as pd
import redis
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pytz
import re
from events import BarEvent, OptionLegEvent

# Load environment variables FIRST before any internal imports
load_dotenv() 

from tick_buffer import (
    REDIS_CLIENT, 
    REST_CLIENT, 
    MassiveLiveManager, 
    TickDataBuffer, 
    TimeBasedBarBufferWithRedis
)
from utils import (
    get_dataset,
    get_symbol_for_data,
    is_tick_timeframe,
    get_schema,
    parse_strategy_params,
)


class TickProducer:
    def __init__(self):
        # Import here to avoid circular imports
        self.TickDataBuffer = TickDataBuffer  # Store the class, not an instance
        self.strategy = None
        self.tick_buffers = {}
        self.live_manager = MassiveLiveManager()
        self.spreads = {}
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
                max_period = 50
            elif self.strategy == 'supertrend':
                max_period = 50
            else:
                max_period = 1
            dataset = "DELAYED" # Default for Massive migration
            schema = "minute"
            safe_historical_end_time = datetime.now(timezone.utc)
            safe_historical_start_time = safe_historical_end_time - timedelta(days=5)
            
            # Massive date format (YYYY-MM-DD)
            historical_end_time = safe_historical_end_time.strftime("%Y-%m-%d")
            historical_start_time = safe_historical_start_time.strftime("%Y-%m-%d")
            ticker_for_data = get_symbol_for_data(ticker)

            self._setup_symbol_buffer(
                ticker,
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
        ticker,
        ticker_for_data,
        time_frame,
        max_period,
        historical_end_time,
        historical_start_time,
        live_symbols_config,
        dataset,
        schema,
    ):
        if ticker not in self.tick_buffers:
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
                    ticker=ticker,
                    strategy=self.strategy,
                    time_frame=time_frame,
                    max_period=max_period,
                    logger=self.logger,
                )
            self.tick_buffers[ticker] = buffer
            
            # Add bar closed callback for RWR
            if self.strategy == "zeroday":
                buffer.on_bar_closed = self.handle_bar_closed
                
            print(
                f"Initialized buffer for {ticker} of {self.strategy} with timeframe {time_frame}"
            )

        if is_tick_timeframe(time_frame):
            self.tick_buffers[ticker].warmup_with_historical_ticks(
                symbol=ticker_for_data,
                dataset=dataset,
                start=historical_start_time,
                end=historical_end_time,
                schema=schema,
            )
        else:
            self.tick_buffers[ticker].warmup_with_historical_timebars(
                symbol=ticker,
                dataset=dataset,
                start=historical_start_time,
                end=historical_end_time,
                base_schema=schema,
            )

        # Setup live feed
        bars = self.tick_buffers[ticker].processed_bars
        if bars:
            last_ts = pd.Timestamp(bars[-1]["timestamp"])
            replay_start_time = last_ts.value + 1
        else:
            replay_start_time = 0

        live_symbols_config[ticker] = {
            "dataset": dataset,
            "schema": get_schema(time_frame) if not is_tick_timeframe(time_frame) else 'trades',
            "start_time": replay_start_time,
        }

    
    def load_spreads(self, config_path="backtest_positions.json"):
        """Load live/test spreads from a JSON file."""
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.spreads = json.load(f)
                self.logger.info(f"Loaded live spreads for: {list(self.spreads.keys())}")
                
                # Automatically add all legs to monitored symbols
                for ticker, spreads in self.spreads.items():
                    for spread in spreads:
                        for leg_type in ['short', 'long']:
                            leg_symbol = spread[leg_type]['instrument']['symbol']
                            # Add with Massive prefix
                            massive_symbol = f"O:{leg_symbol.replace(' ', '')}"
                            self.add_dynamic_ticker(massive_symbol)
            else:
                self.spreads = {}
        except Exception as e:
            self.logger.error(f"Failed to load spreads: {e}")
            self.spreads = {}

    def handle_bar_closed(self, ticker, bar):
        """Callback invoked when a bar is finalized in any buffer."""
        # If this is an underlying ticker (e.g. SPY), generate and publish a BarEvent
        if ticker in self.spreads:
            self.emit_bar_event(ticker, bar)

    def emit_bar_event(self, ticker, u_bar):
        """Constructs and publishes a unified BarEvent (Market + Legs)."""
        try:
            # 1. Start with the underlying data
            event = BarEvent(
                ticker=ticker,
                timestamp=u_bar.get('timestamp') or datetime.now().isoformat(),
                underlying_price=u_bar['close']
            )
            
            # 2. Add legs from current spreads
            active_spreads = self.spreads.get(ticker, [])
            for spread in active_spreads:
                for leg_type in ['short', 'long']:
                    leg_data = spread[leg_type]
                    symbol = leg_data['instrument']['symbol']
                    massive_symbol = f"O:{symbol.replace(' ', '')}"
                    
                    # Fetch leg price from its own buffer if available
                    leg_price = 0.0
                    leg_vol = 0.0
                    if massive_symbol in self.tick_buffers:
                        l_bars = self.tick_buffers[massive_symbol].processed_bars
                        if l_bars:
                            leg_price = l_bars[-1]['close']
                            leg_vol = l_bars[-1].get('volume', 0)
                    
                    # Metadata (Strike, Expiry, Type)
                    strike, o_type, expiry_dt = self._parse_symbol(symbol)
                    
                    # Calculate expiry years
                    if expiry_dt:
                        expiry_dt_aware = pytz.timezone('US/Eastern').localize(
                            datetime.combine(expiry_dt.date(), datetime.min.time().replace(hour=16))
                        )
                        now_dt = datetime.now(pytz.UTC)
                        diff = (expiry_dt_aware - now_dt).total_seconds()
                        expiry_years = max(1 / (365 * 24 * 3600), diff / (365 * 24 * 3600))
                    else:
                        expiry_years = 0.01
                        
                    leg_event = OptionLegEvent(
                        symbol=symbol,
                        price=leg_price,
                        volume=leg_vol,
                        strike=strike,
                        type=o_type,
                        expiry_years=expiry_years,
                        qty=float(leg_data.get('quantity', 1)) * (-1 if leg_type == 'short' else 1)
                    )
                    event.legs.append(leg_event)
            
            # 3. Publish to Redis
            channel = f"tick_events:{ticker}"
            REDIS_CLIENT.publish(channel, json.dumps(event.to_dict()))
            self.logger.info(f"Published BarEvent for {ticker} with {len(event.legs)} legs.")
            
        except Exception as e:
            self.logger.error(f"Failed to emit BarEvent for {ticker}: {e}", exc_info=True)

    def _parse_symbol(self, symbol):
        """Internal helper to parse option symbol."""
        try:
            strike = float(symbol[-8:]) / 1000.0
            type_char = symbol[-9]
            o_type = 'put' if type_char == 'P' else 'call'
            match = re.search(r'(\d{6})', symbol)
            expiry_dt = datetime.strptime(match.group(1), "%y%m%d") if match else None
            return strike, o_type, expiry_dt
        except:
            return 0.0, 'call', None

    def add_dynamic_ticker(self, ticker, config=None):
        """Dynamically add a new ticker to the monitoring list"""
        if ticker in self.tick_buffers:
            return
            
        self.logger.info(f"Dynamically adding ticker: {ticker}")
        # Default config for RWR options if none provided
        if not config:
            config = {"strategy": self.strategy, "timeframe": "1"} # Default to 1m for RWR
            
        self.setup_tick_buffers({ticker: config})

    async def listen_for_commands(self):
        """Listen for commands from Redis (e.g., dynamic symbol additions)"""
        pubsub = REDIS_CLIENT.pubsub()
        pubsub.subscribe("tick_producer:commands")
        self.logger.info("Listening for commands on 'tick_producer:commands'...")
        
        while True:
            try:
                # We use a thread-safe approach for pubsub in async
                message = pubsub.get_message(ignore_subscribe_messages=True)
                if message:
                    data = json.loads(message['data'].decode('utf-8'))
                    cmd = data.get('command')
                    if cmd == 'subscribe':
                        ticker = data.get('ticker')
                        if ticker:
                            self.add_dynamic_ticker(ticker)
                
                await asyncio.sleep(1) # Don't hammer Redis
            except Exception as e:
                self.logger.error(f"Error in command listener: {e}")
                await asyncio.sleep(5)

    async def start_all(self, tickers_config):
        """Start both the live feeds and the command listener"""
        live_symbols_config = self.setup_tick_buffers(tickers_config)
        
        # Start the initial feeds
        feed_task = asyncio.create_task(
            self.live_manager.start_live_feeds(live_symbols_config, self.tick_buffers)
        )
        
        # Start the dynamic command listener
        command_task = asyncio.create_task(self.listen_for_commands())
        
        await asyncio.gather(feed_task, command_task)

    def run(self, tickers_config, strategy):
        """Main run method (entry point)"""
        self.strategy = strategy
        self.load_spreads() # Load spreads AFTER strategy is set
        self.logger.info("Starting Tick Producer with Dynamic Support...")
        try:
            asyncio.run(self.start_all(tickers_config))
        except KeyboardInterrupt:
            self.logger.info("Tick Producer shutting down...")


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
    return datetime.now(timezone.utc)

def get_historical_start_time(ticker, timeframe, end_time, logger):
    return end_time - timedelta(days=1)

if __name__ == "__main__":
    from utils import load_strategy_configs
    
    # Load ticker configuration from settings/ directory
    _, _, zeroday_config = load_strategy_configs()
    
    if not zeroday_config:
        print("Error: No tickers found in settings/zeroday_ticker_data.json")
    else:
        print(f"Loaded {len(zeroday_config)} tickers for RWR monitoring.")
        producer = TickProducer()
        # Strategy 'zeroday' is used for RWR monitoring
        producer.run(zeroday_config, strategy="zeroday")