import massive as ms
import os
import json
import time
import argparse
import pandas as pd
import pytz
from datetime import datetime, timedelta
from dotenv import load_dotenv
from events import BarEvent, OptionLegEvent
from tick_buffer import REDIS_CLIENT
import logging

load_dotenv()
REST_CLIENT = ms.RESTClient(api_key=os.getenv("MASSIVE_API_KEY"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReplayProducer")

class ReplayProducer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.underlying_bars = []
        self.leg_bars = {} # symbol -> list of bars
        self.positions = []

    def load_positions(self, config_path):
        with open(config_path, 'r') as f:
            all_pos = json.load(f)
            self.positions = all_pos.get(self.ticker, [])
        logger.info(f"Loaded {len(self.positions)} positions for {self.ticker}")

    def fetch_historical_data(self, date_str, cache_file=None):
        if cache_file and os.path.exists(cache_file):
            logger.info(f"Loading data from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                self.underlying_bars = cached['underlying']
                self.leg_bars = cached['legs']
            
            # Verify if we have all legs for current positions
            missing_legs = False
            for pos in self.positions:
                for leg_type in ['short', 'long']:
                    symbol = pos[leg_type]['instrument']['symbol']
                    if symbol not in self.leg_bars:
                        missing_legs = True
                        break
            
            if not missing_legs:
                logger.info("All legs found in cache.")
                return
            else:
                logger.info("Some legs missing from cache, fetching from API...")

        # 1. Fetch Underlying Bars (if not loaded from cache)
        if not self.underlying_bars:
            logger.info(f"Fetching underlying bars for {self.ticker} on {date_str}...")
            u_aggs = REST_CLIENT.list_aggs(
                ticker=self.ticker,
                multiplier=1,
                timespan="minute",
                from_=date_str,
                to=date_str,
                limit=5000
            )
            self.underlying_bars = [
                {'ts': a.timestamp, 'close': a.close} for a in u_aggs
            ]
            logger.info(f"Fetched {len(self.underlying_bars)} underlying bars.")

        # 2. Fetch Option Leg Bars
        for pos in self.positions:
            for leg_type in ['short', 'long']:
                symbol = pos[leg_type]['instrument']['symbol']
                if symbol in self.leg_bars:
                    continue # Skip if already in cache
                
                massive_symbol = f"O:{symbol.replace(' ', '')}"
                logger.info(f"Fetching bars for leg {massive_symbol}...")
                try:
                    l_aggs = REST_CLIENT.list_aggs(
                        ticker=massive_symbol,
                        multiplier=1,
                        timespan="minute",
                        from_=date_str,
                        to=date_str,
                        limit=5000
                    )
                    # Use string keys for timestamps to be consistent with JSON cache
                    self.leg_bars[symbol] = {
                        str(a.timestamp): {'close': a.close, 'volume': a.volume} for a in l_aggs
                    }
                    logger.info(f"Fetched {len(self.leg_bars[symbol])} bars for {symbol}.")
                except Exception as e:
                    logger.error(f"Failed to fetch {massive_symbol}: {e}")
                    self.leg_bars[symbol] = {}

        if cache_file:
            logger.info(f"Saving data to cache: {cache_file}")
            with open(cache_file, 'w') as f:
                json.dump({
                    'underlying': self.underlying_bars,
                    'legs': self.leg_bars
                }, f)
        
        # Diagnostic Summary
        logger.info("--- Data Coverage Summary (EST) ---")
        eastern = pytz.timezone('US/Eastern')
        if self.underlying_bars:
            u_start = datetime.fromtimestamp(self.underlying_bars[0]['ts']/1000, tz=pytz.UTC).astimezone(eastern).strftime('%H:%M:%S')
            u_end = datetime.fromtimestamp(self.underlying_bars[-1]['ts']/1000, tz=pytz.UTC).astimezone(eastern).strftime('%H:%M:%S')
            logger.info(f"Underlying: {len(self.underlying_bars)} bars from {u_start} to {u_end} EST")
        
        for symbol, bars in self.leg_bars.items():
            if not bars:
                logger.warning(f"Leg {symbol}: EMPTY DATA")
                continue
            
            ts_sorted = sorted([int(ts) for ts in bars.keys()])
            l_start = datetime.fromtimestamp(ts_sorted[0]/1000, tz=pytz.UTC).astimezone(eastern).strftime('%H:%M:%S')
            l_end = datetime.fromtimestamp(ts_sorted[-1]/1000, tz=pytz.UTC).astimezone(eastern).strftime('%H:%M:%S')
            logger.info(f"Leg {symbol}: {len(bars)} bars from {l_start} to {l_end} EST")
        logger.info("-----------------------------")

    def run_replay(self, speed=1.0):
        # ... (rest of the method stays the same)
        # Ensure leg_meta uses the corrected qty extraction
        channel = f"tick_events:{self.ticker}"
        logger.info(f"Starting replay for {self.ticker} on channel {channel}...")

        leg_meta = {}
        for pos in self.positions:
            for leg_type in ['short', 'long']:
                leg_data = pos[leg_type]
                symbol = leg_data['instrument']['symbol']
                # Default to 1 if qty is missing, then apply sign
                raw_qty = float(leg_data.get('qty', leg_data.get('quantity', 1)))
                qty = raw_qty * (-1 if leg_type == 'short' else 1)
                
                try:
                    strike = float(symbol[-8:])/1000
                    import re
                    match = re.search(r'\d{6}', symbol)
                    date_str = match.group() if match else "260213"
                    expiry_date = datetime.strptime(date_str, "%y%m%d")
                except:
                    strike = 0.0
                    expiry_date = datetime.now()

                leg_meta[symbol] = {
                    'strike': strike,
                    'expiry_date': expiry_date,
                    'type': 'put' if 'P' in symbol else 'call',
                    'qty': qty
                }

        if not self.underlying_bars:
            logger.warning("No underlying bars to replay.")
            return

        last_leg_prices = {} # symbol -> {'close': price, 'volume': vol}
        
        for i, u_bar in enumerate(self.underlying_bars):
            ts_ms = u_bar['ts']
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=pytz.UTC)
            
            event = BarEvent(
                ticker=self.ticker,
                timestamp=dt.isoformat(),
                underlying_price=u_bar['close']
            )

            for symbol, bars in self.leg_bars.items():
                meta = leg_meta.get(symbol)
                if not meta:
                    continue
                
                # Bar Matching with Fill-Forward
                l_bar = bars.get(str(ts_ms))
                if l_bar:
                    last_leg_prices[symbol] = l_bar
                else:
                    l_bar = last_leg_prices.get(symbol, {'close': 0, 'volume': 0})
                
                eastern = pytz.timezone('US/Eastern')
                expiry_dt = eastern.localize(datetime.combine(meta['expiry_date'].date(), datetime.min.time().replace(hour=16)))
                diff_seconds = (expiry_dt - dt).total_seconds()
                expiry_years = max(1 / (365 * 24 * 3600), diff_seconds / (365 * 24 * 3600))

                leg_event = OptionLegEvent(
                    symbol=symbol,
                    price=float(l_bar.get('close', 0)),
                    volume=int(l_bar.get('volume', 0)),
                    strike=meta['strike'],
                    type=meta['type'],
                    expiry_years=expiry_years,
                    qty=meta['qty']
                )
                event.legs.append(leg_event)

            REDIS_CLIENT.publish(channel, json.dumps(event.to_dict()))
            if i % 10 == 0:
                leg_count = len(event.legs)
                legs_with_price = sum(1 for l in event.legs if l.price > 0)
                est_time = dt.astimezone(eastern).strftime('%H:%M:%S')
                logger.info(f"Published bar {i}/{len(self.underlying_bars)} - {est_time} EST | Legs: {legs_with_price}/{leg_count}")
            
            # Calculate sleep to target speed (bars per second)
            if speed > 0:
                time.sleep(1.0 / speed)
            else:
                # Small yield to prevent saturating Redis and allow consumer to catch up
                time.sleep(0.001) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--date", help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, help="Number of days to go back from today")
    parser.add_argument("--config", default="backtest_positions.json")
    parser.add_argument("--speed", type=float, default=0.0)
    parser.add_argument("--cache", action="store_true", help="Use default cache file")
    parser.add_argument("--cache-file", default="backtest_cache.json", help="Specific cache file to use")
    args = parser.parse_args()

    # Determine date
    replay_date = args.date
    if args.days and not replay_date:
        replay_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    if not replay_date:
        replay_date = "2026-02-13" # Fallback

    producer = ReplayProducer(args.ticker)
    producer.load_positions(args.config)
    
    cache_path = args.cache_file if (args.cache or args.cache_file != "backtest_cache.json") else None
    producer.fetch_historical_data(replay_date, cache_file=cache_path)
    producer.run_replay(speed=args.speed)
