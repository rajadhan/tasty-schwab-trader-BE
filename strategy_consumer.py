import threading
import json
import redis
import pandas as pd
from time import sleep
import logging
from collections import defaultdict
from config import *
from utils import (
    is_tick_timeframe, 
    get_active_exchange_symbol,
    load_strategy_configs,
    configure_logger,
    is_holiday,
    get_current_datetime,
    get_strategy_prarams,
    get_market_hours,
    sleep_base_on_timeframe,
    is_within_time_range
)
from datetime import datetime,timedelta,timezone
import logging
import sys
from gamma_rwr_engine import GammaRWREngine
from gamma_rwr_filters import GammaRWRFilters
from rwr_alert_manager import RWRAlertManager
from brokers.schwab import get_positions, discover_credit_spreads, get_quotes
import argparse
import pytz
from datetime import time as dt_time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.FileHandler("strategy_consumer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
class StrategyConsumer:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.pubsub = self.redis_client.pubsub()
        self.logger = logging.getLogger('StrategyConsumer')
        self.pending_strategies = defaultdict(threading.Event)
        # Type hints for better IDE/linting support
        self.rwr_components: dict[str, dict] = {} 
        self.backtest_positions: dict[str, list] = {}
        self._backtest_ejected = False


    def get_tick_dataframe(self, symbol, period1, period2, timeframe):
        ticker_for_data = get_active_exchange_symbol(symbol)
        if is_tick_timeframe(timeframe):
            zset_key = f"bars_history:{ticker_for_data}"
        else:
            zset_key = f"bars_history:{symbol}"

        bars_count = self.redis_client.zcard(zset_key)
        print(f"{zset_key} has {bars_count} bars saved")

        max_bars = max(period1, period2)
        # Fetch latest bars by rank (newest to oldest), then reverse for oldest → newest
        latest_bars = self.redis_client.zrevrange(zset_key, 0, -1)
        bars = [json.loads(bar.decode('utf-8')) for bar in reversed(latest_bars)]
        self.logger.info(f"Latest bars for {symbol}: {len(bars)}")
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df.to_csv(f"logs/ema/strategy_MNQ.csv")
        # Robust ISO8601 parsing with ns precision and no timezone
        parsed_ts = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True, errors='coerce')
        # Drop rows we couldn't parse
        valid_mask = ~parsed_ts.isna()
        if not valid_mask.any():
            return pd.DataFrame()
        df = df.loc[valid_mask].copy()
        df['timestamp'] = parsed_ts.loc[valid_mask]
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        return df
    
    def strategy(self, strategy, ticker, logger, triggered_by_new_bar=False):
        """Run the specified strategy for the given ticker."""
        logger.info(f"Running strategy {strategy} for {ticker}")
        if strategy == "ema":
            from ema_strategy import ema_strategy
            ema_strategy(ticker, logger)
        elif strategy == "supertrend":
            from supertrend_strategy import supertrend_strategy
            supertrend_strategy(ticker, logger)
        elif strategy == "zeroday":
            from zeroday_strategy import zeroday_strategy
            zeroday_strategy(ticker, logger)
        else:
            logger.error(f"Unknown strategy: {strategy}")

    def subscribe_to_tick_bars(self, symbols):
        """Subscribe to tick bar updates for given symbols"""
        for symbol in symbols:
            channel = f"tick_bars:{symbol}" 
            self.pubsub.subscribe(channel)
        
        self.logger.info(f"Subscribed to tick bars for symbols: {symbols}")

    def listen_for_events(self, tickers):
        """Main event listener for unified BarEvents."""
        for ticker in tickers:
            channel = f"tick_events:{ticker}"
            self.pubsub.subscribe(channel)
            self.logger.info(f"Subscribed to {channel}")

        for message in self.pubsub.listen():
            if message['type'] == 'message':
                try:
                    channel = message['channel'].decode('utf-8')
                    ticker = channel.split(':')[1]
                    event_data = json.loads(message['data'].decode('utf-8'))
                    
                    # Convert UTC ISO string to EST for logging
                    try:
                        utc_dt = datetime.fromisoformat(event_data.get('timestamp').replace('Z', '+00:00'))
                        if utc_dt.tzinfo is None:
                            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
                        est_time = utc_dt.astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        est_time = event_data.get('timestamp')

                    self.logger.info(f"Received BarEvent for {ticker} at {est_time} EST")
                    self.process_bar_event(ticker, event_data)
                    
                except Exception as e:
                    self.logger.error(f"Error in event listener: {e}")

    def process_bar_event(self, ticker, event):
        """Processes a unified BarEvent (Market + Legs)."""
        logger = logging.getLogger(f"RWR:{ticker}")
        if ticker not in self.rwr_components:
            self.rwr_components[ticker] = {
                'engine': GammaRWREngine(),
                'filters': GammaRWRFilters(),
                'alerts': RWRAlertManager(ticker)
            }
        
        comp = self.rwr_components[ticker]
        spot = event['underlying_price']
        legs = event['legs']
        ts = event['timestamp']

        # Detailed data verification log in EST
        try:
            utc_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            if utc_dt.tzinfo is None:
                utc_dt = utc_dt.replace(tzinfo=timezone.utc)
            est_ts = utc_dt.astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')
        except:
            est_ts = ts
            
        leg_summary = ", ".join([f"{l['symbol'][-8:]}: ${l['price']:.2f}" for l in legs])
        self.logger.info(f"Processing {ticker} at {est_ts} EST | Spot: ${spot:.2f} | Legs: {leg_summary}")

        # Calculate Net Greeks using Market-Implied logic
        net_greeks = comp['engine'].calculate_position_greeks(spot, legs)
        gar = net_greeks['gar']
        market_iv = net_greeks.get('market_iv', 0.4)
        
        # Calculate Probability of Touch for the short strike (most at-risk leg)
        short_strike = None
        min_dte = float('inf')
        for leg in legs:
            if leg['qty'] < 0:  # Short leg
                short_strike = leg['strike']
                min_dte = min(min_dte, leg['expiry_years'])
        
        if short_strike:
            pot = comp['engine'].calculate_probability_of_touch(
                spot, short_strike, min_dte, market_iv
            )
            self.logger.info(f"PoT Calc: Strike={short_strike}, Spot={spot:.2f}, DTE={min_dte:.4f}y, IV={market_iv:.2f}, PoT={pot:.3f}")
        else:
            pot = 1.0  # Default to full risk if no short strike identified
        
        # ECCM logic
        comp['filters'].add_snapshot(gar=gar, volume_ratio=1.0, iv_delta=0.0)
        confidence = comp['filters'].calculate_confidence()
        gar_results = {k: float(v) for k, v in comp['filters'].get_multi_window_gar().items()}
        
        threat_level, _ = comp['engine'].classify_threat(gar, pot)
        
        # HUD Handling
        is_ejected = getattr(self, '_backtest_ejected', False)
        # Always render HUD during backtest/replay for monitoring
        if not is_ejected:
            comp['alerts'].render_hud(gar_results, confidence, threat_level, timestamp=est_ts, spot=spot, greeks=net_greeks)
        
        # Alerting
        if comp['filters'].should_eject(threat_level, confidence) and not is_ejected:
            self._backtest_ejected = True
            logger.warning(f"RWR ALERT: !!! EJECT !!! for {ticker} at GAR {gar:.2f} (Time: {ts})")
            comp['alerts'].trigger_alert('LAUNCH', f"G.A.R. Spike: {gar:.2f} @ Conf: {confidence:.2f}")

    def inject_simulated_positions(self, config_path):
        """Load simulated positions for backtesting from a JSON file."""
        try:
            with open(config_path, "r") as f:
                self.backtest_positions = json.load(f)
            self.logger.info(f"Loaded simulated positions for: {list(self.backtest_positions.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to load backtest positions: {e}")

    def run_backtest(self, ticker, logger):
        """
        Backtesting entry point. Now uses the event-driven listener
        to process events from replay_producer.py.
        """
        logger.info(f"STARTING BACKTEST EVENT LISTENER for {ticker}")
        self.listen_for_events([ticker])
        logger.info(f"BACKTEST COMPLETED for {ticker}")

    def run_rwr_monitoring(self, ticker, logger):
        """Executes one cycle of Gamma RWR risk monitoring."""
        if ticker not in self.rwr_components:
            self.rwr_components[ticker] = {
                'engine': GammaRWREngine(),
                'filters': GammaRWRFilters(),
                'alerts': RWRAlertManager(ticker, logger)
            }
        
        comp = self.rwr_components[ticker]
        
        try:
            # 1. Sync Positions and Discover Spreads
            positions = get_positions()
            spreads = discover_credit_spreads(positions)
            
            if ticker not in spreads:
                return # No active spreads for this ticker
                
            active_spreads = spreads[ticker]
            for spread in active_spreads:
                # 2. Get Real-time Data for Greeks
                # For 0DTE, we focus on the short leg's threat
                short_leg = spread['short']
                symbol = short_leg['instrument']['symbol']
                quote = get_quotes(symbol)
                spot = quote.get('last', 0)
                if spot == 0: continue
                
                # Signal TickProducer to ensure we are monitoring this option
                # Massive symbol prefix usually O: for options
                massive_symbol = f"O:{symbol}"
                self.redis_client.publish("tick_producer:commands", json.dumps({
                    "command": "subscribe",
                    "ticker": massive_symbol
                }))
                
                # 3. Fetch latest market data from Redis for Greeks/Volume
                # Underlying symbol for volume analysis
                underlying = spread['short']['instrument']['underlyingSymbol']
                underlying_key = f"bars_history:{underlying}"
                latest_underlying = self.redis_client.zrevrange(underlying_key, 0, 0)
                
                volume_ratio = 1.0
                if latest_underlying:
                    und_bar = json.loads(latest_underlying[0].decode('utf-8'))
                    # Simple volume ratio: current volume vs 5-period avg (placeholder logic)
                    volume_ratio = 1.2 # Placeholder for real ratio logic
                
                # Option legs - try to get Massive Greeks from Redis if available
                legs = []
                for leg_type in ['short', 'long']:
                    leg_data = spread[leg_type]
                    symbol = leg_data['instrument']['symbol']
                    # Massive key format (O:...)
                    massive_key = f"bars_history:{symbol}"
                    latest_leg = self.redis_client.zrevrange(massive_key, 0, 0)
                    
                    strike = float(symbol[-7:])/1000
                    qty = -1 if leg_type == 'short' else 1
                    
                    engine_leg = {
                        'strike': strike,
                        'qty': qty,
                        'type': spread['type'],
                        'expiry_years': 1/365, # Placeholder for real DTE
                        'iv': 0.2
                    }
                    
                    if latest_leg:
                        leg_bar = json.loads(latest_leg[0].decode('utf-8'))
                        if 'gamma' in leg_bar:
                            engine_leg.update({
                                'delta': leg_bar.get('delta'),
                                'gamma': leg_bar.get('gamma'),
                                'theta': leg_bar.get('theta')
                            })
                    legs.append(engine_leg)

                # 4. Calculate Greeks & G.A.R.
                net_greeks = comp['engine'].calculate_position_greeks(spot, legs)
                gar = net_greeks['gar']
                
                # 5. Filter and Validate (ECCM)
                comp['filters'].add_snapshot(gar=gar, volume_ratio=volume_ratio, iv_delta=0.0)
                confidence = comp['filters'].calculate_confidence()
                gar_results = comp['filters'].get_multi_window_gar()
                
                # 6. Alerting
                threat_level, _ = comp['engine'].classify_threat(gar)
                comp['alerts'].render_hud(gar_results, confidence, threat_level)
                
                if comp['filters'].should_eject(gar_results, confidence):
                    comp['alerts'].trigger_alert('LAUNCH', f"G.A.R. Spike: {gar:.2f} @ Conf: {confidence:.2f}")
                    
        except Exception as e:
            logger.error(f"Error in RWR monitoring for {ticker}: {e}")

    def main_strategy_loop(self, ticker, strategy):
        """Main strategy loop (now event-driven)."""
        logger = configure_logger(ticker, strategy)
        logger.info(f"EVENT LISTENER STARTED for {ticker}")
        
        # For RWR, we just listen for events
        if strategy == "zeroday":
            self.listen_for_events([ticker])
        else:
            # Legacy polling strategies
            print(f"Starting main loop for {ticker} with strategy {strategy}")
            try:
                while True:
                    # Check stop flag per strategy
                    try:
                        if self.redis_client.get(f"trading:stop:{strategy}"):
                            self.logger.info(f"Stop signal detected for {strategy}; exiting {ticker} loop")
                            break
                    except Exception:
                        pass
                    while is_within_time_range():
                        try:
                            if self.redis_client.get(f"trading:stop:{strategy}"):
                                self.logger.info(f"Stop signal detected for {strategy}; breaking trading window for {ticker}")
                                break
                        except Exception:
                            pass
                        _, today_date = get_current_datetime()
                        time_frame, *_ = get_strategy_prarams(strategy, ticker, logger)

                        if ticker.startswith("/"):  # Futures
                            if is_holiday(today_date):
                                logger.info("Market closed due to holiday")
                                sleep(60)
                                continue
                            run_strategy = True

                        else:  # Stocks
                            market_hours, status = get_market_hours(today_date)
                            if not market_hours:
                                logger.info(status)
                                sleep(60)
                                continue
                            current_time, _ = get_current_datetime()
                            if not (market_hours[0] <= current_time <= market_hours[1]):
                                if current_time >= market_hours[1]:
                                    logger.info("Market closed")
                                    break
                                else:
                                    sleep(60)
                                    continue
                            run_strategy = True

                        if run_strategy:
                            print(f"Running strategy for {ticker} with time frame {time_frame}")
                            # Move sleep before strategy or after? Original code has it before strategy call.
                            sleep_base_on_timeframe(time_frame)
                            self.strategy(strategy, ticker, logger)
                            
                            # Run RWR Monitoring cycle after strategy execution
                            # This block is now only for legacy polling strategies,
                            # zeroday is handled by event listener.
                            if strategy == "zeroday":
                                self.run_rwr_monitoring(ticker, logger)

            except Exception as e:
                logger.error(f"Error in main loop for {ticker}: {e}", exc_info=True)


    def run(self, tickers_config, strategy):
        # Start strategy threads for each ticker
        threads = []
        for ticker in tickers_config.keys():
            thread = threading.Thread(target=self.main_strategy_loop, args=(ticker, strategy), daemon=True)
            threads.append(thread)
            thread.start()
        
        # Keep main thread alive
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("Shutting down strategy consumer...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy Consumer CLI")
    parser.add_argument("--backtest", action="store_true", help="Run in backtest mode")
    parser.add_argument("--ticker", type=str, help="Ticker for backtesting (required if --backtest set)")
    parser.add_argument("--config", type=str, default="backtest_positions.json", help="Path to backtest positions JSON")
    args = parser.parse_args()

    consumer = StrategyConsumer()

    if args.backtest:
        if not args.ticker:
            print("Error: --ticker is required for backtest mode.")
            sys.exit(1)
            
        consumer.inject_simulated_positions(args.config)
        logger = configure_logger(args.ticker, "backtest")
        consumer.run_backtest(args.ticker, logger)
    else:
        # Load all strategy configurations from the settings/ directory
        ema_config, super_config, zeroday_config = load_strategy_configs()
        
        configs = [
            ("ema", ema_config),
            ("supertrend", super_config),
            ("zeroday", zeroday_config)
        ]
        
        threads = []
        for strategy_name, config in configs:
            if config:
                print(f"Starting StrategyConsumer for {strategy_name} with {len(config)} tickers.")
                # We use a separate thread for each strategy's group of tickers if needed, 
                # but StrategyConsumer.run already spawns threads for each ticker.
                # So we can just call run sequentially if we want them all in one consumer,
                # or better, just pass the combined config if the consumer supports it.
                # However, StrategyConsumer.run(self, tickers_config, strategy) takes one strategy.
                
                # Start a thread for this strategy group
                t = threading.Thread(target=consumer.run, args=(config, strategy_name), daemon=True)
                threads.append(t)
                t.start()
                
        if not threads:
            print("No active ticker configurations found in settings/*.json")
        else:
            try:
                for t in threads:
                    t.join()
            except KeyboardInterrupt:
                print("Shutting down strategy consumer...")