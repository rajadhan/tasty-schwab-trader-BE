import threading
import json
import redis
import pandas as pd
from time import sleep
import logging
from collections import defaultdict
from config import *
from utils import *
from brokers.schwab import historical_data, place_order
from brokers.tastytrade import place_tastytrade_order
from utils import is_tick_timeframe, get_active_exchange_symbol
from datetime import datetime,timedelta,timezone
import logging
import sys

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
        # Remove self.tick_dataframes - use Redis only for tick data
        self.pending_strategies = defaultdict(threading.Event)  # For triggering strategy on new bars


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

    def listen_for_tick_bars(self, tick_symbols_to_tickers):
        """Listen for new tick bars and trigger strategies"""
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                try:
                    # Parse the channel to get symbol
                    channel = message['channel'].decode('utf-8')
                    symbol = channel.split(':')[1]
                    # Parse bar data
                    bar_data = json.loads(message['data'].decode('utf-8'))
                    self.logger.info(f"Received new bar for {symbol}: close={bar_data['close']}")
                    
                    # Trigger strategy for all tickers using this symbol
                    if symbol in tick_symbols_to_tickers:
                        for ticker in tick_symbols_to_tickers[symbol]:
                            self.pending_strategies[ticker].set()
                            self.logger.debug(f"Triggered strategy event for {ticker}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing tick bar message: {e}")

    def main_strategy_loop(self, ticker, strategy):
        """Main strategy loop for a specific ticker"""
        logger = configure_logger(ticker, strategy)
        logger.info(f"MAIN STRATEGY STARTED for {ticker}")
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
                        sleep_base_on_timeframe(time_frame)
                        self.strategy(strategy, ticker, logger)

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
    # Load ticker configuration
    with open("jsons/tickers.json", "r") as file:
        tickers_config = json.load(file)
    print(f"Loaded {len(tickers_config)} tickers from configuration.")
    # Start strategy consumer
    consumer = StrategyConsumer()
    consumer.run(tickers_config)