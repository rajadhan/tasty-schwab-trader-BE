import threading
import json
from time import sleep
from utils import (
    is_holiday,
    is_within_time_range,
    get_current_datetime,
    get_market_hours,
    get_strategy_prarams,
    configure_logger,
    store_logs,
    sleep_base_on_timeframe,
    get_strategy_for_ticker,
    ticker_data_path_for_strategy
)
import schedule
from ema_strategy import ema_strategy
from supertrend_strategy import supertrend_strategy
from zeroday_strategy import zeroday_strategy


def main_strategy_loop(ticker):
    """Main loop for running the strategy for a specific ticker."""
    strategy, config = get_strategy_for_ticker(ticker)
    logger = configure_logger(ticker, strategy)
    if strategy == "ema":
        ema_strategy(ticker, logger)
    elif strategy == "supertrend":
        supertrend_strategy(
            ticker,
            logger,
        )
    elif strategy == "zeroday":
        zeroday_strategy(ticker, logger)
    else:
        logger.error(f"No strategy configured for {ticker}")

    try:
        while is_within_time_range():
            _, today_date = get_current_datetime()
            if "/" == ticker[0]:
                if is_holiday(today_date):
                    logger.info("Market closed due to holiday")
                    sleep(60)
                else:
                    [timeframe, *_] = get_strategy_prarams(ticker, logger)
                    # sleep_until_next_interval(ticker, timeframe)
                    sleep_base_on_timeframe(timeframe)
                    ema_strategy(ticker, logger)
            else:
                market_hours, status = get_market_hours(today_date)
                if not market_hours:
                    logger.info(status)
                    sleep(60)
                else:
                    while True:
                        current_time, _ = get_current_datetime()

                        if market_hours[0] <= current_time <= market_hours[1]:
                            [timeframe, *_] = get_strategy_prarams(ticker, logger)
                            # sleep_until_next_interval(ticker, timeframe)
                            sleep_base_on_timeframe(timeframe)
                            ema_strategy(ticker, logger)
                        else:
                            if current_time >= market_hours[1]:
                                logger.info("Market closed")
                                store_logs(ticker)
                                logger = None
                                break
                            elif current_time < market_hours[0]:
                                sleep(60)
                            else:
                                break

    except Exception as e:
        logger.error(f"Error in main loop for {ticker}: {e}", exc_info=True)


def run_every_week(strategy):
    """Starts threads for each ticker."""
    TICKER_DATA_PATH = ticker_data_path_for_strategy(strategy)
    with open(TICKER_DATA_PATH, "r") as file:
        ticker_n_tf = json.load(file)
    threads = []
    for ticker in ticker_n_tf.keys():
        thread = threading.Thread(target=main_strategy_loop, args=(ticker,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


def main():
    """Main scheduling function."""
    schedule.every().sunday.at("18:00").do(run_every_week)

    while True:
        schedule.run_pending()
        sleep(1)


# Start the process
if __name__ == "__main__":
    #     main()
    run_every_week()
