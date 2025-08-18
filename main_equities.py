import json
import schedule
import threading
from strategy_consumer import StrategyConsumer
from tick_producer import TickProducer
from time import sleep
from utils import (
    ticker_data_path_for_strategy,
)


def run_every_week(strategy):
    """Starts threads for each ticker."""
    TICKER_DATA_PATH = ticker_data_path_for_strategy(strategy)
    with open(TICKER_DATA_PATH, "r") as file:
        tickers_config = json.load(file)

    print("1")
    producer = TickProducer()
    producer_thread = threading.Thread(target=producer.run, arg=(tickers_config,))
    consumer = StrategyConsumer()
    consumer_thread = threading.Thread(target=consumer.run, arg=(tickers_config, strategy,))
    Print('e')
    producer_thread.start()
    print("2")
    time.sleep(30)
    print("3")
    consumer_thread.start()


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
