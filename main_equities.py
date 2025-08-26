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

    producer = TickProducer()
    producer_thread = threading.Thread(target=producer.run, args=(tickers_config, strategy,))
    consumer = StrategyConsumer()
    consumer_thread = threading.Thread(target=consumer.run, args=(tickers_config, strategy,))
    
    producer_thread.start()
    sleep(15)
    consumer_thread.start()
    
    # Wait for both threads to complete
    # producer_thread.join()
    # consumer_thread.join()


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
