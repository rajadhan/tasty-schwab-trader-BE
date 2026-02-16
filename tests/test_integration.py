import time
import logging
from strategy_consumer import StrategyConsumer
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import pytz

# Mocking the Schwab and Massive API calls
@patch('tick_buffer.REST_CLIENT')
@patch('strategy_consumer.get_positions')
@patch('strategy_consumer.get_quotes')
def run_test(mock_quotes, mock_positions, mock_massive):
    # 1. Mock Massive Snapshot (Simulating Greeks from Polygon)
    mock_snap = MagicMock()
    mock_snap.day = MagicMock(last_price=4995.0, volume=2000000)
    mock_snap.greeks = MagicMock(delta=0.6, gamma=0.08, theta=-0.03)
    # Ensure get_option_contract_snapshot returns our mock
    mock_massive.get_option_contract_snapshot.return_value = mock_snap

    # 2. Setup Mock Positions
    mock_positions.return_value = [
        {
            "longQuantity": 0, "shortQuantity": 1,
            "instrument": {"symbol": "SPX   260215C05000000", "assetType": "OPTION", "underlyingSymbol": "SPX"}
        },
        {
            "longQuantity": 1, "shortQuantity": 0,
            "instrument": {"symbol": "SPX   260215C05100000", "assetType": "OPTION", "underlyingSymbol": "SPX"}
        }
    ]
    
    # 3. Setup Mock Quotes
    mock_quotes.return_value = {'last': 4980}
    
    # Initialize Consumer (this will use the mocked REST_CLIENT)
    consumer = StrategyConsumer()
    logger = logging.getLogger("TEST")
    
    print("\n--- MASSIVE RWR INTEGRATION TEST STARTING ---")
    
    print("\n[SCENARIO 1] Checking Massive Greek Capture")
    # This should now update the radar with mock Greeks
    consumer.run_rwr_monitoring("SPX", logger)
    
    print("\n[SUCCESS] Integration test cycle completed.")

if __name__ == "__main__":
    run_test()
