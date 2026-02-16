import time
import logging
from strategy_consumer import StrategyConsumer
from unittest.mock import patch, MagicMock

# Mocking the Schwab API calls to simulate real scenarios
@patch('strategy_consumer.get_positions')
@patch('strategy_consumer.get_quotes')
def test_integration(mock_quotes, mock_positions):
    # 1. Setup Mock Positions (SPX 5000/5100 Call Credit Spread)
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
    
    # 2. Setup Mock Quotes (Spot moving towards short strike)
    # Start at 4980 (SEARCH)
    mock_quotes.return_value = {'last': 4980}
    
    consumer = StrategyConsumer()
    logger = logging.getLogger("TEST")
    
    print("\n--- RWR INTEGRATION TEST STARTING ---")
    
    print("\n[SCENARIO 1] Spot at 4980 (Safe Zone)")
    consumer.run_rwr_monitoring("SPX", logger)
    
    # 3. Simulate Threat (Spot moving to 4995 - G.A.R. should spike)
    print("\n[SCENARIO 2] Spot at 4995 (Danger Zone)")
    mock_quotes.return_value = {'last': 4995}
    for _ in range(4): # Trigger persistence
        consumer.run_rwr_monitoring("SPX", logger)
        time.sleep(0.1)

if __name__ == "__main__":
    test_integration()
