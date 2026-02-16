import time
import json
import logging
import redis
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from strategy_consumer import StrategyConsumer

# Configure logging for test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RWR_E2E")

def setup_redis_data(r, ticker, price, volume, greeks=None):
    """Utility to inject a mock Massive bar into Redis"""
    zset_key = f"bars_history:{ticker}"
    bar = {
        "symbol": ticker,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "open": price, "high": price + 1, "low": price - 1, "close": price,
        "volume": volume
    }
    if greeks:
        bar.update(greeks)
        
    data_str = json.dumps(bar)
    score = int(time.time())
    r.zadd(zset_key, {data_str: score})

@patch('strategy_consumer.get_positions')
@patch('strategy_consumer.get_quotes')
def test_rwr_e2e(mock_quotes, mock_positions):
    # Connect to REAL Redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    # 1. CLEANUP PREVIOUS DATA
    r.delete("bars_history:SPX")
    r.delete("bars_history:SPX   260215C05000000")
    r.delete("bars_history:SPX   260215C05100000")

    # 2. SETUP MOCK POSITIONS (SPX Call Credit Spread)
    mock_positions.return_value = {
        "SPX": [
            {
                "type": "call",
                "short": {
                    "instrument": {"symbol": "SPX   260215C05000000", "underlyingSymbol": "SPX"}
                },
                "long": {
                    "instrument": {"symbol": "SPX   260215C05100000"}
                }
            }
        ]
    }
    
    # Mock the discover_credit_spreads function as well
    with patch('strategy_consumer.discover_credit_spreads') as mock_ds:
        mock_ds.return_value = mock_positions.return_value
        
        consumer = StrategyConsumer()

        print("\n" + "="*50)
        print("E2E SCENARIO 1: ALL CLEAR (Low G.A.R. & Normal Volume)")
        print("="*50)
        
        # Inject "Safe" data into Redis
        setup_redis_data(r, "SPX", 4980, 10000) 
        setup_redis_data(r, "SPX   260215C05000000", 5.0, 100, {'delta': -0.4, 'gamma': 0.01, 'theta': 0.02})
        setup_redis_data(r, "SPX   260215C05100000", 2.0, 50, {'delta': 0.1, 'gamma': 0.005, 'theta': -0.01})
        
        # Mock Spot Price
        mock_quotes.return_value = {'last': 4980}
        
        # Run Monitoring
        consumer.run_rwr_monitoring("SPX", logger)
        
        print("\n" + "="*50)
        print("E2E SCENARIO 2: WARNING (Gamma Spike & High Confidence)")
        print("="*50)
        
        # Inject "Threat" data (High Gamma)
        setup_redis_data(r, "SPX", 4998, 50000) 
        setup_redis_data(r, "SPX   260215C05000000", 15.0, 500, {'delta': -0.85, 'gamma': 0.15, 'theta': 0.05})
        setup_redis_data(r, "SPX   260215C05100000", 8.0, 200, {'delta': 0.3, 'gamma': 0.02, 'theta': -0.03})
        
        # Mock Spot Price
        mock_quotes.return_value = {'last': 4998}
        
        # Trigger persistence
        for i in range(4):
            print(f"\nTick {i+1}/4...")
            consumer.run_rwr_monitoring("SPX", logger)
            time.sleep(0.1)

    print("\nE2E Real-Redis Test Cycle Finished.")

if __name__ == "__main__":
    try:
        test_rwr_e2e()
    except Exception as e:
        print(f"Test failed: {e}")
