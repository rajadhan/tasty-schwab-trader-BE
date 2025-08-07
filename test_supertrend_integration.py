#!/usr/bin/env python3
"""
Test script to verify get_historical_data integration with supertrend_strategy
"""

import os
import sys
import logging
from datetime import datetime
import pytz

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the functions we want to test
from main_equities import get_historical_data, supertrend_strategy

def setup_test_logger():
    """Set up a test logger"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('test_supertrend')

def test_get_historical_data():
    """Test the get_historical_data function"""
    logger = setup_test_logger()
    
    # Test parameters
    test_ticker = "ESM2"  # Example futures ticker
    test_timeframe = "100tick"  # Tick-based timeframe
    
    logger.info(f"Testing get_historical_data with ticker: {test_ticker}, timeframe: {test_timeframe}")
    
    try:
        # Test the function
        df = get_historical_data(test_ticker, test_timeframe, logger)
        
        # Verify the DataFrame structure
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        logger.info(f"DataFrame head:\n{df.head()}")
        
        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        logger.info("✅ get_historical_data test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ get_historical_data test failed: {str(e)}", exc_info=True)
        return False

def test_supertrend_strategy_integration():
    """Test the supertrend_strategy with the new get_historical_data function"""
    logger = setup_test_logger()
    
    # Test parameters
    test_ticker = "ESM2"
    
    logger.info(f"Testing supertrend_strategy integration with ticker: {test_ticker}")
    
    try:
        # Run the strategy (this will use get_historical_data internally)
        supertrend_strategy(test_ticker, logger)
        
        logger.info("✅ supertrend_strategy integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ supertrend_strategy integration test failed: {str(e)}", exc_info=True)
        return False

def main():
    """Run all tests"""
    logger = setup_test_logger()
    logger.info("Starting supertrend integration tests...")
    
    # Check if DATABENTO_API_KEY is set
    if not os.getenv("DATABENTO_API_KEY"):
        logger.error("❌ DATABENTO_API_KEY environment variable not set!")
        logger.info("Please set your Databento API key before running tests.")
        return False
    
    # Run tests
    test1_passed = test_get_historical_data()
    test2_passed = test_supertrend_strategy_integration()
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
