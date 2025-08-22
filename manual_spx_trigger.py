import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManualSPXTrigger:
    def __init__(self, schwab_client=None, tastytrade_client=None):
        """
        Initialize the manual SPX trigger system
        
        Args:
            schwab_client: Your existing Schwab API client
            tastytrade_client: Your existing TastyTrade API client
        """
        self.schwab_client = schwab_client
        self.tastytrade_client = tastytrade_client
        self.current_position = None
        
    def get_current_spx_price(self) -> Optional[float]:
        """Get current SPX price for ATM strike calculation"""
        try:
            # This would integrate with your existing price feed
            # For now, returning a placeholder
            return 5000.0  # Replace with actual SPX price
        except Exception as e:
            logger.error(f"Error getting SPX price: {e}")
            return None
    
    def calculate_atm_strike(self, price: float) -> int:
        """Calculate at-the-money strike price"""
        # Round to nearest 5 for SPX options
        return round(price / 5) * 5
    
    def get_today_expiration(self) -> str:
        """Get today's date in YYYY-MM-DD format"""
        return datetime.now().strftime("%Y-%m-%d")
    
    def execute_spx_call(self, strike: Optional[int] = None, expiration: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute SPX call option purchase
        
        Args:
            strike: Strike price (if None, uses ATM)
            expiration: Expiration date (if None, uses today)
        
        Returns:
            Dict with success status and trade details
        """
        try:
            # Get current SPX price
            spx_price = self.get_current_spx_price()
            if not spx_price:
                return {"success": False, "error": "Unable to get SPX price"}
            
            # Calculate strike if not provided
            if not strike:
                strike = self.calculate_atm_strike(spx_price)
            
            # Use today if no expiration provided
            if not expiration:
                expiration = self.get_today_expiration()
            
            # Check if we have an existing position to close first
            if self.current_position and self.current_position.get('type') == 'put':
                logger.info("Closing existing PUT position before buying CALL")
                self.close_position()
            
            # Execute the call purchase
            trade_details = {
                "action": "BUY_TO_OPEN",
                "symbol": "SPX",
                "option_type": "CALL",
                "strike": strike,
                "expiration": expiration,
                "quantity": 1,  # Adjust based on your risk management
                "order_type": "MARKET",
                "timestamp": datetime.now().isoformat()
            }
            
            # Here you would integrate with your existing trading logic
            # For Schwab:
            if self.schwab_client:
                # Use your existing Schwab trading methods
                result = self.execute_schwab_trade(trade_details)
            # For TastyTrade:
            elif self.tastytrade_client:
                # Use your existing TastyTrade trading methods
                result = self.execute_tastytrade_trade(trade_details)
            else:
                # Fallback - log the trade for manual execution
                logger.info(f"Manual SPX CALL trade logged: {trade_details}")
                result = {"success": True, "message": "Trade logged for manual execution"}
            
            # Update current position
            if result.get("success"):
                self.current_position = {
                    "type": "call",
                    "strike": strike,
                    "expiration": expiration,
                    "entry_time": datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing SPX call: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_spx_put(self, strike: Optional[int] = None, expiration: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute SPX put option purchase
        
        Args:
            strike: Strike price (if None, uses ATM)
            expiration: Expiration date (if None, uses today)
        
        Returns:
            Dict with success status and trade details
        """
        try:
            # Get current SPX price
            spx_price = self.get_current_spx_price()
            if not spx_price:
                return {"success": False, "error": "Unable to get SPX price"}
            
            # Calculate strike if not provided
            if not strike:
                strike = self.calculate_atm_strike(spx_price)
            
            # Use today if no expiration provided
            if not expiration:
                expiration = self.get_today_expiration()
            
            # Check if we have an existing position to close first
            if self.current_position and self.current_position.get('type') == 'call':
                logger.info("Closing existing CALL position before buying PUT")
                self.close_position()
            
            # Execute the put purchase
            trade_details = {
                "action": "BUY_TO_OPEN",
                "symbol": "SPX",
                "option_type": "PUT",
                "strike": strike,
                "expiration": expiration,
                "quantity": 1,  # Adjust based on your risk management
                "order_type": "MARKET",
                "timestamp": datetime.now().isoformat()
            }
            
            # Here you would integrate with your existing trading logic
            # For Schwab:
            if self.schwab_client:
                result = self.execute_schwab_trade(trade_details)
            # For TastyTrade:
            elif self.tastytrade_client:
                result = self.execute_tastytrade_trade(trade_details)
            else:
                # Fallback - log the trade for manual execution
                logger.info(f"Manual SPX PUT trade logged: {trade_details}")
                result = {"success": True, "message": "Trade logged for manual execution"}
            
            # Update current position
            if result.get("success"):
                self.current_position = {
                    "type": "put",
                    "strike": strike,
                    "expiration": expiration,
                    "entry_time": datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing SPX put: {e}")
            return {"success": False, "error": str(e)}
    
    def close_position(self) -> Dict[str, Any]:
        """
        Close the current SPX option position
        
        Returns:
            Dict with success status and close details
        """
        try:
            if not self.current_position:
                return {"success": False, "error": "No position to close"}
            
            position_type = self.current_position['type']
            strike = self.current_position['strike']
            expiration = self.current_position['expiration']
            
            # Execute position close
            trade_details = {
                "action": "SELL_TO_CLOSE",
                "symbol": "SPX",
                "option_type": position_type.upper(),
                "strike": strike,
                "expiration": expiration,
                "quantity": 1,
                "order_type": "MARKET",
                "timestamp": datetime.now().isoformat()
            }
            
            # Here you would integrate with your existing trading logic
            if self.schwab_client:
                result = self.execute_schwab_trade(trade_details)
            elif self.tastytrade_client:
                result = self.execute_tastytrade_trade(trade_details)
            else:
                logger.info(f"Manual SPX position close logged: {trade_details}")
                result = {"success": True, "message": "Position close logged for manual execution"}
            
            # Clear current position if successful
            if result.get("success"):
                self.current_position = None
            
            return result
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_schwab_trade(self, trade_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade using Schwab API
        
        This is a placeholder - integrate with your existing Schwab trading logic
        """
        try:
            # Integrate with your existing schwab/client.py methods
            logger.info(f"Executing Schwab trade: {trade_details}")
            
            # Example integration:
            # if trade_details["action"] == "BUY_TO_OPEN":
            #     result = self.schwab_client.buy_option(
            #         symbol=trade_details["symbol"],
            #         option_type=trade_details["option_type"],
            #         strike=trade_details["strike"],
            #         expiration=trade_details["expiration"],
            #         quantity=trade_details["quantity"]
            #     )
            
            return {"success": True, "message": "Trade executed via Schwab", "details": trade_details}
            
        except Exception as e:
            logger.error(f"Schwab trade execution error: {e}")
            return {"success": False, "error": f"Schwab execution failed: {str(e)}"}
    
    def execute_tastytrade_trade(self, trade_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade using TastyTrade API
        
        This is a placeholder - integrate with your existing TastyTrade trading logic
        """
        try:
            # Integrate with your existing tastytrade.py methods
            logger.info(f"Executing TastyTrade trade: {trade_details}")
            
            # Example integration:
            # if trade_details["action"] == "BUY_TO_OPEN":
            #     result = self.tastytrade_client.buy_option(
            #         symbol=trade_details["symbol"],
            #         option_type=trade_details["option_type"],
            #         strike=trade_details["strike"],
            #         expiration=trade_details["expiration"],
            #         quantity=trade_details["quantity"]
            #     )
            
            return {"success": True, "message": "Trade executed via TastyTrade", "details": trade_details}
            
        except Exception as e:
            logger.error(f"TastyTrade execution error: {e}")
            return {"success": False, "error": f"TastyTrade execution failed: {str(e)}"}
    
    def get_position_status(self) -> Dict[str, Any]:
        """Get current position status"""
        if self.current_position:
            return {
                "has_position": True,
                "position": self.current_position
            }
        else:
            return {
                "has_position": False,
                "position": None
            }

# Example usage function
def handle_manual_trigger_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle manual trigger requests from the frontend
    
    Args:
        request_data: Dict containing action, symbol, strike, expiration
        
    Returns:
        Dict with execution results
    """
    try:
        action = request_data.get("action")
        symbol = request_data.get("symbol", "SPX")
        strike = request_data.get("strike")
        expiration = request_data.get("expiration")
        
        # Initialize the trigger system
        # You'll need to pass your existing clients here
        trigger = ManualSPXTrigger()
        
        if action == "buy_call":
            result = trigger.execute_spx_call(strike, expiration)
        elif action == "buy_put":
            result = trigger.execute_spx_put(strike, expiration)
        elif action == "close_position":
            result = trigger.close_position()
        else:
            result = {"success": False, "error": f"Unknown action: {action}"}
        
        return result
        
    except Exception as e:
        logger.error(f"Manual trigger request error: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Test the manual trigger system
    trigger = ManualSPXTrigger()
    
    # Test call execution
    result = trigger.execute_spx_call()
    print(f"Call execution result: {result}")
    
    # Test put execution
    result = trigger.execute_spx_put()
    print(f"Put execution result: {result}")
    
    # Test position close
    result = trigger.close_position()
    print(f"Position close result: {result}")
