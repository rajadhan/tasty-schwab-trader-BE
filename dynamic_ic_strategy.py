import time
import pytz
from datetime import datetime
from gamma_rwr_engine import GammaRWREngine
from utils import get_strategy_prarams

def process_dynamic_ic_tick(ticker, event, state, logger):
    """
    Main evaluation loop for the Dynamic Iron Condor.
    Called every 1-minute BarEvent.
    """
    try:
        spot = event.get('underlying_price', 0.0)
        legs = event.get('legs', [])
        ts = event.get('timestamp')
        
        # Load config dynamically to allow hot-reloading
        config = get_strategy_prarams("dynamic_ic", ticker, logger)
        if not config or str(config.get("trade_enabled", "FALSE")).upper() != "TRUE":
            return
            
        trigger_delta = float(config.get("trigger_delta", 0.30))
        confirmation_mins = int(config.get("confirmation_mins", 5))
        max_rolls = int(config.get("max_rolls_per_day", 2))
        
        # Initialize strategy state if empty
        if not state:
            state['call_rolls_today'] = 0
            state['put_rolls_today'] = 0
            state['consecutive_call_breach'] = 0
            state['consecutive_put_breach'] = 0
            state['total_net_credit'] = 0.0 
            state['initial_credit'] = 2.0 # Placeholder
            state['date'] = datetime.now().date().isoformat()
            
        # Reset daily counters if day changed
        current_date = datetime.now().date().isoformat()
        if state['date'] != current_date:
            state['call_rolls_today'] = 0
            state['put_rolls_today'] = 0
            state['consecutive_call_breach'] = 0
            state['consecutive_put_breach'] = 0
            state['date'] = current_date
            
        if not legs:
            logger.debug(f"No active legs found for {ticker} IC.")
            return

        engine = GammaRWREngine()
        # Find the short legs
        short_call = None
        short_put = None
        
        for leg in legs:
            if leg['qty'] < 0:
                if leg['type'].lower() == 'call':
                    short_call = leg
                elif leg['type'].lower() == 'put':
                    short_put = leg
                    
        # Calculate Delta for Short Call
        call_threatened = False
        put_threatened = False
        
        market_iv = 0.4 # Default placeholder, would use real IV solver if available
        
        if short_call:
            call_greeks = engine.black_scholes_greeks(
                spot, short_call['strike'], short_call['expiry_years'], market_iv, 0.05, 0.0, 'call'
            )
            call_delta = call_greeks['delta']
            logger.debug(f"{ticker} Short Call Strike {short_call['strike']} Delta: {call_delta:.3f}")
            if abs(call_delta) > trigger_delta:
                state['consecutive_call_breach'] += 1
                call_threatened = True
            else:
                state['consecutive_call_breach'] = 0
                
        if short_put:
            put_greeks = engine.black_scholes_greeks(
                spot, short_put['strike'], short_put['expiry_years'], market_iv, 0.05, 0.0, 'put'
            )
            put_delta = put_greeks['delta']
            logger.debug(f"{ticker} Short Put Strike {short_put['strike']} Delta: {put_delta:.3f}")
            # Puts have negative delta
            if abs(put_delta) > trigger_delta: 
                state['consecutive_put_breach'] += 1
                put_threatened = True
            else:
                state['consecutive_put_breach'] = 0

        # Check Triggers
        if state['consecutive_call_breach'] >= confirmation_mins:
            logger.warning(f"DYNAMIC IC: CALL SIDE THREATENED FOR {confirmation_mins} MINS. DELTA OVER {trigger_delta}.")
            if state['call_rolls_today'] < max_rolls:
                execute_roll(ticker, 'call', spot, legs, state, logger)
                state['consecutive_call_breach'] = 0 # Reset after attempt
                state['call_rolls_today'] += 1
            else:
                logger.error("DYNAMIC IC: Max Call Rolls exceeded for today. Eject evaluated.")
                
        elif state['consecutive_put_breach'] >= confirmation_mins:
            logger.warning(f"DYNAMIC IC: PUT SIDE THREATENED FOR {confirmation_mins} MINS. DELTA OVER {trigger_delta}.")
            if state['put_rolls_today'] < max_rolls:
                execute_roll(ticker, 'put', spot, legs, state, logger)
                state['consecutive_put_breach'] = 0
                state['put_rolls_today'] += 1
            else:
                logger.error("DYNAMIC IC: Max Put Rolls exceeded for today. Eject evaluated.")

    except Exception as e:
        logger.error(f"Error in dynamic IC loop: {e}", exc_info=True)


def execute_roll(ticker, threatened_side, spot, current_legs, state, logger):
    """
    Executes the roll logic.
    - Closes unchallenged side for credit.
    - Uses credit to close threatened side and reopen further out.
    - Checks Buying Power and IV constraints.
    """
    logger.info(f"### INITIATING ROLL SEQUENCE FOR {threatened_side.upper()} SIDE ###")
    logger.info("TODO: Implement Broker Complex Order Routing (4-leg atomic)")
    logger.info("TODO: Implement Net Credit Exhaustion check")
    logger.info("TODO: Implement Buying Power check")
    logger.info("TODO: Calculate new strikes based on target deltas from config")
