import numpy as np
import pandas as pd
import os

from utils import (
    get_strategy_prarams,
    get_trade_file_path,
    is_tick_timeframe,
    load_json,
    wilders_smoothing,
)

import json
from datetime import datetime
import pytz
from config import *
from strategy_consumer import StrategyConsumer
from tastytrade import place_tastytrade_order

def supertrend_strategy(
    ticker,
    logger,
):
    try:
        logger.info(f"{ticker} strategy started at {datetime.now(pytz.utc)}")

        # Get parameters from frontend/backend
        [
            timeframe,
            qty_schwab,
            trade_enabled,
            qty_tastytrade,
            zigzag_method, # average, high_low
        ] = get_strategy_prarams("supertrend", ticker, logger)

        # Create trades directory if it doesn't exist
        os.makedirs("trades/supertrend", exist_ok=True)
        trades_file = f"trades/supertrend/{ticker[1:] if ticker.startswith('/') else ticker}.json"

        if trade_enabled != "TRUE":
            logger.info(
                f"Trade disabled for {ticker}, skipping execution and clearing trades file if any."
            )
            try:
                with open(trades_file, "r") as f:
                    trades = json.load(f)
                if ticker in trades:
                    trades = {}
                    with open(trades_file, "w") as f:
                        json.dump(trades, f)
            except FileNotFoundError:
                pass
            return

        # Load existing trades or initialize empty
        try:
            with open(trades_file, "r") as f:
                trades = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            trades = {}

        superfast_len = 9
        fast_len = 14
        slow_len = 21

        # ZigZag parameters
        bubble_offset = 0.0005
        percent_amount = 0.01
        rev_amount = 0.05
        atr_reversal = 2.0
        atr_length = 5
        average_length = 5
        
        # Get historical data using the strategy consumer
        from strategy_consumer import StrategyConsumer
        strategy_consumer = StrategyConsumer()

        # Get tick data for the strategy - need enough data for all moving averages
        df = strategy_consumer.get_tick_dataframe(ticker, max(superfast_len, slow_len) + 50, slow_len)

        if df is None or len(df) < max(superfast_len, slow_len) + 10:
            logger.warning(f"Insufficient data for {ticker}")
            return

        # Ensure required columns exist
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                logger.error(f"Missing required price/volume column '{col}' in historical data")
                return

        # Calculate the three MAs (Superfast, Fast, Slow)
        # Calculate the three MAs (Superfast, Fast, Slow) as in ThinkScript
        df["mov_avg9"] = df["close"].ewm(span=superfast_len, adjust=False).mean()
        df["mov_avg14"] = df["close"].ewm(span=fast_len, adjust=False).mean()
        df["mov_avg21"] = df["close"].ewm(span=slow_len, adjust=False).mean()

        # Define the moving averages as in ThinkScript
        df["Superfast"] = df["mov_avg9"]
        df["Fast"] = df["mov_avg14"]
        df["Slow"] = df["mov_avg21"]

        # Buy/Sell conditions from ThinkScript
        df["buy"] = (
            (df["Superfast"] > df["Fast"]) &
            (df["Fast"] > df["Slow"]) &
            (df["low"] > df["Superfast"])
        )
        df["stopbuy"] = df["Superfast"] <= df["Fast"]
        df["buynow"] = df["buy"] & (~df["buy"].shift(1).fillna(False))
        
        # Compound value function for buy signal (ThinkScript CompoundValue equivalent)
        def compound_value_buy():
            signal = [0]
            for i in range(1, len(df)):
                if df["buynow"].iloc[i] and not df["stopbuy"].iloc[i]:
                    signal.append(1)
                elif signal[-1] == 1 and df["stopbuy"].iloc[i]:
                    signal.append(0)
                else:
                    signal.append(signal[-1])
            return pd.Series(signal, index=df.index)

        df["buysignal"] = compound_value_buy()
        df["Buy_Signal"] = (df["buysignal"].shift(1).fillna(0) == 0) & (df["buysignal"] == 1)
        df["Momentum_Down"] = (df["buysignal"].shift(1).fillna(0) == 1) & (df["buysignal"] == 0)

        # Sell conditions
        df["sell"] = (
            (df["Superfast"] < df["Fast"]) &
            (df["Fast"] < df["Slow"]) &
            (df["high"] < df["Superfast"])
        )
        df["stopsell"] = df["Superfast"] >= df["Fast"]
        df["sellnow"] = df["sell"] & (~df["sell"].shift(1).fillna(False))

        def compound_value_sell():
            signal = [0]
            for i in range(1, len(df)):
                if df["sellnow"].iloc[i] and not df["stopsell"].iloc[i]:
                    signal.append(1)
                elif signal[-1] == 1 and df["stopsell"].iloc[i]:
                    signal.append(0)
                else:
                    signal.append(signal[-1])
            return pd.Series(signal, index=df.index)

        df["sellsignal"] = compound_value_sell()
        df["Sell_Signal"] = (df["sellsignal"].shift(1).fillna(0) == 0) & (df["sellsignal"] == 1)
        df["Momentum_Up"] = (df["sellsignal"].shift(1).fillna(0) == 1) & (df["sellsignal"] == 0)

        # ThinkScript colorbars for signal identification
        def calculate_colorbars():
            colorbars = []
            for i in range(len(df)):
                if df["buysignal"].iloc[i] == 1:
                    colorbars.append(1)  # Green badge - Buy signal
                elif df["sellsignal"].iloc[i] == 1:
                    colorbars.append(2)  # Red badge - Sell signal
                elif df["buysignal"].iloc[i] == 0 or df["sellsignal"].iloc[i] == 0:
                    colorbars.append(3)  # Purple badge - Neutral/transition
                else:
                    colorbars.append(0)
            return pd.Series(colorbars, index=df.index)

        df["ColorBars"] = calculate_colorbars()

        # ZigZag High/Low calculation (simplified version focusing on key signals)
        if zigzag_method == "high_low":
            df["priceh"] = df["high"]
            df["pricel"] = df["low"]
        else:
            df["mah"] = df["high"].ewm(span=average_length, adjust=False).mean()
            df["mal"] = df["low"].ewm(span=average_length, adjust=False).mean()
            df["priceh"] = df["mah"]
            df["pricel"] = df["mal"]

        # Calculate ATR for reversal amount
        def calculate_atr(df, length):
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift(1))
            low_close = np.abs(df["low"] - df["close"].shift(1))
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.ewm(span=length, adjust=False).mean()
            return atr

        df["atr"] = calculate_atr(df, atr_length)

        # Calculate reversal amount
        df["reversal_amount"] = np.where(
            (df["close"] * percent_amount) > np.maximum(rev_amount, atr_reversal * df['atr']),
            df["close"] * percent_amount,
            np.where(
                rev_amount < atr_reversal * df["atr"],
                atr_reversal * df["atr"],
                rev_amount
            )
        )

        # Simplified ZigZag implementation focusing on key reversal points
        # This is a simplified version - the full ThinkScript has complex recursive logic
        zigzag_points = []
        zigzag_values = []
        current_direction = None
        last_extreme = None

        for i in range(len(df)):
            if i == 0:
                zigzag_points.append(i)
                zigzag_values.append(df["close"].iloc[i])
                last_extreme = df["close"].iloc[i]
                continue

            current_price = df["close"].iloc[i]
            price_change = abs(current_price - last_extreme)
            reversal_threshold = df["reversal_amount"].iloc[i]

            if price_change >= reversal_threshold:
                if current_price > last_extreme:
                    if current_direction != "up":
                        current_direction = "up"
                        zigzag_points.append(i)
                        zigzag_values.append(current_price)
                        last_extreme = current_price
                    else:
                        # Update high
                        zigzag_values[-1] = current_price
                        last_extreme = current_price
                else:
                    if current_direction != "down":
                        current_direction = "down"
                        zigzag_points.append(i)
                        zigzag_values.append(current_price)
                        last_extreme = current_price
                    else:
                        # Update low
                        zigzag_values[-1] = current_price
                        last_extreme = current_price

        # Create ZigZag series
        df["zigzag"] = np.nan
        for i, point in enumerate(zigzag_points):
            df.loc[df.index[point], "zigzag"] = zigzag_values[i]

        # Determine latest signal states for trading decisions
        last_buy_signal = bool(df["Buy_Signal"].iloc[-1])
        last_sell_signal = bool(df["Sell_Signal"].iloc[-1])
        last_momentum_down = bool(df["Momentum_Down"].iloc[-1])
        last_momentum_up = bool(df["Momentum_Up"].iloc[-1])

        # Get current colorbar state
        current_colorbars = df["ColorBars"].iloc[-1]
        previous_colorbars = df["ColorBars"].iloc[-2] if len(df) > 1 else 0

        logger.info(f"{ticker} - Current ColorBars: {current_colorbars}, Previous: {previous_colorbars}")
        logger.info(f"{ticker} - Buy Signal: {last_buy_signal}, Sell Signal: {last_sell_signal}")
        logger.info(f"{ticker} - Momentum Down: {last_momentum_down}, Momentum Up: {last_momentum_up}")

        # Trading Logic: Execute orders immediately when reversal candles are printed
        if ticker not in trades:
            # No existing position - check for new signals
            if last_buy_signal and current_colorbars == 1:  # Green badge - Buy signal
                logger.info(f"Buy signal for {ticker}: Placing LONG orders immediately.")
                order_id_tasty = (
                    place_tastytrade_order(
                        ticker, qty_tastytrade, "Buy to Open", TASTY_ACCOUNT_ID, logger
                    )
                    if int(qty_tastytrade) > 0
                    else 0
                )
                trades[ticker] = {
                    "position": "LONG",
                    "order_id_tastytrade": order_id_tasty,
                    "entry_price": df["close"].iloc[-1],
                    "entry_time": datetime.now(pytz.utc).isoformat(),
                    "signal_type": "BUY",
                    "colorbars": current_colorbars
                }
                
            elif last_sell_signal and current_colorbars == 2:  # Red badge - Sell signal
                logger.info(f"Sell signal for {ticker}: Placing SHORT orders immediately.")
                order_id_tasty = (
                    place_tastytrade_order(
                        ticker, qty_tastytrade, "Sell to Open", TASTY_ACCOUNT_ID, logger
                    )
                    if int(qty_tastytrade) > 0
                    else 0
                )
                trades[ticker] = {
                    "position": "SHORT",
                    "order_id_tastytrade": order_id_tasty,
                    "entry_price": df["close"].iloc[-1],
                    "entry_time": datetime.now(pytz.utc).isoformat(),
                    "signal_type": "SELL",
                    "colorbars": current_colorbars
                }
                
        else:
            # Existing position - check for reversal or unprinting
            current_position = trades[ticker]["position"]
            current_colorbars = trades[ticker].get("colorbars", 0)
            
            # Check for candle unprinting (trend continues in opposite direction)
            if current_position == "LONG":
                if last_sell_signal and current_colorbars == 2:  # Red badge appears
                    logger.info(f"Reversing LONG to SHORT for {ticker}: Closing LONG, opening SHORT")
                    
                    # Close LONG position
                    close_id_tasty = (
                        place_tastytrade_order(
                            ticker, qty_tastytrade, "Sell to Close", TASTY_ACCOUNT_ID, logger
                        )
                        if int(qty_tastytrade) > 0
                        else 0
                    )
                    
                    # Open SHORT position
                    open_id_tasty = (
                        place_tastytrade_order(
                            ticker, qty_tastytrade, "Sell to Open", TASTY_ACCOUNT_ID, logger
                        )
                        if int(qty_tastytrade) > 0
                        else 0
                    )
                    
                    trades[ticker] = {
                        "position": "SHORT",
                        "order_id_tastytrade": open_id_tasty,
                        "entry_price": df["close"].iloc[-1],
                        "entry_time": datetime.now(pytz.utc).isoformat(),
                        "signal_type": "SELL",
                        "colorbars": current_colorbars,
                        "previous_position": "LONG",
                        "close_order_id": close_id_tasty
                    }
                    
                elif last_momentum_down and current_colorbars == 3:  # Candle unprinted - momentum down
                    logger.info(f"Candle unprinted for {ticker} LONG position: Closing and reversing to SHORT")
                    
                    # Close LONG position due to unprinting
                    close_id_tasty = (
                        place_tastytrade_order(
                            ticker, qty_tastytrade, "Sell to Close", TASTY_ACCOUNT_ID, logger
                        )
                        if int(qty_tastytrade) > 0
                        else 0
                    )
                    
                    # Open SHORT position
                    open_id_tasty = (
                        place_tastytrade_order(
                            ticker, qty_tastytrade, "Sell to Open", TASTY_ACCOUNT_ID, logger
                        )
                        if int(qty_tastytrade) > 0
                        else 0
                    )
                    
                    trades[ticker] = {
                        "position": "SHORT",
                        "order_id_tastytrade": open_id_tasty,
                        "entry_price": df["close"].iloc[-1],
                        "entry_time": datetime.now(pytz.utc).isoformat(),
                        "signal_type": "UNPRINT_SELL",
                        "colorbars": current_colorbars,
                        "previous_position": "LONG",
                        "close_order_id": close_id_tasty
                    }
                    
            elif current_position == "SHORT":
                if last_buy_signal and current_colorbars == 1:  # Green badge appears
                    logger.info(f"Reversing SHORT to LONG for {ticker}: Closing SHORT, opening LONG")
                    
                    # Close SHORT position
                    close_id_tasty = (
                        place_tastytrade_order(
                            ticker, qty_tastytrade, "Buy to Close", TASTY_ACCOUNT_ID, logger
                        )
                        if int(qty_tastytrade) > 0
                        else 0
                    )
                    
                    # Open LONG position
                    open_id_tasty = (
                        place_tastytrade_order(
                            ticker, qty_tastytrade, "Buy to Open", TASTY_ACCOUNT_ID, logger
                        )
                        if int(qty_tastytrade) > 0
                        else 0
                    )
                    
                    trades[ticker] = {
                        "position": "LONG",
                        "order_id_tastytrade": open_id_tasty,
                        "entry_price": df["close"].iloc[-1],
                        "entry_time": datetime.now(pytz.utc).isoformat(),
                        "signal_type": "BUY",
                        "colorbars": current_colorbars,
                        "previous_position": "SHORT",
                        "close_order_id": close_id_tasty
                    }
                    
                elif last_momentum_up and current_colorbars == 3:  # Candle unprinted - momentum up
                    logger.info(f"Candle unprinted for {ticker} SHORT position: Closing and reversing to LONG")
                    
                    # Close SHORT position due to unprinting
                    close_id_tasty = (
                        place_tastytrade_order(
                            ticker, qty_tastytrade, "Buy to Close", TASTY_ACCOUNT_ID, logger
                        )
                        if int(qty_tastytrade) > 0
                        else 0
                    )
                    
                    # Open LONG position
                    open_id_tasty = (
                        place_tastytrade_order(
                            ticker, qty_tastytrade, "Buy to Open", TASTY_ACCOUNT_ID, logger
                        )
                        if int(qty_tastytrade) > 0
                        else 0
                    )
                    
                    trades[ticker] = {
                        "position": "LONG",
                        "order_id_tastytrade": open_id_tasty,
                        "entry_price": df["close"].iloc[-1],
                        "entry_time": datetime.now(pytz.utc).isoformat(),
                        "signal_type": "UNPRINT_BUY",
                        "colorbars": current_colorbars,
                        "previous_position": "SHORT",
                        "close_order_id": close_id_tasty
                    }

        # Save trade states
        with open(trades_file, "w") as f:
            json.dump(trades, f)

        logger.info(f"Reversal candle strategy for {ticker} completed at {datetime.now(pytz.utc)}")
        
    except Exception as e:
        logger.error(f"Exception in reversal candle strategy for {ticker}: {str(e)}", exc_info=True)

