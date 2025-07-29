import threading
import json
import os
from time import sleep
from datetime import datetime
import pytz
from config import *
from utils import is_holiday, is_within_time_range, get_current_datetime, get_market_hours, get_strategy_prarams, sleep_until_next_interval, configure_logger, store_logs, wilders_smoothing, sleep_base_on_timeframe
from schwab.client import historical_data, place_order
import schedule
from tastytrade import place_tastytrade_order

TICKER_DATA_PATH = os.path.join('settings', 'ticker_data.json')

def ema_strategy(ticker, logger):
    """Runs the trading strategy for the specified ticker."""
    try:
        print(ticker + " strategy started")
        [timeframe, schwab_qty, trade_enabled, tasty_qty, trend_line_1, period_1, trend_line_2, period_2] = (
            get_strategy_prarams(ticker, logger)
        )
        if trade_enabled != "TRUE":
            logger.info(f"Skipping strategy for {ticker}, trade flag is FALSE.")
            with open(
                f"trades/{ticker[1:] if '/' == ticker[0] else ticker}.json", "r"
            ) as file:
                trades = json.load(file)
            if ticker in trades:
                trades = {}
                with open(
                    f"trades/{ticker[1:] if '/' == ticker[0] else ticker}.json",
                    "w",
                ) as file:
                    json.dump(trades, file)
            return

        logger.info(
            f"Running strategy for {ticker} at {datetime.now(tz=pytz.timezone(time_zone))} with params: QTY={schwab_qty}, TRENDS=({period_1}, {trend_line_1}), ({period_2}, {trend_line_2})"
        )

        schwab_qty = int(schwab_qty)
        tasty_qty = int(tasty_qty)
        with open(
            f"trades/{ticker[1:] if '/' == ticker[0] else ticker}.json", "r"
        ) as file:
            trades = json.load(file)

        df = historical_data(
            ticker,
            timeframe,
            logger=logger,
        )

        if trend_line_1 == "EMA":
            df["trend1"] = df["close"].ewm(span=int(period_1)).mean()
        elif trend_line_1 == "SMA":
            df["trend1"] = df["close"].rolling(window=int(period_1)).mean()
        elif trend_line_1 == "WilderSmoother":
            df["trend1"] = wilders_smoothing(df, length=int(period_1))

        if trend_line_2 == "EMA":
            df["trend2"] = df["close"].ewm(span=int(period_2)).mean()
        elif trend_line_2 == "SMA":
            df["trend2"] = df["close"].rolling(window=int(period_2)).mean()
        elif trend_line_2 == "WilderSmoother":
            df["trend2"] = wilders_smoothing(df, length=int(period_2))

        Long_condition = (
            df.iloc[-1]["trend1"] > df.iloc[-1]["trend2"]
            and df.iloc[-2]["trend1"] < df.iloc[-2]["trend2"]
        )
        Short_condition = (
            df.iloc[-1]["trend1"] < df.iloc[-1]["trend2"]
            and df.iloc[-2]["trend1"] > df.iloc[-2]["trend2"]
        )

        if ticker not in trades.copy():
            if Long_condition:
                logger.info(f"Long condition triggered for {ticker}")
                order_id_schwab = place_order(ticker, schwab_qty, "BUY", account_id, logger, "OPENING") if schwab_qty > 0 else 0
                order_id_tastytrade = place_tastytrade_order(ticker, tasty_qty, "Buy to Open", account_id, logger) if tasty_qty > 0 else 0
                trades[ticker] = {"action": "LONG", "order_id_schwab": order_id_schwab, "order_id_tastytrade": order_id_tastytrade}
            elif Short_condition:
                logger.info(f"Short condition triggered for {ticker}")
                order_id_schwab = place_order(ticker, schwab_qty, "SELL_SHORT", account_id, logger, "OPENING") if schwab_qty > 0 else 0
                order_id_tastytrade = place_tastytrade_order(ticker, tasty_qty, "Sell to Open", account_id, logger) if tasty_qty > 0 else 0
                trades[ticker] = {"action": "SHORT", "order_id_schwab": order_id_schwab, "order_id_tastytrade": order_id_tastytrade}
        else:
            if trades[ticker]["action"] == "LONG" and Short_condition:
                logger.info(
                    f"Reversing position for {ticker}: Closing LONG, opening SHORT"
                )
                long_order_id_schwab = place_order(ticker, schwab_qty, "SELL", account_id, logger, "CLOSING") if schwab_qty > 0 else 0
                long_order_id_tastytrade = place_tastytrade_order(ticker, tasty_qty, "Sell to Close", account_id, logger) if tasty_qty > 0 else 0
                short_order_id_schwab = place_order(ticker, schwab_qty, "SELL_SHORT", account_id, logger, "OPENING") if schwab_qty > 0 else 0
                short_order_id_tastytrade = place_tastytrade_order(ticker, tasty_qty, "Sell to Open", account_id, logger) if tasty_qty > 0 else 0
                trades[ticker] = {"action": "SHORT", "order_id_schwab": short_order_id_schwab, "order_id_tastytrade": short_order_id_tastytrade}

            elif trades[ticker]["action"] == "SHORT" and Long_condition:
                logger.info(
                    f"Reversing position for {ticker}: Closing SHORT, opening LONG"
                )
                short_order_id_schwab = place_order(ticker, schwab_qty, "BUY_TO_COVER", account_id, logger, "CLOSING") if schwab_qty > 0 else 0
                short_order_id_tastytrade = place_tastytrade_order(ticker, tasty_qty, "Buy to Close", account_id, logger) if tasty_qty > 0 else 0
                long_order_id_schwab = place_order(ticker, schwab_qty, "BUY", account_id, logger, "OPENING") if schwab_qty > 0 else 0
                long_order_id_tastytrade = place_tastytrade_order(ticker, tasty_qty, "Buy to Open", account_id, logger) if tasty_qty > 0 else 0
                trades[ticker] = {"action": "LONG", "order_id_schwab": long_order_id_schwab, "order_id_tastytrade": long_order_id_tastytrade}

        with open(
            f"trades/{ticker[1:] if '/' == ticker[0] else ticker}.json", "w"
        ) as file:
            json.dump(trades.copy(), file)

        logger.info(f"Strategy for {ticker} completed.")

    except Exception as e:
        logger.error(f"Error in strategy for {ticker}: {e}", exc_info=True)

# NOTE: Replace the following helper functions with your real implementations:
#
# get_strategy_params(ticker) -> dict of user parameters (front-end configurable)
# get_historical_data(ticker, timeframe, logger) -> pd.DataFrame with at least ['open', 'high', 'low', 'close', 'volume']
# place_order_api(ticker, quantity, action, account_id, logger, order_type) -> order_id int/str
# wilders_smoothing(df_or_series, length) -> pd.Series of smoothed data

def supertrend_strategy(ticker, logger, get_strategy_params, get_historical_data, place_order_api, account_id):
    try:
        logger.info(f"{ticker} strategy started at {datetime.now(pytz.utc)}")

        # 1. Get parameters from frontend/backend
        params = get_strategy_params(ticker)
        # Example expected params (can be adjusted):
        timeframe = params.get("timeframe", "15min")
        trade_enabled = params.get("trade_enabled", True)
        qty_schwab = int(params.get("quantity_schwab", 10))
        qty_tastytrade = int(params.get("quantity_tastytrade", 10))
        # MA params
        short_ma_len = int(params.get("short_ma_length", 9))
        short_ma_type = params.get("short_ma_type", "EMA")
        mid_ma_len = int(params.get("mid_ma_length", 14))
        mid_ma_type = params.get("mid_ma_type", "EMA")
        long_ma_len = int(params.get("long_ma_length", 21))
        long_ma_type = params.get("long_ma_type", "EMA")
        # ZigZag/ATR params
        zigzag_percent_reversal = float(params.get("zigzag_percent_reversal", 1.0))  # in percent, e.g. 1.0 for 1%
        atr_length = int(params.get("atr_length", 14))
        atr_reversal_mult = float(params.get("zigzag_atr_multiple", 2.0))
        # Fibonacci & Support/Demand
        fibonacci_enabled = bool(params.get("fibonacci_enabled", False))
        support_demand_enabled = bool(params.get("support_demand_enabled", False))
        timezone_str = params.get("timezone", "America/New_York")
        show_volume_bubbles = bool(params.get("show_volume_bubbles", False))
        show_bubbles_price = bool(params.get("show_bubbles_price", False))

        trades_file = f"trades/{ticker[1:] if ticker.startswith('/') else ticker}.json"

        if not trade_enabled:
            logger.info(f"Trade disabled for {ticker}, skipping execution and clearing trades file if any.")
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

        # 2. Load existing trades or initialize empty
        try:
            with open(trades_file, "r") as f:
                trades = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            trades = {}

        # 3. Get historical data
        df = get_historical_data(ticker, timeframe, logger)
        # Ensure required columns exist
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"Missing required price/volume column '{col}' in historical data")

        # 4. Define helper: Moving Average calculators
        def calc_ma(series, length, ma_type):
            if ma_type.upper() == "EMA":
                return series.ewm(span=length, adjust=False).mean()
            elif ma_type.upper() == "SMA":
                return series.rolling(window=length, min_periods=1).mean()
            elif ma_type.lower() == "wildersmoother" or ma_type.lower() == "wilder":
                # Wilder's Smoothing - EMA with alpha=1/length, implemented manually
                alpha = 1 / length
                wma = np.zeros(len(series))
                wma[0] = series.iloc[0]
                for i in range(1, len(series)):
                    wma[i] = wma[i-1] + alpha*(series.iloc[i] - wma[i-1])
                return pd.Series(wma, index=series.index)
            else:
                logger.warning(f"Unknown MA type {ma_type}, defaulting to EMA")
                return series.ewm(span=length, adjust=False).mean()

        # Calculate the three MAs (Superfast, Fast, Slow)
        df["superfast"] = calc_ma(df["close"], short_ma_len, short_ma_type)
        df["fast"] = calc_ma(df["close"], mid_ma_len, mid_ma_type)
        df["slow"] = calc_ma(df["close"], long_ma_len, long_ma_type)

        # 5. Compute Supertrend-like buy/sell and stop conditions
        buy_condition = (df["superfast"] > df["fast"]) & (df["fast"] > df["slow"]) & (df["low"] > df["superfast"])
        stop_buy_condition = df["superfast"] <= df["fast"]

        sell_condition = (df["superfast"] < df["fast"]) & (df["fast"] < df["slow"]) & (df["high"] < df["superfast"])
        stop_sell_condition = df["superfast"] >= df["fast"]

        # Stateful signal computation similar to CompoundValue from TOS
        def stateful_signal(condition_series, stop_condition_series):
            signal = [0]
            for i in range(1, len(condition_series)):
                if condition_series.iloc[i] and not stop_condition_series.iloc[i]:
                    signal.append(1)
                elif signal[-1] == 1 and stop_condition_series.iloc[i]:
                    signal.append(0)
                else:
                    signal.append(signal[-1])
            return pd.Series(signal, index=condition_series.index)

        df["buy_signal"] = stateful_signal(buy_condition, stop_buy_condition)
        df["sell_signal"] = stateful_signal(sell_condition, stop_sell_condition)

        df["Buy_Signal"] = (df["buy_signal"].shift(1).fillna(0) == 0) & (df["buy_signal"] == 1)
        df["Sell_Signal"] = (df["sell_signal"].shift(1).fillna(0) == 0) & (df["sell_signal"] == 1)

        # 6. --- ZigZag and ATR-based swing detection ---

        # Compute ATR
        def atr(df, n):
            high = df["high"]
            low = df["low"]
            close = df["close"]
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_val = tr.rolling(n, min_periods=1).mean()  # Wilder smoothing could be used here
            return atr_val

        df["ATR"] = atr(df, atr_length)

        # ZigZag logic:
        # We track recent swing points (highs/lows), identify reversal when price moves by percent or ATR multiple.

        swing_highs = []
        swing_lows = []
        swing_indices = []
        swing_directions = []  # 1 for upswing, 0 for downswing
        last_extreme_price = None
        last_extreme_index = None
        last_swing_dir = None  # 1 for up, 0 for down

        percent_thresh = zigzag_percent_reversal / 100.0

        for i in range(1, len(df)):
            price = df["high"].iloc[i] if last_swing_dir != 1 else df["low"].iloc[i]
            if last_extreme_price is None:
                last_extreme_price = df["close"].iloc[0]
                last_extreme_index = 0
                last_swing_dir = 1  # start arbitrarily up
                swing_highs.append(np.nan)
                swing_lows.append(np.nan)
                swing_indices.append(np.nan)
                swing_directions.append(np.nan)
                continue

            atr_val = df["ATR"].iloc[i]
            rev_amt = max(price * percent_thresh, atr_reversal_mult * atr_val)

            # Identify if a reversal occurred
            if last_swing_dir == 1:
                # We are looking for down reversal
                if df["low"].iloc[i] < last_extreme_price - rev_amt:
                    # Swing low detected
                    swing_lows.append(df["low"].iloc[i])
                    swing_highs.append(np.nan)
                    swing_indices.append(i)
                    swing_directions.append(0)
                    last_extreme_price = df["low"].iloc[i]
                    last_extreme_index = i
                    last_swing_dir = 0
                else:
                    swing_highs.append(np.nan)
                    swing_lows.append(np.nan)
                    swing_indices.append(np.nan)
                    swing_directions.append(np.nan)

            else:
                # We are looking for up reversal
                if df["high"].iloc[i] > last_extreme_price + rev_amt:
                    # Swing high detected
                    swing_highs.append(df["high"].iloc[i])
                    swing_lows.append(np.nan)
                    swing_indices.append(i)
                    swing_directions.append(1)
                    last_extreme_price = df["high"].iloc[i]
                    last_extreme_index = i
                    last_swing_dir = 1
                else:
                    swing_highs.append(np.nan)
                    swing_lows.append(np.nan)
                    swing_indices.append(np.nan)
                    swing_directions.append(np.nan)

        df["swing_high"] = pd.Series(swing_highs, index=df.index)
        df["swing_low"] = pd.Series(swing_lows, index=df.index)
        df["swing_direction"] = pd.Series(swing_directions, index=df.index)

        # 7. Support and Demand Zones (optional)
        # For each confirmed swing, treat swing high as resistance (supply), low as support (demand)
        # For demo, track last N swing highs and lows

        support_zones = []
        demand_zones = []

        swing_points = df[["swing_high", "swing_low", "swing_direction"]].dropna(subset=["swing_direction"])
        swing_points = swing_points.reset_index()

        max_zones_to_store = 5  # configurable

        for idx, row in swing_points.iterrows():
            if row["swing_direction"] == 1 and support_demand_enabled:
                # swing high = resistance / supply zone
                support_zones.append(row["swing_high"])
                if len(support_zones) > max_zones_to_store:
                    support_zones.pop(0)
            elif row["swing_direction"] == 0 and support_demand_enabled:
                demand_zones.append(row["swing_low"])
                if len(demand_zones) > max_zones_to_store:
                    demand_zones.pop(0)

        # 8. Fibonacci retracement and extension levels based on last swings
        # For demo: calculate using last two significant swings (high and low)

        fib_levels = {}
        if fibonacci_enabled and len(swing_points) >= 2:
            # Find last swing high and low in order
            last_swing = swing_points.iloc[-1]
            second_last_swing = swing_points.iloc[-2]

            # Determine up or down swing
            if last_swing["swing_direction"] == 1 and second_last_swing["swing_direction"] == 0:
                # Up swing from low to high
                swing_low_val = second_last_swing["swing_low"]
                swing_high_val = last_swing["swing_high"]
            elif last_swing["swing_direction"] == 0 and second_last_swing["swing_direction"] == 1:
                # Down swing from high to low
                swing_high_val = second_last_swing["swing_high"]
                swing_low_val = last_swing["swing_low"]
            else:
                swing_low_val = None
                swing_high_val = None

            if swing_low_val is not None and swing_high_val is not None:
                diff = swing_high_val - swing_low_val
                fib_levels = {
                    "fib_0%": swing_low_val,
                    "fib_23.6%": swing_high_val - diff * 0.236,
                    "fib_38.2%": swing_high_val - diff * 0.382,
                    "fib_50%": swing_high_val - diff * 0.5,
                    "fib_61.8%": swing_high_val - diff * 0.618,
                    "fib_78.6%": swing_high_val - diff * 0.786,
                    "fib_100%": swing_high_val,
                    # Extensions beyond 100%
                    "fib_161.8%": swing_high_val + diff * 0.618,
                    "fib_261.8%": swing_high_val + diff * 1.618,
                }

        # 9. Use Supertrend signals and swing info to place/close trades

        last_buy_signal = df["Buy_Signal"].iloc[-1]
        last_sell_signal = df["Sell_Signal"].iloc[-1]

        if ticker not in trades:
            if last_buy_signal:
                logger.info(f"Buy signal for {ticker}: Placing LONG orders.")
                order_id_schwab = place_order_api(ticker, qty_schwab, "BUY", account_id, logger, "OPENING") if qty_schwab > 0 else 0
                order_id_tasty = place_order_api(ticker, qty_tasty, "Buy to Open", account_id, logger) if qty_tasty > 0 else 0
                trades[ticker] = {"position": "LONG", "order_id_schwab": order_id_schwab, "order_id_tastytrade": order_id_tasty}
            elif last_sell_signal:
                logger.info(f"Sell signal for {ticker}: Placing SHORT orders.")
                order_id_schwab = place_order_api(ticker, qty_schwab, "SELL_SHORT", account_id, logger, "OPENING") if qty_schwab > 0 else 0
                order_id_tasty = place_order_api(ticker, qty_tasty, "Sell to Open", account_id, logger) if qty_tasty > 0 else 0
                trades[ticker] = {"position": "SHORT", "order_id_schwab": order_id_schwab, "order_id_tastytrade": order_id_tasty}
        else:
            current_position = trades[ticker]["position"]
            if current_position == "LONG" and last_sell_signal:
                logger.info(f"Reversing LONG to SHORT for {ticker}: Closing LONG, opening SHORT")
                close_id_schwab = place_order_api(ticker, qty_schwab, "SELL", account_id, logger, "CLOSING") if qty_schwab > 0 else 0
                close_id_tasty = place_order_api(ticker, qty_tasty, "Sell to Close", account_id, logger) if qty_tasty > 0 else 0
                open_id_schwab = place_order_api(ticker, qty_schwab, "SELL_SHORT", account_id, logger, "OPENING") if qty_schwab > 0 else 0
                open_id_tasty = place_order_api(ticker, qty_tasty, "Sell to Open", account_id, logger) if qty_tasty > 0 else 0
                trades[ticker] = {"position": "SHORT", "order_id_schwab": open_id_schwab, "order_id_tastytrade": open_id_tasty}

            elif current_position == "SHORT" and last_buy_signal:
                logger.info(f"Reversing SHORT to LONG for {ticker}: Closing SHORT, opening LONG")
                close_id_schwab = place_order_api(ticker, qty_schwab, "BUY_TO_COVER", account_id, logger, "CLOSING") if qty_schwab > 0 else 0
                close_id_tasty = place_order_api(ticker, qty_tasty, "Buy to Close", account_id, logger) if qty_tasty > 0 else 0
                open_id_schwab = place_order_api(ticker, qty_schwab, "BUY", account_id, logger, "OPENING") if qty_schwab > 0 else 0
                open_id_tasty = place_order_api(ticker, qty_tasty, "Buy to Open", account_id, logger) if qty_tasty > 0 else 0
                trades[ticker] = {"position": "LONG", "order_id_schwab": open_id_schwab, "order_id_tastytrade": open_id_tasty}

        # 10. Save trade states
        with open(trades_file, "w") as f:
            json.dump(trades, f)

        # Optionally log important swing/fib/support info for monitoring/debugging
        logger.info(f"{ticker} swings detected: {len(swing_points)}")
        if fibonacci_enabled:
            logger.info(f"{ticker} current Fibonacci levels: {fib_levels}")
        if support_demand_enabled:
            logger.info(f"{ticker} support zones: {support_zones}")
            logger.info(f"{ticker} demand zones: {demand_zones}")

        logger.info(f"Strategy for {ticker} completed at {datetime.now(pytz.utc)}")

    except Exception as e:
        logger.error(f"Exception in strategy for {ticker}: {str(e)}", exc_info=True)

def main_strategy_loop(ticker):
    """Main loop for running the strategy for a specific ticker."""
    logger = configure_logger(ticker)

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


def run_every_week():
    """Starts threads for each ticker."""
    with open(TICKER_DATA_PATH, 'r') as file:
        ticker_n_tf = json.load(file)

    threads = []
    for ticker in ticker_n_tf.keys():
        thread = threading.Thread(target=main_strategy_loop, args=(ticker,))
        threads.append(thread)
        thread.start()
        print(ticker + " started")

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
