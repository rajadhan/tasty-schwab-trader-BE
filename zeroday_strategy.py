from pathlib import Path
from utils import (
    get_strategy_prarams,
    get_trade_file_path,
    load_json,
    wilders_smoothing,
)

import json
from datetime import datetime
import pytz

from config import TIME_ZONE
from strategy_consumer import StrategyConsumer
from brokers.tastytrade import (
    place_option_trade as place_tasty_order,
)
from brokers.schwab import (
    historical_data,
    place_order as place_schwab_order
)


def zeroday_strategy(ticker, logger):
    strategy_consumer = StrategyConsumer()
    """Runs the trading strategy for the specified ticker."""
    try:
        [
            timeframe,
            schwab_qty,
            trade_enabled,
            tasty_qty,
            trend_line_1,
            period_1,
            trend_line_2,
            period_2,
        ] = get_strategy_prarams("zeroday", ticker, logger)

        if trade_enabled != "TRUE":
            logger.info(f"Skipping  strategy for {ticker}, trade flag is FALSE.")
            trade_file = get_trade_file_path(ticker, "zeroday")
            try:
                with open(trade_file, "r") as file:
                    trades = json.load(file)
                if ticker in trades:
                    trades = {}
                    with open(trade_file, "w") as file:
                        json.dump(trades, file)
            except FileNotFoundError:
                trades = {}
            return

        logger.info(
            f"Running strategy for {ticker} at {datetime.now(tz=pytz.timezone(TIME_ZONE))} with params: Tasty_QTY={tasty_qty} TRENDS=({period_1}, {trend_line_1}), ({period_2}, {trend_line_2})"
        )
        schwab_qty = int(schwab_qty) # Convert Schwab quantity into integer type
        tasty_qty = int(tasty_qty) # Convert Tasty quantity into integer type
        # Load trade history
        trade_file = get_trade_file_path(ticker, "zeroday")
        trades = load_json(trade_file)

        logger.info(f"Using tick data for {ticker}")
        df = historical_data(ticker, timeframe, logger)

        if df is None:
            logger.warning(f"No tick data available for {ticker}")
            return

        if df is None or len(df) < max(int(period_1), int(period_2)):
            logger.warning(f"Insufficient data for {ticker}")
            return
        # Calculate trend lines
        ## Trend Line 1
        if trend_line_1 == "EMA":
            df["trend1"] = df["close"].ewm(span=int(period_1)).mean()
        elif trend_line_1 == "SMA":
            df["trend1"] = df["close"].rolling(window=int(period_1)).mean()
        elif trend_line_1 == "WilderSmoother":
            df["trend1"] = wilders_smoothing(df, length=int(period_1))
        ## Trend Line 2
        if trend_line_2 == "EMA":
            df["trend2"] = df["close"].ewm(span=int(period_2)).mean()
        elif trend_line_2 == "SMA":
            df["trend2"] = df["close"].rolling(window=int(period_2)).mean()
        elif trend_line_2 == "WilderSmoother":
            df["trend2"] = wilders_smoothing(df, length=int(period_2))

        latest = {
            "time": df.index[-1],
            "trend1": df.iloc[-1]["trend1"],
            "trend2": df.iloc[-1]["trend2"],
        }
        prev = {
            "time": df.index[-2],
            "trend1": df.iloc[-2]["trend1"],
            "trend2": df.iloc[-2]["trend2"],
        }
        logger.info(f"latest data {ticker}: {latest}, {prev}")
        Long_condition = (
            df.iloc[-1]["trend1"] > df.iloc[-1]["trend2"]
            and df.iloc[-2]["trend1"] < df.iloc[-2]["trend2"]
        )
        logger.info(f"Long condition for {ticker}: {Long_condition}")
        Short_condition = (
            df.iloc[-1]["trend1"] < df.iloc[-1]["trend2"]
            and df.iloc[-2]["trend1"] > df.iloc[-2]["trend2"]
        )
        logger.info(f"Short condition for {ticker}: {Short_condition}")

        # Shift previous values of trend1 and trend2
        prev_trend1 = df["trend1"].shift(1)
        prev_trend2 = df["trend2"].shift(1)
        df["method"] = ""
        df.loc[
            (df["trend1"] > df["trend2"]) & (prev_trend1 < prev_trend2), "method"
        ] = "LONG"
        df.loc[
            (df["trend1"] < df["trend2"]) & (prev_trend1 > prev_trend2), "method"
        ] = "SHORT"
        df.to_csv(
            f"logs/zeroday/{ticker[1:] if '/' == ticker[0] else ticker}.csv",
            index=True,
            index_label="timestamp",
        )

        # Execute trades based on conditions
        if ticker not in trades:
            if Long_condition:
                logger.info(
                    f"Long condition triggered for {ticker} (zeroday strategy) - Buying CALL option"
                )
                # Buy at-the-money CALL option
                order_id_schwab = (
                    place_schwab_order(
                        ticker, schwab_qty, "BUY", logger, "OPENING"
                    )
                    if schwab_qty > 0
                    else 0
                )
                order_id_tastytrade = (
                    place_tasty_order(
                        ticker,
                        "CALL",
                        "Buy to Open",
                        tasty_qty,
                        logger,
                    )
                    if tasty_qty > 0
                    else 0
                )
                trades[ticker] = {
                    "action": "LONG",
                    "option_type": "CALL",
                    "order_id_schwab": order_id_schwab,
                    "order_id_tastytrade": order_id_tastytrade,
                    "entry_time": datetime.now(pytz.utc).isoformat(),
                    "entry_price": df.iloc[-1]["close"],
                }
            elif Short_condition:
                logger.info(
                    f"Short condition triggered for {ticker} (zeroday strategy) - Buying PUT option"
                )
                # Buy at-the-money PUT option
                order_id_schwab = place_schwab_order(
                    ticker, schwab_qty, "SELL_SHORT", logger, "OPENING"
                ) if schwab_qty > 0 else 0
                order_id_tastytrade = (
                    place_tasty_order(
                        ticker,
                        "PUT",
                        "Buy to Open",
                        tasty_qty,
                        logger,
                    )
                    if tasty_qty > 0
                    else 0
                )
                trades[ticker] = {
                    "action": "SHORT",
                    "option_type": "PUT",
                    "order_id_schwab": order_id_schwab,
                    "order_id_tastytrade": order_id_tastytrade,
                    "entry_time": datetime.now(pytz.utc).isoformat(),
                    "entry_price": df.iloc[-1]["close"],
                }
        else:
            if trades[ticker]["action"] == "LONG" and Short_condition:
                logger.info(
                    f"Reversing position for {ticker}: Closing CALL, opening PUT (zeroday strategy)"
                )
                # Close current CALL position
                close_order_id_schwab = place_schwab_order(
                    ticker, schwab_qty, "SELL", logger, "CLOSING"
                ) if schwab_qty > 0 else 0
                close_order_id = (
                    place_tasty_order(
                        ticker,
                        "CALL",
                        "Sell to Close",
                        tasty_qty,
                        logger,
                    )
                    if tasty_qty > 0
                    else 0
                )
                # Open new PUT position
                open_order_id_schwab = place_schwab_order(
                    ticker, schwab_qty, "SELL_SHORT", logger, "OPENING"
                ) if schwab_qty > 0 else 0
                open_order_id = (
                    place_tasty_order(
                        ticker,
                        "PUT",
                        "Buy to Open",
                        tasty_qty,
                        logger,
                    )
                    if tasty_qty > 0
                    else 0
                )
                trades[ticker] = {
                    "action": "SHORT",
                    "option_type": "PUT",
                    "order_id_schwab": open_order_id_schwab,
                    "order_id_tastytrade": open_order_id,
                    "entry_time": datetime.now(pytz.utc).isoformat(),
                    "entry_price": df.iloc[-1]["close"],
                }

            elif trades[ticker]["action"] == "SHORT" and Long_condition:
                logger.info(
                    f"Reversing position for {ticker}: Closing PUT, opening CALL (zeroday strategy)"
                )
                # Close current PUT position
                close_order_id_schwab = place_schwab_order(
                    ticker, schwab_qty, "BUY_TO_COVER", logger, "CLOSING"
                ) if schwab_qty > 0 else 0
                close_order_id = (
                    place_tasty_order(
                        ticker,
                        "PUT",
                        "Sell to Close",
                        tasty_qty,
                        logger,
                    )
                    if tasty_qty > 0
                    else 0
                )
                # Open new CALL position
                open_order_id_schwab = place_schwab_order(
                    ticker, schwab_qty, "BUY", logger, "OPENING"
                ) if schwab_qty > 0 else 0
                open_order_id = (
                    place_tasty_order(
                        ticker,
                        "CALL",
                        "Buy to Open",
                        tasty_qty,
                        logger,
                    )
                    if tasty_qty > 0
                    else 0
                )
                trades[ticker] = {
                    "action": "LONG",
                    "option_type": "CALL",
                    "order_id_schwab": open_order_id_schwab,
                    "order_id_tastytrade": open_order_id,
                    "entry_time": datetime.now(pytz.utc).isoformat(),
                    "entry_price": df.iloc[-1]["close"],
                }
        with open(
            f"trades/zeroday/{ticker[1:] if '/' == ticker[0] else ticker}.json", "w"
        ) as file:
            json.dump(trades.copy(), file)

        logger.info(f"Strategy for {ticker} completed.")

    except Exception as e:
        logger.error(f"Error in strategy for {ticker}: {e}", exc_info=True)
