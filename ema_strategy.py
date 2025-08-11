from utils import (
    get_strategy_prarams,
    wilders_smoothing,
    get_trade_file_path,
    load_json
)

import json
from datetime import datetime
import pytz
from config import *
from get_historical_data import get_historical_data


def ema_strategy(ticker, logger):
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
        ] = get_strategy_prarams("ema", ticker, logger)

        if trade_enabled != "TRUE":
            logger.info(f"Skipping  strategy for {ticker}, trade flag is FALSE.")
            trade_file = get_trade_file_path(ticker, "ema")
            print("trade_file", trade_file)  # TODO
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
            f"Running strategy for {ticker} at {datetime.now(tz=pytz.timezone(TIME_ZONE))} with params: Schwab_QTY={schwab_qty}, Tasty_QTY={tasty_qty} TRENDS=({period_1}, {trend_line_1}), ({period_2}, {trend_line_2})"
        )
        schwab_qty = int(schwab_qty)
        tasty_qty = int(tasty_qty)
        trade_file = get_trade_file_path(ticker, "ema")
        trades = load_json(trade_file)

        df = get_historical_data(
            ticker,
            timeframe,
            logger=logger,
        )
        print("df", df)  # TODO

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
                # order_id_schwab = (
                #     place_order(
                #         ticker, schwab_qty, "BUY", schwab_account_id, logger, "OPENING"
                #     )
                #     if schwab_qty > 0
                #     else 0
                # )
                order_id_tastytrade = (
                    place_tastytrade_order(
                        ticker, tasty_qty, "Buy to Open", TASTY_ACCOUNT_ID, logger
                    )
                    if tasty_qty > 0
                    else 0
                )
                trades[ticker] = {
                    "action": "LONG",
                    # "order_id_schwab": order_id_schwab,
                    "order_id_tastytrade": order_id_tastytrade,
                }
            elif Short_condition:
                logger.info(f"Short condition triggered for {ticker}")
                order_id_schwab = (
                    place_order(
                        ticker,
                        schwab_qty,
                        "SELL_SHORT",
                        schwab_account_id,
                        logger,
                        "OPENING",
                    )
                    if schwab_qty > 0
                    else 0
                )
                order_id_tastytrade = (
                    place_tastytrade_order(
                        ticker, tasty_qty, "Sell to Open", TASTY_ACCOUNT_ID, logger
                    )
                    if tasty_qty > 0
                    else 0
                )
                trades[ticker] = {
                    "action": "SHORT",
                    "order_id_schwab": order_id_schwab,
                    "order_id_tastytrade": order_id_tastytrade,
                }
        else:
            if trades[ticker]["action"] == "LONG" and Short_condition:
                logger.info(
                    f"Reversing position for {ticker}: Closing LONG, opening SHORT"
                )
                long_order_id_schwab = (
                    place_order(
                        ticker, schwab_qty, "SELL", schwab_account_id, logger, "CLOSING"
                    )
                    if schwab_qty > 0
                    else 0
                )
                long_order_id_tastytrade = (
                    place_tastytrade_order(
                        ticker, tasty_qty, "Sell to Close", TASTY_ACCOUNT_ID, logger
                    )
                    if tasty_qty > 0
                    else 0
                )
                short_order_id_schwab = (
                    place_order(
                        ticker,
                        schwab_qty,
                        "SELL_SHORT",
                        schwab_account_id,
                        logger,
                        "OPENING",
                    )
                    if schwab_qty > 0
                    else 0
                )
                short_order_id_tastytrade = (
                    place_tastytrade_order(
                        ticker, tasty_qty, "Sell to Open", TASTY_ACCOUNT_ID, logger
                    )
                    if tasty_qty > 0
                    else 0
                )
                trades[ticker] = {
                    "action": "SHORT",
                    "order_id_schwab": short_order_id_schwab,
                    "order_id_tastytrade": short_order_id_tastytrade,
                }

            elif trades[ticker]["action"] == "SHORT" and Long_condition:
                logger.info(
                    f"Reversing position for {ticker}: Closing SHORT, opening LONG"
                )
                short_order_id_schwab = (
                    place_order(
                        ticker,
                        schwab_qty,
                        "BUY_TO_COVER",
                        schwab_account_id,
                        logger,
                        "CLOSING",
                    )
                    if schwab_qty > 0
                    else 0
                )
                short_order_id_tastytrade = (
                    place_tastytrade_order(
                        ticker, tasty_qty, "Buy to Close", TASTY_ACCOUNT_ID, logger
                    )
                    if tasty_qty > 0
                    else 0
                )
                long_order_id_schwab = (
                    place_order(
                        ticker, schwab_qty, "BUY", schwab_account_id, logger, "OPENING"
                    )
                    if schwab_qty > 0
                    else 0
                )
                long_order_id_tastytrade = (
                    place_tastytrade_order(
                        ticker, tasty_qty, "Buy to Open", TASTY_ACCOUNT_ID, logger
                    )
                    if tasty_qty > 0
                    else 0
                )
                trades[ticker] = {
                    "action": "LONG",
                    "order_id_schwab": long_order_id_schwab,
                    "order_id_tastytrade": long_order_id_tastytrade,
                }

        with open(
            f"trades/{ticker[1:] if '/' == ticker[0] else ticker}.json", "w"
        ) as file:
            json.dump(trades.copy(), file)

        logger.info(f"Strategy for {ticker} completed.")

    except Exception as e:
        logger.error(f"Error in strategy for {ticker}: {e}", exc_info=True)
