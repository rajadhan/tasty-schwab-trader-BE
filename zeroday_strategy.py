def zeroday_strategy(ticker, logger):
    """Implements the zero-day options strategy for SPX.
    This strategy is designed to be manually triggered and uses EMA crossovers
    for entry and exit signals.
    """
    try:
        logger.info(f"{ticker} zeroday strategy started at {datetime.now(pytz.utc)}")

        # 1. Get parameters from frontend/backend
        [
            timeframe,
            schwab_qty,
            trade_enabled,
            tasty_qty,
            trend_line_1,
            period_1,
            trend_line_2,
            period_2,
            call_enabled,
            put_enabled,
        ] = get_strategy_prarams("zeroday", ticker, logger)

        if trade_enabled != "TRUE":
            logger.info(f"Skipping zeroday strategy for {ticker}, trade flag is FALSE.")
            trade_file = f"trades/zeroday/{ticker}.json"
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
            f"Running zeroday strategy for {ticker} at {datetime.now(tz=pytz.timezone(time_zone))} "
            f"with params: Schwab_QTY={schwab_qty}, Tasty_QTY={tasty_qty} "
            f"TRENDS=({period_1}, {trend_line_1}), ({period_2}, {trend_line_2})"
        )

        schwab_qty = int(schwab_qty)
        tasty_qty = int(tasty_qty)

        # Create trades directory if it doesn't exist
        os.makedirs(f"trades/zeroday", exist_ok=True)
        trade_file = f"trades/zeroday/{ticker}.json"
        
        try:
            with open(trade_file, "r") as file:
                trades = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            trades = {}

        # Get historical data
        df = historical_data(
            ticker,
            timeframe,
            logger=logger,
        )

        # Calculate trend lines
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

        # Define entry conditions for zero-day options
        Long_condition = (
            df.iloc[-1]["trend1"] > df.iloc[-1]["trend2"]
            and df.iloc[-2]["trend1"] < df.iloc[-2]["trend2"]
        )
        Short_condition = (
            df.iloc[-1]["trend1"] < df.iloc[-1]["trend2"]
            and df.iloc[-2]["trend1"] > df.iloc[-2]["trend2"]
        )

        # Execute trades based on conditions
        if ticker not in trades:
            if Long_condition:
                logger.info(f"Long condition triggered for {ticker} (zeroday strategy)")
                order_id_schwab = (
                    place_order(
                        ticker, schwab_qty, "BUY", schwab_account_id, logger, "OPENING"
                    )
                    if schwab_qty > 0
                    else 0
                )
                order_id_tastytrade = (
                    place_tastytrade_order(
                        ticker, tasty_qty, "Buy to Open", TASTY_ACCOUNT_ID, logger
                    )
                    if tasty_qty > 0
                    else 0
                )
                trades[ticker] = {
                    "action": "LONG",
                    "order_id_schwab": order_id_schwab,
                    "order_id_tastytrade": order_id_tastytrade,
                    "entry_time": datetime.now(pytz.utc).isoformat(),
                    "entry_price": df.iloc[-1]["close"]
                }
            elif Short_condition:
                logger.info(f"Short condition triggered for {ticker} (zeroday strategy)")
                order_id_schwab = (
                    place_order(
                        ticker, schwab_qty, "SELL_SHORT", schwab_account_id, logger, "OPENING"
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
                    "entry_time": datetime.now(pytz.utc).isoformat(),
                    "entry_price": df.iloc[-1]["close"]
                }
        else:
            if trades[ticker]["action"] == "LONG" and Short_condition:
                logger.info(
                    f"Reversing position for {ticker}: Closing LONG, opening SHORT (zeroday strategy)"
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
                        ticker, schwab_qty, "SELL_SHORT", schwab_account_id, logger, "OPENING"
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
                    "entry_time": datetime.now(pytz.utc).isoformat(),
                    "entry_price": df.iloc[-1]["close"]
                }

            elif trades[ticker]["action"] == "SHORT" and Long_condition:
                logger.info(
                    f"Reversing position for {ticker}: Closing SHORT, opening LONG (zeroday strategy)"
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
                    "entry_time": datetime.now(pytz.utc).isoformat(),
                    "entry_price": df.iloc[-1]["close"]
                }

        # Save trade data
        with open(trade_file, "w") as file:
            json.dump(trades, file)

        logger.info(f"Zeroday strategy for {ticker} completed at {datetime.now(pytz.utc)}")

    except Exception as e:
        logger.error(f"Error in zeroday strategy for {ticker}: {e}", exc_info=True)

