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
            show_supply_demand, # Pivot, Arrow, None
            use_manual_fib_skip, # Yes, No
            fib_skip,
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
        df["mov_avg9"] = df["close"].ewm(span=superfast_len, adjust=False).mean()
        df["mov_avg14"] = df["close"].ewm(span=fast_len, adjust=False).mean()
        df["mov_avg21"] = df["close"].ewm(span=slow_len, adjust=False).mean()

        # Define the moving averages as in ThinkScript
        df["Superfast"] = df["mov_avg9"]
        df["Fast"] = df["mov_avg14"]
        df["Slow"] = df["mov_avg21"]

        df["buy"] = (
            (df["Superfast"] > df["Fast"]) &
            (df["Fast"] > df["Slow"]) &
            (df["low"] > df["Superfast"])
        )
        df["stopbuy"] = df["Superfast"] <= df["Fast"]
        df["buynow"] = df["buy"] & (~df["buy"].shift(1).fillna(False))
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

        # Momentum down detection
        df["Momentum_Down"] = (df["buysignal"].shift(1).fillna(0) == 1) & (df["buysignal"] == 0)

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
        
        # Momentum up detection
        df["Momentum_Up"] = (df["sellsignal"].shift(1).fillna(0) == 1) & (df["sellsignal"] == 0)

        # Thinkscript colorbars
        def calculate_colorbars():
            colorbars = []
            for i in range(len(df)):
                if df["buysignal"].iloc[i] == 1:
                    colorbars.append(1)
                elif df["sellsignal"].iloc[i] == 1:
                    colorbars.append(2)
                elif df["buysignal"].iloc[i] == 0 or df["sellsignal"].iloc[i] == 0:
                    colorbars.append(3)
                else:
                    colorbars.append(0)
            return pd.Series(colorbars, index=df.index)
        df["ColorBars"] = calculate_colorbars()
        
        # ThinkScript ZigZag High/Low 
        bubble_offset = 0.0005
        percent_amount = 0.01
        rev_amount = 0.05
        atr_reversal = 2.0
        atr_length = 5
        average_length = 5
        
        df["mah"] = df["high"].ewm(span=average_length, adjust=False).mean()
        df["mal"] = df["low"].ewm(span=average_length, adjust=False).mean()

        if zigzag_method == "high_low":
            df["priceh"] = df["high"]
            df["pricel"] = df["low"]
        else:
            df["priceh"] = df["mah"]
            df["pricel"] = df["mal"]
        
        # Calculate ATR (Average True Range)
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

        # Initialize ZigZag arrays
        zigzag_points = []
        zigzag_values = []
        zigzag_direction = []

        # Find ZigZag points
        for i in range(1, len(df)):
            if i == 1:
                # First point
                zigzag_points.append(i)
                zigzag_values.append(df["priceh"].iloc[i])
                zigzag_direction.append(1)
                continue
            current_high = df["priceh"].iloc[i]
            current_low = df["pricel"].iloc[i]
            last_zigzag_value = zigzag_values[-1]
            last_direction = zigzag_direction[-1]
            reversal_threshold = df["reversal_amount"].iloc[i]

            if last_direction == 1:
                # Last point was a high
                if current_high > last_zigzag_value:
                    # New high - update the last point
                    zigzag_values[-1] = current_high
                    zigzag_points[-1] = i
                elif (last_zigzag_value - current_low) >= reversal_threshold:
                    zigzag_points.append(i)
                    zigzag_values.append(current_low)
                    zigzag_direction.append(0)
            else: # Last point was a low
                if current_low < last_zigzag_value:
                    zigzag_values[-1] = current_low
                    zigzag_points[-1] = i
                elif (current_high - last_zigzag_value) >= reversal_threshold:
                    zigzag_points.append(i)
                    zigzag_values.append(current_high)
                    zigzag_direction.append(1)
        # Create ZigZag series (pivot-only) and forward-filled storage
        df["EI"] = np.nan              # pivot-only values (NaN between pivots)
        df["zigzag_direction"] = np.nan

        for point, value, direction in zip(zigzag_points, zigzag_values, zigzag_direction):
            df.loc[df.index[point], "EI"] = value
            df.loc[df.index[point], "zigzag_direction"] = direction

        # Forward fill pivot values for easier comparisons (ThinkScript EISave)
        df["EISave"] = df["EI"].ffill()
        # Also provide a continuously filled zigzag value for change computations
        df["zigzag"] = df["EISave"]
        df["zigzag_direction"] = df["zigzag_direction"].fillna(method="ffill")

        # Calculate change and confirmation (ThinkScript logic)
        df["zigzag_change"] = df["zigzag"] - df["zigzag"].shift(1)
        df["is_up"] = df["zigzag_change"] >= 0
        # Back-compat with ThinkScript variable naming used below
        df["isUp"] = df["is_up"]

        # Calculate confirmation (change >= reversal amount)
        df["is_confirmed"] = (
            (df["zigzag_change"].abs() >= df["reversal_amount"]) | 
            (df["zigzag_direction"].shift(1).isna() & df["zigzag_direction"].shift(1).isna())
        )
        df["zigzag_final"] = np.where(df["is_up"], 1, 0)

        # Equivalent to: EnhancedLines = EI when EId <= 1 else NaN (keep as EI here)
        df["EnhancedLines"] = df["EI"]

        # xxhigh/xxlow: last priceh/pricel captured when EISave equals priceh/pricel, forward-filled otherwise
        df["xxhigh"] = np.where(df["EISave"].eq(df["priceh"]), df["priceh"], np.nan)
        df["xxhigh"] = df["xxhigh"].ffill()

        df["xxlow"] = np.where(df["EISave"].eq(df["pricel"]), df["pricel"], np.nan)
        df["xxlow"] = df["xxlow"].ffill()

        # chghigh/chglow: current price minus prior captured level
        df["chghigh"] = df["priceh"] - df["xxhigh"].shift(1)
        df["chglow"] = df["pricel"] - df["xxlow"].shift(1)

        # Prepare "bubbles" (annotation points) like AddChartBubble; no plotting, just data
        # showBubbleschange = no → keep it switchable
        show_bubbles_change = False

        if show_bubbles_change:
            # BarNumber() != 1  → exclude first bar (index position 0)
            idx_pos = np.arange(len(df))
            cond = (~df["EI"].isna()) & (idx_pos != 0)

            bubble_y = np.where(
                df["isUp"],
                df["priceh"] * (1.0 + float(bubble_offset)),
                df["pricel"] * (1.0 - float(bubble_offset)),
            )

            # Color logic from ThinkScript
            def bubble_color(row):
                if row["isUp"]:
                    if row["chghigh"] > 0:
                        return "green"
                    elif row["chghigh"] < 0:
                        return "red"
                    else:
                        return "yellow"
                else:
                    if row["chglow"] > 0:
                        return "green"
                    elif row["chglow"] < 0:
                        return "red"
                    else:
                        return "yellow"

            bubble_df = pd.DataFrame(
                {
                    "x": df.index,
                    "y": bubble_y,
                    "text": df["chg"].map(lambda v: f"${v:.4f}" if pd.notna(v) else ""),
                },
                index=df.index,
            )
            bubble_df["color"] = df.apply(bubble_color, axis=1)
            bubble_df = bubble_df[cond]

        # Prereqs in df: 'EI', 'isUp', 'priceh', 'pricel', 'chghigh', 'chglow'
        # Also define bubbleoffset (e.g., 0.0005)
        showBubblesprice = False  # set True to enable

        # Bubble y-position: above high when up, below low when down
        bubble_y_price = np.where(
            df["isUp"],
            df["priceh"] * (1.0 + float(bubble_offset)),
            df["pricel"] * (1.0 - float(bubble_offset)),
        )

        # Bubble text: dollar value of the priceh/pricel
        bubble_text_price = np.where(
            df["isUp"],
            df["priceh"].map(lambda v: f"${v:.4f}" if pd.notna(v) else ""),
            df["pricel"].map(lambda v: f"${v:.4f}" if pd.notna(v) else ""),
        )

        # Bubble color logic from ThinkScript
        def bubble_color_price(row):
            if row["isUp"]:
                if row["chghigh"] > 0:
                    return "green"
                elif row["chghigh"] < 0:
                    return "red"
                else:
                    return "yellow"
            else:
                if row["chglow"] > 0:
                    return "green"
                elif row["chglow"] < 0:
                    return "red"
                else:
                    return "yellow"

        # BarNumber() != 1 → exclude first bar (index 0)
        idx_pos = np.arange(len(df))
        cond_price = showBubblesprice & (~df["EI"].isna()) & (idx_pos != 0)

        bubble_price_df = pd.DataFrame(
            {
                "x": df.index,
                "y": bubble_y_price,
                "text": bubble_text_price,
                "color": [bubble_color_price(r) for _, r in df.iterrows()],
                "up": df["isUp"].astype(bool),  # orientation flag (ThinkScript last param)
            },
            index=df.index,
        )

        # Keep only rows where bubbles should be shown
        bubble_price_df = bubble_price_df[cond_price]

        # EIcount: bar count since EISave last changed
        eicount = [np.nan] * len(df)
        for i in range(len(df)):
            if i == 0:
                eicount[i] = 1
            else:
                if df["EISave"].iloc[i-1] != df["EISave"].iloc[i]:
                    eicount[i] = 1
                else:
                    eicount[i] = eicount[i-1] + 1
        df["EIcount"] = pd.Series(eicount, index=df.index)

        # EIcounthilo: running count only while EISave equals priceh or pricel
        # starts 1 on first bar that matches after being 0
        eicounthilo = [0] * len(df)
        for i in range(len(df)):
            is_hilo = (df["EISave"].iloc[i] == df["priceh"].iloc[i]) or (df["EISave"].iloc[i] == df["pricel"].iloc[i])
            if i == 0:
                eicounthilo[i] = 1 if is_hilo else 0
            else:
                if (eicounthilo[i-1] == 0) and is_hilo:
                    eicounthilo[i] = 1
                elif is_hilo:
                    eicounthilo[i] = eicounthilo[i-1] + 1
                else:
                    eicounthilo[i] = eicounthilo[i-1]
        df["EIcounthilo"] = pd.Series(eicounthilo, index=df.index)

        # EIhilo: passthrough in TS; mirrors ThinkScript line
        df["EIhilo"] = np.where(
            (df["EISave"].eq(df["priceh"])) | (df["EISave"].eq(df["pricel"])),
            df["EIcounthilo"],
            df["EIcounthilo"] + 1,
        )

        # EIcounthigh / EIcountlow: EIcount[1] when EISave equals corresponding side
        df["EIcounthigh"] = np.where(df["EISave"].eq(df["priceh"]), df["EIcount"].shift(1), np.nan)
        df["EIcountlow"]  = np.where(df["EISave"].eq(df["pricel"]), df["EIcount"].shift(1), np.nan)

        # Optional bubbles (disabled by default)
        showBubblesbarcount = False
        if showBubblesbarcount:
            idx_pos = np.arange(len(df))
            cond = (~df["EI"].isna()) & (idx_pos != 0)

            bubble_y = np.where(
                df["isUp"],
                df["priceh"] * (1.0 + float(bubble_offset)),
                df["pricel"] * (1.0 - float(bubble_offset)),
            )

            bubble_text = np.where(
                df["EISave"].eq(df["priceh"]),
                df["EIcounthigh"].map(lambda v: f"{int(v)}" if pd.notna(v) else ""),
                df["EIcountlow"].map(lambda v: f"{int(v)}" if pd.notna(v) else ""),
            )

            def bubble_color(row):
                if row["isUp"]:
                    if row["chghigh"] > 0: return "green"
                    if row["chghigh"] < 0: return "red"
                    return "yellow"
                else:
                    if row["chglow"] > 0: return "green"
                    if row["chglow"] < 0: return "red"
                    return "yellow"

            bubbles_barcount_df = pd.DataFrame(
                {
                    "x": df.index,
                    "y": bubble_y,
                    "text": bubble_text,
                    "color": [bubble_color(r) for _, r in df.iterrows()],
                    "up": df["isUp"].astype(bool),
                },
                index=df.index,
            )[cond]

        # Prereqs in df:
        # - df["EI"]            # ZigZag series (NaN between pivots)
        # - df["EISave"]        # EI forward-filled
        # - df["priceh"], df["pricel"]
        # - df["isUp"]          # bool: chg >= 0 from earlier step
        # - df["low"], df["high"]
        # - df["ColorBars"]     # 1/2/3 mapping you computed earlier

        # EIL, EIH (recursive)
        EIL = [np.nan] * len(df)
        EIH = [np.nan] * len(df)
        for i in range(len(df)):
            if pd.notna(df["EI"].iloc[i]) and (not df["isUp"].iloc[i]):
                EIL[i] = df["pricel"].iloc[i]
            else:
                EIL[i] = EIL[i-1] if i > 0 else np.nan

            if pd.notna(df["EI"].iloc[i]) and df["isUp"].iloc[i]:
                EIH[i] = df["priceh"].iloc[i]
            else:
                EIH[i] = EIH[i-1] if i > 0 else np.nan

        df["EIL"] = pd.Series(EIL, index=df.index)
        df["EIH"] = pd.Series(EIH, index=df.index)

        # dir (CompoundValue)
        dir_vals = [0]
        for i in range(1, len(df)):
            prev_dir = dir_vals[-1]
            eil_changed = (df["EIL"].iloc[i] != df["EIL"].shift(1).iloc[i])
            eih_changed = (df["EIH"].iloc[i] != df["EIH"].shift(1).iloc[i])

            cond_up = eil_changed or ((df["pricel"].iloc[i] == df["EIL"].shift(1).iloc[i]) and (df["pricel"].iloc[i] == df["EISave"].iloc[i]))
            cond_dn = eih_changed or ((df["priceh"].iloc[i] == df["EIH"].shift(1).iloc[i]) and (df["priceh"].iloc[i] == df["EISave"].iloc[i]))

            if cond_up:
                dir_vals.append(1)
            elif cond_dn:
                dir_vals.append(-1)
            else:
                dir_vals.append(prev_dir)

        df["dir"] = pd.Series(dir_vals, index=df.index)

        # signal (CompoundValue)
        signal_vals = [0]
        for i in range(1, len(df)):
            prev_sig = signal_vals[-1]
            if (df["dir"].iloc[i] > 0) and (df["pricel"].iloc[i] > df["EIL"].iloc[i]):
                signal_vals.append(1 if prev_sig <= 0 else prev_sig)
            elif (df["dir"].iloc[i] < 0) and (df["priceh"].iloc[i] < df["EIH"].iloc[i]):
                signal_vals.append(-1 if prev_sig >= 0 else prev_sig)
            else:
                signal_vals.append(prev_sig)

        df["signal"] = pd.Series(signal_vals, index=df.index)

        # Arrow events
        showarrows = True
        df["U1"] = showarrows & (df["signal"] > 0) & (df["signal"].shift(1).fillna(0) <= 0)
        df["D1"] = showarrows & (df["signal"] < 0) & (df["signal"].shift(1).fillna(0) >= 0)

        # barnumber = BarNumber()[10] → “bar index >= 10”
        idx_pos = np.arange(len(df))
        barnumber_ok = idx_pos >= 10

        # Reversal bubbles (optional)
        # Text and color mimic ThinkScript:
        # - Text: "Reversal:" + low/high at U1/D1
        # - Color: plum if ColorBars==3 else uptick/downtick (green/red)
        def bubble_row_up(i):
            return {
                "x": df.index[i],
                "y": df["low"].iloc[i] if df["isUp"].iloc[i] else df["high"].iloc[i],
                "text": f"Reversal:{df['low'].iloc[i]:.4f}",
                "color": "plum" if df["ColorBars"].iloc[i] == 3 else "green",
                "up": False,  # last TS arg (orientation)
            }

        def bubble_row_dn(i):
            return {
                "x": df.index[i],
                "y": df["low"].iloc[i] if df["isUp"].iloc[i] else df["high"].iloc[i],
                "text": f"Reversal:{df['high'].iloc[i]:.4f}",
                "color": "plum" if df["ColorBars"].iloc[i] == 3 else "red",
                "up": True,
            }

        u_idx = np.where(barnumber_ok & df["U1"].values)[0]
        d_idx = np.where(barnumber_ok & df["D1"].values)[0]

        reversal_bubbles = pd.DataFrame([*(bubble_row_up(i) for i in u_idx),
                                        *(bubble_row_dn(i) for i in d_idx)])
        
        # barnumber ~ BarNumber()[10] → index >= 10
        idx_pos = np.arange(len(df))
        barnumber_ok = idx_pos >= 10

        # Prepare arrays
        revLineTop = [np.nan] * len(df)
        revLineBot = [np.nan] * len(df)

        # Helper: ColorBars[1] and [2] (1 and 2 bars ago)
        colorbars_1 = df["ColorBars"].shift(1)
        colorbars_2 = df["ColorBars"].shift(2)

        # Iterate bars
        for i in range(len(df)):
            if barnumber_ok[i] and bool(df["D1"].iloc[i]):
                # Start/refresh bottom line at prior high
                revLineBot[i] = df["high"].shift(1).iloc[i]
                revLineTop[i] = np.nan
            elif barnumber_ok[i] and bool(df["U1"].iloc[i]):
                # Start/refresh top line at prior low
                revLineTop[i] = df["low"].shift(1).iloc[i]
                revLineBot[i] = np.nan
            elif i > 0 and not np.isnan(revLineBot[i-1]) and ((colorbars_2.iloc[i] == 2) or (colorbars_1.iloc[i] == 2)):
                # Keep extending bottom line during sell regime
                revLineBot[i] = revLineBot[i-1]
                revLineTop[i] = np.nan
            elif i > 0 and not np.isnan(revLineTop[i-1]) and ((colorbars_2.iloc[i] == 1) or (colorbars_1.iloc[i] == 1)):
                # Keep extending top line during buy regime
                revLineTop[i] = revLineTop[i-1]
                revLineBot[i] = np.nan
            else:
                revLineTop[i] = np.nan
                revLineBot[i] = np.nan

        df["revLineTop"] = pd.Series(revLineTop, index=df.index)
        df["revLineBot"] = pd.Series(revLineBot, index=df.index)

        # Plot displacement: botLine = revLineBot[-1], topLine = revLineTop[-1]
        # In ThinkScript, [-1] is one bar forward; in pandas use shift(-1)
        df["botLine"] = df["revLineBot"].shift(-1)
        df["topLine"] = df["revLineTop"].shift(-1)

        # Optional colors for plotting:
        # botLine -> light green, topLine -> light red

        # Config inputs
        usealerts = False  # not used here; wire to your alert system if needed
        numbersuppdemandtoshow = 0
        # Respect incoming config for supply/demand rendering mode
        idx = 1 if str(show_supply_demand).lower() == "pivot" else 0

        # data1: CompoundValue(1, if (EISave == priceh or EISave == pricel) then data1[1] + 1 else data1[1], 0)
        # If you already have EISave, uncomment the next lines and compute exactly; otherwise, use a neutral counter.
        # is_hilo = (df["EISave"].eq(df["priceh"])) | (df["EISave"].eq(df["pricel"]))
        # data1_vals = [0]
        # for i in range(1, len(df)):
        #     data1_vals.append(data1_vals[-1] + 1 if is_hilo.iloc[i] else data1_vals[-1])
        # df["data1"] = pd.Series(data1_vals, index=df.index)

        # If EISave not available, emulate a monotonically increasing counter so HighestAll(data1) works:
        df["data1"] = np.arange(len(df))

        # datacount1 = HighestAll(data1) - data1[1]
        highest_all_data1 = df["data1"].max()
        df["datacount1"] = highest_all_data1 - df["data1"].shift(1)

        # signal crosses 0: any sign change through zero
        prev_signal = df["signal"].shift(1).fillna(0)
        crosses0 = ((df["signal"] > 0) & (prev_signal <= 0)) | ((df["signal"] < 0) & (prev_signal >= 0))

        # rLow / rHigh with index offset idx
        rLow = [np.nan] * len(df)
        rHigh = [np.nan] * len(df)
        for i in range(len(df)):
            if crosses0.iloc[i]:
                j = i - idx
                if j >= 0:
                    rLow[i] = df["pricel"].iloc[j]
                    rHigh[i] = df["priceh"].iloc[j]
                else:
                    rLow[i] = np.nan
                    rHigh[i] = np.nan
            else:
                rLow[i] = rLow[i-1] if i > 0 else np.nan
                rHigh[i] = rHigh[i-1] if i > 0 else np.nan

        df["rLow"] = pd.Series(rLow, index=df.index)
        df["rHigh"] = pd.Series(rHigh, index=df.index)

        # HighLine / LowLine gates
        valid_show = (show_supply_demand != "None") & (~df["close"].isna())
        df["HighLine"] = np.where(
            (df["datacount1"] <= numbersuppdemandtoshow) & valid_show & (df["rHigh"].fillna(0) != 0),
            df["rHigh"],
            np.nan,
        )
        df["LowLine"] = np.where(
            (df["datacount1"] <= numbersuppdemandtoshow) & valid_show & (df["rLow"].fillna(0) != 0),
            df["rLow"],
            np.nan,
        )

        # hlUp / hlDn selection by signal
        df["hlUp"] = np.where(df["signal"] > 0, df["HighLine"], np.nan)
        df["hlDn"] = np.where(df["signal"] < 0, df["HighLine"], np.nan)

        # showsupplydemandcloud = no → no plotting here.
        # If plotting, create clouds between (hlUp, LowLine) and (hlDn, LowLine) with light green/red colors.

        # EIsave1: forward-filled EISave
        df["EIsave1"] = df["EISave"].ffill()

        # EIsave2: same as EIsave1
        df["EIsave2"] = df["EIsave1"]

        # priorEI1: when EIsave2 changes, capture previous EIsave2; else carry forward
        prior1 = [np.nan]
        for i in range(1, len(df)):
            if df["EIsave2"].iloc[i] != df["EIsave2"].iloc[i-1]:
                prior1.append(df["EIsave2"].iloc[i-1])
            else:
                prior1.append(prior1[-1])
        df["priorEI1"] = pd.Series(prior1, index=df.index)

        # priorEI2: when priorEI1 changes, capture previous priorEI1; else carry forward
        prior2 = [np.nan]
        for i in range(1, len(df)):
            if not pd.isna(df["priorEI1"].iloc[i]) and df["priorEI1"].iloc[i] != df["priorEI1"].iloc[i-1]:
                prior2.append(df["priorEI1"].iloc[i-1])
            else:
                prior2.append(prior2[-1])
        df["priorEI2"] = pd.Series(prior2, index=df.index)

        # priorEI3: when priorEI2 changes, capture previous priorEI2; else carry forward
        prior3 = [np.nan]
        for i in range(1, len(df)):
            if not pd.isna(df["priorEI2"].iloc[i]) and df["priorEI2"].iloc[i] != df["priorEI2"].iloc[i-1]:
                prior3.append(df["priorEI2"].iloc[i-1])
            else:
                prior3.append(prior3[-1])
        df["priorEI3"] = pd.Series(prior3, index=df.index)

        # Settings (match ThinkScript inputs)
        numberextfibstoshow = 2
        showFibExtLines = False   # set True to enable extension outputs
        showtodayonly = False     # if True, only for today's bars
        b = 8                     # fibextbubblespacesinexpansion

        # Helper flags
        cpo = (df["dir"].shift(1) == df["dir"]).astype(int)   # 0 when dir changes, else 1
        if showtodayonly:
            # emulate GetDay() == GetLastDay(): use the max date of index
            if isinstance(df.index, pd.DatetimeIndex):
                last_day = df.index.normalize().max()
                today_flag = df.index.normalize().eq(last_day)
            else:
                today_flag = True
        else:
            today_flag = True

        # data: increments only when EISave equals priceh or pricel
        is_hilo = (df["EISave"].eq(df["priceh"])) | (df["EISave"].eq(df["pricel"]))
        data_vals = [0]
        for i in range(1, len(df)):
            data_vals.append(data_vals[-1] + 1 if is_hilo.iloc[i] else data_vals[-1])
        df["data"] = pd.Series(data_vals, index=df.index)
        # datacount = HighestAll(data) - data[1]
        df["datacount"] = df["data"].max() - df["data"].shift(1)

        # Bases for bearish (from priceh) and bullish (from pricel)
        rng = (df["priorEI2"] - df["priorEI1"]).abs()

        # Bearish bases (when current pivot equals priceh), carry-forward otherwise
        df["extfib1_base"]   = np.where(df["EISave"].eq(df["priceh"]), df["priceh"] - rng * 1.000, np.nan)
        df["extfib1a_base"]  = np.where(df["EISave"].eq(df["priceh"]), df["priceh"] - rng * 0.382, np.nan)
        df["extfib2_base"]   = np.where(df["EISave"].eq(df["priceh"]), df["priceh"] - rng * 0.618, np.nan)
        df["extfib3_base"]   = np.where(df["EISave"].eq(df["priceh"]), df["priceh"] - rng * 1.618, np.nan)
        df["extfib3a_base"]  = np.where(df["EISave"].eq(df["priceh"]), df["priceh"] - rng * 2.000, np.nan)
        df["extfib4_base"]   = np.where(df["EISave"].eq(df["priceh"]), df["priceh"] - rng * 2.618, np.nan)
        df["extfib5_base"]   = np.where(df["EISave"].eq(df["priceh"]), df["priceh"] - rng * 3.618, np.nan)
        df[["extfib1_base","extfib1a_base","extfib2_base","extfib3_base","extfib3a_base","extfib4_base","extfib5_base"]] = \
            df[["extfib1_base","extfib1a_base","extfib2_base","extfib3_base","extfib3a_base","extfib4_base","extfib5_base"]].ffill()

        # Bullish bases (when current pivot equals pricel), carry-forward otherwise
        df["extfib1b_base"]  = np.where(df["EISave"].eq(df["pricel"]), df["pricel"] + rng * 1.000, np.nan)
        df["extfib1ab_base"] = np.where(df["EISave"].eq(df["pricel"]), df["pricel"] + rng * 0.382, np.nan)
        df["extfib2b_base"]  = np.where(df["EISave"].eq(df["pricel"]), df["pricel"] + rng * 0.618, np.nan)
        df["extfib3b_base"]  = np.where(df["EISave"].eq(df["pricel"]), df["pricel"] + rng * 1.618, np.nan)
        df["extfib3ab_base"] = np.where(df["EISave"].eq(df["pricel"]), df["pricel"] + rng * 2.000, np.nan)
        df["extfib4b_base"]  = np.where(df["EISave"].eq(df["pricel"]), df["pricel"] + rng * 2.618, np.nan)
        df["extfib5b_base"]  = np.where(df["EISave"].eq(df["pricel"]), df["pricel"] + rng * 3.618, np.nan)
        df[["extfib1b_base","extfib1ab_base","extfib2b_base","extfib3b_base","extfib3ab_base","extfib4b_base","extfib5b_base"]] = \
            df[["extfib1b_base","extfib1ab_base","extfib2b_base","extfib3b_base","extfib3ab_base","extfib4b_base","extfib5b_base"]].ffill()

        # Gate conditions (ThinkScript: datacount <= numberextfibstoshow and today and showFibExtLines and !IsNaN(base) and dir < 0/ > 0 and cpo != 0)
        gate_common = (df["datacount"] <= numberextfibstoshow) & showFibExtLines & pd.Series(today_flag, index=df.index) & (cpo != 0)

        bear_gate = gate_common & (df["dir"] < 0)
        bull_gate = gate_common & (df["dir"] > 0)

        # Output lines use previous base value [1]
        df["extfib100"]  = np.where(bear_gate & df["extfib1_base"].notna(),  df["extfib1_base"].shift(1),  np.nan)
        df["extfib382"]  = np.where(bear_gate & df["extfib1a_base"].notna(), df["extfib1a_base"].shift(1), np.nan)
        df["extfib618"]  = np.where(bear_gate & df["extfib2_base"].notna(),  df["extfib2_base"].shift(1),  np.nan)
        df["extfib1618"] = np.where(bear_gate & df["extfib3_base"].notna(),  df["extfib3_base"].shift(1),  np.nan)
        df["extfib2000"] = np.where(bear_gate & df["extfib3a_base"].notna(), df["extfib3a_base"].shift(1), np.nan)
        df["extfib2618"] = np.where(bear_gate & df["extfib4_base"].notna(),  df["extfib4_base"].shift(1),  np.nan)
        df["extfib3618"] = np.where(bear_gate & df["extfib5_base"].notna(),  df["extfib5_base"].shift(1),  np.nan)

        df["extfib100_"]  = np.where(bull_gate & df["extfib1b_base"].notna(),  df["extfib1b_base"].shift(1),  np.nan)
        df["extfib382_"]  = np.where(bull_gate & df["extfib1ab_base"].notna(), df["extfib1ab_base"].shift(1), np.nan)
        df["extfib618_"]  = np.where(bull_gate & df["extfib2b_base"].notna(),  df["extfib2b_base"].shift(1),  np.nan)
        df["extfib1618_"] = np.where(bull_gate & df["extfib3b_base"].notna(),  df["extfib3b_base"].shift(1),  np.nan)
        df["extfib2000_"] = np.where(bull_gate & df["extfib3ab_base"].notna(), df["extfib3ab_base"].shift(1), np.nan)
        df["extfib2618_"] = np.where(bull_gate & df["extfib4b_base"].notna(),  df["extfib4b_base"].shift(1),  np.nan)
        df["extfib3618_"] = np.where(bull_gate & df["extfib5b_base"].notna(),  df["extfib5b_base"].shift(1),  np.nan)

        # Optional “bubbles” in ThinkScript rely on expansion bars (future bars), which don't exist in a static DataFrame.
        # If you want labels near the most recent bar instead, you can create them at the last index:
        if showFibExtLines and len(df) > (b + 2):
            last = df.index[-1]
            labels = []
            if df["isUp"].iloc[-(b+2)] is False:
                # bearish (red)
                for col, txt in [("extfib100", "100%"), ("extfib382", "38.2%"), ("extfib618", "61.8%"),
                                ("extfib1618", "161.8%"), ("extfib2000", "200%"), ("extfib2618", "261.8%"), ("extfib3618", "361.8%")]:
                    val = df[col].iloc[-(b+2)]
                    if pd.notna(val):
                        labels.append({"x": last, "y": val, "text": txt, "color": "red"})
            else:
                # bullish (green)
                for col, txt in [("extfib100_", "100%"), ("extfib382_", "38.2%"), ("extfib618_", "61.8%"),
                                ("extfib1618_", "161.8%"), ("extfib2000_", "200%"), ("extfib2618_", "261.8%"), ("extfib3618_", "361.8%")]:
                    val = df[col].iloc[-(b+2)]
                    if pd.notna(val):
                        labels.append({"x": last, "y": val, "text": txt, "color": "green"})
            fibext_bubbles = pd.DataFrame(labels) if labels else pd.DataFrame(columns=["x","y","text","color"])
        else:
            fibext_bubbles = pd.DataFrame(columns=["x","y","text","color"])

        # Cumulative volume (ThinkScript: vol)
        df["vol"] = df["volume"].cumsum()

        # vol1: first bar volume, carried forward
        first_bar_vol = df["volume"].iloc[0] if len(df) else 0
        df["vol1"] = first_bar_vol

        # xxvol: cumulative volume at last pivot (when EISave == priceh or pricel), else hold prior
        pivot_mask = df["EISave"].eq(df["priceh"]) | df["EISave"].eq(df["pricel"])
        df["xxvol"] = np.where(pivot_mask, df["vol"], np.nan)
        df["xxvol"] = df["xxvol"].ffill()

        # chgvol: volume since previous pivot (ThinkScript logic)
        delta = df["xxvol"] - df["xxvol"].shift(1)
        df["chgvol"] = np.where((delta + df["vol1"]) == df["vol"], df["vol"], delta)

        # Optional bubbles (disabled by default)
        showBubblesVolume = False
        if showBubblesVolume:
            idx_pos = np.arange(len(df))
            cond = (~df["EI"].isna()) & (idx_pos != 0)

            bubble_y = np.where(
                df["isUp"],
                df["priceh"] * (1.0 + float(bubble_offset)),
                df["pricel"] * (1.0 - float(bubble_offset)),
            )

            def bubble_color(row):
                if row["isUp"]:
                    if row["chghigh"] > 0: return "green"
                    if row["chghigh"] < 0: return "red"
                    return "yellow"
                else:
                    if row["chglow"] > 0: return "green"
                    if row["chglow"] < 0: return "red"
                    return "yellow"

            bubbles_volume_df = pd.DataFrame(
                {
                    "x": df.index,
                    "y": bubble_y,
                    "text": df["chgvol"].map(lambda v: f"{int(v)}" if pd.notna(v) else ""),
                    "color": [bubble_color(r) for _, r in df.iterrows()],
                    "up": df["isUp"].astype(bool),
                },
                index=df.index,
            )[cond]

        # Settings converted from ThinkScript inputs
        use_manual_fib_skip = False  # usemanualfibskip
        fib_skip = 0.50              # fibskip (as ratio, e.g., 0.50 = 50%)
        show_bubbles_fib_ratio = False  # showBubblesfibratio
        show_fib_label = False          # showFibLabel
        show_fib_lines = False          # showfiblines

        # Fibonacci levels
        fib1level = 0.236
        fib2level = 0.382
        fibMlevel = 0.500
        fib3level = 0.618
        fib4level = 0.786

        # Example: pick skip amount (manual vs preprogrammed)
        # "preprogrammed" can be whatever logic you use (e.g., 1.0 default)
        preprogrammed_fib_skip = 1.0
        effective_fib_skip = fib_skip if use_manual_fib_skip else preprogrammed_fib_skip

        # datacount2 = HighestAll(data1) - data1[1]
        df["datacount2"] = df["data1"].max() - df["data1"].shift(1)

        numberfibretracementstoshow = 2

        # fibskipit: if usemanualfibskip == no then (close > 800 ? .25 : .5) else fibskip
        auto_skip = np.where(df["close"] > 800, 0.25, 0.50)
        df["fibskipit"] = np.where(use_manual_fib_skip, float(fib_skip), auto_skip)

        # EIfibh/EIfibl (recursive)
        EIfibh = [np.nan] * len(df)
        EIfibl = [np.nan] * len(df)
        for i in range(len(df)):
            prev_hi = EIfibh[i-1] if i > 0 else np.nan
            prev_lo = EIfibl[i-1] if i > 0 else np.nan
            prev_EI = df["EISave"].shift(1).iloc[i]
            priceh_i = df["priceh"].iloc[i]
            pricel_i = df["pricel"].iloc[i]
            EISave_i = df["EISave"].iloc[i]
            skip_pct = df["fibskipit"].iloc[i] * 0.01  # convert to percent

            # New fib high only when pivot is priceh AND move > priceh * fibskipit%
            if pd.notna(EISave_i) and (EISave_i == priceh_i) and pd.notna(prev_EI):
                if abs(EISave_i - prev_EI) > (priceh_i * skip_pct):
                    EIfibh[i] = priceh_i
                else:
                    EIfibh[i] = prev_hi
            else:
                EIfibh[i] = prev_hi

            # New fib low only when pivot is pricel AND move > priceh * fibskipit% (matches TS)
            if pd.notna(EISave_i) and (EISave_i == pricel_i) and pd.notna(prev_EI):
                if abs(EISave_i - prev_EI) > (priceh_i * skip_pct):
                    EIfibl[i] = pricel_i
                else:
                    EIfibl[i] = prev_lo
            else:
                EIfibl[i] = prev_lo

        df["EIfibh"] = pd.Series(EIfibh, index=df.index)
        df["EIfibl"] = pd.Series(EIfibl, index=df.index)

        # range
        df["fib_range"] = df["EIfibh"] - df["EIfibl"]

        # Gate: showfiblines == no → NaN; else if datacount2 <= numberfibretracementstoshow → show
        gate = show_fib_lines & (df["datacount2"] <= numberfibretracementstoshow)

        df["fibH"] = np.where(gate, df["EIfibh"], np.nan)
        df["fibL"] = np.where(gate, df["EIfibl"], np.nan)
        df["fibM"] = np.where(gate, df["EIfibl"] + df["fib_range"] * float(fibMlevel), np.nan)
        df["fib1"] = np.where(gate, df["EIfibl"] + df["fib_range"] * float(fib1level), np.nan)
        df["fib2"] = np.where(gate, df["EIfibl"] + df["fib_range"] * float(fib2level), np.nan)
        df["fib3"] = np.where(gate, df["EIfibl"] + df["fib_range"] * float(fib3level), np.nan)
        df["fib4"] = np.where(gate, df["EIfibl"] + df["fib_range"] * float(fib4level), np.nan)

        # Label: "Current Fib Level ..." (optional)
        if show_fib_label:
            df["current_fib_level_pct"] = (df["close"] - df["EIfibl"]) / df["fib_range"]  # fraction (0..1+)
            # format as percent on render

        # Bubble for fib ratio (optional)
        if show_bubbles_fib_ratio:
            idx_pos = np.arange(len(df))
            cond = (~df["EI"].isna()) & (idx_pos != 0)
            bubble_y = np.where(
                df["isUp"],
                df["priceh"] * (1.0 + float(bubble_offset)),
                df["pricel"] * (1.0 - float(bubble_offset)),
            )
            # text = percent of distance from EIfibl to priceh/pricel
            up_frac = (df["priceh"] - df["EIfibl"]) / df["fib_range"]
            dn_frac = (df["pricel"] - df["EIfibl"]) / df["fib_range"]
            bubble_text = np.where(df["isUp"], up_frac, dn_frac)

            def bubble_color(row):
                if row["isUp"]:
                    if row["chghigh"] > 0: return "green"
                    if row["chghigh"] < 0: return "red"
                    return "green"
                else:
                    if row["chglow"] > 0: return "green"
                    if row["chglow"] < 0: return "red"
                    return "red"

            fib_ratio_bubbles = pd.DataFrame(
                {
                    "x": df.index,
                    "y": bubble_y,
                    "text": bubble_text.map(lambda v: f"{v:.2%}" if pd.notna(v) else ""),
                    "color": [bubble_color(r) for _, r in df.iterrows()],
                    "up": df["isUp"].astype(bool),
                },
                index=df.index,
            )[cond]


        # Determine latest signal states
        last_buy_signal = bool(df["Buy_Signal"].iloc[-1])
        last_sell_signal = bool(df["Sell_Signal"].iloc[-1])

        if ticker not in trades:
            if last_buy_signal:
                logger.info(f"Buy signal for {ticker}: Placing LONG orders.")
                # order_id_schwab = (
                #     place_order(
                #         ticker, qty_schwab, "BUY", schwab_account_id, logger, "OPENING"
                #     )
                #     if qty_schwab > 0
                #     else 0
                # )
                order_id_tasty = (
                    place_tastytrade_order(
                        ticker, qty_tastytrade, "Buy to Open", TASTY_ACCOUNT_ID, logger
                    )
                    if qty_tastytrade > 0
                    else 0
                )
                trades[ticker] = {
                    "position": "LONG",
                    # "order_id_schwab": order_id_schwab,
                    "order_id_tastytrade": order_id_tasty,
                }
            elif last_sell_signal:
                logger.info(f"Sell signal for {ticker}: Placing SHORT orders.")
                order_id_schwab = (
                    # place_order(
                    #     ticker, qty_schwab, "SELL_SHORT", schwab_account_id, logger, "OPENING"
                    # )
                    # if qty_schwab > 0
                    # else 0
                )
                order_id_tasty = (
                    place_tastytrade_order(
                        ticker, qty_tastytrade, "Sell to Open", TASTY_ACCOUNT_ID, logger
                    )
                    if qty_tastytrade > 0
                    else 0
                )
                trades[ticker] = {
                    "position": "SHORT",
                    # "order_id_schwab": order_id_schwab,
                    "order_id_tastytrade": order_id_tasty,
                }
        else:
            current_position = trades[ticker]["position"]
            if current_position == "LONG" and last_sell_signal:
                logger.info(
                    f"Reversing LONG to SHORT for {ticker}: Closing LONG, opening SHORT"
                )
                # close_id_schwab = (
                #     place_order(
                #         ticker, qty_schwab, "SELL", schwab_account_id, logger, "CLOSING"
                #     )
                #     if qty_schwab > 0
                #     else 0
                # )
                close_id_tasty = (
                    place_tastytrade_order(
                        ticker, qty_tastytrade, "Sell to Close", TASTY_ACCOUNT_ID, logger
                    )
                    if qty_tastytrade > 0
                    else 0
                )
                # open_id_schwab = (
                #     place_order(
                #         ticker, qty_schwab, "SELL_SHORT", schwab_account_id, logger, "OPENING"
                #     )
                #     if qty_schwab > 0
                #     else 0
                # )
                open_id_tasty = (
                    place_tastytrade_order(
                        ticker, qty_tastytrade, "Sell to Open", TASTY_ACCOUNT_ID, logger
                    )
                    if qty_tastytrade > 0
                    else 0
                )
                trades[ticker] = {
                    "position": "SHORT",
                    # "order_id_schwab": open_id_schwab,
                    "order_id_tastytrade": open_id_tasty,
                }

            elif current_position == "SHORT" and last_buy_signal:
                logger.info(
                    f"Reversing SHORT to LONG for {ticker}: Closing SHORT, opening LONG"
                )
                # close_id_schwab = (
                #     place_order(
                #         ticker,
                #         qty_schwab,
                #         "BUY_TO_COVER",
                #         schwab_account_id,
                #         logger,
                #         "CLOSING",
                #     )
                #     if qty_schwab > 0
                #     else 0
                # )
                close_id_tasty = (
                    place_tastytrade_order(
                        ticker, qty_tastytrade, "Buy to Close", TASTY_ACCOUNT_ID, logger
                    )
                    if qty_tastytrade > 0
                    else 0
                )
                # open_id_schwab = (
                #     place_order(
                #         ticker, qty_schwab, "BUY", schwab_account_id, logger, "OPENING"
                #     )
                #     if qty_schwab > 0
                #     else 0
                # )
                open_id_tasty = (
                    place_tastytrade_order(
                        ticker, qty_tastytrade, "Buy to Open", TASTY_ACCOUNT_ID, logger
                    )
                    if qty_tastytrade > 0
                    else 0
                )
                trades[ticker] = {
                    "position": "LONG",
                    # "order_id_schwab": open_id_schwab,
                    "order_id_tastytrade": open_id_tasty,
                }

        # Save trade states
        with open(trades_file, "w") as f:
            json.dump(trades, f)

        # Optional: add any summary logs here if needed

        logger.info(f"Strategy for {ticker} completed at {datetime.now(pytz.utc)}")
    except Exception as e:
        logger.error(f"Exception in strategy for {ticker}: {str(e)}", exc_info=True)
