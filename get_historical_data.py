import os
import json
import redis
import pandas as pd
from datetime import datetime, timezone, timedelta
from databento import Historical
from typing import Optional
from tick_producer import TickDataBufferWithRedis 

def get_historical_data(ticker: str, timeframe: str, logger) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for EMA crossover, Supertrend, or 0DTE SPX strategies.
    - Uses Redis caching for faster repeated access.
    - Supports tick-based timeframes ('516t', '1160t', '1600t') and time-based timeframes.
    - Pulls from Databento when cache miss occurs.
    """

    # ----- Helper: tick or time-based -----
    def _is_tick(tf: str) -> bool:
        return tf.lower().endswith("t")

    # ----- Helper: timeframe → pandas resample -----
    resample_map = {
        "2Min": "2min",
        "5Min": "5min",
        "15Min": "15min",
        "30Min": "30min",
        "4Hour": "4H",
    }

    # ----- Helper: timeframe → lookback -----
    def _compute_lookback(tf: str) -> timedelta:
        if _is_tick(tf):
            return timedelta(days=7)
        if tf in {"1Min", "2Min", "5Min", "15Min", "30Min"}:
            return timedelta(days=30)
        if tf in {"1Hour", "4Hour"}:
            return timedelta(days=90)
        if tf == "1Day":
            return timedelta(days=365)
        return timedelta(days=30)

    # ----- Redis connection -----
    def _get_redis() -> Optional[redis.Redis]:
        try:
            client = redis.Redis(host="localhost", port=6379, db=0)
            client.ping()
            return client
        except Exception:
            return None

    # ----- Load from Redis -----
    def _load_from_redis(symbol: str, tf: str, start_dt: datetime, end_dt: datetime) -> Optional[pd.DataFrame]:
        client = _get_redis()
        if not client:
            return None
        key = f"ohlcv_history:{symbol}:{tf}"
        start_score = int(start_dt.timestamp())
        end_score = int(end_dt.timestamp())
        try:
            raw_items = client.zrangebyscore(key, start_score, end_score)
            if not raw_items:
                return None
            records = [json.loads(item) for item in raw_items]
            df = pd.DataFrame(records)
            if df.empty:
                return None
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.sort_values("timestamp", inplace=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception:
            return None

    # ----- Save to Redis -----
    def _save_to_redis(df: pd.DataFrame, symbol: str, tf: str):
        client = _get_redis()
        if not client or df.empty:
            return
        key = f"ohlcv_history:{symbol}:{tf}"
        pipe = client.pipeline(transaction=False)
        for _, row in df.iterrows():
            ts = int(pd.Timestamp(row["timestamp"]).timestamp())
            payload = {
                "symbol": symbol,
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
            }
            pipe.zadd(key, {json.dumps(payload): ts})
        pipe.execute()

    # ----- Prepare time window -----
    now_utc = datetime.now(timezone.utc)
    lookback = _compute_lookback(timeframe)
    start_dt = now_utc - lookback

    # ----- Try Redis first -----
    cached_df = _load_from_redis(ticker, timeframe, start_dt, now_utc)
    if cached_df is not None:
        return cached_df

    # ----- Databento API key -----
    db_api_key = os.getenv("DATABENTO_API_KEY")
    if not db_api_key:
        raise EnvironmentError("DATABENTO_API_KEY not set.")

    dataset = "GLBX.MDP3" if ticker.startswith("/") else "XNAS.ITCH"  # futures vs stocks

    # ===============================
    # TICK-BASED DATA
    # ===============================
    if _is_tick(timeframe):
        tick_size = int("".join(filter(str.isdigit, timeframe)))
        redis_client = _get_redis()
        buffer = TickDataBufferWithRedis(
            ticker=ticker,
            tick_size=tick_size,
            redis_client=redis_client,
            db_api_key=db_api_key,
            max_period=10,
        )
        buffer.warmup_with_historical_ticks(
            symbol=ticker,
            dataset=dataset,
            start=start_dt.isoformat(),
            end=now_utc.isoformat(),
            schema="trades",
        )
        df = buffer.get_dataframe(min_bars=5)
        if df is None or df.empty:
            raise ValueError(f"No historical tick bars for {ticker} ({timeframe}).")
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.to_datetime(df.index)
        _save_to_redis(df, ticker, timeframe)
        return df

    # ===============================
    # TIME-BASED DATA
    # ===============================
    db_client = Historical(key=db_api_key)

    schema_map = {
        "1Min": "ohlcv-1m",
        "2Min": "ohlcv-1m",
        "5Min": "ohlcv-1m",
        "15Min": "ohlcv-1m",
        "30Min": "ohlcv-1m",
        "1Hour": "ohlcv-1h",
        "4Hour": "ohlcv-1h",
        "1Day": "ohlcv-1d",
    }

    if timeframe not in schema_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    df = db_client.timeseries.get_range(
        dataset=dataset,
        symbols=[ticker],
        schema=schema_map[timeframe],
        start=start_dt.isoformat(),
        end=now_utc.isoformat(),
    ).to_df()

    if df.empty:
        raise ValueError(f"No historical data for {ticker} ({timeframe}).")

    # Normalize prices (Databento fixed-point)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col] / 1e9

    df["timestamp"] = pd.to_datetime(df.index)

    # Resample if needed
    if timeframe in resample_map:
        df = (
            df.set_index("timestamp")
            .resample(resample_map[timeframe])
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )

    _save_to_redis(df, ticker, timeframe)
    return df
