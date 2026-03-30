"""
data.py — OHLCV data fetcher with local parquet cache.

Fetches BTC/USDT and ETH/USDT 1h candles from Binance via ccxt.
Results are cached as parquet files so the research loop never re-fetches
what it already has. Cache is refreshed on stale or missing data only.

Usage:
    from data import fetch_ohlcv
    df = fetch_ohlcv("BTC/USDT", "1h", days=730)
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

log = logging.getLogger(__name__)

# ── config ──────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_EXCHANGE = "binance"
DEFAULT_LIMIT    = 1000          # candles per ccxt request (Binance max)
STALE_HOURS      = 4             # re-fetch if cache is older than this


def _cache_path(symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "_")
    return CACHE_DIR / f"{safe}_{timeframe}.parquet"


def _load_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception as e:
        log.warning("Cache read failed (%s), will re-fetch: %s", path.name, e)
        return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path)


def _fetch_all(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    """Page through ccxt to collect all OHLCV between since_ms and until_ms."""
    rows: list[list] = []
    cursor = since_ms

    max_retries = 5
    retry_count = 0
    while cursor < until_ms:
        try:
            batch = exchange.fetch_ohlcv(
                symbol, timeframe, since=cursor, limit=DEFAULT_LIMIT
            )
            retry_count = 0  # reset on success
        except ccxt.NetworkError as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise RuntimeError(
                    f"Exchange unreachable after {max_retries} attempts: {e}\n"
                    "If running in a sandboxed environment, place pre-downloaded "
                    "parquet files in data_cache/ and they will be used automatically."
                ) from e
            log.warning("Network error (attempt %d/%d), retrying in 5s: %s", retry_count, max_retries, e)
            time.sleep(5)
            continue
        except ccxt.RateLimitExceeded:
            time.sleep(exchange.rateLimit / 1000)
            continue

        if not batch:
            break

        rows.extend(batch)
        last_ts = batch[-1][0]

        if last_ts <= cursor:          # no forward progress — stop
            break
        cursor = last_ts + 1

        # polite pause between pages
        time.sleep(exchange.rateLimit / 1000)

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # trim to requested window
    df = df[df.index < pd.Timestamp(until_ms, unit="ms", tz="UTC")]
    return df


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 730,
    exchange_id: str = DEFAULT_EXCHANGE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of OHLCV candles for the past `days` days.

    Columns: open, high, low, close, volume  (float64)
    Index:   UTC-aware DatetimeIndex

    Data is cached locally. Re-fetches only if:
      - cache is missing
      - cache doesn't cover the requested window
      - cache is older than STALE_HOURS and the end of the window is recent
    """
    cache_path = _cache_path(symbol, timeframe)
    now_utc    = datetime.now(tz=timezone.utc)
    since_dt   = now_utc - timedelta(days=days)
    until_dt   = now_utc

    cached = None if force_refresh else _load_cache(cache_path)

    if cached is not None:
        coverage_ok = (
            cached.index.min() <= pd.Timestamp(since_dt).tz_convert("UTC") + timedelta(hours=48)
            and cached.index.max() >= pd.Timestamp(until_dt).tz_convert("UTC") - timedelta(hours=STALE_HOURS)
        )
        if coverage_ok:
            log.info("Cache hit for %s %s (%d rows)", symbol, timeframe, len(cached))
            return cached[(cached.index >= pd.Timestamp(since_dt).tz_convert("UTC"))].copy()

    log.info("Fetching %s %s from exchange (days=%d)…", symbol, timeframe, days)

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    since_ms = int(since_dt.timestamp() * 1000)
    until_ms = int(until_dt.timestamp() * 1000)

    try:
        fresh = _fetch_all(exchange, symbol, timeframe, since_ms, until_ms)
    except RuntimeError as e:
        # Exchange unreachable (sandboxed env). Fall back to stale cache if available.
        if cached is not None and not cached.empty:
            log.warning("Exchange unreachable — using stale cache (%d rows, newest=%s): %s",
                        len(cached), cached.index.max(), e)
            return cached[(cached.index >= pd.Timestamp(since_dt).tz_convert("UTC"))].copy()
        raise

    if fresh.empty:
        raise RuntimeError(f"No data returned for {symbol} {timeframe}")

    # merge with any existing cache so we don't lose older bars
    if cached is not None and not cached.empty:
        merged = pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fresh

    _save_cache(merged, cache_path)
    log.info("Fetched and cached %d rows for %s %s", len(fresh), symbol, timeframe)

    since_ts = pd.Timestamp(since_dt).tz_localize("UTC") if pd.Timestamp(since_dt).tzinfo is None else pd.Timestamp(since_dt).tz_convert("UTC")
    return fresh[(fresh.index >= since_ts)].copy()


def _fetch_all_funding(
    exchange: ccxt.Exchange,
    symbol: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    """Page through ccxt to collect all funding rates between since_ms and until_ms."""
    rows: list[dict] = []
    cursor = since_ms

    max_retries = 5
    retry_count = 0
    while cursor < until_ms:
        try:
            batch = exchange.fetch_funding_rate_history(
                symbol, since=cursor, limit=DEFAULT_LIMIT
            )
            retry_count = 0
        except ccxt.NetworkError as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise RuntimeError(
                    f"Exchange unreachable after {max_retries} attempts: {e}\n"
                    "If running in a sandboxed environment, place pre-downloaded "
                    "parquet files in data_cache/ and they will be used automatically."
                ) from e
            log.warning("Network error (attempt %d/%d), retrying in 5s: %s", retry_count, max_retries, e)
            time.sleep(5)
            continue
        except ccxt.RateLimitExceeded:
            time.sleep(exchange.rateLimit / 1000)
            continue

        if not batch:
            break

        for entry in batch:
            rows.append({
                "timestamp": entry["timestamp"],
                "funding_rate": entry["fundingRate"],
            })

        last_ts = batch[-1]["timestamp"]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1

        time.sleep(exchange.rateLimit / 1000)

    if not rows:
        return pd.DataFrame(columns=["funding_rate"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df.index < pd.Timestamp(until_ms, unit="ms", tz="UTC")]
    return df


def fetch_funding_rates(
    symbol: str = "BTC/USDT:USDT",
    days: int = 730,
    exchange_id: str = DEFAULT_EXCHANGE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of perpetual funding rates for the past `days` days.

    Columns: funding_rate (float64, decimal — e.g. 0.0001 = 0.01% per 8h)
    Index:   UTC-aware DatetimeIndex at 8h intervals

    Data is cached locally. Re-fetches only if cache is missing or stale.

    Args:
        symbol:       Perpetual symbol (e.g. "BTC/USDT:USDT" for Binance linear).
        days:         Number of days of history to fetch.
        exchange_id:  ccxt exchange to use (must be a futures exchange).
        force_refresh: Bypass cache entirely.
    """
    safe_name = symbol.replace("/", "_").replace(":", "_")
    cache_path = CACHE_DIR / f"{safe_name}_funding.parquet"
    now_utc = datetime.now(tz=timezone.utc)
    since_dt = now_utc - timedelta(days=days)
    until_dt = now_utc

    cached = None if force_refresh else _load_cache(cache_path)

    if cached is not None:
        coverage_ok = (
            cached.index.min() <= pd.Timestamp(since_dt).tz_convert("UTC") + timedelta(hours=48)
            and cached.index.max() >= pd.Timestamp(until_dt).tz_convert("UTC") - timedelta(hours=STALE_HOURS)
        )
        if coverage_ok:
            log.info("Cache hit for %s funding (%d rows)", symbol, len(cached))
            return cached[(cached.index >= pd.Timestamp(since_dt).tz_convert("UTC"))].copy()

    log.info("Fetching %s funding rates from exchange (days=%d)…", symbol, days)

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    since_ms = int(since_dt.timestamp() * 1000)
    until_ms = int(until_dt.timestamp() * 1000)

    try:
        fresh = _fetch_all_funding(exchange, symbol, since_ms, until_ms)
    except RuntimeError as e:
        if cached is not None and not cached.empty:
            log.warning("Exchange unreachable — using stale cache (%d rows, newest=%s): %s",
                        len(cached), cached.index.max(), e)
            return cached[(cached.index >= pd.Timestamp(since_dt).tz_convert("UTC"))].copy()
        raise

    if fresh.empty:
        raise RuntimeError(f"No funding rate data returned for {symbol}")

    if cached is not None and not cached.empty:
        merged = pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fresh

    _save_cache(merged, cache_path)
    log.info("Fetched and cached %d funding rate entries for %s", len(fresh), symbol)

    since_ts = pd.Timestamp(since_dt).tz_localize("UTC") if pd.Timestamp(since_dt).tzinfo is None else pd.Timestamp(since_dt).tz_convert("UTC")
    return fresh[(fresh.index >= since_ts)].copy()


# ── Open interest history ─────────────────────────────────────────────────────

def _fetch_all_oi(
    exchange: ccxt.Exchange,
    symbol_raw: str,
    period: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    """Page through Binance's fapiData/openInterestHist endpoint directly.

    ccxt's fetchOpenInterestHistory wrapper has a parameter-mapping bug
    that causes Binance to reject startTime. We call the raw endpoint.
    """
    rows: list[dict] = []
    cursor = since_ms
    oi_limit = 500  # Binance max for this endpoint

    max_retries = 5
    retry_count = 0
    while cursor < until_ms:
        try:
            batch = exchange.fapiDataGetOpenInterestHist({
                "symbol": symbol_raw,
                "period": period,
                "limit": oi_limit,
                "startTime": int(cursor),
                "endTime": int(until_ms),
            })
            retry_count = 0
        except ccxt.NetworkError as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise RuntimeError(
                    f"Exchange unreachable after {max_retries} attempts: {e}\n"
                    "Place pre-downloaded parquet in data_cache/ as fallback."
                ) from e
            log.warning("Network error (attempt %d/%d), retrying in 5s: %s",
                        retry_count, max_retries, e)
            time.sleep(5)
            continue
        except ccxt.RateLimitExceeded:
            time.sleep(exchange.rateLimit / 1000)
            continue

        if not batch:
            break

        for entry in batch:
            rows.append({
                "timestamp": int(entry["timestamp"]),
                "open_interest": float(entry["sumOpenInterestValue"]),
            })

        last_ts = int(batch[-1]["timestamp"])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1

        time.sleep(exchange.rateLimit / 1000)

    if not rows:
        return pd.DataFrame(columns=["open_interest"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df.index < pd.Timestamp(until_ms, unit="ms", tz="UTC")]
    return df


def fetch_open_interest(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "1h",
    days: int = 730,
    exchange_id: str = DEFAULT_EXCHANGE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of open interest history for the past `days` days.

    Columns: open_interest (float64, quote-currency value)
    Index:   UTC-aware DatetimeIndex
    """
    safe_name = symbol.replace("/", "_").replace(":", "_")
    cache_path = CACHE_DIR / f"{safe_name}_oi_{timeframe}.parquet"
    now_utc = datetime.now(tz=timezone.utc)
    since_dt = now_utc - timedelta(days=days)
    until_dt = now_utc

    cached = None if force_refresh else _load_cache(cache_path)

    if cached is not None:
        coverage_ok = (
            cached.index.min() <= pd.Timestamp(since_dt).tz_convert("UTC") + timedelta(hours=48)
            and cached.index.max() >= pd.Timestamp(until_dt).tz_convert("UTC") - timedelta(hours=STALE_HOURS)
        )
        if coverage_ok:
            log.info("Cache hit for %s OI (%d rows)", symbol, len(cached))
            return cached[(cached.index >= pd.Timestamp(since_dt).tz_convert("UTC"))].copy()

    log.info("Fetching %s open interest from exchange (days=%d)…", symbol, days)

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    since_ms = int(since_dt.timestamp() * 1000)
    until_ms = int(until_dt.timestamp() * 1000)

    symbol_raw = symbol.replace("/", "").replace(":USDT", "")  # BTC/USDT:USDT -> BTCUSDT

    try:
        fresh = _fetch_all_oi(exchange, symbol_raw, timeframe, since_ms, until_ms)
    except RuntimeError as e:
        if cached is not None and not cached.empty:
            log.warning("Exchange unreachable — using stale OI cache (%d rows): %s",
                        len(cached), e)
            return cached[(cached.index >= pd.Timestamp(since_dt).tz_convert("UTC"))].copy()
        raise

    if fresh.empty:
        raise RuntimeError(f"No open interest data returned for {symbol}")

    if cached is not None and not cached.empty:
        merged = pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fresh

    _save_cache(merged, cache_path)
    log.info("Fetched and cached %d OI entries for %s", len(fresh), symbol)

    since_ts = pd.Timestamp(since_dt).tz_localize("UTC") if pd.Timestamp(since_dt).tzinfo is None else pd.Timestamp(since_dt).tz_convert("UTC")
    return fresh[(fresh.index >= since_ts)].copy()


# ── Perpetual futures OHLCV (for basis computation) ───────────────────────────

def fetch_perp_ohlcv(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "1h",
    days: int = 730,
    exchange_id: str = DEFAULT_EXCHANGE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame of perpetual futures OHLCV candles.

    Same format as fetch_ohlcv() but from the futures exchange.
    Used to compute basis spread (perp_close - spot_close).
    """
    safe_name = symbol.replace("/", "_").replace(":", "_")
    cache_path = CACHE_DIR / f"{safe_name}_{timeframe}.parquet"
    now_utc = datetime.now(tz=timezone.utc)
    since_dt = now_utc - timedelta(days=days)
    until_dt = now_utc

    cached = None if force_refresh else _load_cache(cache_path)

    if cached is not None:
        coverage_ok = (
            cached.index.min() <= pd.Timestamp(since_dt).tz_convert("UTC") + timedelta(hours=48)
            and cached.index.max() >= pd.Timestamp(until_dt).tz_convert("UTC") - timedelta(hours=STALE_HOURS)
        )
        if coverage_ok:
            log.info("Cache hit for %s perp OHLCV (%d rows)", symbol, len(cached))
            return cached[(cached.index >= pd.Timestamp(since_dt).tz_convert("UTC"))].copy()

    log.info("Fetching %s perp OHLCV from exchange (days=%d)…", symbol, days)

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    since_ms = int(since_dt.timestamp() * 1000)
    until_ms = int(until_dt.timestamp() * 1000)

    try:
        fresh = _fetch_all(exchange, symbol, timeframe, since_ms, until_ms)
    except RuntimeError as e:
        if cached is not None and not cached.empty:
            log.warning("Exchange unreachable — using stale perp cache (%d rows): %s",
                        len(cached), e)
            return cached[(cached.index >= pd.Timestamp(since_dt).tz_convert("UTC"))].copy()
        raise

    if fresh.empty:
        raise RuntimeError(f"No perp OHLCV data returned for {symbol}")

    if cached is not None and not cached.empty:
        merged = pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fresh

    _save_cache(merged, cache_path)
    log.info("Fetched and cached %d perp OHLCV rows for %s", len(fresh), symbol)

    since_ts = pd.Timestamp(since_dt).tz_localize("UTC") if pd.Timestamp(since_dt).tzinfo is None else pd.Timestamp(since_dt).tz_convert("UTC")
    return fresh[(fresh.index >= since_ts)].copy()


# ── Multi-timeframe augmentation (V3 §6.5.1) ─────────────────────────────────

def augment_with_timeframes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1h OHLCV to 4h and 1d, compute features at each timeframe,
    and add them as prefixed columns (h4_*, d1_*) to the original 1h DataFrame.

    Features per timeframe:
      - close, sma_50, sma_200: price and trend filters
      - vol: Garman-Klass realized volatility (annualized)
      - momentum: close / close.shift(1) - 1  (single-bar return at that TF)
      - range_pct: (high - low) / close  (bar range as fraction of price)

    All lower-frequency columns are forward-filled to align with the 1h index,
    so the strategy only sees information available at each 1h bar.
    """
    import numpy as np

    timeframes = [
        ("h4", "4h"),
        ("d1", "1D"),
    ]

    for prefix, rule in timeframes:
        ohlcv_resamp = df[["open", "high", "low", "close", "volume"]].resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        close_tf = ohlcv_resamp["close"]
        high_tf = ohlcv_resamp["high"]
        low_tf = ohlcv_resamp["low"]
        open_tf = ohlcv_resamp["open"]

        features = pd.DataFrame(index=ohlcv_resamp.index)
        features[f"{prefix}_close"] = close_tf
        features[f"{prefix}_sma_50"] = close_tf.rolling(50, min_periods=1).mean()
        features[f"{prefix}_sma_200"] = close_tf.rolling(200, min_periods=1).mean()

        log_hl = np.log(high_tf / low_tf)
        log_co = np.log(close_tf / open_tf)
        gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        bars_per_year = 365 * 24 if rule == "1D" else 365 * 6
        features[f"{prefix}_vol"] = np.sqrt(gk_var.rolling(20, min_periods=1).mean() * bars_per_year)

        features[f"{prefix}_momentum"] = close_tf.pct_change()
        features[f"{prefix}_range_pct"] = (high_tf - low_tf) / close_tf

        df = df.join(features, how="left")
        for col in features.columns:
            df[col] = df[col].ffill()

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = fetch_ohlcv("BTC/USDT", "1h", days=730)
    print(f"Rows: {len(df)}")
    print(f"From: {df.index.min()}  To: {df.index.max()}")
    print(df.tail(3))
