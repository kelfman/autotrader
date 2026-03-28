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
            return cached[(cached.index >= pd.Timestamp(since_dt, tz="UTC"))].copy()

    log.info("Fetching %s %s from exchange (days=%d)…", symbol, timeframe, days)

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    since_ms = int(since_dt.timestamp() * 1000)
    until_ms = int(until_dt.timestamp() * 1000)

    fresh = _fetch_all(exchange, symbol, timeframe, since_ms, until_ms)

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

    return fresh[(fresh.index >= pd.Timestamp(since_dt, tz="UTC"))].copy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = fetch_ohlcv("BTC/USDT", "1h", days=730)
    print(f"Rows: {len(df)}")
    print(f"From: {df.index.min()}  To: {df.index.max()}")
    print(df.tail(3))
