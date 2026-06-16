"""Pure Binance Spot REST client and kline normalizer."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from src.config import BINANCE_KLINES_ENDPOINTS, BINANCE_KLINES_URL, BINANCE_LIMIT, INTERVAL, SYMBOL


def fetch_recent_btcusdt_klines(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    limit: int = BINANCE_LIMIT,
    timeout: int = 10,
    max_retries: int = 3,
    endpoint_url: str = BINANCE_KLINES_URL,
) -> list:
    """Fetch recent raw klines from Binance Spot REST API."""
    if limit > 1000:
        raise ValueError("Binance klines limit must be <= 1000")
    if limit <= 0:
        raise ValueError("Binance klines limit must be positive")
    if max_retries <= 0:
        raise ValueError("max_retries must be positive")

    params = {"symbol": symbol, "interval": interval, "limit": limit}
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(endpoint_url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                raise RuntimeError("Binance returned an empty kline response")
            return payload
        except requests.exceptions.Timeout as exc:
            last_error = exc
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", "unknown")
            raise RuntimeError(f"Binance HTTP error: {status}") from exc
        except requests.exceptions.RequestException as exc:
            last_error = exc
        except ValueError as exc:
            raise RuntimeError("Binance returned invalid JSON") from exc
        except RuntimeError as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(min(0.25 * attempt, 1.0))

    raise RuntimeError(f"Could not fetch Binance klines after {max_retries} attempts: {last_error}")


def fetch_recent_btcusdt_klines_with_source(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    limit: int = BINANCE_LIMIT,
    timeout: int = 10,
    max_retries: int = 3,
    endpoints: tuple[dict[str, str], ...] = BINANCE_KLINES_ENDPOINTS,
) -> tuple[list, str]:
    """Fetch recent raw klines, falling back across compatible Binance endpoints."""
    errors: list[str] = []
    for endpoint in endpoints:
        name = endpoint["name"]
        try:
            raw_klines = fetch_recent_btcusdt_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                timeout=timeout,
                max_retries=max_retries,
                endpoint_url=endpoint["url"],
            )
            return raw_klines, name
        except RuntimeError as exc:
            errors.append(f"{name}: {exc}")

    raise RuntimeError("No compatible Binance endpoint responded. " + " | ".join(errors))


def normalize_binance_klines(raw_klines: list) -> pd.DataFrame:
    """Normalize Binance kline arrays to timestamp,close using open_time as timestamp."""
    if not raw_klines:
        raise ValueError("No Binance klines to normalize")

    rows: list[dict[str, Any]] = []
    for position, candle in enumerate(raw_klines):
        if not isinstance(candle, (list, tuple)) or len(candle) <= 6:
            raise ValueError(f"Invalid Binance kline at position {position}")
        rows.append(
            {
                "timestamp": pd.to_datetime(int(candle[0]), unit="ms", utc=True),
                "close": float(candle[4]),
                "_close_time": pd.to_datetime(int(candle[6]), unit="ms", utc=True),
            }
        )

    df = pd.DataFrame(rows)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close", "_close_time"])
    if (df["close"] <= 0).any():
        raise ValueError("Binance klines contain non-positive close prices")

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return drop_incomplete_candles(df)


def drop_incomplete_candles(
    df: pd.DataFrame,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Drop the last kline if its close_time indicates it is not safely closed yet."""
    if df.empty:
        return df.loc[:, ["timestamp", "close"]].copy() if {"timestamp", "close"}.issubset(df.columns) else df.copy()

    working = df.copy()
    if "_close_time" in working.columns:
        now = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        last_close_time = pd.to_datetime(working["_close_time"].iloc[-1], utc=True)
        if last_close_time > now - pd.Timedelta(seconds=10):
            working = working.iloc[:-1].reset_index(drop=True)

    return working.loc[:, ["timestamp", "close"]].reset_index(drop=True)


def fetch_and_normalize_recent_btcusdt(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    limit: int = BINANCE_LIMIT,
) -> pd.DataFrame:
    """Fetch and normalize recent BTCUSDT klines."""
    raw_klines = fetch_recent_btcusdt_klines(symbol=symbol, interval=interval, limit=limit)
    return normalize_binance_klines(raw_klines)


def fetch_and_normalize_recent_btcusdt_with_source(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    limit: int = BINANCE_LIMIT,
) -> tuple[pd.DataFrame, str]:
    """Fetch and normalize recent BTCUSDT klines, returning the endpoint used."""
    raw_klines, source_name = fetch_recent_btcusdt_klines_with_source(
        symbol=symbol,
        interval=interval,
        limit=limit,
    )
    return normalize_binance_klines(raw_klines), source_name
