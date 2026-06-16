from __future__ import annotations

import pandas as pd
import pytest
import requests

from src import binance_client


def raw_kline(open_time: int, close: str = "42505.00", close_time: int | None = None) -> list:
    return [
        open_time,
        "42500.00",
        "42520.00",
        "42490.00",
        close,
        "1.234",
        close_time if close_time is not None else open_time + 299_999,
        "52500.00",
        100,
        "0.5",
        "21250",
        "0",
    ]


def test_normalize_binance_klines_uses_open_time_as_timestamp() -> None:
    open_time = 1_700_000_000_000
    close_time = open_time + 299_999
    df = binance_client.normalize_binance_klines([raw_kline(open_time, close_time=close_time)])
    assert df["timestamp"].iloc[0] == pd.Timestamp(open_time, unit="ms", tz="UTC")
    assert df["timestamp"].iloc[0] != pd.Timestamp(close_time, unit="ms", tz="UTC")


def test_normalize_binance_klines_returns_only_timestamp_close() -> None:
    df = binance_client.normalize_binance_klines([raw_kline(1_700_000_000_000)])
    assert list(df.columns) == ["timestamp", "close"]


def test_drop_incomplete_candles_removes_open_last_candle() -> None:
    now = pd.Timestamp("2026-06-04 12:00:00", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-06-04 11:50:00", tz="UTC"),
                pd.Timestamp("2026-06-04 11:55:00", tz="UTC"),
            ],
            "close": [100.0, 101.0],
            "_close_time": [
                pd.Timestamp("2026-06-04 11:54:59", tz="UTC"),
                pd.Timestamp("2026-06-04 11:59:59", tz="UTC"),
            ],
        }
    )
    cleaned = binance_client.drop_incomplete_candles(df, now_utc=now)
    assert len(cleaned) == 1
    assert cleaned["close"].tolist() == [100.0]
    assert list(cleaned.columns) == ["timestamp", "close"]


def test_drop_incomplete_candles_keeps_closed_candles() -> None:
    now = pd.Timestamp("2026-06-04 12:00:20", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-06-04 11:55:00", tz="UTC")],
            "close": [101.0],
            "_close_time": [pd.Timestamp("2026-06-04 11:59:59", tz="UTC")],
        }
    )
    cleaned = binance_client.drop_incomplete_candles(df, now_utc=now)
    assert len(cleaned) == 1
    assert cleaned["close"].iloc[0] == 101.0


def test_fetch_recent_rejects_limit_over_1000() -> None:
    with pytest.raises(ValueError):
        binance_client.fetch_recent_btcusdt_klines(limit=1001)


def test_fetch_recent_handles_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list:
            return []

    monkeypatch.setattr(binance_client.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match="empty kline response|Could not fetch"):
        binance_client.fetch_recent_btcusdt_klines(max_retries=1)


def test_fetch_recent_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status_code = 500

        def raise_for_status(self) -> None:
            raise requests.exceptions.HTTPError(response=self)

    monkeypatch.setattr(binance_client.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match="Binance HTTP error: 500"):
        binance_client.fetch_recent_btcusdt_klines()
