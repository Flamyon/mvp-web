from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

from src.config import SAMPLE_DIR, SAMPLE_FILENAME
from src.data_loader import load_sample_data, normalize_price_dataframe


def test_load_sample_data() -> None:
    df = load_sample_data(SAMPLE_DIR / SAMPLE_FILENAME)
    assert list(df.columns) == ["timestamp", "close"]
    assert not df.empty
    assert (df["close"] > 0).all()


def test_normalize_price_dataframe_accepts_timestamp_close() -> None:
    df = pd.DataFrame({"timestamp": ["2026-06-04 12:00:00"], "close": ["100.5"]})
    normalized = normalize_price_dataframe(df)
    assert list(normalized.columns) == ["timestamp", "close"]
    assert normalized["timestamp"].iloc[0] == pd.Timestamp("2026-06-04 12:00:00", tz="UTC")
    assert normalized["close"].iloc[0] == 100.5


def test_normalize_price_dataframe_accepts_open_time_close() -> None:
    df = pd.DataFrame({"open_time": ["2026-06-04 12:00:00"], "close": [100.5]})
    normalized = normalize_price_dataframe(df)
    assert normalized["timestamp"].iloc[0] == pd.Timestamp("2026-06-04 12:00:00", tz="UTC")


def test_normalize_price_dataframe_rejects_missing_close() -> None:
    df = pd.DataFrame({"timestamp": ["2026-06-04 12:00:00"], "open": [100.5]})
    with pytest.raises(ValueError, match="close"):
        normalize_price_dataframe(df)


def test_normalize_price_dataframe_rejects_missing_timestamp() -> None:
    df = pd.DataFrame({"close": [100.5]})
    with pytest.raises(ValueError, match="timestamp"):
        normalize_price_dataframe(df)


def test_normalize_price_dataframe_sorts_and_deduplicates() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-06-04 12:05:00",
                "2026-06-04 12:00:00",
                "2026-06-04 12:05:00",
            ],
            "close": [101.0, 100.0, 102.0],
        }
    )
    normalized = normalize_price_dataframe(df)
    assert normalized["timestamp"].tolist() == [
        pd.Timestamp("2026-06-04 12:00:00", tz="UTC"),
        pd.Timestamp("2026-06-04 12:05:00", tz="UTC"),
    ]
    assert normalized["close"].tolist() == [100.0, 102.0]


def test_normalize_price_dataframe_accepts_unix_milliseconds() -> None:
    millis = 1_700_000_000_000
    df = pd.DataFrame({"time": [millis], "close": [100.0]})
    normalized = normalize_price_dataframe(df)
    assert normalized["timestamp"].iloc[0] == pd.Timestamp(millis, unit="ms", tz="UTC")


def test_load_user_csv_like_timestamp_close() -> None:
    from src.data_loader import load_user_csv

    normalized = load_user_csv(StringIO("timestamp,close\n2026-06-04 12:00:00,100.0\n"))
    assert list(normalized.columns) == ["timestamp", "close"]
    assert normalized["close"].iloc[0] == 100.0
