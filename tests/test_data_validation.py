from __future__ import annotations

import pandas as pd

from src.data_validation import detect_frequency_and_gaps, summarize_price_data, validate_price_data


def clean_5m_data(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-04 00:00:00", periods=n, freq="5min", tz="UTC"),
            "close": [100.0 + index for index in range(n)],
        }
    )


def test_validate_price_data_clean_5m_data() -> None:
    result = validate_price_data(clean_5m_data())
    assert result.valid
    assert result.errors == []


def test_validate_price_data_rejects_missing_columns() -> None:
    result = validate_price_data(pd.DataFrame({"timestamp": ["2026-06-04 00:00:00"]}))
    assert not result.valid
    assert result.errors


def test_validate_price_data_rejects_non_positive_close() -> None:
    df = clean_5m_data()
    df.loc[3, "close"] = 0.0
    result = validate_price_data(df)
    assert not result.valid
    assert any("close <= 0" in error for error in result.errors)


def test_detect_frequency_and_gaps_detects_5m() -> None:
    info = detect_frequency_and_gaps(clean_5m_data()["timestamp"])
    assert info["modal_frequency"] == pd.Timedelta(minutes=5)
    assert info["is_expected_5m"]
    assert info["n_gaps"] == 0


def test_detect_frequency_and_gaps_detects_gap() -> None:
    timestamps = pd.Series(
        [
            pd.Timestamp("2026-06-04 00:00:00", tz="UTC"),
            pd.Timestamp("2026-06-04 00:05:00", tz="UTC"),
            pd.Timestamp("2026-06-04 00:15:00", tz="UTC"),
        ]
    )
    info = detect_frequency_and_gaps(timestamps)
    assert info["n_gaps"] == 1
    assert info["largest_gap"] == pd.Timedelta(minutes=10)


def test_summarize_price_data_returns_expected_fields() -> None:
    summary = summarize_price_data(clean_5m_data())
    expected = {
        "n_rows",
        "timestamp_start",
        "timestamp_end",
        "time_range",
        "price_min",
        "price_max",
        "last_close",
        "timezone",
        "utc_notice",
    }
    assert expected.issubset(summary)
    assert summary["timezone"] == "UTC"
