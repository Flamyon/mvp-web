"""Load and normalize non-Binance price data sources."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MAX_UPLOAD_ROWS


TIME_COLUMN_CANDIDATES = ("timestamp", "open_time", "date", "datetime", "time")


def load_sample_data(sample_path: Path) -> pd.DataFrame:
    """Load the packaged sample CSV and normalize it to timestamp,close."""
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    df = pd.read_csv(sample_path)
    missing = {"timestamp", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"Sample data missing required columns: {sorted(missing)}")
    return normalize_price_dataframe(df)


def load_user_csv(uploaded_file: Any) -> pd.DataFrame:
    """Load a user CSV-like object and normalize it to timestamp,close."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {exc}") from exc
    return normalize_price_dataframe(df)


def normalize_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame with price data to exactly timestamp,close."""
    if df.empty:
        raise ValueError("Input data is empty")

    columns_by_lower = {str(column).strip().lower(): column for column in df.columns}
    close_column = columns_by_lower.get("close")
    if close_column is None:
        raise ValueError("Missing required close column")

    time_column = find_time_column(columns_by_lower)
    if time_column is None:
        raise ValueError(
            "Missing timestamp column. Accepted names: "
            + ", ".join(TIME_COLUMN_CANDIDATES)
        )

    normalized = pd.DataFrame(
        {
            "timestamp": parse_timestamps(df[time_column]),
            "close": pd.to_numeric(df[close_column], errors="coerce"),
        }
    )
    normalized = normalized.dropna(subset=["timestamp", "close"])
    normalized = normalized[normalized["close"] > 0].copy()
    if normalized.empty:
        raise ValueError("No valid rows after timestamp/close normalization")

    normalized = (
        normalized.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    if len(normalized) > MAX_UPLOAD_ROWS:
        warnings.warn(
            f"Input has {len(normalized)} rows; keeping the last {MAX_UPLOAD_ROWS}",
            RuntimeWarning,
            stacklevel=2,
        )
        normalized = normalized.tail(MAX_UPLOAD_ROWS).reset_index(drop=True)

    return normalized.loc[:, ["timestamp", "close"]]


def find_time_column(columns_by_lower: dict[str, Any]) -> Any | None:
    """Find the first recognized time column, preferring timestamp."""
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in columns_by_lower:
            return columns_by_lower[candidate]
    return None


def parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse datetime strings or clearly numeric Unix timestamps to UTC."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_fraction = numeric.notna().mean() if len(series) else 0.0
    if numeric_fraction >= 0.9:
        median_abs = numeric.dropna().abs().median()
        if pd.notna(median_abs) and median_abs > 1e11:
            return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        if pd.notna(median_abs) and median_abs > 1e9:
            return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")
