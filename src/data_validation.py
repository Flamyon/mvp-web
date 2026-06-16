"""Validation helpers for normalized BTC price data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


EXPECTED_FREQUENCY = pd.Timedelta(minutes=5)
GAP_THRESHOLD = pd.Timedelta(minutes=5.5)
MIN_ROWS = 50
MAX_HAR_FEATURE_WINDOW = 288
UTC_NOTICE = "Todos los timestamps se interpretan y muestran en UTC."


@dataclass
class ValidationResult:
    """Structured result for price-data validation."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    info: dict[str, Any]


def validate_price_data(df: pd.DataFrame) -> ValidationResult:
    """Validate a normalized price DataFrame without mutating it."""
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}

    required = {"timestamp", "close"}
    missing = required - set(df.columns)
    if missing:
        return ValidationResult(
            valid=False,
            errors=[f"Faltan columnas requeridas: {sorted(missing)}"],
            warnings=[],
            info={},
        )

    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    invalid_timestamps = int(timestamps.isna().sum())
    invalid_closes = int(close.isna().sum())
    if invalid_timestamps:
        errors.append(f"Hay {invalid_timestamps} timestamps invalidos")
    if invalid_closes:
        errors.append(f"Hay {invalid_closes} cierres no numericos")

    valid_mask = timestamps.notna() & close.notna()
    working = pd.DataFrame({"timestamp": timestamps[valid_mask], "close": close[valid_mask]})

    if len(working) < MIN_ROWS:
        errors.append(f"Insuficientes datos: {len(working)} < {MIN_ROWS}")

    if not working.empty and (working["close"] <= 0).any():
        errors.append("Hay precios close <= 0")

    if not working.empty:
        duplicate_count = int(working["timestamp"].duplicated().sum())
        info["duplicate_timestamps"] = duplicate_count
        if duplicate_count >= len(working) - 1:
            errors.append("Los timestamps estan completamente duplicados")
        elif duplicate_count:
            warnings.append(f"Se detectaron {duplicate_count} timestamps duplicados")

        if working["timestamp"].nunique() <= 1:
            errors.append("El rango temporal es nulo")

        if not working["timestamp"].is_monotonic_increasing:
            warnings.append("Los timestamps no estan ordenados ascendentemente")

        now_utc = pd.Timestamp.now(tz="UTC")
        max_timestamp = working["timestamp"].max()
        if max_timestamp > now_utc + pd.Timedelta(hours=1):
            warnings.append("Hay timestamps mas de 1 hora en el futuro respecto a UTC")

        frequency = detect_frequency_and_gaps(working["timestamp"])
        info["frequency"] = frequency
        if not frequency["is_expected_5m"]:
            warnings.append(f"Frecuencia modal distinta de 5 minutos: {frequency['modal_frequency']}")
        if int(frequency["n_gaps"]) > 0:
            warnings.append(
                f"Gaps detectados: {frequency['n_gaps']} de {frequency['n_intervals']} intervalos"
            )

        summary = summarize_price_data(working)
        info["summary"] = summary
        if summary["time_range"] < pd.Timedelta(hours=4):
            warnings.append("Rango temporal corto para diagnostico estable")

        info["expected_har_feature_rows"] = max(0, len(working) - MAX_HAR_FEATURE_WINDOW)

        returns = (working["close"] / working["close"].shift(1)).apply(_safe_log)
        max_abs_return = returns.abs().max(skipna=True)
        info["max_abs_return"] = None if pd.isna(max_abs_return) else float(max_abs_return)
        if pd.notna(max_abs_return) and max_abs_return > 0.5:
            warnings.append(f"Cambio extremo de precio detectado: {max_abs_return:.2%}")

    info["utc_notice"] = UTC_NOTICE
    return ValidationResult(valid=not errors, errors=errors, warnings=warnings, info=info)


def detect_frequency_and_gaps(timestamps: pd.Series) -> dict[str, Any]:
    """Detect modal frequency and gaps in a timestamp series."""
    parsed = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().sort_values()
    diffs = parsed.diff().dropna()
    if diffs.empty:
        return {
            "modal_frequency": None,
            "expected_frequency": EXPECTED_FREQUENCY,
            "n_intervals": 0,
            "n_gaps": 0,
            "gap_percentage": 0.0,
            "largest_gap": None,
            "is_expected_5m": False,
        }

    modal_frequency = diffs.mode().iloc[0]
    gaps = diffs[diffs > GAP_THRESHOLD]
    n_intervals = int(len(diffs))
    n_gaps = int(len(gaps))
    return {
        "modal_frequency": modal_frequency,
        "expected_frequency": EXPECTED_FREQUENCY,
        "n_intervals": n_intervals,
        "n_gaps": n_gaps,
        "gap_percentage": 100.0 * n_gaps / n_intervals if n_intervals else 0.0,
        "largest_gap": diffs.max(),
        "is_expected_5m": modal_frequency == EXPECTED_FREQUENCY,
    }


def summarize_price_data(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact summary for normalized price data."""
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    valid = pd.DataFrame({"timestamp": timestamps, "close": close}).dropna()
    if valid.empty:
        return {
            "n_rows": 0,
            "timestamp_start": None,
            "timestamp_end": None,
            "time_range": pd.Timedelta(0),
            "price_min": None,
            "price_max": None,
            "last_close": None,
            "timezone": "UTC",
            "utc_notice": UTC_NOTICE,
        }
    return {
        "n_rows": int(len(valid)),
        "timestamp_start": valid["timestamp"].min(),
        "timestamp_end": valid["timestamp"].max(),
        "time_range": valid["timestamp"].max() - valid["timestamp"].min(),
        "price_min": float(valid["close"].min()),
        "price_max": float(valid["close"].max()),
        "last_close": float(valid["close"].iloc[-1]),
        "timezone": str(valid["timestamp"].dt.tz),
        "utc_notice": UTC_NOTICE,
    }


def _safe_log(value: float) -> float:
    if pd.isna(value) or value <= 0:
        return float("nan")
    import math

    return math.log(float(value))
