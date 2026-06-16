"""Feature engineering for realized-volatility models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import EPSILON_LOG_RV, HORIZON_BARS


FEATURE_COLUMNS = [
    "timestamp",
    "close",
    "log_close",
    "r",
    "r_squared",
    "rv_past_12",
    "rv_past_48",
    "rv_past_288",
    "log_rv_past_12",
    "log_rv_past_48",
    "log_rv_past_288",
    "z_log_rv_past_12",
]


def load_preprocessing_params(path: Path) -> dict[str, Any]:
    """Load and validate the preprocessing constants used by the local models."""
    with path.open(encoding="utf-8") as handle:
        params = json.load(handle)
    required = {"x_mean_train", "x_std_train", "epsilon_for_log_rv", "horizon_bars"}
    missing = required - set(params)
    if missing:
        raise ValueError(f"Missing preprocessing params: {sorted(missing)}")

    params["x_mean_train"] = float(params["x_mean_train"])
    params["x_std_train"] = float(params["x_std_train"])
    params["epsilon_for_log_rv"] = float(params["epsilon_for_log_rv"])
    params["horizon_bars"] = int(params["horizon_bars"])

    if params["x_std_train"] <= 0:
        raise ValueError("x_std_train must be positive")
    if params["epsilon_for_log_rv"] <= 0:
        raise ValueError("epsilon_for_log_rv must be positive")
    if params["horizon_bars"] != HORIZON_BARS:
        raise ValueError(f"horizon_bars must be {HORIZON_BARS}")
    return params


def engineer_features(
    price_df: pd.DataFrame,
    x_mean_train: float,
    x_std_train: float,
    epsilon: float = EPSILON_LOG_RV,
    rv_window: int = 12,
) -> pd.DataFrame:
    """Build realized-volatility features from normalized close prices."""
    if x_std_train <= 0:
        raise ValueError("x_std_train must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if rv_window <= 0:
        raise ValueError("rv_window must be positive")
    missing = {"timestamp", "close"} - set(price_df.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")

    df = price_df.loc[:, ["timestamp", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"])
    df = df[df["close"] > 0].copy()
    if df.empty:
        raise ValueError("No valid positive close prices")

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    df["log_close"] = np.log(df["close"])
    df["r"] = df["log_close"].diff()
    df["r_squared"] = df["r"] ** 2
    df["rv_past_12"] = df["r_squared"].rolling(window=rv_window, min_periods=rv_window).sum()
    df["rv_past_48"] = df["r_squared"].rolling(window=48, min_periods=48).sum()
    df["rv_past_288"] = df["r_squared"].rolling(window=288, min_periods=288).sum()
    df["log_rv_past_12"] = np.log(df["rv_past_12"] + epsilon)
    df["log_rv_past_48"] = np.log(df["rv_past_48"] + epsilon)
    df["log_rv_past_288"] = np.log(df["rv_past_288"] + epsilon)
    df["z_log_rv_past_12"] = (df["log_rv_past_12"] - x_mean_train) / x_std_train

    feature_df = df.dropna(subset=["log_rv_past_12", "z_log_rv_past_12"]).reset_index(drop=True)
    return feature_df.loc[:, FEATURE_COLUMNS]


def add_future_evaluation_target(
    feature_df: pd.DataFrame,
    horizon_bars: int = HORIZON_BARS,
    epsilon: float = EPSILON_LOG_RV,
) -> pd.DataFrame:
    """Add observed future RV columns for evaluation only.

    The added target uses future returns already present inside the loaded
    window. It must never be used as model input or for runtime prediction.
    """
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if "r_squared" not in feature_df.columns:
        raise ValueError("feature_df must contain r_squared")

    result = feature_df.copy()
    future_r_squared = pd.to_numeric(result["r_squared"], errors="coerce").shift(-1)
    result["rv_future_12"] = (
        future_r_squared.iloc[::-1]
        .rolling(window=horizon_bars, min_periods=horizon_bars)
        .sum()
        .iloc[::-1]
    )
    result["log_rv_future_12"] = np.log(result["rv_future_12"] + epsilon)
    return result


def check_model_availability(
    feature_df: pd.DataFrame,
    ar_order: int = 49,
    tau: int = 137,
    m: int = 5,
    recommended_knn_values: int = 750,
) -> dict[str, dict[str, object]]:
    """Check whether engineered features are sufficient for each future model."""
    if "log_rv_past_12" in feature_df.columns:
        log_rv_12 = pd.to_numeric(feature_df["log_rv_past_12"], errors="coerce")
        n_values = int(np.isfinite(log_rv_12).sum())
    else:
        n_values = 0
    har_columns = ["log_rv_past_12", "log_rv_past_48", "log_rv_past_288"]
    has_har_columns = set(har_columns) <= set(feature_df.columns)
    if has_har_columns:
        har_values = feature_df[har_columns].apply(pd.to_numeric, errors="coerce")
        har_valid = np.isfinite(har_values).all(axis=1)
        n_har_values = int(har_valid.sum())
    else:
        n_har_values = 0
    knn_required = (m - 1) * tau + 1
    mini_har_required = 300 + HORIZON_BARS
    mini_har_recommended = 700 + HORIZON_BARS
    return {
        "har_global": {
            "available": n_har_values >= 1,
            "required": 1,
            "available_values": n_har_values,
            "message": (
                "Disponible: modelo practico recomendado"
                if n_har_values >= 1
                else "Requiere features HAR validas: log_rv_past_12/48/288"
            ),
        },
        "har_mini": {
            "available": n_har_values >= mini_har_required,
            "recommended": n_har_values >= mini_har_recommended,
            "required": mini_har_required,
            "recommended_required": mini_har_recommended,
            "available_values": n_har_values,
            "message": (
                "Disponible con ventana adecuada para recalibracion local experimental"
                if n_har_values >= mini_har_recommended
                else f"Disponible, pero por debajo del umbral recomendado de {mini_har_recommended} filas HAR"
                if n_har_values >= mini_har_required
                else f"Requiere al menos {mini_har_required} filas con features HAR validas"
            ),
        },
        "persistence": {
            "available": n_values >= 1,
            "required": 1,
            "available_values": n_values,
            "message": "Disponible" if n_values >= 1 else "Requiere al menos 1 valor de log_rv_past_12",
        },
        "ar49": {
            "available": n_values >= ar_order,
            "required": ar_order,
            "available_values": n_values,
            "message": "Disponible" if n_values >= ar_order else f"Requiere {ar_order} valores utiles",
        },
        "knn": {
            "available": n_values >= knn_required,
            "recommended": n_values >= recommended_knn_values,
            "required": knn_required,
            "recommended_required": recommended_knn_values,
            "available_values": n_values,
            "message": (
                "Disponible con margen recomendado"
                if n_values >= recommended_knn_values
                else "Disponible matematicamente, pero por debajo del umbral recomendado"
                if n_values >= knn_required
                else f"Requiere {knn_required} valores utiles"
            ),
        },
    }
