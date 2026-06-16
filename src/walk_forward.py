"""Recent walk-forward evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import EPSILON_LOG_RV, HORIZON_BARS, HORIZON_MINUTES
from src.feature_engineering import add_future_evaluation_target
from src.predictors import HAR_FEATURES, HAR_TARGET, fit_har_logrv_ols, predict_har_logrv


WALK_FORWARD_MODEL_NAMES = ("persistence", "knn", "ar49", "har_global", "har_mini")
MINI_HAR_MIN_TRAIN_ROWS = 200

RESULT_COLUMNS = [
    "timestamp",
    "horizon_start",
    "horizon_end",
    "y_true_log_rv_future_12",
    "y_pred_log_rv_future_12",
    "error",
    "abs_error",
    "squared_error",
    "rv_true_future_12",
    "rv_pred_future_12",
    "sqrt_rv_true_percent",
    "sqrt_rv_pred_percent",
]


def run_walk_forward_predictions(
    feature_df: pd.DataFrame,
    models: dict[str, object],
    epsilon: float = EPSILON_LOG_RV,
    max_points_per_model: int = 300,
) -> dict[str, pd.DataFrame]:
    """Run recent walk-forward predictions using future target only for scoring."""
    if max_points_per_model <= 0:
        raise ValueError("max_points_per_model must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    required = {"timestamp", "log_rv_past_12", "r_squared"}
    missing = required - set(feature_df.columns)
    if missing:
        raise ValueError(f"feature_df missing columns: {sorted(missing)}")

    evaluation_df = (
        feature_df.copy()
        if {"rv_future_12", "log_rv_future_12"} <= set(feature_df.columns)
        else add_future_evaluation_target(feature_df, epsilon=epsilon)
    )
    evaluation_df = evaluation_df.reset_index(drop=True)
    results: dict[str, pd.DataFrame] = {}

    for model_name in WALK_FORWARD_MODEL_NAMES:
        model = models.get(model_name)
        if model is None and model_name != "har_mini":
            results[model_name] = pd.DataFrame(columns=RESULT_COLUMNS)
            continue
        indices = _evaluable_indices(evaluation_df, model_name, model)
        if len(indices) > max_points_per_model:
            indices = indices[-max_points_per_model:]
        rows = [
            _predict_one(evaluation_df, model_name, model, index, epsilon)
            for index in indices
        ]
        results[model_name] = pd.DataFrame(rows, columns=RESULT_COLUMNS)

    return results


def summarize_walk_forward_results(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize recent walk-forward metrics for each model."""
    rows: list[dict[str, Any]] = []
    for model_name in WALK_FORWARD_MODEL_NAMES:
        df = results.get(model_name, pd.DataFrame())
        if df.empty:
            rows.append(
                {
                    "model": model_name,
                    "n_predictions": 0,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "bias": np.nan,
                    "first_timestamp": None,
                    "last_timestamp": None,
                }
            )
            continue
        rows.append(
            {
                "model": model_name,
                "n_predictions": int(len(df)),
                "rmse": float(np.sqrt(df["squared_error"].mean())),
                "mae": float(df["abs_error"].mean()),
                "bias": float(df["error"].mean()),
                "first_timestamp": pd.to_datetime(df["timestamp"].iloc[0], utc=True),
                "last_timestamp": pd.to_datetime(df["timestamp"].iloc[-1], utc=True),
            }
        )
    return pd.DataFrame(rows)


def combine_walk_forward_results(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-model walk-forward result tables with a model column."""
    frames = []
    for model_name, df in results.items():
        if df.empty:
            continue
        model_df = df.copy()
        model_df.insert(0, "model", model_name)
        frames.append(model_df)
    if not frames:
        return pd.DataFrame(columns=["model", *RESULT_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def _evaluable_indices(feature_df: pd.DataFrame, model_name: str, model: object) -> list[int]:
    """Return row indices where a model has enough past history and future target."""
    target_mask = feature_df["log_rv_future_12"].notna()
    har_current_mask = pd.Series(False, index=feature_df.index)
    if set(HAR_FEATURES) <= set(feature_df.columns):
        har_current_values = feature_df[HAR_FEATURES].apply(pd.to_numeric, errors="coerce")
        har_current_mask = np.isfinite(har_current_values).all(axis=1)

    if model_name == "persistence":
        start_index = 0
    elif model_name in {"har_global", "har_mini"}:
        if not set(HAR_FEATURES) <= set(feature_df.columns):
            return []
        start_index = 0 if model_name == "har_global" else HORIZON_BARS + MINI_HAR_MIN_TRAIN_ROWS - 1
    elif model_name == "ar49":
        start_index = 48
    elif model_name == "knn":
        start_index = int(getattr(model, "required_history_values")) - 1
    else:
        start_index = len(feature_df)

    indices = []
    train_mask = target_mask & har_current_mask
    train_counts = train_mask.cumsum()
    for index in range(max(0, start_index), len(feature_df)):
        if not bool(target_mask.iloc[index]):
            continue
        if model_name in {"har_global", "har_mini"} and not bool(har_current_mask.iloc[index]):
            continue
        if model_name == "har_mini":
            train_last_index = index - HORIZON_BARS
            if train_last_index < 0 or int(train_counts.iloc[train_last_index]) < MINI_HAR_MIN_TRAIN_ROWS:
                continue
        indices.append(index)
    return indices


def _predict_one(
    feature_df: pd.DataFrame,
    model_name: str,
    model: object,
    index: int,
    epsilon: float,
) -> dict[str, Any]:
    """Predict and score a single walk-forward row."""
    history = np.asarray(feature_df["log_rv_past_12"].iloc[: index + 1], dtype=float)
    if model_name == "har_global":
        y_pred = model.predict(feature_df.iloc[index])  # type: ignore[attr-defined]
    elif model_name == "har_mini":
        y_pred = _predict_mini_har_one(feature_df, index)
    elif model_name == "persistence":
        y_pred = model.predict(float(history[-1]))  # type: ignore[attr-defined]
    else:
        y_pred = model.predict(history)  # type: ignore[attr-defined]

    y_true = float(feature_df["log_rv_future_12"].iloc[index])
    y_pred = float(y_pred)
    error = y_pred - y_true
    rv_true = float(np.exp(y_true) - epsilon)
    rv_pred = float(np.exp(y_pred) - epsilon)
    timestamp = pd.to_datetime(feature_df["timestamp"].iloc[index], utc=True)
    horizon_start = timestamp
    horizon_end = timestamp + pd.Timedelta(minutes=HORIZON_MINUTES)

    return {
        "timestamp": timestamp,
        "horizon_start": horizon_start,
        "horizon_end": horizon_end,
        "y_true_log_rv_future_12": y_true,
        "y_pred_log_rv_future_12": y_pred,
        "error": error,
        "abs_error": abs(error),
        "squared_error": error**2,
        "rv_true_future_12": rv_true,
        "rv_pred_future_12": rv_pred,
        "sqrt_rv_true_percent": float(np.sqrt(max(rv_true, 0.0)) * 100),
        "sqrt_rv_pred_percent": float(np.sqrt(max(rv_pred, 0.0)) * 100),
    }


def _predict_mini_har_one(feature_df: pd.DataFrame, index: int) -> float:
    """Fit Mini-HAR with targets observable before index and predict row index."""
    train_last_index = index - HORIZON_BARS
    if train_last_index < MINI_HAR_MIN_TRAIN_ROWS - 1:
        raise ValueError("Mini-HAR walk-forward does not have enough past observed targets")

    train = feature_df.iloc[: train_last_index + 1].dropna(subset=[*HAR_FEATURES, HAR_TARGET]).copy()
    for column in [*HAR_FEATURES, HAR_TARGET]:
        train[column] = pd.to_numeric(train[column], errors="coerce")
    train = train[np.isfinite(train[[*HAR_FEATURES, HAR_TARGET]]).all(axis=1)]
    if len(train) < MINI_HAR_MIN_TRAIN_ROWS:
        raise ValueError("Mini-HAR walk-forward does not have enough finite training rows")

    model = fit_har_logrv_ols(train[HAR_FEATURES].to_numpy(), train[HAR_TARGET].to_numpy())
    return float(predict_har_logrv(model, pd.DataFrame([feature_df.iloc[index]]))[0])
