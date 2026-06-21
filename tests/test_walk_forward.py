from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, SAMPLE_DIR, SAMPLE_FILENAME
from src.data_loader import load_sample_data
from src.feature_engineering import add_future_evaluation_target, engineer_features
from src.model_registry import ModelRegistry, load_preprocessing_params
from src.predictors import AR49Model, HAR_FEATURES, HAR_TARGET, HARLogRVGlobalModel, KNNModel, PersistenceModel
from src.walk_forward import (
    MINI_HAR_MIN_TRAIN_ROWS,
    RESULT_COLUMNS,
    run_walk_forward_predictions,
    summarize_walk_forward_results,
)


def synthetic_feature_df(rows: int = 70) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-04", periods=rows, freq="5min", tz="UTC"),
            "log_rv_past_12": np.linspace(-12.0, -10.0, rows),
            "log_rv_past_48": np.linspace(-11.8, -9.8, rows),
            "log_rv_past_288": np.linspace(-11.5, -9.5, rows),
            "r_squared": np.linspace(1e-6, 2e-6, rows),
        }
    )


def synthetic_models() -> dict[str, object]:
    coefficients = {lag: 0.0 for lag in range(1, 50)}
    coefficients[1] = 0.5
    return {
        "har_global": HARLogRVGlobalModel(
            intercept=-2.0,
            coefficients={
                "log_rv_past_12": 0.4,
                "log_rv_past_48": 0.3,
                "log_rv_past_288": 0.2,
            },
            features=HAR_FEATURES,
            target=HAR_TARGET,
        ),
        "persistence": PersistenceModel(),
        "ar49": AR49Model(coefficients, x_mean=-11.0, x_std=1.0),
        "knn": KNNModel(
            train_vectors=np.zeros((3, 3)),
            train_targets=np.array([-11.0, -10.5, -10.0]),
            tau=2,
            m=3,
            x_mean=-11.0,
            x_std=1.0,
            k=1,
        ),
    }


def test_walk_forward_persistence_starts_early() -> None:
    features = add_future_evaluation_target(synthetic_feature_df(), horizon_bars=3)
    results = run_walk_forward_predictions(features, synthetic_models())

    assert len(results["har_global"]) == len(results["persistence"])
    assert results["har_mini"].empty
    assert len(results["persistence"]) > len(results["ar49"])
    assert len(results["persistence"]) > len(results["knn"])


def test_walk_forward_ar49_requires_49_values() -> None:
    features = add_future_evaluation_target(synthetic_feature_df(), horizon_bars=3)
    results = run_walk_forward_predictions(features, synthetic_models())

    assert results["ar49"]["timestamp"].iloc[0] == features["timestamp"].iloc[48]


def test_walk_forward_knn_requires_embedding_history() -> None:
    features = add_future_evaluation_target(synthetic_feature_df(), horizon_bars=3)
    models = synthetic_models()
    results = run_walk_forward_predictions(features, models)
    required = models["knn"].required_history_values

    assert results["knn"]["timestamp"].iloc[0] == features["timestamp"].iloc[required - 1]


def test_walk_forward_mini_har_requires_observed_past_targets() -> None:
    features = add_future_evaluation_target(synthetic_feature_df(rows=260))
    results = run_walk_forward_predictions(features, synthetic_models())

    assert not results["har_mini"].empty
    assert results["har_mini"]["timestamp"].iloc[0] == features["timestamp"].iloc[MINI_HAR_MIN_TRAIN_ROWS + 11]


def test_walk_forward_returns_expected_columns() -> None:
    features = add_future_evaluation_target(synthetic_feature_df(rows=260))
    results = run_walk_forward_predictions(features, synthetic_models(), max_points_per_model=10)

    assert list(results["har_global"].columns) == RESULT_COLUMNS
    assert list(results["har_mini"].columns) == RESULT_COLUMNS
    assert list(results["persistence"].columns) == RESULT_COLUMNS
    assert len(results["har_global"]) == 10
    assert len(results["har_mini"]) == 10
    assert len(results["persistence"]) == 10


def test_walk_forward_horizon_covers_the_twelve_future_candle_opens() -> None:
    features = add_future_evaluation_target(synthetic_feature_df())
    result = run_walk_forward_predictions(
        features,
        synthetic_models(),
        max_points_per_model=1,
    )["persistence"].iloc[0]

    assert result["horizon_start"] - result["timestamp"] == pd.Timedelta(minutes=5)
    assert result["horizon_end"] - result["timestamp"] == pd.Timedelta(minutes=60)


def test_summarize_walk_forward_results() -> None:
    features = add_future_evaluation_target(synthetic_feature_df(rows=260))
    results = run_walk_forward_predictions(features, synthetic_models(), max_points_per_model=10)
    summary = summarize_walk_forward_results(results)

    assert list(summary.columns) == [
        "model",
        "n_predictions",
        "rmse",
        "mae",
        "bias",
        "first_timestamp",
        "last_timestamp",
    ]
    assert summary.loc[summary["model"] == "persistence", "n_predictions"].iloc[0] == 10
    assert summary.loc[summary["model"] == "har_global", "n_predictions"].iloc[0] == 10
    assert summary.loc[summary["model"] == "har_mini", "n_predictions"].iloc[0] == 10
    assert math.isfinite(summary.loc[summary["model"] == "persistence", "rmse"].iloc[0])


def test_walk_forward_with_real_sample() -> None:
    params = load_preprocessing_params(ARTIFACTS_DIR / "preprocessing_params.json")
    price_df = load_sample_data(SAMPLE_DIR / SAMPLE_FILENAME)
    features = engineer_features(
        price_df,
        x_mean_train=params["x_mean_train"],
        x_std_train=params["x_std_train"],
        epsilon=params["epsilon_for_log_rv"],
        rv_window=params["horizon_bars"],
    )
    evaluated = add_future_evaluation_target(features)
    results = run_walk_forward_predictions(
        evaluated,
        ModelRegistry(ARTIFACTS_DIR).load_all_models(),
        max_points_per_model=20,
    )

    assert not results["har_global"].empty
    assert not results["har_mini"].empty
    assert not results["persistence"].empty
    assert not results["ar49"].empty
    assert not results["knn"].empty
