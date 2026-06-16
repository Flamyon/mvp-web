from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config import ARTIFACTS_DIR
from src.feature_engineering import (
    add_future_evaluation_target,
    check_model_availability,
    engineer_features,
    load_preprocessing_params,
)


def price_data(n: int = 320, log_step: float = 0.01) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-04 00:00:00", periods=n, freq="5min", tz="UTC"),
            "close": [math.exp(log_step * index) for index in range(n)],
        }
    )


def test_engineer_features_adds_expected_columns() -> None:
    features = engineer_features(price_data(), x_mean_train=-11.0, x_std_train=1.0)
    assert list(features.columns) == [
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


def test_engineer_features_uses_only_past_returns() -> None:
    features = engineer_features(price_data(n=320, log_step=0.01), x_mean_train=-11.0, x_std_train=1.0)
    first = features.iloc[0]
    expected_rv_12 = 12 * (0.01**2)
    expected_rv_48 = 48 * (0.01**2)
    expected_rv_288 = 288 * (0.01**2)
    first_har = features.dropna(subset=["log_rv_past_288"]).iloc[0]

    assert first["timestamp"] == pd.Timestamp("2026-06-04 01:00:00", tz="UTC")
    assert np.isclose(first["rv_past_12"], expected_rv_12)
    assert pd.isna(first["rv_past_48"])
    assert pd.isna(first["rv_past_288"])
    assert np.isclose(first["log_rv_past_12"], math.log(expected_rv_12 + 1e-12))
    assert first_har["timestamp"] == pd.Timestamp("2026-06-05 00:00:00", tz="UTC")
    assert np.isclose(first_har["rv_past_48"], expected_rv_48)
    assert np.isclose(first_har["rv_past_288"], expected_rv_288)
    assert np.isclose(first_har["log_rv_past_48"], math.log(expected_rv_48 + 1e-12))
    assert np.isclose(first_har["log_rv_past_288"], math.log(expected_rv_288 + 1e-12))


def test_engineer_features_drops_initial_nan_rows() -> None:
    raw = price_data(n=320)
    features = engineer_features(raw, x_mean_train=-11.0, x_std_train=1.0)
    assert len(features) == len(raw) - 12
    assert features["log_rv_past_12"].notna().all()
    assert features["log_rv_past_48"].notna().sum() == len(raw) - 48
    assert features["log_rv_past_288"].notna().sum() == len(raw) - 288


def test_engineer_features_rejects_non_positive_std() -> None:
    with pytest.raises(ValueError, match="x_std_train"):
        engineer_features(price_data(), x_mean_train=-11.0, x_std_train=0.0)


def test_check_model_availability() -> None:
    small = pd.DataFrame({"log_rv_past_12": range(10)})
    ar_ready = availability_features(49)
    mini_ready = availability_features(312)
    knn_ready = availability_features(549)
    recommended = availability_features(750)
    mini_recommended = availability_features(712)

    assert check_model_availability(small)["persistence"]["available"]
    assert not check_model_availability(small)["har_global"]["available"]
    assert not check_model_availability(small)["ar49"]["available"]
    assert check_model_availability(ar_ready)["har_global"]["available"]
    assert check_model_availability(ar_ready)["ar49"]["available"]
    assert check_model_availability(mini_ready)["har_mini"]["available"]
    assert not check_model_availability(mini_ready)["har_mini"]["recommended"]
    assert check_model_availability(knn_ready)["knn"]["available"]
    assert not check_model_availability(knn_ready)["knn"]["recommended"]
    assert check_model_availability(mini_recommended)["har_mini"]["recommended"]
    assert check_model_availability(recommended)["knn"]["recommended"]


def test_load_preprocessing_params() -> None:
    params = load_preprocessing_params(ARTIFACTS_DIR / "preprocessing_params.json")
    assert params["x_std_train"] > 0
    assert params["epsilon_for_log_rv"] > 0
    assert params["horizon_bars"] == 12


def test_add_future_evaluation_target_uses_future_returns() -> None:
    features = pd.DataFrame(
        {
            "r_squared": [1.0, 2.0, 3.0, 4.0, 5.0],
            "log_rv_past_12": [-10.0, -9.0, -8.0, -7.0, -6.0],
        }
    )
    evaluated = add_future_evaluation_target(features, horizon_bars=2, epsilon=1e-12)

    assert evaluated.loc[0, "rv_future_12"] == 5.0
    assert evaluated.loc[1, "rv_future_12"] == 7.0
    assert np.isclose(evaluated.loc[0, "log_rv_future_12"], math.log(5.0 + 1e-12))


def test_add_future_evaluation_target_leaves_last_horizon_nan() -> None:
    features = pd.DataFrame({"r_squared": [1.0, 2.0, 3.0, 4.0]})
    evaluated = add_future_evaluation_target(features, horizon_bars=2)

    assert evaluated["rv_future_12"].tail(2).isna().all()
    assert evaluated["log_rv_future_12"].tail(2).isna().all()


def test_add_future_evaluation_target_does_not_change_past_features() -> None:
    features = engineer_features(price_data(n=320), x_mean_train=-11.0, x_std_train=1.0)
    before = features["log_rv_past_12"].copy()
    evaluated = add_future_evaluation_target(features, horizon_bars=3)

    pd.testing.assert_series_equal(before, evaluated["log_rv_past_12"], check_names=False)


def test_engineer_features_1000_rows_keeps_expected_har_rows() -> None:
    features = engineer_features(price_data(n=1000), x_mean_train=-11.0, x_std_train=1.0)

    assert len(features) == 988
    assert features["log_rv_past_12"].notna().sum() == 988
    assert features["log_rv_past_288"].notna().sum() == 712
    assert {"log_rv_past_12", "log_rv_past_48", "log_rv_past_288"} <= set(features.columns)


def test_model_availability_counts_each_model_independently() -> None:
    features = engineer_features(price_data(n=1000), x_mean_train=-11.0, x_std_train=1.0)
    availability = check_model_availability(features)

    assert availability["knn"]["available_values"] == 988
    assert availability["knn"]["recommended"]
    assert availability["har_mini"]["available_values"] == 712
    assert availability["har_mini"]["recommended"]


def availability_features(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_rv_past_12": np.linspace(-12.0, -10.0, rows),
            "log_rv_past_48": np.linspace(-11.8, -9.8, rows),
            "log_rv_past_288": np.linspace(-11.5, -9.5, rows),
        }
    )
