from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.predictors import (
    AR49Model,
    HAR_FEATURES,
    HAR_TARGET,
    HARLogRVGlobalModel,
    KNNModel,
    PersistenceModel,
    fit_har_logrv_ols,
    fit_mini_har_from_recent_features,
    predict_har_logrv,
)


def ar_coefficients(lag1: float = 0.5) -> dict[int, float]:
    coefficients = {lag: 0.0 for lag in range(1, 50)}
    coefficients[1] = lag1
    return coefficients


def test_persistence_returns_current_value() -> None:
    assert PersistenceModel().predict(-11.25) == -11.25


def test_persistence_rejects_nan() -> None:
    with pytest.raises(ValueError, match="finite"):
        PersistenceModel().predict(float("nan"))


def test_ar49_rejects_short_history() -> None:
    model = AR49Model(ar_coefficients(), x_mean=0.0, x_std=1.0)
    with pytest.raises(ValueError, match="at least 49"):
        model.predict([0.0] * 48)


def test_ar49_predicts_finite_value() -> None:
    model = AR49Model(ar_coefficients(lag1=0.75), x_mean=-11.0, x_std=2.0)
    prediction = model.predict(np.linspace(-12.0, -10.0, 80))
    assert math.isfinite(prediction)


def test_ar49_requires_49_coefficients() -> None:
    with pytest.raises(ValueError, match="lags 1..49"):
        AR49Model({lag: 0.0 for lag in range(1, 49)}, x_mean=0.0, x_std=1.0)


def test_ar49_rejects_non_positive_std() -> None:
    with pytest.raises(ValueError, match="x_std"):
        AR49Model(ar_coefficients(), x_mean=0.0, x_std=0.0)


def test_knn_build_embedding_convention() -> None:
    model = KNNModel(
        train_vectors=np.array([[0.0, 0.0, 0.0]]),
        train_targets=np.array([-10.0]),
        tau=2,
        m=3,
        x_mean=0.0,
        x_std=1.0,
        k=1,
    )
    embedding = model.build_embedding(np.arange(10.0))
    np.testing.assert_array_equal(embedding, np.array([9.0, 7.0, 5.0]))


def test_knn_rejects_short_history() -> None:
    model = KNNModel(
        train_vectors=np.array([[0.0, 0.0, 0.0]]),
        train_targets=np.array([-10.0]),
        tau=2,
        m=3,
        x_mean=0.0,
        x_std=1.0,
        k=1,
    )
    with pytest.raises(ValueError, match="at least 5"):
        model.predict([0.0, 1.0, 2.0, 3.0])


def test_knn_predicts_mean_of_nearest_targets() -> None:
    model = KNNModel(
        train_vectors=np.array([[0.0, 0.0], [10.0, 10.0], [1.0, 1.0]]),
        train_targets=np.array([2.0, 20.0, 4.0]),
        tau=1,
        m=2,
        x_mean=0.0,
        x_std=1.0,
        k=2,
    )
    assert model.predict([0.0, 0.0]) == 3.0


def test_knn_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="2D"):
        KNNModel(
            train_vectors=np.array([0.0, 1.0]),
            train_targets=np.array([-10.0]),
            tau=1,
            m=1,
            x_mean=0.0,
            x_std=1.0,
            k=1,
        )

    with pytest.raises(ValueError, match="same length"):
        KNNModel(
            train_vectors=np.array([[0.0], [1.0]]),
            train_targets=np.array([-10.0]),
            tau=1,
            m=1,
            x_mean=0.0,
            x_std=1.0,
            k=1,
        )


def test_knn_rejects_k_too_large() -> None:
    with pytest.raises(ValueError, match="k"):
        KNNModel(
            train_vectors=np.array([[0.0], [1.0]]),
            train_targets=np.array([-10.0, -11.0]),
            tau=1,
            m=1,
            x_mean=0.0,
            x_std=1.0,
            k=3,
        )


def test_har_global_predicts_from_har_features() -> None:
    model = HARLogRVGlobalModel(
        intercept=1.0,
        coefficients={
            "log_rv_past_12": 0.5,
            "log_rv_past_48": 0.25,
            "log_rv_past_288": -0.1,
        },
        features=HAR_FEATURES,
        target=HAR_TARGET,
    )

    prediction = model.predict(
        {
            "log_rv_past_12": -10.0,
            "log_rv_past_48": -9.0,
            "log_rv_past_288": -8.0,
        }
    )

    assert np.isclose(prediction, 1.0 + 0.5 * -10.0 + 0.25 * -9.0 - 0.1 * -8.0)


def test_har_global_rejects_wrong_features() -> None:
    with pytest.raises(ValueError, match="HAR features"):
        HARLogRVGlobalModel(
            intercept=0.0,
            coefficients={"log_rv_past_12": 1.0},
            features=["log_rv_past_12"],
            target=HAR_TARGET,
        )


def test_fit_har_logrv_ols_and_predict() -> None:
    x_values = np.array(
        [
            [-12.0, -11.0, -10.0],
            [-11.0, -10.0, -9.0],
            [-10.0, -9.0, -8.0],
            [-9.0, -8.0, -7.0],
        ]
    )
    y_values = 2.0 + 0.2 * x_values[:, 0] + 0.3 * x_values[:, 1] + 0.4 * x_values[:, 2]
    model = fit_har_logrv_ols(x_values, y_values)
    predictions = predict_har_logrv(
        model,
        [
            {
                "log_rv_past_12": -8.0,
                "log_rv_past_48": -7.0,
                "log_rv_past_288": -6.0,
            }
        ],
    )

    assert predictions.shape == (1,)
    assert math.isfinite(predictions[0])


def test_mini_har_uses_future_targets_for_local_evaluation_only() -> None:
    rows = 340
    feature_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-04", periods=rows, freq="5min", tz="UTC"),
            "log_rv_past_12": np.linspace(-12.0, -10.0, rows),
            "log_rv_past_48": np.linspace(-11.8, -9.8, rows),
            "log_rv_past_288": np.linspace(-11.5, -9.5, rows),
            "r_squared": np.linspace(1e-7, 5e-6, rows),
        }
    )

    result = fit_mini_har_from_recent_features(feature_df)

    assert result["status"] == "ok"
    assert math.isfinite(result["prediction"])
    assert result["train_n"] >= 200
    assert result["test_n"] > 0
    assert {"rmse", "mae", "r2_oos", "bias"} <= set(result["local_metrics"])
