from __future__ import annotations

import math

import pandas as pd

from src.config import ARTIFACTS_DIR, SAMPLE_DIR, SAMPLE_FILENAME
from src.data_loader import load_sample_data
from src.feature_engineering import check_model_availability, engineer_features
from src.model_registry import (
    ModelRegistry,
    load_ar_coefficients,
    load_embedding_params,
    load_har_logrv_artifact,
    load_knn_reference,
    load_preprocessing_params,
    run_available_predictions,
)
from src.predictors import AR49Model, HARLogRVGlobalModel, KNNModel, PersistenceModel


def test_load_ar_coefficients_real_artifact() -> None:
    coefficients = load_ar_coefficients(ARTIFACTS_DIR / "ar49_coefficients.csv")
    assert len(coefficients) == 49
    assert set(coefficients) == set(range(1, 50))


def test_load_preprocessing_params_real_artifact() -> None:
    params = load_preprocessing_params(ARTIFACTS_DIR / "preprocessing_params.json")
    assert params["x_std_train"] > 0
    assert params["horizon_bars"] == 12


def test_load_embedding_params_real_artifact() -> None:
    params = load_embedding_params(ARTIFACTS_DIR / "embedding_params.json")
    assert params["tau_selected"] == 137
    assert params["m_selected"] == 5
    assert params["horizon_bars"] == 12


def test_load_knn_reference_real_artifact() -> None:
    reference = load_knn_reference(ARTIFACTS_DIR / "knn_reference_train.npz")
    assert set(reference) == {"vectors", "targets"}
    assert reference["vectors"].ndim == 2
    assert reference["targets"].ndim == 1
    assert reference["vectors"].shape[1] == 5
    assert len(reference["vectors"]) == len(reference["targets"])


def test_load_har_logrv_artifact_real_artifact() -> None:
    model = load_har_logrv_artifact(ARTIFACTS_DIR / "har_logrv_model.json")
    assert isinstance(model, HARLogRVGlobalModel)


def test_model_registry_load_all_models() -> None:
    models = ModelRegistry(ARTIFACTS_DIR).load_all_models()
    assert isinstance(models["persistence"], PersistenceModel)
    assert isinstance(models["ar49"], AR49Model)
    assert isinstance(models["knn"], KNNModel)
    assert isinstance(models["har_global"], HARLogRVGlobalModel)
    assert models["knn"].k == 200


def test_run_available_predictions_with_sample() -> None:
    params = load_preprocessing_params(ARTIFACTS_DIR / "preprocessing_params.json")
    sample = load_sample_data(SAMPLE_DIR / SAMPLE_FILENAME)
    feature_df = engineer_features(
        sample,
        x_mean_train=params["x_mean_train"],
        x_std_train=params["x_std_train"],
        epsilon=params["epsilon_for_log_rv"],
        rv_window=params["horizon_bars"],
    )
    models = ModelRegistry(ARTIFACTS_DIR).load_all_models()
    availability = check_model_availability(feature_df)
    payload = run_available_predictions(feature_df, models, availability)

    predictions = payload["predictions"]
    for model_name in ("har_global", "har_mini", "persistence", "ar49", "knn"):
        result = predictions[model_name]
        assert result["status"] == "ok"
        assert math.isfinite(result["log_rv_future_12"])
        assert math.isfinite(result["rv_future_12"])
        assert math.isfinite(result["sqrt_rv_percent"])

    metadata = payload["metadata"]
    assert metadata["horizon_start"] - metadata["timestamp_used"] == pd.Timedelta(minutes=5)
    assert metadata["horizon_end"] - metadata["timestamp_used"] == pd.Timedelta(minutes=60)
    assert str(metadata["timestamp_used"].tzinfo) == "UTC"
