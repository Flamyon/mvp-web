"""Model artifact loading and prediction orchestration."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import EPSILON_LOG_RV, HORIZON_BARS, HORIZON_MINUTES, load_json
from src.predictors import (
    AR_ORDER,
    HAR_FEATURES,
    HAR_TARGET,
    AR49Model,
    HARLogRVGlobalModel,
    KNNModel,
    PersistenceModel,
    fit_mini_har_from_recent_features,
)


EXPECTED_TAU = 137
EXPECTED_M = 5

MODEL_NOTES = {
    "persistence": "Baseline: replica la volatilidad realizada actual.",
    "ar49": "Benchmark lineal fuerte entrenado historicamente.",
    "knn": "Predictor local no lineal experimental.",
    "har_global": "Modelo practico recomendado, entrenado con historico largo.",
    "har_mini": "Recalibracion local experimental con la ventana cargada.",
}


def _validate_finite_float(value: object, name: str) -> float:
    """Validate a finite float value."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def load_ar_coefficients(path: Path) -> dict[int, float]:
    """Load and validate AR(49) coefficients from CSV."""
    if not path.exists():
        raise FileNotFoundError(path)

    coefficients: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or {"lag", "coefficient"} - set(reader.fieldnames):
            raise ValueError("AR coefficients CSV must contain lag and coefficient columns")
        for row in reader:
            lag = int(row["lag"])
            coefficient = _validate_finite_float(row["coefficient"], f"coefficient lag {lag}")
            coefficients[lag] = coefficient

    if set(coefficients) != set(range(1, AR_ORDER + 1)):
        raise ValueError("AR coefficients must contain exactly lags 1..49")
    return coefficients


def load_preprocessing_params(path: Path) -> dict[str, Any]:
    """Load and validate preprocessing parameters."""
    params = load_json(path)
    required = {"x_mean_train", "x_std_train", "epsilon_for_log_rv", "horizon_bars"}
    missing = required - set(params)
    if missing:
        raise ValueError(f"Missing preprocessing params: {sorted(missing)}")

    params["x_mean_train"] = _validate_finite_float(params["x_mean_train"], "x_mean_train")
    params["x_std_train"] = _validate_finite_float(params["x_std_train"], "x_std_train")
    params["epsilon_for_log_rv"] = _validate_finite_float(
        params["epsilon_for_log_rv"],
        "epsilon_for_log_rv",
    )
    params["horizon_bars"] = int(params["horizon_bars"])

    if params["x_std_train"] <= 0:
        raise ValueError("x_std_train must be positive")
    if params["epsilon_for_log_rv"] <= 0:
        raise ValueError("epsilon_for_log_rv must be positive")
    if params["horizon_bars"] != HORIZON_BARS:
        raise ValueError(f"horizon_bars must be {HORIZON_BARS}")
    return params


def load_embedding_params(path: Path) -> dict[str, Any]:
    """Load and validate delay-embedding parameters."""
    params = load_json(path)
    required = {"tau_selected", "m_selected", "horizon_bars", "x_mean_train", "x_std_train"}
    missing = required - set(params)
    if missing:
        raise ValueError(f"Missing embedding params: {sorted(missing)}")

    params["tau_selected"] = int(params["tau_selected"])
    params["m_selected"] = int(params["m_selected"])
    params["horizon_bars"] = int(params["horizon_bars"])
    params["x_mean_train"] = _validate_finite_float(params["x_mean_train"], "x_mean_train")
    params["x_std_train"] = _validate_finite_float(params["x_std_train"], "x_std_train")

    if params["tau_selected"] != EXPECTED_TAU:
        raise ValueError(f"tau_selected must be {EXPECTED_TAU}")
    if params["m_selected"] != EXPECTED_M:
        raise ValueError(f"m_selected must be {EXPECTED_M}")
    if params["horizon_bars"] != HORIZON_BARS:
        raise ValueError(f"horizon_bars must be {HORIZON_BARS}")
    if params["x_std_train"] <= 0:
        raise ValueError("x_std_train must be positive")
    return params


def load_knn_reference(path: Path) -> dict[str, np.ndarray]:
    """Load and validate exported kNN reference arrays."""
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path) as data:
        missing = {"vectors", "targets"} - set(data.files)
        if missing:
            raise ValueError(f"Missing kNN reference arrays: {sorted(missing)}")
        vectors = np.asarray(data["vectors"], dtype=float).copy()
        targets = np.asarray(data["targets"], dtype=float).copy()

    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")
    if targets.ndim != 1:
        raise ValueError("targets must be a 1D array")
    if len(vectors) != len(targets):
        raise ValueError("vectors and targets must have the same length")
    if vectors.shape[1] != EXPECTED_M:
        raise ValueError(f"vectors must have dimension {EXPECTED_M}")
    if not np.isfinite(vectors).all():
        raise ValueError("vectors must contain only finite values")
    if not np.isfinite(targets).all():
        raise ValueError("targets must contain only finite values")
    return {"vectors": vectors, "targets": targets}


def load_har_logrv_artifact(path: Path) -> HARLogRVGlobalModel:
    """Load and validate the exported HAR-logRV compact artifact."""
    if not path.exists():
        raise FileNotFoundError(path)
    artifact = load_json(path)
    if artifact.get("model_name") != "har_logrv_compact":
        raise ValueError("HAR artifact model_name must be har_logrv_compact")
    if artifact.get("target") != HAR_TARGET:
        raise ValueError(f"HAR artifact target must be {HAR_TARGET}")
    features = list(artifact.get("features", []))
    if features != HAR_FEATURES:
        raise ValueError(f"HAR artifact features must be {HAR_FEATURES}")
    if int(artifact.get("horizon_bars", HORIZON_BARS)) != HORIZON_BARS:
        raise ValueError(f"HAR artifact horizon_bars must be {HORIZON_BARS}")
    intercept = _validate_finite_float(artifact.get("intercept"), "har intercept")
    coefficients_payload = artifact.get("coefficients")
    if not isinstance(coefficients_payload, dict):
        vector = artifact.get("coefficient_vector")
        if not isinstance(vector, list):
            raise ValueError("HAR artifact must contain coefficients or coefficient_vector")
        coefficients_payload = {
            feature: vector[index]
            for index, feature in enumerate(features)
        }
    coefficients = {
        feature: _validate_finite_float(coefficients_payload.get(feature), f"har coefficient {feature}")
        for feature in features
    }
    return HARLogRVGlobalModel(
        intercept=intercept,
        coefficients=coefficients,
        features=features,
        target=HAR_TARGET,
    )


class ModelRegistry:
    """Centralized loader for local MVP model artifacts."""

    def __init__(self, artifacts_dir: Path):
        """Create a registry rooted at the local model artifacts directory."""
        self.artifacts_dir = Path(artifacts_dir)

    def load_all_models(self, k_knn: int = 200) -> dict[str, object]:
        """Load all MVP predictors from local artifacts."""
        preprocessing = load_preprocessing_params(self.artifacts_dir / "preprocessing_params.json")
        embedding = load_embedding_params(self.artifacts_dir / "embedding_params.json")
        coefficients = load_ar_coefficients(self.artifacts_dir / "ar49_coefficients.csv")
        knn_reference = load_knn_reference(self.artifacts_dir / "knn_reference_train.npz")
        har_model = load_har_logrv_artifact(self.artifacts_dir / "har_logrv_model.json")

        if embedding["x_mean_train"] != preprocessing["x_mean_train"]:
            raise ValueError("Embedding and preprocessing x_mean_train differ")
        if embedding["x_std_train"] != preprocessing["x_std_train"]:
            raise ValueError("Embedding and preprocessing x_std_train differ")
        if knn_reference["vectors"].shape[1] != embedding["m_selected"]:
            raise ValueError("kNN vector dimension does not match m_selected")

        return {
            "persistence": PersistenceModel(),
            "ar49": AR49Model(
                coefficients=coefficients,
                x_mean=preprocessing["x_mean_train"],
                x_std=preprocessing["x_std_train"],
                horizon_bars=preprocessing["horizon_bars"],
            ),
            "knn": KNNModel(
                train_vectors=knn_reference["vectors"],
                train_targets=knn_reference["targets"],
                tau=embedding["tau_selected"],
                m=embedding["m_selected"],
                x_mean=preprocessing["x_mean_train"],
                x_std=preprocessing["x_std_train"],
                k=k_knn,
            ),
            "har_global": har_model,
        }

    def load_historical_metrics(self) -> dict[str, Any]:
        """Load historical model metrics from local artifacts."""
        return load_json(self.artifacts_dir / "historical_metrics.json")


def run_available_predictions(
    feature_df: pd.DataFrame,
    models: dict[str, object],
    availability: dict[str, dict[str, object]],
    epsilon: float = EPSILON_LOG_RV,
) -> dict[str, Any]:
    """Run available models and return predictions plus UTC horizon metadata."""
    if "log_rv_past_12" not in feature_df.columns:
        raise ValueError("feature_df must contain log_rv_past_12")
    if "timestamp" not in feature_df.columns:
        raise ValueError("feature_df must contain timestamp")
    if feature_df.empty:
        raise ValueError("feature_df must not be empty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    history = np.asarray(feature_df["log_rv_past_12"], dtype=float)
    if len(history) == 0 or not np.isfinite(history).all():
        raise ValueError("log_rv_past_12 history must contain only finite values")

    last_timestamp = pd.to_datetime(feature_df["timestamp"].iloc[-1], utc=True)
    metadata = {
        "timestamp_used": last_timestamp,
        "horizon_start": last_timestamp,
        "horizon_end": last_timestamp + pd.Timedelta(minutes=HORIZON_MINUTES),
    }

    predictions: dict[str, dict[str, Any]] = {}
    for model_name in ("har_global", "har_mini", "ar49", "knn", "persistence"):
        model_availability = availability.get(model_name, {})
        if not bool(model_availability.get("available", False)):
            predictions[model_name] = {
                "status": "unavailable",
                "log_rv_future_12": None,
                "rv_future_12": None,
                "sqrt_rv_percent": None,
                "note": str(model_availability.get("message", "Modelo no disponible")),
            }
            continue

        model = models.get(model_name)
        if model is None and model_name != "har_mini":
            predictions[model_name] = {
                "status": "error",
                "log_rv_future_12": None,
                "rv_future_12": None,
                "sqrt_rv_percent": None,
                "note": "Modelo no cargado",
            }
            continue

        try:
            if model_name == "har_global":
                log_prediction = model.predict(feature_df.iloc[-1])  # type: ignore[attr-defined]
            elif model_name == "har_mini":
                mini_result = fit_mini_har_from_recent_features(feature_df)
                if mini_result.get("status") != "ok":
                    predictions[model_name] = {
                        "status": "unavailable",
                        "log_rv_future_12": None,
                        "rv_future_12": None,
                        "sqrt_rv_percent": None,
                        "note": str(mini_result.get("note", "Mini-HAR no disponible")),
                        **{key: mini_result.get(key) for key in ("n_effective_rows", "train_n", "test_n")},
                    }
                    continue
                log_prediction = float(mini_result["prediction"])
            elif model_name == "persistence":
                log_prediction = model.predict(float(history[-1]))  # type: ignore[attr-defined]
            else:
                log_prediction = model.predict(history)  # type: ignore[attr-defined]
            predictions[model_name] = _format_ok_prediction(log_prediction, epsilon, MODEL_NOTES[model_name])
            if model_name == "har_mini":
                predictions[model_name].update(
                    {
                        "local_metrics": mini_result.get("local_metrics"),
                        "coefficients": mini_result.get("coefficients"),
                        "intercept": mini_result.get("intercept"),
                        "n_effective_rows": mini_result.get("n_effective_rows"),
                        "train_n": mini_result.get("train_n"),
                        "test_n": mini_result.get("test_n"),
                    }
                )
        except Exception as exc:
            predictions[model_name] = {
                "status": "error",
                "log_rv_future_12": None,
                "rv_future_12": None,
                "sqrt_rv_percent": None,
                "note": str(exc),
            }

    return {"predictions": predictions, "metadata": metadata}


def _format_ok_prediction(log_prediction: float, epsilon: float, note: str) -> dict[str, Any]:
    """Convert a log-scale model output into display-ready prediction values."""
    log_prediction = _validate_finite_float(log_prediction, "log_prediction")
    rv_future_12 = float(np.exp(log_prediction) - epsilon)
    if not np.isfinite(rv_future_12) or rv_future_12 < 0:
        raise ValueError("Predicted rv_future_12 must be finite and non-negative")
    sqrt_rv_percent = float(np.sqrt(rv_future_12) * 100)
    if not np.isfinite(sqrt_rv_percent):
        raise ValueError("Predicted sqrt_rv_percent must be finite")
    return {
        "status": "ok",
        "log_rv_future_12": log_prediction,
        "rv_future_12": rv_future_12,
        "sqrt_rv_percent": sqrt_rv_percent,
        "note": note,
    }
