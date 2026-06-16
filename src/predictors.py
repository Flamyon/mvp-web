"""Pure model predictors for realized-volatility forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering import add_future_evaluation_target


AR_ORDER = 49
HAR_FEATURES = ["log_rv_past_12", "log_rv_past_48", "log_rv_past_288"]
HAR_TARGET = "log_rv_future_12"


def _as_1d_finite_array(values: list[float] | np.ndarray, name: str) -> np.ndarray:
    """Convert input values to a finite 1D float array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_finite_scalar(value: float, name: str) -> float:
    """Validate and normalize a finite scalar float."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


class PersistenceModel:
    """Baseline predictor: future realized volatility equals current value."""

    def predict(self, log_rv_past_12_current: float) -> float:
        """Return the current log realized-volatility value unchanged."""
        return _validate_finite_scalar(log_rv_past_12_current, "log_rv_past_12_current")


class AR49Model:
    """Autoregressive order-49 predictor with recursive 12-step forecast."""

    def __init__(
        self,
        coefficients: dict[int, float],
        x_mean: float,
        x_std: float,
        horizon_bars: int = 12,
    ):
        """Create an AR(49) predictor from exported coefficients."""
        if set(coefficients) != set(range(1, AR_ORDER + 1)):
            raise ValueError("AR49Model requires coefficients for lags 1..49")

        self.coefficients = {
            lag: _validate_finite_scalar(coefficients[lag], f"coefficient lag {lag}")
            for lag in range(1, AR_ORDER + 1)
        }
        self.x_mean = _validate_finite_scalar(x_mean, "x_mean")
        self.x_std = _validate_finite_scalar(x_std, "x_std")
        self.horizon_bars = int(horizon_bars)

        if self.x_std <= 0:
            raise ValueError("x_std must be positive")
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")

    def predict(self, log_rv_past_12_history: list[float] | np.ndarray) -> float:
        """Predict log_rv_future_12 from an ascending log_rv_past_12 history."""
        history = _as_1d_finite_array(log_rv_past_12_history, "log_rv_past_12_history")
        if len(history) < AR_ORDER:
            raise ValueError(f"AR49Model requires at least {AR_ORDER} history values")

        z_predictions = ((history[-AR_ORDER:] - self.x_mean) / self.x_std).astype(float).tolist()
        for _ in range(self.horizon_bars):
            z_recent = z_predictions[-AR_ORDER:]
            z_pred = sum(self.coefficients[lag] * z_recent[-lag] for lag in range(1, AR_ORDER + 1))
            z_pred = _validate_finite_scalar(z_pred, "z_pred")
            z_predictions.append(z_pred)

        log_rv_pred = z_predictions[-1] * self.x_std + self.x_mean
        return _validate_finite_scalar(log_rv_pred, "log_rv_pred")


class KNNModel:
    """Local k-nearest-neighbor predictor in delay-embedding space."""

    def __init__(
        self,
        train_vectors: np.ndarray,
        train_targets: np.ndarray,
        tau: int,
        m: int,
        x_mean: float,
        x_std: float,
        k: int = 50,
    ):
        """Create a kNN predictor from exported train vectors and targets."""
        self.tau = int(tau)
        self.m = int(m)
        self.k = int(k)
        self.x_mean = _validate_finite_scalar(x_mean, "x_mean")
        self.x_std = _validate_finite_scalar(x_std, "x_std")

        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.m <= 0:
            raise ValueError("m must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.x_std <= 0:
            raise ValueError("x_std must be positive")

        vectors = np.asarray(train_vectors, dtype=float)
        targets = np.asarray(train_targets, dtype=float)
        if vectors.ndim != 2:
            raise ValueError("train_vectors must be a 2D matrix")
        if vectors.shape[1] != self.m:
            raise ValueError(f"train_vectors must have {self.m} columns")
        if targets.ndim != 1:
            raise ValueError("train_targets must be a 1D array")
        if len(vectors) != len(targets):
            raise ValueError("train_vectors and train_targets must have the same length")
        if len(vectors) == 0:
            raise ValueError("train_vectors must not be empty")
        if self.k > len(vectors):
            raise ValueError("k must be <= len(train_vectors)")
        if not np.isfinite(vectors).all():
            raise ValueError("train_vectors must contain only finite values")
        if not np.isfinite(targets).all():
            raise ValueError("train_targets must contain only finite values")

        self.train_vectors = vectors
        self.train_targets = targets

    @property
    def required_history_values(self) -> int:
        """Minimum useful history values for the configured embedding."""
        return (self.m - 1) * self.tau + 1

    def build_embedding(self, z_sequence: list[float] | np.ndarray) -> np.ndarray:
        """Build X_t=[z_t,z_{t-tau},...,z_{t-(m-1)tau}] from ascending z history."""
        z_values = _as_1d_finite_array(z_sequence, "z_sequence")
        if len(z_values) < self.required_history_values:
            raise ValueError(f"KNNModel requires at least {self.required_history_values} history values")
        return np.asarray([z_values[-1 - offset * self.tau] for offset in range(self.m)], dtype=float)

    def predict(self, log_rv_history: list[float] | np.ndarray) -> float:
        """Predict log_rv_future_12 as the mean target of the k nearest neighbors."""
        result = self.predict_with_neighbors(log_rv_history)
        return float(result["prediction"])

    def predict_with_neighbors(self, log_rv_history: list[float] | np.ndarray) -> dict[str, Any]:
        """Predict and return neighbor diagnostics for the current embedding."""
        history = _as_1d_finite_array(log_rv_history, "log_rv_history")
        z_history = (history - self.x_mean) / self.x_std
        embedding = self.build_embedding(z_history)

        deltas = self.train_vectors - embedding
        squared_distances = np.einsum("ij,ij->i", deltas, deltas)
        nearest = np.argpartition(squared_distances, self.k - 1)[: self.k]
        nearest = nearest[np.argsort(squared_distances[nearest])]
        nearest_distances = np.sqrt(squared_distances[nearest])
        prediction = float(np.mean(self.train_targets[nearest]))
        prediction = _validate_finite_scalar(prediction, "prediction")

        return {
            "prediction": prediction,
            "mean_neighbor_distance": float(np.mean(nearest_distances)),
            "min_neighbor_distance": float(np.min(nearest_distances)),
            "neighbor_indices": nearest.tolist(),
        }


class HARLogRVGlobalModel:
    """Compact HAR-logRV predictor loaded from the exported Phase 14 artifact."""

    def __init__(
        self,
        intercept: float,
        coefficients: dict[str, float] | list[float],
        features: list[str],
        target: str,
    ):
        """Create a HAR-logRV model from intercept and feature coefficients."""
        if features != HAR_FEATURES:
            raise ValueError(f"HAR features must be {HAR_FEATURES}")
        if target != HAR_TARGET:
            raise ValueError(f"HAR target must be {HAR_TARGET}")

        self.intercept = _validate_finite_scalar(intercept, "intercept")
        self.features = features[:]
        self.target = target
        if isinstance(coefficients, dict):
            if set(coefficients) != set(self.features):
                raise ValueError("HAR coefficients must match HAR features")
            self.coefficients = {
                feature: _validate_finite_scalar(coefficients[feature], f"coefficient {feature}")
                for feature in self.features
            }
        else:
            if len(coefficients) != len(self.features):
                raise ValueError("HAR coefficient vector length does not match features")
            self.coefficients = {
                feature: _validate_finite_scalar(value, f"coefficient {feature}")
                for feature, value in zip(self.features, coefficients)
            }

    def predict(self, row_or_dataframe: pd.Series | pd.DataFrame | dict[str, Any]) -> float:
        """Predict log_rv_future_12 from one row containing HAR features."""
        row = _last_row_mapping(row_or_dataframe)
        prediction = self.intercept
        for feature in self.features:
            prediction += self.coefficients[feature] * _validate_finite_scalar(row[feature], feature)
        return _validate_finite_scalar(prediction, "har_logrv_prediction")


def fit_har_logrv_ols(X: list[list[float]] | np.ndarray, y: list[float] | np.ndarray) -> dict[str, Any]:
    """Fit compact HAR-logRV by normal equations with tiny numerical ridge fallback."""
    x_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 2 or x_array.shape[1] != len(HAR_FEATURES):
        raise ValueError("X must be a 2D matrix with three HAR features")
    if y_array.ndim != 1 or len(y_array) != len(x_array) or len(y_array) == 0:
        raise ValueError("y must be a non-empty vector aligned with X")
    if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
        raise ValueError("HAR training data must be finite")

    design = np.column_stack([np.ones(len(x_array)), x_array])
    ridge = 0.0
    try:
        beta = np.linalg.solve(design.T @ design, design.T @ y_array)
    except np.linalg.LinAlgError:
        ridge = 1e-8
        normal = design.T @ design
        beta = np.linalg.solve(normal + ridge * np.eye(normal.shape[0]), design.T @ y_array)

    intercept = _validate_finite_scalar(beta[0], "har_intercept")
    coefficients = {
        feature: _validate_finite_scalar(value, f"har_beta_{feature}")
        for feature, value in zip(HAR_FEATURES, beta[1:])
    }
    return {
        "intercept": intercept,
        "coefficients": coefficients,
        "features": HAR_FEATURES[:],
        "target": HAR_TARGET,
        "ridge_lambda_used": ridge,
    }


def predict_har_logrv(model: dict[str, Any] | HARLogRVGlobalModel, rows_or_df: pd.DataFrame | list[dict[str, Any]]) -> np.ndarray:
    """Predict HAR-logRV for multiple rows."""
    model_obj = (
        model
        if isinstance(model, HARLogRVGlobalModel)
        else HARLogRVGlobalModel(
            intercept=float(model["intercept"]),
            coefficients=model["coefficients"],
            features=list(model.get("features", HAR_FEATURES)),
            target=str(model.get("target", HAR_TARGET)),
        )
    )
    rows = rows_or_df if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    predictions = [model_obj.predict(row) for _, row in rows.iterrows()]
    return np.asarray(predictions, dtype=float)


def fit_mini_har_from_recent_features(
    feature_df: pd.DataFrame,
    train_fraction: float = 0.7,
    min_train_rows: int = 200,
) -> dict[str, Any]:
    """Fit a local HAR on loaded data for experimental diagnostics and final prediction.

    Future targets are created only for past rows where the future is inside the loaded
    window. The final prediction uses only the latest row's past HAR features.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    required = {"timestamp", *HAR_FEATURES, "r_squared"}
    missing = required - set(feature_df.columns)
    if missing:
        return {
            "status": "unavailable",
            "prediction": None,
            "note": f"Faltan columnas para Mini-HAR: {sorted(missing)}",
        }
    if feature_df.empty:
        return {
            "status": "unavailable",
            "prediction": None,
            "note": "Mini-HAR requiere filas con features HAR",
        }

    last_row = feature_df.iloc[-1]
    try:
        _ = [_validate_finite_scalar(last_row[feature], feature) for feature in HAR_FEATURES]
    except Exception as exc:
        return {
            "status": "unavailable",
            "prediction": None,
            "note": f"Ultima fila sin features HAR validas: {exc}",
        }

    evaluation_df = add_future_evaluation_target(feature_df)
    effective = evaluation_df.dropna(subset=[*HAR_FEATURES, HAR_TARGET]).copy()
    for column in [*HAR_FEATURES, HAR_TARGET]:
        effective[column] = pd.to_numeric(effective[column], errors="coerce")
    effective = effective[np.isfinite(effective[[*HAR_FEATURES, HAR_TARGET]]).all(axis=1)].reset_index(drop=True)

    if len(effective) < 300:
        return {
            "status": "unavailable",
            "prediction": None,
            "n_effective_rows": int(len(effective)),
            "note": "Mini-HAR requiere al menos 300 filas con target futuro evaluable",
        }

    train_n = int(len(effective) * train_fraction)
    test_n = len(effective) - train_n
    if train_n < min_train_rows or test_n <= 0:
        return {
            "status": "unavailable",
            "prediction": None,
            "n_effective_rows": int(len(effective)),
            "train_n": int(train_n),
            "test_n": int(test_n),
            "note": "Mini-HAR no tiene suficientes filas para split local",
        }

    train = effective.iloc[:train_n]
    test = effective.iloc[train_n:]
    model = fit_har_logrv_ols(train[HAR_FEATURES].to_numpy(), train[HAR_TARGET].to_numpy())
    y_true = test[HAR_TARGET].to_numpy(dtype=float)
    y_pred = predict_har_logrv(model, test)
    mean_train = float(train[HAR_TARGET].mean())
    errors = y_pred - y_true
    squared_errors = errors**2
    denominator = float(np.sum((y_true - mean_train) ** 2))
    r2_oos = 1.0 - float(np.sum(squared_errors)) / denominator if denominator > 0 else np.nan

    final_prediction = float(predict_har_logrv(model, pd.DataFrame([last_row]))[0])
    return {
        "status": "ok",
        "prediction": _validate_finite_scalar(final_prediction, "mini_har_prediction"),
        "coefficients": model["coefficients"],
        "intercept": model["intercept"],
        "local_metrics": {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(squared_errors))),
            "r2_oos": _validate_finite_scalar(r2_oos, "mini_har_r2_oos") if np.isfinite(r2_oos) else None,
            "bias": float(np.mean(errors)),
        },
        "n_effective_rows": int(len(effective)),
        "train_n": int(train_n),
        "test_n": int(test_n),
        "note": "Experimental: reentrena HAR con la ventana cargada",
    }


def _last_row_mapping(row_or_dataframe: pd.Series | pd.DataFrame | dict[str, Any]) -> dict[str, Any] | pd.Series:
    if isinstance(row_or_dataframe, pd.DataFrame):
        if row_or_dataframe.empty:
            raise ValueError("HAR prediction dataframe must not be empty")
        return row_or_dataframe.iloc[-1]
    if isinstance(row_or_dataframe, pd.Series):
        return row_or_dataframe
    if isinstance(row_or_dataframe, dict):
        return row_or_dataframe
    raise ValueError("HAR prediction input must be a row, dict, or dataframe")
