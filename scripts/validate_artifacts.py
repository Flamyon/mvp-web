#!/usr/bin/env python3
"""Validate exported MVP artifacts without reading btc-volatility."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from export_model_artifacts import (
    HORIZON_BARS,
    inspect_npz,
    read_ar_coefficients,
    read_json,
    read_npz_numeric_array,
    validate_knn_npz,
)


PREDICTION_MIN = -20.0
PREDICTION_MAX = -5.0
RUNTIME_KNN_K = 200
HAR_FEATURES = ["log_rv_past_12", "log_rv_past_48", "log_rv_past_288"]


def validate_artifacts(mvp_path: Path) -> dict[str, float]:
    mvp_path = mvp_path.resolve()
    artifacts_dir = mvp_path / "data" / "model_artifacts"
    sample_path = mvp_path / "data" / "sample" / "btcusdt_5m_recent_sample.csv"

    required_files = [
        artifacts_dir / "ar49_coefficients.csv",
        artifacts_dir / "preprocessing_params.json",
        artifacts_dir / "embedding_params.json",
        artifacts_dir / "knn_reference_train.npz",
        artifacts_dir / "historical_metrics.json",
        artifacts_dir / "har_logrv_model.json",
        sample_path,
    ]
    for path in required_files:
        assert_exists(path)

    preprocessing = read_json(artifacts_dir / "preprocessing_params.json")
    embedding = read_json(artifacts_dir / "embedding_params.json")
    historical = read_json(artifacts_dir / "historical_metrics.json")
    har_artifact = read_json(artifacts_dir / "har_logrv_model.json")
    validate_json_no_runtime_btc_refs(preprocessing, "preprocessing_params.json")
    validate_json_no_runtime_btc_refs(embedding, "embedding_params.json")
    validate_json_no_runtime_btc_refs(historical, "historical_metrics.json")
    validate_json_no_runtime_btc_refs(har_artifact, "har_logrv_model.json")

    if float(preprocessing["x_std_train"]) <= 0.0:
        raise AssertionError("x_std_train must be positive")
    if int(embedding["tau_selected"]) != 137:
        raise AssertionError("tau_selected must be 137")
    if int(embedding["m_selected"]) != 5:
        raise AssertionError("m_selected must be 5")
    if int(embedding["horizon_bars"]) != HORIZON_BARS:
        raise AssertionError("horizon_bars must be 12")
    if int(embedding["minimum_useful_log_rv_values"]) != 549:
        raise AssertionError("minimum_useful_log_rv_values must be 549")
    validate_har_artifact(har_artifact)

    coeffs = read_ar_coefficients(artifacts_dir / "ar49_coefficients.csv")
    if len(coeffs) != 49:
        raise AssertionError(f"Expected 49 AR coefficients, got {len(coeffs)}")

    npz_info = inspect_npz(artifacts_dir / "knn_reference_train.npz")
    if not {"vectors", "targets"}.issubset(npz_info["keys"]):
        raise AssertionError("knn_reference_train.npz must contain vectors and targets")
    validate_knn_npz(artifacts_dir / "knn_reference_train.npz", expected_dim=5)
    vectors, vector_shape = read_npz_numeric_array(artifacts_dir / "knn_reference_train.npz", "vectors")
    targets, target_shape = read_npz_numeric_array(artifacts_dir / "knn_reference_train.npz", "targets")
    if vector_shape[1] != int(embedding["m_selected"]):
        raise AssertionError("vectors.shape[1] does not match m_selected")
    if target_shape[0] != vector_shape[0]:
        raise AssertionError("targets length does not match vectors")

    timestamps, closes = read_sample(sample_path)
    if len(timestamps) < 1000:
        raise AssertionError(f"Sample is too short: {len(timestamps)} rows")
    if any(close <= 0.0 for close in closes):
        raise AssertionError("Sample contains non-positive prices")

    epsilon = float(preprocessing["epsilon_for_log_rv"])
    log_rv_values = compute_log_rv_past(closes, 12, epsilon)
    log_rv_past_48 = compute_log_rv_past(closes, 48, epsilon)
    log_rv_past_288 = compute_log_rv_past(closes, 288, epsilon)
    if len(log_rv_values) < int(embedding["minimum_useful_log_rv_values"]):
        raise AssertionError("Not enough useful log_rv_past_12 values for kNN")

    persistence_pred = log_rv_values[-1]
    ar_pred = predict_ar49(
        log_rv_values[-49:],
        coeffs,
        float(preprocessing["x_mean_train"]),
        float(preprocessing["x_std_train"]),
        HORIZON_BARS,
    )
    knn_pred = predict_knn(
        log_rv_values,
        vectors,
        targets,
        tau=int(embedding["tau_selected"]),
        dim=int(embedding["m_selected"]),
        k=RUNTIME_KNN_K,
        x_mean=float(embedding["x_mean_train"]),
        x_std=float(embedding["x_std_train"]),
    )
    har_pred = predict_har(
        har_artifact,
        {
            "log_rv_past_12": log_rv_values[-1],
            "log_rv_past_48": log_rv_past_48[-1],
            "log_rv_past_288": log_rv_past_288[-1],
        },
    )

    predictions = {
        "har_global": har_pred,
        "persistence": persistence_pred,
        "ar49": ar_pred,
        f"knn_k{RUNTIME_KNN_K}": knn_pred,
    }
    for name, value in predictions.items():
        if not math.isfinite(value):
            raise AssertionError(f"{name} prediction is not finite")
        if not (PREDICTION_MIN < value < PREDICTION_MAX):
            raise AssertionError(f"{name} prediction out of expected broad range: {value}")

    return predictions


def validate_har_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("model_name") != "har_logrv_compact":
        raise AssertionError("HAR artifact model_name must be har_logrv_compact")
    if artifact.get("target") != "log_rv_future_12":
        raise AssertionError("HAR artifact target must be log_rv_future_12")
    if artifact.get("features") != HAR_FEATURES:
        raise AssertionError(f"HAR artifact features must be {HAR_FEATURES}")
    if int(artifact.get("horizon_bars", HORIZON_BARS)) != HORIZON_BARS:
        raise AssertionError("HAR artifact horizon_bars must be 12")
    _finite_float(artifact.get("intercept"), "HAR intercept")
    coefficients = artifact.get("coefficients")
    if not isinstance(coefficients, dict):
        raise AssertionError("HAR artifact coefficients must be an object")
    for feature in HAR_FEATURES:
        _finite_float(coefficients.get(feature), f"HAR coefficient {feature}")


def _finite_float(value: Any, label: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{label} must be numeric") from exc
    if not math.isfinite(scalar):
        raise AssertionError(f"{label} must be finite")
    return scalar


def assert_exists(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Required file not found: {path}")


def validate_json_no_runtime_btc_refs(value: Any, label: str) -> None:
    bad_fragments = ("/home/", "../btc-volatility", "btc-volatility/")
    if isinstance(value, dict):
        for nested in value.values():
            validate_json_no_runtime_btc_refs(nested, label)
    elif isinstance(value, list):
        for nested in value:
            validate_json_no_runtime_btc_refs(nested, label)
    elif isinstance(value, str):
        if value.startswith("/") or any(fragment in value for fragment in bad_fragments):
            raise AssertionError(f"{label} contains a runtime reference to btc-volatility or an absolute path: {value}")


def read_sample(path: Path) -> tuple[list[str], list[float]]:
    timestamps: list[str] = []
    closes: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["timestamp", "close"]:
            raise AssertionError(f"Sample columns must be exactly timestamp,close; got {reader.fieldnames}")
        for row in reader:
            timestamps.append(row["timestamp"])
            close = float(row["close"])
            if not math.isfinite(close):
                raise AssertionError("Sample contains non-finite close")
            closes.append(close)
    return timestamps, closes


def compute_log_rv_past(closes: list[float], window: int, epsilon: float) -> list[float]:
    returns: list[float] = [float("nan")]
    for previous, current in zip(closes, closes[1:]):
        returns.append(math.log(current / previous))
    log_rv: list[float] = []
    for index in range(window, len(returns)):
        current_window = returns[index - window + 1:index + 1]
        rv = sum(value * value for value in current_window)
        log_rv.append(math.log(rv + epsilon))
    return log_rv


def predict_har(artifact: dict[str, Any], features: dict[str, float]) -> float:
    prediction = float(artifact["intercept"])
    for feature in HAR_FEATURES:
        prediction += float(artifact["coefficients"][feature]) * features[feature]
    return prediction


def predict_ar49(
    log_rv_history: list[float],
    coeffs: list[float],
    x_mean: float,
    x_std: float,
    horizon: int,
) -> float:
    if len(log_rv_history) < len(coeffs):
        raise AssertionError("Not enough history for AR(49)")
    z_values = [(value - x_mean) / x_std for value in log_rv_history[-len(coeffs):]]
    for _step in range(horizon):
        recent = z_values[-len(coeffs):]
        forecast = 0.0
        for lag, coeff in enumerate(coeffs, start=1):
            forecast += coeff * recent[-lag]
        z_values.append(forecast)
    return z_values[-1] * x_std + x_mean


def predict_knn(
    log_rv_history: list[float],
    train_vectors: list[Any],
    train_targets: list[Any],
    tau: int,
    dim: int,
    k: int,
    x_mean: float,
    x_std: float,
) -> float:
    min_values = (dim - 1) * tau + 1
    if len(log_rv_history) < min_values:
        raise AssertionError(f"Not enough history for kNN: {len(log_rv_history)} < {min_values}")
    z_values = [(value - x_mean) / x_std for value in log_rv_history]
    query = tuple(z_values[-1 - coord * tau] for coord in range(dim))
    best: list[tuple[float, int]] = []
    for index, vector in enumerate(train_vectors):
        distance = 0.0
        for left, right in zip(query, vector):
            diff = left - right
            distance += diff * diff
            if len(best) == k and distance >= -best[0][0]:
                break
        item = (-distance, index)
        if len(best) < k:
            import heapq
            heapq.heappush(best, item)
        elif distance < -best[0][0]:
            import heapq
            heapq.heapreplace(best, item)
    neighbor_indices = [index for _neg_distance, index in best]
    return sum(float(train_targets[index]) for index in neighbor_indices) / len(neighbor_indices)


def build_parser() -> argparse.ArgumentParser:
    script_path = Path(__file__).resolve()
    default_mvp_path = script_path.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvp-web-path", type=Path, default=default_mvp_path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    predictions = validate_artifacts(args.mvp_web_path)
    print("Artifact validation passed.")
    for name, value in predictions.items():
        print(f"{name}: {value:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
