#!/usr/bin/env python3
"""Exporta los artefactos del estudio tecnico al arbol del MVP."""

from __future__ import annotations

import argparse
import ast
import csv
import heapq
import json
import math
import shutil
import struct
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TRAIN_END = "2025-06-30 23:55:00"
VALIDATION_END = "2025-12-31 23:55:00"
HORIZON_BARS = 12
BAR_MINUTES = 5
EPSILON_FOR_LOG_RV = 1e-12
DEFAULT_SAMPLE_ROWS = 1500
DEFAULT_K = 50


@dataclass(frozen=True)
class NeighborSet:
    distances: list[float]
    targets: list[float]
    indices: list[int]


class KDNode:
    __slots__ = ("point", "axis", "left", "right")

    def __init__(self, point: int, axis: int, left: "KDNode | None", right: "KDNode | None") -> None:
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    """Exact KD-tree for low-dimensional neighbor queries."""

    def __init__(self, vectors: list[tuple[float, ...]]) -> None:
        if not vectors:
            raise ValueError("Cannot build an empty KDTree")
        self.vectors = vectors
        self.dim = len(vectors[0])
        self.root = self._build(list(range(len(vectors))), depth=0)

    def query(
        self,
        vector: tuple[float, ...],
        k: int,
        query_index: int,
        candidate_indices: list[int],
        theiler_window: int,
        horizon: int,
    ) -> list[tuple[float, int]]:
        heap: list[tuple[float, int]] = []

        def eligible(candidate_position: int) -> bool:
            candidate_index = candidate_indices[candidate_position]
            if candidate_index + horizon > query_index:
                return False
            return abs(query_index - candidate_index) > theiler_window

        def visit(node: KDNode | None) -> None:
            if node is None:
                return
            point = node.point
            axis = node.axis
            point_vector = self.vectors[point]
            diff_axis = vector[axis] - point_vector[axis]
            near = node.left if diff_axis <= 0.0 else node.right
            far = node.right if diff_axis <= 0.0 else node.left

            visit(near)

            current_limit = -heap[0][0] if len(heap) >= k else float("inf")
            if eligible(point):
                distance_sq = distance_sq_bounded(vector, point_vector, current_limit)
                if distance_sq < current_limit or len(heap) < k:
                    item = (-distance_sq, -point)
                    if len(heap) < k:
                        heapq.heappush(heap, item)
                    else:
                        heapq.heapreplace(heap, item)

            current_limit = -heap[0][0] if len(heap) >= k else float("inf")
            if len(heap) < k or diff_axis * diff_axis < current_limit:
                visit(far)

        visit(self.root)
        rows = [(-distance_sq, -position) for distance_sq, position in heap]
        rows.sort(key=lambda item: item[0])
        return rows

    def _build(self, points: list[int], depth: int) -> KDNode | None:
        if not points:
            return None
        axis = depth % self.dim
        points.sort(key=lambda point: self.vectors[point][axis])
        median = len(points) // 2
        return KDNode(
            point=points[median],
            axis=axis,
            left=self._build(points[:median], depth + 1),
            right=self._build(points[median + 1 :], depth + 1),
        )


class ArtifactExporter:
    def __init__(
        self,
        btc_volatility_path: Path,
        mvp_web_path: Path,
        sample_rows: int,
        skip_deep_knn_validation: bool,
    ) -> None:
        self.btc_path = btc_volatility_path.resolve()
        self.mvp_path = mvp_web_path.resolve()
        self.sample_rows = sample_rows
        self.skip_deep_knn_validation = skip_deep_knn_validation

        self.artifacts_dir = self.mvp_path / "data" / "model_artifacts"
        self.sample_dir = self.mvp_path / "data" / "sample"
        self.assets_dir = self.mvp_path / "assets" / "historical_validation"

        self._phase11_summary: dict[str, Any] | None = None
        self._phase8_params: dict[str, Any] | None = None

        for directory in [self.artifacts_dir, self.sample_dir, self.assets_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> None:
        print("Starting artifact export")
        self.export_ar49_coefficients()
        self.export_preprocessing_params()
        self.export_embedding_params()
        self.export_knn_reference()
        self.export_har_logrv_artifact()
        self.export_historical_metrics()
        self.export_sample_data()
        self.export_figures()

        self.run_historical_validation()

        print(f"Export complete: {self.mvp_path}")

    def export_ar49_coefficients(self) -> None:
        source = self.btc_path / "reports" / "tables" / "phase6_ar_coefficients.csv"
        destination = self.artifacts_dir / "ar49_coefficients.csv"
        require_file(source)
        shutil.copy2(source, destination)
        coeffs = read_ar_coefficients(destination)
        if len(coeffs) != 49:
            raise ValueError(f"Expected 49 AR coefficients, got {len(coeffs)}")

    def export_preprocessing_params(self) -> None:
        summary = self.phase11_summary()
        split_train = next(row for row in summary["split_summary"] if row["split"] == "train")
        params = {
            "series": "log_rv_past_12",
            "target": "log_rv_future_12",
            "x_mean_train": float(summary["x_mean_train"]),
            "x_std_train": float(summary["x_std_train"]),
            "epsilon_for_log_rv": EPSILON_FOR_LOG_RV,
            "horizon_bars": int(summary["horizon_bars"]),
            "horizon_minutes": int(summary["horizon_minutes"]),
            "source_phase": "phase11_local_state_space_prediction",
            "source_artifacts": [
                "reports/tables/phase11_prediction_summary.json",
                "reports/tables/phase1_shape_summary.csv",
            ],
            "train_period_start": split_train["row_start_time"],
            "train_period_end": split_train["row_end_time"],
            "train_sample_size": int(split_train["row_n"]),
        }
        if params["x_std_train"] <= 0.0:
            raise ValueError("x_std_train must be positive")
        destination = self.artifacts_dir / "preprocessing_params.json"
        write_json(destination, params)

    def export_embedding_params(self) -> None:
        phase8 = self.phase8_params()
        phase11 = self.phase11_summary()
        tau = int(phase11["tau"])
        dim = int(phase11["m"])
        effective_history_bars = (dim - 1) * tau
        params = {
            "series": phase8.get("series", "z_log_rv_past_12"),
            "source_series": phase8.get("source_series", "log_rv_past_12"),
            "target": "log_rv_future_12",
            "tau_selected": tau,
            "m_selected": dim,
            "selected_k_by_validation_rmse": int(phase11["selected_k_by_validation_rmse"]),
            "theiler_window": int(phase11["theiler_window"]),
            "embedding_convention": phase11["phase8_reference"]["embedding_convention"],
            "horizon_bars": int(phase11["horizon_bars"]),
            "horizon_minutes": int(phase11["horizon_minutes"]),
            "x_mean_train": float(phase11["x_mean_train"]),
            "x_std_train": float(phase11["x_std_train"]),
            "train_start": phase8.get("train_start"),
            "train_end": phase8.get("train_end"),
            "train_size": int(phase8["train_size"]),
            "train_embedding_vectors": int(phase8["train_embedding_vectors"]),
            "effective_history_bars": effective_history_bars,
            "minimum_useful_log_rv_values": effective_history_bars + 1,
            "recommended_useful_log_rv_values": 750,
            "source_phase": "phase8_state_space_reconstruction + phase11_local_state_space_prediction",
            "source_artifacts": [
                "reports/tables/phase8_selected_embedding_params.json",
                "reports/tables/phase11_prediction_summary.json",
            ],
        }
        if params["tau_selected"] != 137 or params["m_selected"] != 5:
            raise ValueError("Unexpected embedding parameters; expected tau=137 and m=5")
        destination = self.artifacts_dir / "embedding_params.json"
        write_json(destination, params)

    def export_knn_reference(self) -> None:
        source = self.btc_path / "data" / "processed" / "phase8_embedding_train.npz"
        destination = self.artifacts_dir / "knn_reference_train.npz"
        require_file(source)

        inspected = inspect_npz(source)
        if {"vectors", "targets"}.issubset(inspected["keys"]):
            shutil.copy2(source, destination)
            validate_knn_npz(destination, expected_dim=5)
        else:
            print("Source Phase 8 NPZ lacks vectors/targets keys required by MVP; regenerating compatible NPZ")
            self.regenerate_knn_reference(source, destination)

    def regenerate_knn_reference(self, source_npz: Path, destination: Path) -> None:
        phase11 = self.phase11_summary()
        tau = int(phase11["tau"])
        dim = int(phase11["m"])
        x_mean = float(phase11["x_mean_train"])
        x_std = float(phase11["x_std_train"])
        data = read_prediction_data(self.btc_path / "data" / "processed" / "btc_5m_features.csv")
        train_end = train_end_index(data["times"], TRAIN_END)
        z_values = [(value - x_mean) / x_std for value in data["x"]]
        train_indices = list(range(train_end))
        vectors, indices, times, targets = build_embedding_with_targets(
            z_values,
            data["y"],
            data["times"],
            train_indices,
            tau,
            dim,
        )
        write_knn_npz(destination, vectors, targets, indices, times, metadata={
            "source": "rebuilt from data/processed/btc_5m_features.csv",
            "source_phase8_npz": "data/processed/phase8_embedding_train.npz",
            "embedding_convention": phase11["phase8_reference"]["embedding_convention"],
            "tau": tau,
            "m": dim,
            "target": "log_rv_future_12",
            "train_end": TRAIN_END,
        })
        validate_knn_npz(destination, expected_dim=dim)

        if "X" in inspect_npz(source_npz)["keys"]:
            source_x, source_shape = read_npz_numeric_array(source_npz, "X")
            source_indices, _ = read_npz_numeric_array(source_npz, "indices")
            if source_shape != (len(vectors), dim):
                raise ValueError(f"Rebuilt vectors {len(vectors), dim} do not match source X shape {source_shape}")
            max_abs_diff = 0.0
            for rebuilt_row, source_row in zip(vectors, source_x):
                for left, right in zip(rebuilt_row, source_row):
                    max_abs_diff = max(max_abs_diff, abs(left - right))
            if [int(value) for value in source_indices] != indices:
                raise ValueError("Rebuilt embedding indices do not match Phase 8 source indices")
            if max_abs_diff > 1e-12:
                raise ValueError(f"Rebuilt embedding differs from Phase 8 X, max_abs_diff={max_abs_diff}")

    def export_historical_metrics(self) -> None:
        summary = self.phase11_summary()
        test_rows = summary["test_metrics"]
        selected_k = int(summary["selected_k_by_validation_rmse"])
        persistence = find_metric_row(test_rows, "persistence")
        ar49 = find_metric_row(test_rows, "ar49_horizon12")
        knn = find_metric_row(test_rows, f"knn_mean_k{selected_k}")
        phase14_rows = read_csv_rows(self.btc_path / "reports" / "tables" / "phase14_test_metrics.csv")
        har = find_metric_row_by_split(phase14_rows, "har_logrv_compact", "test_knn_comparable_sample")
        knn_k200 = find_metric_row_by_split(phase14_rows, "knn_tau137_m5_k200", "test_knn_comparable_sample")
        metrics = {
            "horizon_bars": int(summary["horizon_bars"]),
            "horizon_minutes": int(summary["horizon_minutes"]),
            "target": "log_rv_future_12",
            "models": {
                "persistence": metric_payload(persistence),
                "ar49": metric_payload(ar49),
                "har_logrv_global": metric_payload(har),
                "knn_k200": metric_payload(knn_k200),
                f"knn_k{selected_k}": metric_payload(knn),
            },
            "recommended_model": "har_logrv_global",
            "source_phase": "phase14_har_logrv_mvp",
            "source_artifacts": [
                "reports/tables/phase14_prediction_summary.json",
                "reports/tables/phase14_test_metrics.csv",
                "reports/tables/phase11_prediction_summary.json",
                "reports/tables/phase11_test_metrics.csv",
                "reports/tables/phase12_test_metrics.csv",
            ],
        }
        destination = self.artifacts_dir / "historical_metrics.json"
        write_json(destination, metrics)

    def export_har_logrv_artifact(self) -> None:
        source = self.btc_path / "data" / "model_artifacts" / "har_logrv_model.json"
        if not source.exists():
            source = self.btc_path / "reports" / "tables" / "phase14_har_model_artifact.json"
        destination = self.artifacts_dir / "har_logrv_model.json"
        require_file(source)
        artifact = read_json(source)
        validate_har_artifact(artifact)
        shutil.copy2(source, destination)

    def export_sample_data(self) -> None:
        source = self.btc_path / "data" / "processed" / "btc_5m_features.csv"
        require_file(source)
        destination = self.sample_dir / "btcusdt_5m_recent_sample.csv"
        last_rows: deque[dict[str, str]] = deque(maxlen=self.sample_rows)
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"open_time", "close"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing columns in feature CSV: {sorted(missing)}")
            for row in reader:
                close = float(row["close"])
                if close <= 0.0 or not math.isfinite(close):
                    continue
                last_rows.append({"timestamp": row["open_time"], "close": format(close, ".12g")})
        if len(last_rows) < self.sample_rows:
            raise ValueError(f"Only {len(last_rows)} valid sample rows available")
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["timestamp", "close"])
            writer.writeheader()
            writer.writerows(last_rows)

    def export_figures(self) -> None:
        figures = {
            "phase4_log_rv_past_12_acf_288.svg": "phase4_volatility_acf.svg",
            "phase8_embedding_2d.svg": "phase8_embedding_2d.svg",
            "phase11_test_real_vs_predicted.svg": "phase11_test_real_vs_predicted.svg",
            "phase12_test_metrics_comparison.svg": "phase12_comparison_models.svg",
            "phase13_prediction_metrics.svg": "phase13_prediction_metrics.svg",
        }
        figures_dir = self.btc_path / "reports" / "figures"
        for source_name, destination_name in figures.items():
            source = figures_dir / source_name
            destination = self.assets_dir / destination_name
            if source.exists():
                shutil.copy2(source, destination)

    def run_historical_validation(self) -> None:
        print("Running historical validation")
        data = read_prediction_data(self.btc_path / "data" / "processed" / "btc_5m_features.csv")
        summary = self.phase11_summary()
        x_mean = float(summary["x_mean_train"])
        x_std = float(summary["x_std_train"])
        z_values = [(value - x_mean) / x_std for value in data["x"]]
        mean_y_train = float(summary["mean_y_train_known"])
        coeffs = read_ar_coefficients(self.artifacts_dir / "ar49_coefficients.csv")

        prediction_sample = self.btc_path / "reports" / "tables" / "phase11_predictions_test_sample.csv"
        query_indices = read_query_indices(prediction_sample)
        y_true = [data["y"][index] for index in query_indices]
        ar_pred = ar_recursive_forecast(z_values, query_indices, coeffs, HORIZON_BARS, x_mean, x_std)
        ar_metrics = evaluate_metrics(y_true, ar_pred, mean_y_train)
        expected_ar = find_metric_row(summary["test_metrics"], "ar49_horizon12")
        compare_metrics_or_raise("AR(49) Phase 11 test", ar_metrics, expected_ar, tolerance=1e-9)

        if self.skip_deep_knn_validation:
            return

        print("Running exact kNN validation against Phase 11 validation split (this can take a little while)")
        params = read_json(self.artifacts_dir / "embedding_params.json")
        tau = int(params["tau_selected"])
        dim = int(params["m_selected"])
        selected_k = int(params["selected_k_by_validation_rmse"])
        theiler = int(params["theiler_window"])
        vectors, _ = read_npz_numeric_array(self.artifacts_dir / "knn_reference_train.npz", "vectors")
        targets, _ = read_npz_numeric_array(self.artifacts_dir / "knn_reference_train.npz", "targets")
        indices, _ = read_npz_numeric_array(self.artifacts_dir / "knn_reference_train.npz", "indices")
        train_vectors = [tuple(row) for row in vectors]
        train_targets = [float(value) for value in targets]
        train_indices = [int(value) for value in indices]

        validation_indices = [
            index for index, time in enumerate(data["times"])
            if TRAIN_END < time <= VALIDATION_END
        ]
        val_vectors, val_indices, _val_times, val_targets = build_embedding_with_targets(
            z_values,
            data["y"],
            data["times"],
            validation_indices,
            tau,
            dim,
        )
        positions = sample_positions(len(val_vectors), int(summary["eval_sample_size_requested"]))
        tree = KDTree(train_vectors)
        predictions: list[float] = []
        selected_targets: list[float] = []
        for counter, position in enumerate(positions, start=1):
            neighbors = tree.query(
                tuple(val_vectors[position]),
                selected_k,
                val_indices[position],
                train_indices,
                theiler,
                HORIZON_BARS,
            )
            if not neighbors:
                raise ValueError("No eligible kNN neighbors found during historical validation")
            neighbor_targets = [train_targets[candidate_position] for _distance, candidate_position in neighbors]
            predictions.append(sum(neighbor_targets[:selected_k]) / min(selected_k, len(neighbor_targets)))
            selected_targets.append(val_targets[position])
            if counter % 1000 == 0 or counter == len(positions):
                print(f"kNN historical validation: {counter}/{len(positions)}")

        knn_metrics = evaluate_metrics(selected_targets, predictions, mean_y_train)
        expected_row = next(
            row for row in summary["validation_k_selection"]
            if int(row["k"]) == selected_k
        )
        compare_metrics_or_raise("kNN Phase 11 validation", knn_metrics, expected_row, tolerance=1e-9)


    def phase11_summary(self) -> dict[str, Any]:
        if self._phase11_summary is None:
            self._phase11_summary = read_json(self.btc_path / "reports" / "tables" / "phase11_prediction_summary.json")
        return self._phase11_summary

    def phase8_params(self) -> dict[str, Any]:
        if self._phase8_params is None:
            self._phase8_params = read_json(self.btc_path / "reports" / "tables" / "phase8_selected_embedding_params.json")
        return self._phase8_params

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def read_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    require_file(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=True) + "\n", encoding="utf-8")


def validate_har_artifact(artifact: dict[str, Any]) -> None:
    expected_features = ["log_rv_past_12", "log_rv_past_48", "log_rv_past_288"]
    if artifact.get("model_name") != "har_logrv_compact":
        raise ValueError("HAR artifact model_name must be har_logrv_compact")
    if artifact.get("target") != "log_rv_future_12":
        raise ValueError("HAR artifact target must be log_rv_future_12")
    if artifact.get("features") != expected_features:
        raise ValueError("HAR artifact features do not match compact HAR features")
    if int(artifact.get("horizon_bars", HORIZON_BARS)) != HORIZON_BARS:
        raise ValueError("HAR artifact horizon_bars must be 12")
    finite_float(artifact.get("intercept"), "HAR intercept")
    coefficients = artifact.get("coefficients")
    if not isinstance(coefficients, dict):
        raise ValueError("HAR artifact coefficients must be an object")
    for feature in expected_features:
        finite_float(coefficients.get(feature), f"HAR coefficient {feature}")


def finite_float(value: Any, label: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{label} must be finite")
    return scalar


def read_ar_coefficients(path: Path) -> list[float]:
    coeffs_by_lag: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"lag", "coefficient"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing AR coefficient columns: {sorted(missing)}")
        for row in reader:
            lag = int(row["lag"])
            coeff = float(row["coefficient"])
            if not math.isfinite(coeff):
                raise ValueError(f"Non-finite AR coefficient at lag {lag}")
            coeffs_by_lag[lag] = coeff
    if not coeffs_by_lag:
        raise ValueError("No AR coefficients found")
    order = max(coeffs_by_lag)
    return [coeffs_by_lag[lag] for lag in range(1, order + 1)]


def read_prediction_data(path: Path) -> dict[str, list[Any]]:
    require_file(path)
    times: list[str] = []
    closes: list[float] = []
    x_values: list[float] = []
    y_values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"open_time", "close", "log_rv_past_12", "log_rv_future_12"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        for row in reader:
            close = float(row["close"])
            x_value = float(row["log_rv_past_12"])
            y_value = float(row["log_rv_future_12"])
            if all(math.isfinite(value) for value in [close, x_value, y_value]):
                times.append(row["open_time"])
                closes.append(close)
                x_values.append(x_value)
                y_values.append(y_value)
    return {"times": times, "close": closes, "x": x_values, "y": y_values}


def train_end_index(times: list[str], train_end: str) -> int:
    index = 0
    while index < len(times) and times[index] <= train_end:
        index += 1
    if index == 0:
        raise ValueError("No training observations found")
    return index


def build_embedding_with_targets(
    z_values: list[float],
    y_values: list[float],
    times: list[str],
    indices: Iterable[int],
    tau: int,
    dim: int,
) -> tuple[list[tuple[float, ...]], list[int], list[str], list[float]]:
    vectors: list[tuple[float, ...]] = []
    vector_indices: list[int] = []
    vector_times: list[str] = []
    targets: list[float] = []
    min_index = (dim - 1) * tau
    for index in indices:
        if index < min_index:
            continue
        vector = tuple(z_values[index - coord * tau] for coord in range(dim))
        target = y_values[index]
        if all(math.isfinite(value) for value in vector) and math.isfinite(target):
            vectors.append(vector)
            vector_indices.append(index)
            vector_times.append(times[index])
            targets.append(target)
    return vectors, vector_indices, vector_times, targets


def write_knn_npz(
    path: Path,
    vectors: list[tuple[float, ...]],
    targets: list[float],
    indices: list[int],
    times: list[str],
    metadata: dict[str, Any],
) -> None:
    if not vectors:
        raise ValueError("Cannot write empty kNN reference")
    if len(vectors) != len(targets) or len(vectors) != len(indices) or len(vectors) != len(times):
        raise ValueError("kNN vectors/targets/indices/times length mismatch")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_npy_float64_2d(archive, "vectors.npy", vectors)
        write_npy_float64_1d(archive, "targets.npy", targets)
        write_npy_int64_1d(archive, "indices.npy", indices)
        write_npy_string_1d(archive, "times.npy", times, width=19)
        archive.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=True))


def write_npy_float64_2d(archive: zipfile.ZipFile, name: str, data: list[tuple[float, ...]]) -> None:
    rows = len(data)
    cols = len(data[0]) if rows else 0
    with archive.open(name, "w") as handle:
        handle.write(npy_header("<f8", (rows, cols)))
        for row in data:
            handle.write(struct.pack("<" + "d" * cols, *row))


def write_npy_float64_1d(archive: zipfile.ZipFile, name: str, data: list[float]) -> None:
    with archive.open(name, "w") as handle:
        handle.write(npy_header("<f8", (len(data),)))
        for chunk in chunks(data, 4096):
            handle.write(struct.pack("<" + "d" * len(chunk), *chunk))


def write_npy_int64_1d(archive: zipfile.ZipFile, name: str, data: list[int]) -> None:
    with archive.open(name, "w") as handle:
        handle.write(npy_header("<i8", (len(data),)))
        for chunk in chunks(data, 4096):
            handle.write(struct.pack("<" + "q" * len(chunk), *chunk))


def write_npy_string_1d(archive: zipfile.ZipFile, name: str, data: list[str], width: int) -> None:
    with archive.open(name, "w") as handle:
        handle.write(npy_header(f"|S{width}", (len(data),)))
        for value in data:
            encoded = value.encode("ascii")[:width]
            handle.write(encoded.ljust(width, b"\x00"))


def npy_header(descr: str, shape: tuple[int, ...]) -> bytes:
    shape_text = "(" + ", ".join(str(item) for item in shape)
    if len(shape) == 1:
        shape_text += ","
    shape_text += ")"
    header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape_text + ", }"
    header_bytes = header.encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def inspect_npz(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    keys = {name[:-4] for name in names if name.endswith(".npy")}
    return {"names": names, "keys": keys}


def read_npz_numeric_array(path: Path, key: str) -> tuple[list[Any], tuple[int, ...]]:
    with zipfile.ZipFile(path) as archive:
        name = f"{key}.npy"
        if name not in archive.namelist():
            raise KeyError(f"{key} not found in {path}")
        with archive.open(name) as handle:
            descr, shape = parse_npy_header(handle)
            count = product(shape)
            raw = handle.read()
    if descr == "<f8":
        values = [item[0] for item in struct.iter_unpack("<d", raw[: count * 8])]
    elif descr == "<i8":
        values = [item[0] for item in struct.iter_unpack("<q", raw[: count * 8])]
    else:
        raise ValueError(f"Unsupported numeric dtype {descr} in {key}")
    if len(shape) == 1:
        return values, shape
    if len(shape) == 2:
        rows, cols = shape
        return [tuple(values[start:start + cols]) for start in range(0, rows * cols, cols)], shape
    raise ValueError(f"Unsupported shape for {key}: {shape}")


def parse_npy_header(handle: Any) -> tuple[str, tuple[int, ...]]:
    magic = handle.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("Invalid NPY magic")
    major = handle.read(1)[0]
    minor = handle.read(1)[0]
    if (major, minor) != (1, 0):
        raise ValueError(f"Unsupported NPY version {(major, minor)}")
    header_len = int.from_bytes(handle.read(2), "little")
    header = ast.literal_eval(handle.read(header_len).decode("latin1").strip())
    if header.get("fortran_order"):
        raise ValueError("Fortran-ordered NPY arrays are not supported")
    return str(header["descr"]), tuple(header["shape"])


def product(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def validate_knn_npz(path: Path, expected_dim: int) -> None:
    inspected = inspect_npz(path)
    required = {"vectors", "targets"}
    missing = required - inspected["keys"]
    if missing:
        raise ValueError(f"kNN NPZ missing keys: {sorted(missing)}")
    vectors, vector_shape = read_npz_numeric_array(path, "vectors")
    targets, target_shape = read_npz_numeric_array(path, "targets")
    if len(vector_shape) != 2:
        raise ValueError(f"vectors must be 2D, got {vector_shape}")
    if vector_shape[1] != expected_dim:
        raise ValueError(f"vectors second dimension must be {expected_dim}, got {vector_shape[1]}")
    if target_shape != (vector_shape[0],):
        raise ValueError(f"targets shape {target_shape} does not match vectors rows {vector_shape[0]}")
    for row in vectors:
        for value in row:
            if not math.isfinite(float(value)):
                raise ValueError("Non-finite value in vectors")
    for value in targets:
        if not math.isfinite(float(value)):
            raise ValueError("Non-finite value in targets")


def read_query_indices(path: Path) -> list[int]:
    require_file(path)
    indices: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "index" not in (reader.fieldnames or []):
            raise ValueError("Prediction sample lacks index column")
        for row in reader:
            indices.append(int(row["index"]))
    return indices


def ar_recursive_forecast(
    z_values: list[float],
    query_indices: list[int],
    coefficients: list[float],
    horizon: int,
    x_mean_train: float,
    x_std_train: float,
) -> list[float]:
    predictions: list[float] = []
    for query_index in query_indices:
        future: dict[int, float] = {}
        for step in range(1, horizon + 1):
            target_index = query_index + step
            forecast = 0.0
            for lag, coefficient in enumerate(coefficients, start=1):
                source_index = target_index - lag
                source_value = z_values[source_index] if source_index <= query_index else future[source_index]
                forecast += coefficient * source_value
            future[target_index] = forecast
        predictions.append(future[query_index + horizon] * x_std_train + x_mean_train)
    return predictions


def evaluate_metrics(y_true: list[float], y_pred: list[float], mean_y_train: float) -> dict[str, float]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Invalid y_true/y_pred for metrics")
    errors = [pred - true for true, pred in zip(y_true, y_pred)]
    abs_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    mse = sum(squared_errors) / len(squared_errors)
    denominator = sum((true - mean_y_train) ** 2 for true in y_true)
    return {
        "n": float(len(y_true)),
        "mae": sum(abs_errors) / len(abs_errors),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2_oos": 1.0 - sum(squared_errors) / denominator if denominator > 0.0 else float("nan"),
        "bias_yhat_minus_y": sum(errors) / len(errors),
        "error_std": sample_std(errors),
    }


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def compare_metrics_or_raise(label: str, computed: dict[str, float], expected: dict[str, Any], tolerance: float) -> None:
    key_map = {"mae": "mae", "mse": "mse", "rmse": "rmse", "r2_oos": "r2_oos"}
    differences = {}
    for computed_key, expected_key in key_map.items():
        diff = abs(float(computed[computed_key]) - float(expected[expected_key]))
        differences[computed_key] = diff
        if diff > tolerance:
            raise ValueError(
                f"{label} does not reproduce historical {computed_key}: "
                f"computed={computed[computed_key]}, expected={expected[expected_key]}, diff={diff}"
            )


def find_metric_row(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    return next(row for row in rows if row["model"] == model)


def find_metric_row_by_split(rows: list[dict[str, Any]], model: str, split: str) -> dict[str, Any]:
    return next(row for row in rows if row["model"] == model and row.get("split") == split)


def metric_payload(row: dict[str, Any]) -> dict[str, float]:
    return {
        "test_rmse": float(row["rmse"]),
        "test_mae": float(row["mae"]),
        "test_r2": float(row["r2_oos"]),
    }


def sample_positions(length: int, sample_size: int) -> list[int]:
    if length <= 0:
        return []
    if length <= sample_size:
        return list(range(length))
    if sample_size <= 1:
        return [0]
    positions: list[int] = []
    previous = -1
    for output_index in range(sample_size):
        position = round((length - 1) * output_index / (sample_size - 1))
        if position != previous:
            positions.append(position)
            previous = position
    return positions


def distance_sq_bounded(left: tuple[float, ...], right: tuple[float, ...], limit: float) -> float:
    distance = 0.0
    for left_value, right_value in zip(left, right):
        diff = left_value - right_value
        distance += diff * diff
        if distance >= limit:
            break
    return distance


def build_parser() -> argparse.ArgumentParser:
    script_path = Path(__file__).resolve()
    default_mvp_path = script_path.parents[1]
    default_btc_path = default_mvp_path.parent / "btc-volatility"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-volatility-path", type=Path, default=default_btc_path)
    parser.add_argument("--mvp-web-path", type=Path, default=default_mvp_path)
    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    parser.add_argument(
        "--skip-deep-knn-validation",
        action="store_true",
        help="Skip exact Phase 11 kNN validation, which is slower than structural validation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    exporter = ArtifactExporter(
        btc_volatility_path=args.btc_volatility_path,
        mvp_web_path=args.mvp_web_path,
        sample_rows=args.sample_rows,
        skip_deep_knn_validation=args.skip_deep_knn_validation,
    )
    exporter.run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
