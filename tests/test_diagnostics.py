from __future__ import annotations

import pandas as pd

from src.data_validation import ValidationResult
from src.diagnostics import compute_simple_quality_score


def feature_df(rows: int = 800) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-04", periods=rows, freq="5min", tz="UTC"),
            "log_rv_past_12": [-12.0] * rows,
            "log_rv_past_48": [-11.5] * rows,
            "log_rv_past_288": [-11.0] * rows,
        }
    )


def availability(
    knn_available: bool = True,
    knn_recommended: bool = True,
    har_global_available: bool = True,
    har_mini_available: bool = True,
    har_mini_recommended: bool = True,
) -> dict[str, dict[str, object]]:
    return {
        "har_global": {"available": har_global_available},
        "har_mini": {"available": har_mini_available, "recommended": har_mini_recommended},
        "persistence": {"available": True},
        "ar49": {"available": True},
        "knn": {"available": knn_available, "recommended": knn_recommended},
    }


def valid_result(n_gaps: int = 0, gap_percentage: float = 0.0) -> ValidationResult:
    return ValidationResult(
        valid=True,
        errors=[],
        warnings=[],
        info={
            "frequency": {
                "n_gaps": n_gaps,
                "gap_percentage": gap_percentage,
                "is_expected_5m": True,
            }
        },
    )


def test_quality_score_green() -> None:
    quality = compute_simple_quality_score(valid_result(), availability(), feature_df())
    assert quality["status"] == "Fiable"
    assert quality["score"] == 100


def test_quality_score_yellow_when_knn_unavailable() -> None:
    quality = compute_simple_quality_score(valid_result(), availability(knn_available=False), feature_df())
    assert quality["status"] == "Fiabilidad limitada"
    assert quality["score"] < 100


def test_quality_score_does_not_penalize_mini_har_recommendation() -> None:
    quality = compute_simple_quality_score(
        valid_result(),
        availability(har_mini_recommended=False),
        feature_df(),
    )
    assert quality["status"] == "Fiable"
    assert quality["score"] == 100


def test_quality_score_yellow_when_many_gaps() -> None:
    quality = compute_simple_quality_score(valid_result(n_gaps=5, gap_percentage=2.0), availability(), feature_df())
    assert quality["status"] == "Fiabilidad limitada"
    assert any("Gaps" in message for message in quality["messages"])


def test_quality_score_red_on_validation_error() -> None:
    validation = ValidationResult(valid=False, errors=["Faltan columnas"], warnings=[], info={})
    quality = compute_simple_quality_score(validation, availability(), feature_df())
    assert quality["status"] == "No fiable"
    assert quality["score"] == 20
    assert quality["messages"] == ["Faltan columnas"]
