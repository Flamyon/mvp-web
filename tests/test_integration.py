from __future__ import annotations

import math

import plotly.graph_objects as go

from src.config import ARTIFACTS_DIR, SAMPLE_DIR, SAMPLE_FILENAME
from src.data_loader import load_sample_data
from src.data_validation import validate_price_data
from src.diagnostics import compute_simple_quality_score
from src.feature_engineering import add_future_evaluation_target, check_model_availability, engineer_features
from src.model_registry import ModelRegistry, load_preprocessing_params, run_available_predictions
from src.plotting import plot_recent_volatility_with_predictions, plot_walk_forward_model
from src.walk_forward import run_walk_forward_predictions, summarize_walk_forward_results


def test_sample_end_to_end_pipeline() -> None:
    params = load_preprocessing_params(ARTIFACTS_DIR / "preprocessing_params.json")
    price_df = load_sample_data(SAMPLE_DIR / SAMPLE_FILENAME)
    validation = validate_price_data(price_df)
    assert validation.valid

    feature_df = engineer_features(
        price_df,
        x_mean_train=params["x_mean_train"],
        x_std_train=params["x_std_train"],
        epsilon=params["epsilon_for_log_rv"],
        rv_window=params["horizon_bars"],
    )
    availability = check_model_availability(feature_df)
    models = ModelRegistry(ARTIFACTS_DIR).load_all_models()
    payload = run_available_predictions(feature_df, models, availability)
    quality = compute_simple_quality_score(validation, availability, feature_df)
    figure = plot_recent_volatility_with_predictions(
        feature_df,
        payload["predictions"],
        payload["metadata"],
    )
    evaluated = add_future_evaluation_target(feature_df)
    walk_forward = run_walk_forward_predictions(evaluated, models, max_points_per_model=10)
    walk_forward_summary = summarize_walk_forward_results(walk_forward)
    ar49_walk_forward_figure = plot_walk_forward_model("ar49", walk_forward["ar49"])

    assert quality["status"] == "Fiable"
    assert isinstance(figure, go.Figure)
    assert len(figure.data) >= 5
    assert isinstance(ar49_walk_forward_figure, go.Figure)
    assert not walk_forward["har_global"].empty
    assert not walk_forward["har_mini"].empty
    assert not walk_forward["persistence"].empty
    assert not walk_forward["ar49"].empty
    assert not walk_forward["knn"].empty
    assert set(walk_forward_summary["model"]) == {"har_global", "har_mini", "persistence", "ar49", "knn"}

    for model_name in ("har_global", "har_mini", "persistence", "ar49", "knn"):
        result = payload["predictions"][model_name]
        assert result["status"] == "ok"
        assert math.isfinite(result["log_rv_future_12"])
        assert math.isfinite(result["rv_future_12"])
        assert math.isfinite(result["sqrt_rv_percent"])
