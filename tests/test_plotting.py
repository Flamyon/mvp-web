from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.plotting import plot_recent_volatility_with_predictions, plot_walk_forward_model


def test_plot_recent_volatility_with_predictions_returns_plotly_figure() -> None:
    feature_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-04", periods=20, freq="5min", tz="UTC"),
            "log_rv_past_12": [-12.0 + index * 0.01 for index in range(20)],
        }
    )
    metadata = {
        "horizon_start": pd.Timestamp("2026-06-04 01:35:00", tz="UTC"),
        "horizon_end": pd.Timestamp("2026-06-04 02:35:00", tz="UTC"),
    }
    predictions = {
        "har_global": {"status": "ok", "log_rv_future_12": -11.6},
        "har_mini": {"status": "ok", "log_rv_future_12": -11.65},
        "persistence": {"status": "ok", "log_rv_future_12": -11.9},
        "ar49": {"status": "ok", "log_rv_future_12": -11.8},
        "knn": {"status": "ok", "log_rv_future_12": -11.7},
    }

    figure = plot_recent_volatility_with_predictions(feature_df, predictions, metadata)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 5
    assert figure.layout.xaxis.title.text == "Tiempo UTC"


def test_plot_walk_forward_model_returns_plotly_figure() -> None:
    results_df = pd.DataFrame(
        {
            "horizon_end": pd.date_range("2026-06-04 01:00:00", periods=10, freq="5min", tz="UTC"),
            "y_true_log_rv_future_12": [-12.0 + index * 0.01 for index in range(10)],
            "y_pred_log_rv_future_12": [-11.9 + index * 0.01 for index in range(10)],
        }
    )

    figure = plot_walk_forward_model("ar49", results_df)

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 2
    assert figure.layout.yaxis.title.text == "log_rv_future_12"
