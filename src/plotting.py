"""Plotting helpers for the MVP Streamlit app."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go


MODEL_LABELS = {
    "persistence": "Persistencia",
    "knn": "kNN local",
    "ar49": "AR(49)",
    "har_global": "HAR-logRV global",
    "har_mini": "Mini-HAR local",
}

MODEL_COLORS = {
    "har_global": "#d62728",
    "har_mini": "#9467bd",
    "persistence": "#6c757d",
    "ar49": "#1f77b4",
    "knn": "#2ca02c",
}


def model_display_name(model_name: str) -> str:
    """Return a user-facing model name."""
    return MODEL_LABELS.get(model_name, model_name)


def plot_recent_volatility_with_predictions(
    feature_df: pd.DataFrame,
    predictions: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    recent_points: int = 250,
) -> go.Figure:
    """Plot recent log_rv_past_12 and horizontal forecast lines in UTC."""
    required_columns = {"timestamp", "log_rv_past_12"}
    missing = required_columns - set(feature_df.columns)
    if missing:
        raise ValueError(f"feature_df missing columns: {sorted(missing)}")
    if feature_df.empty:
        raise ValueError("feature_df must not be empty")
    if recent_points <= 0:
        raise ValueError("recent_points must be positive")

    recent = feature_df.loc[:, ["timestamp", "log_rv_past_12"]].tail(recent_points).copy()
    recent["timestamp"] = pd.to_datetime(recent["timestamp"], utc=True)

    horizon_start = pd.to_datetime(metadata["horizon_start"], utc=True)
    horizon_end = pd.to_datetime(metadata["horizon_end"], utc=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["timestamp"],
            y=recent["log_rv_past_12"],
            mode="lines",
            name="log_rv_past_12 reciente",
            line={"color": "#111827", "width": 2},
        )
    )

    for model_name in ("persistence", "knn", "ar49", "har_global"):
        result = predictions.get(model_name, {})
        if result.get("status") != "ok" or result.get("log_rv_future_12") is None:
            continue
        y_value = float(result["log_rv_future_12"])
        fig.add_trace(
            go.Scatter(
                x=[horizon_start, horizon_end],
                y=[y_value, y_value],
                mode="lines+markers",
                name=MODEL_LABELS[model_name],
                line={
                    "color": MODEL_COLORS[model_name],
                    "width": 3,
                    "dash": "dash",
                },
                marker={"size": 7},
            )
        )

    fig.update_layout(
        title="Volatilidad realizada reciente y predicciones a 1 hora",
        xaxis_title="Tiempo UTC",
        yaxis_title="log_rv_past_12 / log_rv_future_12",
        legend_title="Serie / modelo",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="x unified",
    )
    return fig


def plot_walk_forward_model(
    model_name: str,
    results_df: pd.DataFrame,
    recent_points: int = 300,
) -> go.Figure:
    """Plot observed vs predicted recent walk-forward values for one model."""
    display_columns = ["horizon_end", "y_true_log_rv_future_12", "y_pred_log_rv_future_12"]
    required_columns = set(display_columns)
    missing = required_columns - set(results_df.columns)
    if missing:
        raise ValueError(f"results_df missing columns: {sorted(missing)}")
    if recent_points <= 0:
        raise ValueError("recent_points must be positive")

    recent = results_df.loc[:, display_columns].tail(recent_points).copy()
    recent["horizon_end"] = pd.to_datetime(recent["horizon_end"], utc=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["horizon_end"],
            y=recent["y_true_log_rv_future_12"],
            mode="lines",
            name="Real observado",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent["horizon_end"],
            y=recent["y_pred_log_rv_future_12"],
            mode="lines",
            name="Prediccion",
        )
    )
    fig.update_layout(
        title=f"Evaluacion reciente - {model_display_name(model_name)}",
        xaxis_title="Tiempo UTC",
        yaxis_title="log_rv_future_12",
        legend_title="Serie",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        hovermode="x unified",
    )
    return fig
