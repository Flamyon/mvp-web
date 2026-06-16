from __future__ import annotations

import pandas as pd
import streamlit as st

from src.binance_client import fetch_and_normalize_recent_btcusdt
from src.config import ARTIFACTS_DIR, CACHE_TTL_SECONDS, INTERVAL, SAMPLE_DIR, SAMPLE_FILENAME, SYMBOL
from src.data_loader import load_sample_data, load_user_csv
from src.data_validation import summarize_price_data, validate_price_data
from src.feature_engineering import (
    add_future_evaluation_target,
    check_model_availability,
    engineer_features,
    load_preprocessing_params,
)
from src.model_registry import ModelRegistry, run_available_predictions
from src.plotting import plot_recent_volatility_with_predictions, plot_walk_forward_model
from src.ui_components import render_disclaimer, render_header
from src.walk_forward import (
    combine_walk_forward_results,
    run_walk_forward_predictions,
    summarize_walk_forward_results,
)


render_header(
    "Prediccion de volatilidad",
    "Carga datos, construye features runtime y genera predicciones a 1 hora con modelos locales.",
)


MODEL_LABELS = {
    "persistence": "Persistencia",
    "knn": "kNN local",
    "ar49": "AR(49)",
    "har_global": "HAR-logRV global",
    "har_mini": "Mini-HAR local",
}

MODEL_DISPLAY_ORDER = ("persistence", "knn", "ar49", "har_global", "har_mini")
PREDICTION_MODEL_ORDER = MODEL_DISPLAY_ORDER
WALK_FORWARD_MODEL_ORDER = MODEL_DISPLAY_ORDER

STATUS_LABELS = {
    "ok": "OK",
    "unavailable": "No disponible",
    "error": "Error",
}


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_fetch_binance() -> pd.DataFrame:
    """Fetch Binance data with Streamlit-side cache only."""
    return fetch_and_normalize_recent_btcusdt(symbol=SYMBOL, interval=INTERVAL)


@st.cache_resource(show_spinner=False)
def cached_load_models(k_knn: int = 200) -> dict[str, object]:
    """Load local model artifacts with Streamlit resource caching."""
    return ModelRegistry(ARTIFACTS_DIR).load_all_models(k_knn=k_knn)


def store_and_show_data(df: pd.DataFrame, source_label: str) -> None:
    """Store normalized price data, validate it, and build runtime features."""
    st.session_state.pop("feature_data", None)
    st.session_state.pop("model_availability", None)
    st.session_state.pop("predictions", None)
    st.session_state.pop("prediction_metadata", None)
    st.session_state.pop("walk_forward_results", None)
    st.session_state.pop("walk_forward_summary", None)
    st.session_state["price_data"] = df
    st.session_state["price_source_label"] = source_label
    st.success(
        "Datos cargados correctamente. La construccion de volatilidad realizada "
        "ya puede evaluarse y usarse para prediccion."
    )
    show_current_data()


def show_current_data() -> None:
    """Render the currently loaded price data from session state."""
    df = st.session_state["price_data"]
    source_label = st.session_state.get("price_source_label", "Datos cargados")
    st.info("Todos los timestamps se muestran en UTC. No se convierten a hora local de España.")

    st.subheader("Resumen de datos")
    left, middle, right = st.columns(3)
    left.metric("Fuente", source_label)
    middle.metric("Filas", f"{len(df):,}")
    right.metric("Ultimo timestamp", str(df["timestamp"].iloc[-1]))

    summary = summarize_price_data(df)
    st.session_state["price_summary"] = summary
    st.write(f"Rango temporal: `{summary['timestamp_start']}` a `{summary['timestamp_end']}`")

    st.write("Primeras filas")
    st.dataframe(df.head(), hide_index=True, width="stretch")
    st.write("Ultimas filas")
    st.dataframe(df.tail(), hide_index=True, width="stretch")

    validation = validate_price_data(df)
    st.session_state["validation_result"] = validation
    st.subheader("Validacion")
    if validation.errors:
        for error in validation.errors:
            st.error(error)
    if validation.warnings:
        for warning in validation.warnings:
            st.warning(warning)
    if validation.valid:
        st.success("Validacion de datos superada.")
        build_and_show_features(df)
    else:
        st.info("Corrige los errores de datos antes de construir features.")


def build_and_show_features(df: pd.DataFrame) -> None:
    """Build features from valid price data and show availability checks."""
    params = load_preprocessing_params(ARTIFACTS_DIR / "preprocessing_params.json")
    feature_df = engineer_features(
        df,
        x_mean_train=params["x_mean_train"],
        x_std_train=params["x_std_train"],
        epsilon=params["epsilon_for_log_rv"],
        rv_window=params["horizon_bars"],
    )
    availability = check_model_availability(feature_df)
    st.session_state["feature_data"] = feature_df
    st.session_state["model_availability"] = availability

    st.subheader("Features runtime")
    har_valid_rows = int(
        feature_df[["log_rv_past_12", "log_rv_past_48", "log_rv_past_288"]]
        .notna()
        .all(axis=1)
        .sum()
    )
    left, middle_left, middle_right, right = st.columns(4)
    left.metric("Filas originales", f"{len(df):,}")
    middle_left.metric("Filas log_rv_12", f"{len(feature_df):,}")
    middle_right.metric("Filas HAR validas", f"{har_valid_rows:,}")
    right.metric("Ventana HAR max", "288 velas")

    latest = feature_df.iloc[-1]
    st.write(f"Ultimo `log_rv_past_12`: `{latest['log_rv_past_12']:.8f}`")
    st.write(f"Ultimo `log_rv_past_48`: `{latest['log_rv_past_48']:.8f}`")
    st.write(f"Ultimo `log_rv_past_288`: `{latest['log_rv_past_288']:.8f}`")
    st.write(f"Ultimo `z_log_rv_past_12`: `{latest['z_log_rv_past_12']:.8f}`")

    st.write("Disponibilidad de modelos futuros")
    st.dataframe(availability_table(availability), hide_index=True, width="stretch")
    st.success("Features calculadas correctamente. Ya puedes generar predicciones con los modelos locales.")
    render_prediction_controls(feature_df, availability)


def availability_table(availability: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Convert availability dict to a display table."""
    rows = []
    for model in ordered_model_names(availability):
        info = availability[model]
        rows.append(
            {
                "Modelo": MODEL_LABELS.get(model, model),
                "Disponible": "Si" if info.get("available") else "No",
                "Valores disponibles": str(info.get("available_values", "-")),
                "Requerido": str(info.get("required", "-")),
                "Recomendado": str(info.get("recommended_required", "-")),
                "Mensaje": str(info.get("message", "")),
            }
        )
    return pd.DataFrame(rows)


def render_prediction_controls(
    feature_df: pd.DataFrame,
    availability: dict[str, dict[str, object]],
) -> None:
    """Render prediction action, horizon metadata, and current prediction table."""
    st.subheader("Predicciones actuales")
    st.caption(
        "Orden visual: Persistencia, kNN, AR(49), HAR-logRV global y Mini-HAR local. "
        "HAR-logRV global es el modelo practico recomendado; AR(49) es el benchmark "
        "lineal fuerte, kNN es experimental no lineal y Mini-HAR es recalibracion local experimental."
    )
    if st.button("Generar predicciones", type="primary"):
        try:
            with st.spinner("Cargando artefactos locales y ejecutando modelos..."):
                models = cached_load_models(k_knn=200)
                payload = run_available_predictions(feature_df, models, availability)
            st.session_state["predictions"] = payload["predictions"]
            st.session_state["prediction_metadata"] = payload["metadata"]
            st.success("Predicciones calculadas correctamente.")
        except Exception as exc:
            st.error(f"No se pudieron generar predicciones: {exc}")

    predictions = st.session_state.get("predictions")
    metadata = st.session_state.get("prediction_metadata")
    if predictions and metadata:
        render_prediction_metadata(metadata)
        st.success("Modelo recomendado practico: HAR-logRV global.")
        if predictions.get("har_mini", {}).get("status") == "ok":
            st.info("Mini-HAR se ha reentrenado solo con la ventana cargada; interpretar como experimental.")
        st.dataframe(predictions_table(predictions), hide_index=True, width="stretch")
        figure = plot_recent_volatility_with_predictions(feature_df, predictions, metadata)
        st.plotly_chart(figure, width="stretch")
        st.download_button(
            "Descargar predicciones CSV",
            data=predictions_csv(predictions, metadata),
            file_name="btc_volatility_predictions.csv",
            mime="text/csv",
        )
        st.warning(
            "La columna sqrt_rv_percent es una escala aproximada de la ventana de 1 hora. "
            "No esta anualizada, no predice direccion, no demuestra caos determinista "
            "y no es una recomendacion financiera."
        )
        render_walk_forward_section(feature_df)


def render_walk_forward_section(feature_df: pd.DataFrame) -> None:
    """Render recent walk-forward evaluation controls and outputs."""
    st.subheader("Evaluacion reciente walk-forward")
    st.caption(
        "Se generan predicciones en puntos pasados de la ventana cargada y se comparan con "
        "la volatilidad realizada observada una hora despues. El target futuro solo mide error; "
        "no se usa para predecir. En Mini-HAR, cada punto recalibra el modelo solo con targets "
        "que ya estarian observados antes de ese punto."
    )
    max_points = st.selectbox(
        "Numero maximo de puntos evaluados por modelo",
        [100, 200, 300],
        index=2,
    )
    if st.button("Calcular evaluacion reciente", type="primary"):
        try:
            with st.spinner("Calculando evaluacion walk-forward reciente..."):
                models = cached_load_models(k_knn=200)
                evaluation_df = add_future_evaluation_target(feature_df)
                results = run_walk_forward_predictions(
                    evaluation_df,
                    models,
                    max_points_per_model=int(max_points),
                )
                summary = summarize_walk_forward_results(results)
            st.session_state["walk_forward_results"] = results
            st.session_state["walk_forward_summary"] = summary
            st.success("Evaluacion reciente calculada correctamente.")
        except Exception as exc:
            st.error(f"No se pudo calcular la evaluacion reciente: {exc}")

    results = st.session_state.get("walk_forward_results")
    summary = st.session_state.get("walk_forward_summary")
    if results is None or summary is None:
        return

    st.write("Metricas recientes sobre la ventana cargada")
    st.caption("No sustituyen la validacion historica formal del proyecto.")
    st.dataframe(walk_forward_summary_table(summary), hide_index=True, width="stretch")

    tabs = st.tabs([MODEL_LABELS[model_name] for model_name in WALK_FORWARD_MODEL_ORDER])
    for tab, model_name in zip(tabs, WALK_FORWARD_MODEL_ORDER):
        with tab:
            model_df = results.get(model_name, pd.DataFrame())
            if model_df.empty:
                st.info("No hay puntos evaluables para este modelo con la ventana cargada.")
                continue
            st.plotly_chart(plot_walk_forward_model(model_name, model_df), width="stretch")

    st.download_button(
        "Descargar evaluacion walk-forward CSV",
        data=walk_forward_csv(results),
        file_name="btc_volatility_walk_forward_predictions.csv",
        mime="text/csv",
    )


def render_prediction_metadata(metadata: dict[str, object]) -> None:
    """Render UTC horizon metadata for the current forecast."""
    st.write("Horizonte de prediccion en UTC")
    left, middle, right = st.columns(3)
    left.metric("Ultima vela usada", format_timestamp(metadata["timestamp_used"]))
    middle.metric("Inicio horizonte", format_timestamp(metadata["horizon_start"]))
    right.metric("Fin horizonte", format_timestamp(metadata["horizon_end"]))


def predictions_table(predictions: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Convert prediction payload to a display-safe table."""
    rows = []
    for model_name in PREDICTION_MODEL_ORDER:
        result = predictions.get(model_name, {})
        note = str(result.get("note", ""))
        metrics = result.get("local_metrics")
        if model_name == "har_mini" and isinstance(metrics, dict):
            note = (
                f"{note} | RMSE local={format_float(metrics.get('rmse'), decimals=6)}, "
                f"R2 local={format_float(metrics.get('r2_oos'), decimals=6)}, "
                f"train/test={result.get('train_n', '-')}/{result.get('test_n', '-')}"
            )
        rows.append(
            {
                "Modelo": MODEL_LABELS.get(model_name, model_name),
                "Estado": STATUS_LABELS.get(str(result.get("status", "")), str(result.get("status", ""))),
                "log_rv_future_12": format_float(result.get("log_rv_future_12"), decimals=8),
                "rv_future_12": format_float(result.get("rv_future_12"), decimals=10),
                "sqrt_rv_percent": format_float(result.get("sqrt_rv_percent"), decimals=6),
                "Nota": note,
            }
        )
    return pd.DataFrame(rows)


def predictions_csv(predictions: dict[str, dict[str, object]], metadata: dict[str, object]) -> str:
    """Build a CSV download with raw prediction values and UTC horizon."""
    rows = []
    for model_name in PREDICTION_MODEL_ORDER:
        result = predictions.get(model_name, {})
        metrics = result.get("local_metrics") if isinstance(result.get("local_metrics"), dict) else {}
        rows.append(
            {
                "model": model_name,
                "status": result.get("status"),
                "log_rv_future_12": result.get("log_rv_future_12"),
                "rv_future_12": result.get("rv_future_12"),
                "sqrt_rv_percent": result.get("sqrt_rv_percent"),
                "note": result.get("note"),
                "local_rmse": metrics.get("rmse"),
                "local_mae": metrics.get("mae"),
                "local_r2_oos": metrics.get("r2_oos"),
                "train_n": result.get("train_n"),
                "test_n": result.get("test_n"),
                "n_effective_rows": result.get("n_effective_rows"),
                "timestamp_used_utc": format_timestamp(metadata["timestamp_used"]),
                "horizon_start_utc": format_timestamp(metadata["horizon_start"]),
                "horizon_end_utc": format_timestamp(metadata["horizon_end"]),
            }
        )
    return pd.DataFrame(rows).to_csv(index=False)


def walk_forward_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Format recent walk-forward metrics for display."""
    rows = []
    summary_by_model = {
        str(row["model"]): row
        for row in summary.to_dict(orient="records")
    }
    for model_name in ordered_model_names(summary_by_model):
        row = summary_by_model[model_name]
        rows.append(
            {
                "Modelo": MODEL_LABELS.get(model_name, model_name),
                "Predicciones": int(row["n_predictions"]),
                "RMSE": format_float(row["rmse"], decimals=6),
                "MAE": format_float(row["mae"], decimals=6),
                "Bias": format_float(row["bias"], decimals=6),
                "Primer timestamp": "-" if pd.isna(row["first_timestamp"]) else format_timestamp(row["first_timestamp"]),
                "Ultimo timestamp": "-" if pd.isna(row["last_timestamp"]) else format_timestamp(row["last_timestamp"]),
            }
        )
    return pd.DataFrame(rows)


def walk_forward_csv(results: dict[str, pd.DataFrame]) -> str:
    """Build the combined walk-forward CSV download."""
    ordered_results = {
        model_name: results[model_name]
        for model_name in ordered_model_names(results)
    }
    return combine_walk_forward_results(ordered_results).to_csv(index=False)


def ordered_model_names(mapping: dict[str, object]) -> list[str]:
    """Return model keys in the page display order, followed by unknown extras."""
    ordered = [model_name for model_name in MODEL_DISPLAY_ORDER if model_name in mapping]
    ordered.extend(model_name for model_name in mapping if model_name not in MODEL_DISPLAY_ORDER)
    return ordered


def format_float(value: object, decimals: int) -> str:
    """Format optional numeric values for Streamlit tables."""
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{decimals}g}"


def format_timestamp(value: object) -> str:
    """Format timestamps as UTC strings."""
    return str(pd.to_datetime(value, utc=True))


data_rendered = False

source = st.radio(
    "Fuente de datos",
    ["Binance API", "Dataset de ejemplo", "Subir CSV"],
    horizontal=True,
)

if source == "Binance API":
    st.write(f"Descarga prevista: ultimas velas de `{SYMBOL}` con intervalo `{INTERVAL}`.")
    if st.button("Descargar ultimas velas", type="primary"):
        try:
            with st.spinner("Descargando y normalizando datos de Binance..."):
                data = cached_fetch_binance()
            store_and_show_data(data, "Binance API")
            data_rendered = True
        except Exception as exc:
            st.error(f"No se pudieron descargar datos de Binance: {exc}")

elif source == "Dataset de ejemplo":
    sample_path = SAMPLE_DIR / SAMPLE_FILENAME
    st.write(f"Dataset local: `{sample_path}`")
    if st.button("Cargar dataset de ejemplo", type="primary"):
        try:
            data = load_sample_data(sample_path)
            store_and_show_data(data, "Dataset de ejemplo")
            data_rendered = True
        except Exception as exc:
            st.error(f"No se pudo cargar el dataset de ejemplo: {exc}")

else:
    uploaded_file = st.file_uploader(
        "Sube un archivo CSV",
        type=["csv"],
        help="Formato minimo: timestamp,close. Tambien se aceptan open_time/date/datetime/time + close.",
    )
    if uploaded_file is not None and st.button("Cargar CSV subido", type="primary"):
        try:
            data = load_user_csv(uploaded_file)
            store_and_show_data(data, "CSV subido")
            data_rendered = True
        except Exception as exc:
            st.error(f"No se pudo normalizar el CSV: {exc}")

if "price_data" in st.session_state and not data_rendered:
    show_current_data()

render_disclaimer()
