from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_validation import UTC_NOTICE, detect_frequency_and_gaps, summarize_price_data, validate_price_data
from src.diagnostics import compute_simple_quality_score
from src.ui_components import render_disclaimer, render_header


def display_value(value: object) -> str:
    """Return a stable string value for mixed-type Streamlit summary tables."""
    if isinstance(value, float):
        return f"{value:.8f}".rstrip("0").rstrip(".")
    return str(value)


render_header(
    "Diagnostico de datos",
    "Validacion de datos cargados y suficiencia para las siguientes etapas.",
)

st.info(UTC_NOTICE)

if "price_data" not in st.session_state:
    st.warning("Primero carga datos en la pagina Prediccion de volatilidad.")
    render_disclaimer()
else:
    price_df = st.session_state["price_data"]
    validation = validate_price_data(price_df)
    summary = summarize_price_data(price_df)
    frequency = detect_frequency_and_gaps(price_df["timestamp"])
    feature_df = st.session_state.get("feature_data", pd.DataFrame())
    availability = st.session_state.get("model_availability", {})
    quality = compute_simple_quality_score(validation, availability, feature_df)

    st.subheader("Fiabilidad simple")
    left, right = st.columns([1, 3])
    left.metric("Score", f"{quality['score']}/100")
    status = str(quality["status"])
    status_message = f"Fiabilidad: {status}"
    if status == "Fiable":
        right.success(status_message)
    elif status == "Fiabilidad limitada":
        right.warning(status_message)
    else:
        right.error(status_message)
    for message in quality["messages"]:
        st.write(f"- {message}")

    st.subheader("Checks principales")
    checks = [
        {"Check": "Columnas timestamp,close", "Estado": "OK", "Detalle": "Datos normalizados"},
        {
            "Check": "Validacion general",
            "Estado": "OK" if validation.valid else "Error",
            "Detalle": "Sin errores" if validation.valid else "; ".join(validation.errors),
        },
        {
            "Check": "Warnings",
            "Estado": "Avisos" if validation.warnings else "OK",
            "Detalle": "; ".join(validation.warnings) if validation.warnings else "Sin avisos",
        },
        {
            "Check": "Frecuencia 5m",
            "Estado": "OK" if frequency["is_expected_5m"] else "Aviso",
            "Detalle": str(frequency["modal_frequency"]),
        },
        {
            "Check": "Gaps",
            "Estado": "OK" if int(frequency["n_gaps"]) == 0 else "Aviso",
            "Detalle": f"{frequency['n_gaps']} gaps ({frequency['gap_percentage']:.2f}%)",
        },
    ]
    st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")

    st.subheader("Resumen")
    summary_rows = [
        {"Campo": "Filas", "Valor": display_value(summary["n_rows"])},
        {"Campo": "Inicio", "Valor": display_value(summary["timestamp_start"])},
        {"Campo": "Final", "Valor": display_value(summary["timestamp_end"])},
        {"Campo": "Rango temporal", "Valor": display_value(summary["time_range"])},
        {"Campo": "Precio minimo", "Valor": display_value(summary["price_min"])},
        {"Campo": "Precio maximo", "Valor": display_value(summary["price_max"])},
        {"Campo": "Ultimo close", "Valor": display_value(summary["last_close"])},
        {"Campo": "Timezone", "Valor": display_value(summary["timezone"])},
    ]
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    st.subheader("Frecuencia y gaps")
    frequency_rows = [
        {"Campo": "Frecuencia modal", "Valor": display_value(frequency["modal_frequency"])},
        {"Campo": "Frecuencia esperada", "Valor": display_value(frequency["expected_frequency"])},
        {"Campo": "Intervalos", "Valor": display_value(frequency["n_intervals"])},
        {"Campo": "Gaps", "Valor": display_value(frequency["n_gaps"])},
        {"Campo": "Porcentaje gaps", "Valor": f"{frequency['gap_percentage']:.2f}%"},
        {"Campo": "Mayor gap", "Valor": display_value(frequency["largest_gap"])},
    ]
    st.dataframe(pd.DataFrame(frequency_rows), hide_index=True, width="stretch")

    if "feature_data" in st.session_state:
        st.subheader("Disponibilidad de modelos futuros")
        rows = []
        for model, info in availability.items():
            rows.append(
                {
                    "Modelo": model,
                    "Disponible": "Si" if info.get("available") else "No",
                    "Valores disponibles": display_value(info.get("available_values", "-")),
                    "Requerido": display_value(info.get("required", "-")),
                    "Mensaje": display_value(info.get("message", "")),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("Carga datos validos en la pagina de prediccion para calcular features y disponibilidad.")

    render_disclaimer()
