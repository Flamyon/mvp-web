"""Streamlit entry point for the BTC volatility MVP."""

from __future__ import annotations

import streamlit as st

from src.config import (
    HORIZON_BARS,
    HORIZON_MINUTES,
    INTERVAL,
    SYMBOL,
    get_artifact_status,
    verify_artifacts,
)
from src.ui_components import (
    render_artifact_status_table,
    render_disclaimer,
    render_header,
    render_metric_card,
)

st.set_page_config(
    page_title="Predictor de volatilidad en BTC",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    render_header(
        "Prediccion de volatilidad realizada de BTCUSDT a 1 hora",
    )


    left, middle, right = st.columns([1.2, 1.2, 1.4])
    with left:
        render_metric_card("Simbolo", SYMBOL, "Par previsto para la fuente principal Binance.")
        render_metric_card("Frecuencia", INTERVAL, "Velas de 5 minutos.")
    with middle:
        render_metric_card("Horizonte", f"{HORIZON_BARS} velas", f"{HORIZON_MINUTES} minutos.")
        render_metric_card("Modelos", "5", "Persistencia, kNN, AR(49), HAR global y Mini-HAR.")

    st.subheader("Resumen")
    st.write(
        "Esta es la implementacion como resultado de el proyecto TFG en el que se ha analizado el comportamiento de la volatilidad en BTCUSDT. "
    )
    st.write(
        "El usuario puede probar modelos de volatilidad con datos actuales: Persistencia, kNN local, AR(49), HAR-logRV global y Mini-HAR local experimental. "
        "HAR-logRV global es la recomendacion practica; AR(49) queda como benchmark lineal fuerte y kNN como contraste no lineal experimental."
    )

    st.subheader("Archivos locales usados por el MVP")
    render_artifact_status_table(get_artifact_status())
    artifacts_ok, artifact_errors = verify_artifacts()
    if not artifacts_ok:
        st.error("Faltan artefactos locales obligatorios.")
        for error in artifact_errors:
            st.write(f"- {error}")
        st.code(
            "python scripts/export_model_artifacts.py\n"
            "python scripts/validate_artifacts.py",
            language="bash",
        )
        st.stop()

    st.success("Artefactos locales verificados. El MVP puede arrancar sin btc-volatility en runtime.")
    render_disclaimer()

# navegación lateral

paginas = [
    st.Page(main, title="Inicio", default=True),
    st.Page("pages/1_Prediccion_de_volatilidad.py", title="Predicción de volatilidad"),
    st.Page("pages/2_Diagnostico_de_datos.py", title="Diagnóstico de datos"),
    st.Page("pages/3_Comparacion_de_modelos.py", title="Comparación de modelos"),
    st.Page("pages/4_Ayuda_y_limitaciones.py", title="Ayuda y limitaciones"),
]

pg = st.navigation(paginas)
pg.run()
