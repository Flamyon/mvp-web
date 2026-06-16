"""Static model comparison and historical-validation page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import ARTIFACTS_DIR, HISTORICAL_VALIDATION_DIR, load_json
from src.ui_components import render_disclaimer, render_header


MODEL_LABELS = {
    "persistence": "Persistencia",
    "historical_mean": "Media historica",
    "ar49": "AR(49)",
    "har_logrv_global": "HAR-logRV global",
    "knn_k200": "kNN local k=200",
    "knn": "kNN local",
    "knn_k50": "kNN local k=50",
}

MODEL_METRIC_ORDER = [
    "har_logrv_global",
    "ar49",
    "knn_k200",
    "persistence",
    "historical_mean",
]

MODEL_INFO = [
    {
        "Modelo": "HAR-logRV global",
        "Tipo": "HAR lineal multiescala",
        "Rol en el MVP": "Modelo practico recomendado.",
        "Interpretacion": "Combina `log_rv_past_12`, `log_rv_past_48` y `log_rv_past_288`.",
    },
    {
        "Modelo": "Persistencia",
        "Tipo": "Baseline",
        "Rol en el MVP": "Referencia simple.",
        "Interpretacion": "Replica `log_rv_past_12` como prediccion futura.",
    },
    {
        "Modelo": "AR(49)",
        "Tipo": "Lineal autorregresivo",
        "Rol en el MVP": "Benchmark lineal fuerte.",
        "Interpretacion": "Usa los ultimos 49 valores de `log_rv_past_12`.",
    },
    {
        "Modelo": "kNN local",
        "Tipo": "No lineal experimental",
        "Rol en el MVP": "Contraste en espacio reconstruido.",
        "Interpretacion": "Busca estados similares con embedding `tau=137`, `m=5`, `k=200`.",
    },
]

KNN_PARAMS = [
    {
        "Parametro": "tau = 137",
        "Origen": "Estudio Técnico - Fase 8",
        "Justificacion": "Delay seleccionado con AMI; equivale a unas 11.4 horas en velas de 5 minutos.",
    },
    {
        "Parametro": "m = 5",
        "Origen": "Estudio Técnico - Fase 8",
        "Justificacion": "Dimension de embedding elegida como compromiso operativo tras FNN/Cao.",
    },
    {
        "Parametro": "memoria efectiva = 548 velas",
        "Origen": "Estudio Técnico - Fase 8",
        "Justificacion": "`(m - 1) * tau`; el estado reconstruido resume unas 45h40m de historia.",
    },
    {
        "Parametro": "k = 200",
        "Origen": "Estudio Técnico - Fase 12",
        "Justificacion": "Mejor configuracion practica tras ampliar la rejilla de sensibilidad.",
    },
]

FIGURE_ORDER = [
    "phase4_volatility_acf.svg",
    "phase8_embedding_2d.svg",
    "phase14_ar_knn_har_metrics.svg",
    "phase14_ar_knn_har_real_vs_predicted.svg",
    "phase14_ar_knn_har_error_distribution.svg",
    "phase13_prediction_metrics.svg",
]

FIGURE_DETAILS = {
    "phase4_volatility_acf.svg": {
        "title": "Estudio Técnico - Fase 4 - Persistencia de volatilidad",
        "shows": (
            "ACF de la serie principal `log_rv_past_12` hasta 288 retardos, equivalente "
            "a un dia de velas de 5 minutos."
        ),
        "why": (
            "Se incluye porque la prediccion de volatilidad solo tiene sentido si existe "
            "persistencia temporal. En la validacion historica, la ACF de `log_rv_past_12` "
            "fue muy alta: lag 1 ~= 0.9814, lag 12 ~= 0.7139 y lag 288 ~= 0.4193."
        ),
        "reading": (
            "La volatilidad realizada no se comporta como ruido independiente. Esto justifica "
            "comparar una baseline de persistencia, un modelo AR y un modelo HAR. Parte de "
            "esta persistencia viene del solapamiento de ventanas de 12 velas, por lo que "
            "no debe interpretarse como prueba de caos."
        ),
    },
    "phase8_embedding_2d.svg": {
        "title": "Estudio Técnico - Fase 8 - Espacio reconstruido para kNN",
        "shows": (
            "Proyeccion 2D del embedding reconstruido con `tau=137` y `m=5` a partir "
            "de `z_log_rv_past_12`."
        ),
        "why": (
            "Se incluye porque el kNN local no trabaja directamente sobre la serie original, "
            "sino sobre estados reconstruidos por retardos. La Fase 8 selecciona `tau=137` "
            "mediante informacion mutua y adopta `m=5` como dimension practica, aunque Cao "
            "sugeria `m=14`."
        ),
        "reading": (
            "La figura debe leerse como una visualizacion operativa, no como prueba de "
            "atractor determinista. La nube no muestra una geometria caotica limpia, pero "
            "si organiza observaciones en regiones asociadas a estados o regimenes de "
            "volatilidad. Esto justifica usar kNN como contraste no lineal experimental."
        ),
    },
    "phase11_test_real_vs_predicted.svg": {
        "title": "Estudio Técnico - Fase 11 - Prediccion historica en test",
        "shows": (
            "Comparacion temporal entre `log_rv_future_12` observado y predicciones de los "
            "modelos sobre el tramo test historico."
        ),
        "why": (
            "Es la figura directamente conectada con las metricas formales exportadas al MVP: "
            "Persistencia, AR(49) y kNN con `k=50`."
        ),
        "reading": (
            "En test, AR(49) queda como mejor modelo por RMSE. kNN mejora a Persistencia, "
            "pero no supera al modelo lineal principal; esto apoya una lectura predictiva "
            "prudente, no una conclusion fuerte de no linealidad explotable."
        ),
    },
    "phase12_comparison_models.svg": {
        "title": "Estudio Técnico - Fase 12 - Robustez de la comparacion",
        "shows": (
            "Comparacion de MAE/RMSE al ampliar la sensibilidad del kNN, incluyendo otros "
            "valores de `k` y la revision de `m=5` frente a `m=14`."
        ),
        "why": (
            "Esta figura no es la fuente exacta de la tabla de metricas del MVP; sirve para "
            "comprobar si la conclusion de Fase 11 cambia al ampliar la rejilla."
        ),
        "reading": (
            "La conclusion se mantiene: el kNN mejora a Persistencia, pero AR(49) sigue siendo "
            "la referencia lineal fuerte. El aumento de dimension no aporta una mejora "
            "clara y encarece la busqueda local."
        ),
    },
    "phase13_prediction_metrics.svg": {
        "title": "Estudio Técnico - Fase 13 - Control sintetico con mapa logistico",
        "shows": (
            "Comparacion de modelos en un sistema caotico sintetico conocido: mapa logistico "
            "limpio y con ruido."
        ),
        "why": (
            "Se incluye para separar dos ideas. En un sistema no lineal controlado, kNN si "
            "deberia ganar a una referencia lineal si la reconstruccion captura estructura "
            "util. En BTC, en cambio, HAR y AR son mas competitivos."
        ),
        "reading": (
            "En el mapa logistico, kNN gana claramente porque la dinamica no lineal es limpia "
            "y controlada. En BTC, HAR-logRV y AR(49) son mas fuertes, lo que indica que la "
            "volatilidad financiera mezcla persistencia multiescala, ruido, heterocedasticidad "
            "y cambios de regimen. Por tanto, el resultado del MVP no debe interpretarse como "
            "demostracion de caos en BTC."
        ),
    },
    "phase14_ar_knn_har_metrics.svg": {
        "title": "Estudio Técnico - Fase 14 - Comparacion central de modelos",
        "shows": (
            "Comparacion de MAE y RMSE para AR(49), kNN local `tau=137,m=5,k=200` "
            "y HAR-logRV global sobre la misma muestra test comparable de 5000 puntos."
        ),
        "why": (
            "Es la figura principal de la pagina porque resume el cierre predictivo del MVP. "
            "A diferencia de las figuras antiguas de Fase 11/12, esta comparacion ya incluye "
            "HAR-logRV y usa la configuracion final del kNN."
        ),
        "reading": (
            "HAR-logRV obtiene el menor RMSE: 0.860811. AR(49) queda muy cerca con RMSE "
            "0.864067. kNN queda algo por detras con RMSE 0.876532. La diferencia HAR vs "
            "AR es pequena, asi que no debe venderse como superioridad amplia. La razon "
            "para recomendar HAR es que combina rendimiento competitivo con simplicidad, "
            "velocidad e interpretabilidad."
        ),
    },
    "phase14_ar_knn_har_real_vs_predicted.svg": {
        "title": "Estudio Técnico - Fase 14 - Real vs predicho",
        "shows": (
            "Ventana representativa del test historico con `log_rv_future_12` observado "
            "y predicciones de AR(49), kNN k=200 y HAR-logRV."
        ),
        "why": (
            "Se incluye porque las metricas agregadas no muestran como se comportan los "
            "modelos en el tiempo. Esta figura permite ver si los modelos siguen el nivel "
            "general de volatilidad o si fallan en episodios bruscos."
        ),
        "reading": (
            "Los tres modelos siguen parcialmente el nivel de volatilidad futura, pero "
            "ninguno anticipa perfectamente los saltos. HAR y AR son mas suaves y capturan "
            "memoria temporal; kNN introduce vecindad local en el espacio reconstruido, "
            "pero en BTC real no supera a las referencias lineales fuertes. Esto refuerza "
            "una lectura prudente: hay estructura predictiva, pero no una dinamica caotica "
            "limpia explotable."
        ),
    },
    "phase14_ar_knn_har_error_distribution.svg": {
        "title": "Estudio Técnico - Fase 14 - Distribucion de errores",
        "shows": (
            "Distribucion del error absoluto de AR(49), kNN k=200 y HAR-logRV sobre "
            "la muestra test comparable."
        ),
        "why": (
            "Se incluye para complementar el RMSE. Dos modelos pueden tener RMSE parecido "
            "pero comportarse de forma distinta en la mediana de error o en episodios extremos."
        ),
        "reading": (
            "Si HAR aparece ligeramente por debajo de AR y kNN, la mejora es coherente con "
            "las metricas agregadas. Si las cajas son parecidas, la conclusion debe seguir "
            "siendo prudente: HAR no domina de forma aplastante, sino que ofrece una mejora "
            "marginal junto con ventajas practicas claras para el MVP."
        ),
    },
    "phase14_test_metrics_comparison.svg": {
        "title": "Estudio Técnico - Fase 14 - HAR-logRV compacto",
        "shows": (
            "Comparacion MAE/RMSE entre media historica, Persistencia, AR(49), "
            "kNN `tau=137,m=5,k=200` y HAR-logRV global."
        ),
        "why": (
            "Esta figura justifica el cambio de recomendacion practica del MVP: HAR-logRV "
            "es rapido, interpretable, exportable y queda ligeramente por delante de AR(49)."
        ),
        "reading": (
            "La mejora frente a AR(49) es pequena, pero consistente en el test comparable. "
            "Por eso HAR-logRV se presenta como modelo operativo recomendado, no como "
            "prueba de superioridad no lineal."
        ),
    },
}


def render_figure(path: Path) -> None:
    """Render a historical validation figure if Streamlit supports it."""
    try:
        st.image(str(path), width="stretch")
    except Exception:
        st.write(f"No se pudo renderizar directamente: `{path}`")


def render_figure_details(figure_name: str) -> None:
    """Render the technical explanation for a historical figure."""
    details = FIGURE_DETAILS.get(figure_name)
    if not details:
        st.caption("Figura historica exportada para contexto del MVP.")
        return

    st.write("**Qué muestra:**")
    st.write(details["shows"])
    st.write("**Por qué está aquí:**")
    st.write(details["why"])
    st.write("**Lectura técnica:**")
    st.write(details["reading"])


def ordered_historical_figures() -> list[Path]:
    """Return historical figures in project-logical order."""
    figures_by_name = {
        path.name: path
        for path in HISTORICAL_VALIDATION_DIR.glob("*")
        if path.suffix.lower() in {".svg", ".png", ".jpg", ".jpeg"}
    }
    return [figures_by_name[name] for name in FIGURE_ORDER if name in figures_by_name]


def format_metric(value: object, decimals: int) -> str:
    """Format numeric metrics for display while tolerating missing values."""
    if value is None:
        return ""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def historical_metric_rows(payload: dict) -> list[dict[str, str]]:
    """Return historical metrics in the display order used by the MVP."""
    models = payload.get("models", {})
    rows = []
    for model_name in MODEL_METRIC_ORDER:
        metrics = models.get(model_name)
        if not metrics:
            continue
        rows.append(
            {
                "Modelo": MODEL_LABELS.get(model_name, model_name),
                "RMSE test": format_metric(metrics.get("test_rmse"), 6),
                "MAE test": format_metric(metrics.get("test_mae"), 6),
                "R2 test": format_metric(metrics.get("test_r2"), 4),
            }
        )
    return rows


render_header(
    "Comparacion de modelos",
    "Modelos del MVP, metricas historicas formales y figuras de validacion.",
)

st.subheader("Que se predice")
st.write(
    "`log_rv_past_12` resume la volatilidad realizada de la hora pasada: 12 velas de 5 minutos. "
    "`log_rv_future_12` es el target historico: la volatilidad realizada de la hora siguiente. "
    "La app predice ese target en escala logaritmica."
)
st.write(
    "En el estudio técnico se demuestra persistencia temporal en la volatilidad. Por eso tiene sentido comparar "
    "una baseline de persistencia, un modelo lineal AR(49), un modelo no lineal kNN "
    "y el HAR-logRV multiescala."
)
st.info(
    "La pagina no intenta mostrar todas las fases del TFG. Selecciona las figuras que "
    "explican por que existe senal predictiva, como se construye el kNN y cual es la "
    "comparacion final de modelos tras anadir HAR-logRV."
)

st.subheader("Informacion de cada modelo")
st.dataframe(pd.DataFrame(MODEL_INFO), hide_index=True, width="stretch")

st.subheader("Parametros del kNN local")
st.dataframe(pd.DataFrame(KNN_PARAMS), hide_index=True, width="stretch")

st.subheader("Metricas historicas formales")
st.caption(
    "Estas metricas proceden de la validacion historica del proyecto y no dependen "
    "de la ventana cargada en la sesion actual."
)
metrics_path = ARTIFACTS_DIR / "historical_metrics.json"
if not metrics_path.exists():
    st.error("No se encontro historical_metrics.json.")
else:
    payload = load_json(metrics_path)
    model_rows = historical_metric_rows(payload)
    if model_rows:
        st.dataframe(pd.DataFrame(model_rows), hide_index=True, width="stretch")
    else:
        st.warning("El archivo de metricas existe, pero no contiene modelos.")

    st.success("Modelo práctico recomendado: HAR-logRV global")
    st.caption(
        "HAR-logRV se recomienda porque queda ligeramente por delante de AR(49) y kNN "
        "en la validacion historica comparable, y ademas es el modelo mas simple, rapido "
        "e interpretable para el uso operativo del MVP."
    )
    st.info(
        "La comparacion principal debe leerse sobre la muestra test comparable de 5000 "
        "puntos, porque es la unica en la que AR(49), kNN y HAR-logRV se evaluan bajo "
        "el mismo subconjunto. HAR tambien se evalua en test completo, pero para compararlo "
        "con kNN se usa la muestra comparable."
    )

st.subheader("Validacion historica")
st.write(
    "Estas figuras resumen evidencias usadas para justificar la herramienta: persistencia "
    "de volatilidad, reconstruccion para kNN y comparacion predictiva actual con HAR."
)
st.write(
    "El orden de las figuras sigue la logica del proyecto: primero persistencia de "
    "volatilidad, despues reconstruccion del espacio de estados, despues comparacion "
    "predictiva final y finalmente control sintetico con mapa logistico."
)
st.info(
    "Estas figuras contextualizan la validacion historica del modelo. La evaluacion reciente "
    "de la ventana cargada se muestra en la pagina de Prediccion de volatilidad."
)

figures = ordered_historical_figures()
for figure in figures:
    figure_title = FIGURE_DETAILS.get(figure.name, {}).get("title", figure.stem.replace("_", " "))
    st.subheader(figure_title)
    render_figure(figure)
    render_figure_details(figure.name)

st.subheader("Nota metodologica")
st.write(
    "La validacion historica formal pertenece al proyecto experimental. La evaluacion "
    "walk-forward reciente se consulta en la pagina de Prediccion de volatilidad. "
    "Las metricas recientes y las metricas historicas no son equivalentes."
)
st.info("Esta pagina no es un visor completo de todas las fases experimentales del estudio técnico.")

render_disclaimer()
