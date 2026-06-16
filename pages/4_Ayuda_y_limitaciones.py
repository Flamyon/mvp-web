from __future__ import annotations

import streamlit as st

from src.config import BINANCE_LIMIT, HORIZON_BARS, HORIZON_MINUTES, INTERVAL, SYMBOL
from src.ui_components import render_disclaimer, render_header


render_header(
    "Ayuda y limitaciones",
)

st.subheader("Que predice")
st.write(
    f"En esta implementacion se estima la volatilidad realizada esperada de {SYMBOL} durante la proxima "
    f"hora: {HORIZON_BARS} velas de {INTERVAL}, equivalentes a {HORIZON_MINUTES} minutos. "
)
st.write("Todos los tiempos se interpretan y muestran en UTC."
)

st.subheader("Que no predice")
st.write("No predice direccion del precio, retornos esperados, rentabilidad ni senales de trading. ")
st.write("Tampoco reentrena los modelos historicos desde la interfaz, no anualiza volatilidad y no demuestra caos determinista en BTC. ")
st.write("Mini-HAR es la unica recalibracion local, marcada como experimental."
)

st.subheader("Formato CSV esperado")
st.code(
    "timestamp,close\n"
    "2026-04-30 19:45:00+00:00,76442.71\n"
    "2026-04-30 19:50:00+00:00,76415.00",
    language="csv",
)
st.write("Tambien se aceptan columnas temporales como `open_time`, `date`, `datetime` o `time`.")

st.subheader("Fuente API Binance")
st.write(
    f"La integracion con la API de Binance usara las ultimas {BINANCE_LIMIT} velas de {SYMBOL} con intervalo {INTERVAL}."
)
st.write(
    "En despliegues donde Binance Global bloquee la IP con HTTP 451, la app intentara Binance.US automaticamente."
)

st.subheader("Limitaciones")
st.write(
    "Los resultados historicos no garantizan comportamiento futuro, ya que la volatilidad de mercado cambia por regimenes, liquidez, noticias y microestructura."
)

render_disclaimer()
