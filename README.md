# BTC Volatility MVP

App Streamlit para estimar la volatilidad realizada esperada de BTCUSDT durante la próxima hora. Trabaja con velas de 5 minutos, horizonte de 12 velas y timestamps en UTC.

No predice dirección del precio, no genera señales de trading, no anualiza volatilidad y no es asesoramiento financiero.

## Qué hace

- Carga datos desde Binance, dataset de ejemplo o CSV.
- Normaliza datos a `timestamp,close` y valida frecuencia, gaps, nulos, orden temporal y precios positivos.
- Calcula `log_rv_past_12`, `log_rv_past_48`, `log_rv_past_288` y `z_log_rv_past_12`.
- Genera predicciones de `log_rv_future_12` con Persistencia, kNN local, AR(49), HAR-logRV global y Mini-HAR local.
- Muestra horizonte UTC, tabla de predicciones, gráfico reciente, diagnóstico de fiabilidad y descargas CSV.
- Incluye evaluación walk-forward reciente con métricas y un gráfico separado por modelo.
- Muestra comparación histórica formal con figuras AR/kNN/HAR actualizadas tras añadir HAR.

## Instalación

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests:

```bash
python -m pytest tests/
```

## Uso rápido

1. Abrir `Predicción de volatilidad`.
2. Elegir `Binance API`, `Dataset de ejemplo` o `Subir CSV`.
3. Cargar datos y revisar validación/disponibilidad.
4. Pulsar `Generar predicciones`.
5. Revisar tabla, gráfico y horizonte UTC.
6. Descargar `btc_volatility_predictions.csv` si hace falta.
7. Opcional: pulsar `Calcular evaluación reciente` y descargar `btc_volatility_walk_forward_predictions.csv`.

CSV mínimo:

```csv
timestamp,close
2026-04-30 19:55:00+00:00,95000.0
```

También se aceptan columnas temporales `open_time`, `date`, `datetime` o `time`. Los CSV subidos se limitan a las últimas 2000 filas; Binance usa hasta 1000 velas cerradas recientes.

## Modelos

| Modelo | Papel |
| --- | --- |
| Persistencia | Baseline: replica `log_rv_past_12`. |
| kNN local | Modelo no lineal experimental con `tau=137`, `m=5`, `k=200`. |
| AR(49) | Benchmark lineal fuerte entrenado históricamente. |
| HAR-logRV global | Modelo práctico recomendado; usa memoria de 1h, 4h y 24h. |
| Mini-HAR local | Recalibración experimental con la ventana cargada. |

Disponibilidad:
- Persistencia, AR(49) y kNN usan filas con `log_rv_past_12`.
- kNN requiere 549 valores útiles y recomienda 750.
- HAR global y Mini-HAR requieren `log_rv_past_12/48/288`.
- Mini-HAR requiere 312 filas HAR y recomienda 712.

## Páginas

| Página | Contenido |
| --- | --- |
| Inicio | Resumen y verificación de artefactos locales. |
| Predicción de volatilidad | Carga de datos, features, predicción actual, walk-forward, gráficos y CSV. |
| Diagnóstico de datos | Score `Fiable / Fiabilidad limitada / No fiable`, frecuencia, gaps y disponibilidad por modelo. |
| Comparación de modelos | Target, modelos, parámetros kNN, métricas históricas y figuras ACF, embedding, AR/kNN/HAR y control sintético. |
| Ayuda y limitaciones | Uso, CSV, Binance, UTC y restricciones. |

## Salidas

Predicción actual:
- `log_rv_future_12`: predicción en escala logarítmica.
- `rv_future_12`: transformación inversa de la predicción logarítmica.
- `sqrt_rv_percent`: escala aproximada de volatilidad para la ventana de 1 hora, no anualizada.
- `horizon_start_utc` y `horizon_end_utc`: intervalo previsto en UTC.

Walk-forward reciente:
- Compara predicciones pasadas contra `log_rv_future_12` observado una hora después.
- Devuelve RMSE, MAE, bias, errores, valores reales/predichos y `sqrt_rv_*_percent`.
- El target futuro solo se usa para medir error, no para generar la predicción.

## Artefactos locales

La app es autónoma en runtime y no importa código de `btc-volatility`. Usa archivos ya exportados en `data/model_artifacts/`:

- `ar49_coefficients.csv`
- `embedding_params.json`
- `knn_reference_train.npz`
- `preprocessing_params.json`
- `har_logrv_model.json`
- `historical_metrics.json`

El dataset de ejemplo está en `data/sample/btcusdt_5m_recent_sample.csv`.

## Comparación histórica

La página `Comparación de modelos` no es un visor de todas las fases del TFG. Resume solo lo necesario:
- Persistencia de volatilidad (`phase4_volatility_acf.svg`).
- Reconstrucción del espacio de estados para kNN (`phase8_embedding_2d.svg`).
- Comparación final AR(49), kNN k=200 y HAR-logRV (`phase14_ar_knn_har_*`).
- Control sintético con mapa logístico (`phase13_prediction_metrics.svg`).

HAR-logRV global queda como modelo práctico recomendado porque mejora ligeramente a AR(49) y kNN en la muestra histórica comparable, y además es rápido e interpretable.

## Limitaciones

- No predice si BTC sube o baja.
- No da señales de compra/venta ni promete rentabilidad.
- No calcula intervalos de confianza.
- No anualiza volatilidad.
- No demuestra caos determinista en BTC.
- No reentrena modelos históricos desde la web; solo Mini-HAR se recalibra localmente como experimento.
- La evaluación walk-forward reciente sirve para inspección de la ventana cargada y no sustituye la validación histórica formal.

## Estado

MVP funcional cerrado. Con el dataset de ejemplo genera predicciones para los cinco modelos, diagnóstico fiable, gráfico actual, walk-forward reciente y descargas CSV.
