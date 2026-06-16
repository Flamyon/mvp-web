"""Simple reliability diagnostics for the MVP."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_validation import ValidationResult


def compute_simple_quality_score(
    validation_result: ValidationResult,
    model_availability: dict[str, dict[str, object]],
    feature_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute a compact reliability signal from reliable to not reliable."""
    messages: list[str] = []

    if validation_result.errors or not validation_result.valid:
        return {
            "score": 20,
            "status": "No fiable",
            "messages": validation_result.errors or ["Datos no validos para prediccion."],
        }

    if feature_df.empty or "log_rv_past_12" not in feature_df.columns:
        return {
            "score": 30,
            "status": "No fiable",
            "messages": ["No hay features utiles para ejecutar modelos."],
        }

    score = 100
    frequency = validation_result.info.get("frequency", {})
    n_gaps = int(frequency.get("n_gaps", 0) or 0)
    gap_percentage = float(frequency.get("gap_percentage", 0.0) or 0.0)
    if n_gaps >= 3 or gap_percentage >= 1.0:
        score -= 30
        messages.append(f"Gaps relevantes detectados: {n_gaps} ({gap_percentage:.2f}%).")
    elif n_gaps > 0:
        score -= 10
        messages.append(f"Gaps menores detectados: {n_gaps}.")

    if frequency and not bool(frequency.get("is_expected_5m", False)):
        score -= 15
        messages.append("La frecuencia modal no es exactamente 5 minutos.")

    if not bool(model_availability.get("har_global", {}).get("available", False)):
        score -= 25
        messages.append("HAR-logRV global no disponible: faltan features HAR.")

    if not bool(model_availability.get("knn", {}).get("available", False)):
        score -= 25
        messages.append("kNN no disponible: faltan valores utiles para el embedding.")
    elif not bool(model_availability.get("knn", {}).get("recommended", True)):
        score -= 10
        messages.append("kNN disponible, pero por debajo del margen recomendado.")

    if not bool(model_availability.get("ar49", {}).get("available", False)):
        score -= 20
        messages.append("AR(49) no disponible: faltan 49 valores utiles.")

    if validation_result.warnings and not messages:
        score -= 10
        messages.append("Hay avisos de validacion no bloqueantes.")

    score = max(0, min(100, int(score)))
    if score >= 90:
        status = "Fiable"
        if not messages:
            messages.append("Datos suficientes y sin gaps relevantes.")
    elif score >= 50:
        status = "Fiabilidad limitada"
    else:
        status = "No fiable"

    return {"score": score, "status": status, "messages": messages}
