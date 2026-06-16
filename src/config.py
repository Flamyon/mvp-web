"""Application configuration for the standalone MVP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MVP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = MVP_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "model_artifacts"
SAMPLE_DIR = DATA_DIR / "sample"
ASSETS_DIR = MVP_ROOT / "assets"
HISTORICAL_VALIDATION_DIR = ASSETS_DIR / "historical_validation"

SYMBOL = "BTCUSDT"
INTERVAL = "5m"
BINANCE_LIMIT = 1000
HORIZON_BARS = 12
HORIZON_MINUTES = 60
EPSILON_LOG_RV = 1e-12
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_KLINES_ENDPOINTS = (
    {"name": "Binance Global", "url": BINANCE_KLINES_URL},
    {"name": "Binance.US", "url": "https://api.binance.us/api/v3/klines"},
)
CACHE_TTL_SECONDS = 300
MAX_UPLOAD_ROWS = 2000

REQUIRED_ARTIFACT_FILENAMES = (
    "ar49_coefficients.csv",
    "preprocessing_params.json",
    "embedding_params.json",
    "knn_reference_train.npz",
    "historical_metrics.json",
    "har_logrv_model.json",
)
SAMPLE_FILENAME = "btcusdt_5m_recent_sample.csv"


def get_required_artifacts() -> list[Path]:
    """Return required local artifact paths."""
    artifact_paths = [ARTIFACTS_DIR / filename for filename in REQUIRED_ARTIFACT_FILENAMES]
    artifact_paths.append(SAMPLE_DIR / SAMPLE_FILENAME)
    return artifact_paths


def verify_artifacts() -> tuple[bool, list[str]]:
    """Check whether all required local artifacts exist."""
    errors: list[str] = []
    for path in get_required_artifacts():
        if not path.exists():
            errors.append(f"Falta artefacto obligatorio: {path.relative_to(MVP_ROOT)}")
        elif path.is_file() and path.stat().st_size == 0:
            errors.append(f"Artefacto vacio: {path.relative_to(MVP_ROOT)}")
    return (not errors, errors)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file from a local path."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected in {path}")
    return payload


def get_artifact_status() -> list[dict[str, Any]]:
    """Return existence and size metadata for required artifacts."""
    rows: list[dict[str, Any]] = []
    for path in get_required_artifacts():
        exists = path.exists()
        rows.append(
            {
                "name": path.name,
                "path": str(path.relative_to(MVP_ROOT)),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists and path.is_file() else 0,
            }
        )
    return rows
