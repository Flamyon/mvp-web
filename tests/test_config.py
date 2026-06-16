from __future__ import annotations

from pathlib import Path

from src import config


def test_mvp_root_exists() -> None:
    assert config.MVP_ROOT.exists()
    assert config.MVP_ROOT.name == "mvp-web"


def test_artifacts_dir_exists() -> None:
    assert config.ARTIFACTS_DIR.exists()
    assert config.ARTIFACTS_DIR.is_dir()


def test_required_artifacts_are_internal_paths() -> None:
    required = config.get_required_artifacts()
    assert required
    for path in required:
        path.relative_to(config.MVP_ROOT)


def test_required_artifacts_do_not_reference_btc_volatility() -> None:
    for path in config.get_required_artifacts():
        assert "btc-volatility" not in str(path)


def test_verify_artifacts_passes_with_current_artifacts() -> None:
    ok, errors = config.verify_artifacts()
    assert ok
    assert errors == []


def test_get_artifact_status_returns_rows() -> None:
    rows = config.get_artifact_status()
    assert rows
    assert len(rows) == len(config.get_required_artifacts())
    for row in rows:
        assert {"name", "path", "exists", "size_bytes"}.issubset(row)
