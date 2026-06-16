"""Small Streamlit UI helpers shared by the MVP pages."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_header(title: str, subtitle: str | None = None) -> None:
    """Render a compact page header."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def render_disclaimer() -> None:
    """Render the required financial and methodological disclaimer."""
    st.warning(
        "Esta herramienta estima volatilidad realizada esperada, no la direccion del precio. "
        "No es asesoramiento financiero, no genera señales de compra o venta."
    )


def render_artifact_status_table(status_rows: list[dict[str, Any]]) -> None:
    """Render a status table for required local artifacts."""
    display_rows = []
    for row in status_rows:
        display_rows.append(
            {
                "Artefacto": row["name"],
                "Ruta": row["path"],
                "Estado": "OK" if row["exists"] else "Falta",
                "Tamano": format_bytes(int(row["size_bytes"])),
            }
        )
    st.dataframe(pd.DataFrame(display_rows), hide_index=True, width="stretch")


def render_metric_card(label: str, value: str, help_text: str | None = None) -> None:
    """Render a simple metric card."""
    st.metric(label=label, value=value, help=help_text)


def format_bytes(size_bytes: int) -> str:
    """Format a byte count for display."""
    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size_bytes} B"
