"""Fase 5 — Exportación de tablas y figuras para el reporte LaTeX.

`main.tex` hace `\\input{tables/<name>.tex}` y `\\includegraphics{figures/<name>.pdf}`.
Como este módulo genera ambos directo desde los DataFrames de metrics.py,
correr el pipeline de nuevo regenera el reporte con números frescos sin
tocar el .tex a mano.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


def to_latex_table(
    df: pd.DataFrame,
    name: str,
    caption: str,
    label: str,
    float_format: str = "%.1f",
    index: bool = False,
) -> Path:
    out_dir = config.ruta("report_tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.tex"
    latex = df.to_latex(
        index=index,
        float_format=float_format,
        caption=caption,
        label=f"tab:{label}",
        escape=True,
    )
    out_path.write_text(latex, encoding="utf-8")
    return out_path


def save_figure(fig, name: str, fmt: str = "pdf", dpi: int = 300) -> Path:
    out_dir = config.ruta("report_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path
