"""Fase 1b — Validación. 6 reglas obligatorias, cada una como función pura
(df_o_gdf) -> (df_limpio, reporte_dict). `validate_dataset` las encadena y
arma el data quality report que Fase 4 muestra en el panel de calidad.

Orden deliberado: null/zero -> swap lat/lon (recupera antes de descartar) ->
bbox Perú -> polígono declarado (flag, no drop) -> duplicados -> encoding.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src import config


# ---------------------------------------------------------------------- #
# Regla 1 — coordenadas nulas / cero
# ---------------------------------------------------------------------- #
def check_missing_coords(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    is_null = gdf.geometry.isna()
    is_zero = gdf.geometry.apply(lambda g: g is not None and not g.is_empty and g.x == 0 and g.y == 0)
    bad = is_null | is_zero
    dropped = int(bad.sum())
    return gdf.loc[~bad].copy(), {
        "regla": "coordenadas_nulas_o_cero",
        "accion": "drop",
        "registros_afectados": dropped,
    }


# ---------------------------------------------------------------------- #
# Regla 2/3 combinadas en orden: swap primero (recupera), bbox después (drop)
# ---------------------------------------------------------------------- #
def _in_bbox(x: float, y: float, bbox: dict[str, float]) -> bool:
    return bbox["lon_min"] <= x <= bbox["lon_max"] and bbox["lat_min"] <= y <= bbox["lat_max"]


def check_lat_lon_swap(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Si (lon, lat) cae fuera del bbox de Perú pero el swap (lat, lon) sí cae
    dentro, se corrige in-place. Es recuperación documentada, no un drop."""
    bbox = config.peru_bbox()
    fixed = 0

    def _maybe_fix(geom: Point) -> Point:
        nonlocal fixed
        if geom is None or geom.is_empty:
            return geom
        x, y = geom.x, geom.y
        if _in_bbox(x, y, bbox):
            return geom
        if _in_bbox(y, x, bbox):
            fixed += 1
            return Point(y, x)
        return geom

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(_maybe_fix)
    return gdf, {
        "regla": "lat_lon_invertidos",
        "accion": "swap_si_cae_en_bbox",
        "registros_afectados": fixed,
    }


def check_peru_bbox(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    bbox = config.peru_bbox()
    inside = gdf.geometry.apply(lambda g: g is not None and not g.is_empty and _in_bbox(g.x, g.y, bbox))
    dropped = int((~inside).sum())
    return gdf.loc[inside].copy(), {
        "regla": "fuera_de_bbox_peru",
        "accion": "drop",
        "registros_afectados": dropped,
    }


# ---------------------------------------------------------------------- #
# Regla 4 — punto fuera del polígono distrital que declara su propio registro
# (flag, no drop: puede ser error del polígono, no del punto)
# ---------------------------------------------------------------------- #
def check_point_in_declared_polygon(
    gdf: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame | None,
    point_district_col: str,
    polygon_district_col: str = "ubigeo",
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    gdf = gdf.copy()
    if polygon_gdf is None or point_district_col not in gdf.columns:
        gdf["flag_fuera_de_poligono"] = False
        return gdf, {
            "regla": "punto_fuera_de_poligono_distrital",
            "accion": "flag",
            "registros_afectados": 0,
            "nota": "sin capa de polígonos disponible — regla no evaluada",
        }

    # drop_duplicates defensivo: si el shapefile trajera más de una fila por
    # distrito (p.ej. un polígono multi-parte no disuelto), un merge normal
    # multiplicaría filas de `gdf` y el `gdf["flag..."] = flags` de abajo
    # quedaría desalineado en silencio (asume len(merged) == len(gdf)). No
    # ocurre con los shapefiles usados aquí (verificado: 0 ubigeo
    # duplicados en los 3 departamentos), pero el join queda protegido.
    polygons = polygon_gdf[[polygon_district_col, "geometry"]].drop_duplicates(
        subset=polygon_district_col
    ).rename(columns={"geometry": "_poly"})
    merged = gdf.merge(polygons, left_on=point_district_col, right_on=polygon_district_col, how="left")
    assert len(merged) == len(gdf), "merge de polígono distrital multiplicó filas inesperadamente"
    flags = [
        (row["_poly"] is not None and not row["_poly"].contains(row["geometry"]))
        for _, row in merged.iterrows()
    ]
    gdf["flag_fuera_de_poligono"] = flags
    return gdf, {
        "regla": "punto_fuera_de_poligono_distrital",
        "accion": "flag",
        "registros_afectados": int(sum(flags)),
    }


# ---------------------------------------------------------------------- #
# Regla 5 — códigos duplicados
# ---------------------------------------------------------------------- #
def check_duplicate_ids(
    gdf: gpd.GeoDataFrame, id_col: str, updated_at_col: str | None = None
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    before = len(gdf)
    completeness = gdf.notna().sum(axis=1)
    gdf = gdf.assign(_completeness=completeness)
    sort_cols = [id_col]
    ascending = [True]
    if updated_at_col and updated_at_col in gdf.columns:
        sort_cols.append(updated_at_col)
        ascending.append(False)
    sort_cols.append("_completeness")
    ascending.append(False)
    gdf = gdf.sort_values(sort_cols, ascending=ascending).drop_duplicates(subset=id_col, keep="first")
    gdf = gdf.drop(columns="_completeness")
    dropped = before - len(gdf)
    return gdf, {
        "regla": "codigos_duplicados",
        "accion": "dedup_keep_mas_completo_o_reciente",
        "registros_afectados": int(dropped),
    }


# ---------------------------------------------------------------------- #
# Regla 6 — encoding UTF-8 vs latin-1
# ---------------------------------------------------------------------- #
def check_encoding(gdf: gpd.GeoDataFrame, text_cols: list[str]) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    gdf = gdf.copy()
    fixed = 0

    def _fix(value: Any) -> Any:
        nonlocal fixed
        if not isinstance(value, str):
            return value
        try:
            candidate = value.encode("latin1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return value
        if candidate != value:
            fixed += 1
            return candidate
        return value

    for col in text_cols:
        if col in gdf.columns:
            gdf[col] = gdf[col].apply(_fix)

    return gdf, {
        "regla": "encoding_utf8_vs_latin1",
        "accion": "reencode_donde_aplica",
        "registros_afectados": fixed,
    }


# ---------------------------------------------------------------------- #
# Normalización de categoría + bandera "resolutivo" (Fase 1, no es una de
# las 6 reglas genéricas, pero es requisito explícito del issue)
# ---------------------------------------------------------------------- #
def normalize_categoria(gdf: gpd.GeoDataFrame, col: str = "categoria") -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    mapping = config.normalizacion_categorias()
    normalized = gdf[col].astype(str).str.strip()
    normalized = normalized.replace(mapping)
    normalized = normalized.replace({k.strip(): v for k, v in mapping.items()})
    gdf[col] = normalized
    gdf["resolutivo"] = gdf[col].isin(config.categorias_resolutivas())
    return gdf


# ---------------------------------------------------------------------- #
# Orquestador
# ---------------------------------------------------------------------- #
def validate_dataset(
    gdf: gpd.GeoDataFrame,
    dataset_name: str,
    id_col: str,
    text_cols: list[str] | None = None,
    updated_at_col: str | None = None,
    point_district_col: str | None = None,
    polygon_gdf: gpd.GeoDataFrame | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    text_cols = text_cols or [c for c in gdf.columns if gdf[c].dtype == object and c != "geometry"]
    n_start = len(gdf)
    reglas = []

    gdf, r1 = check_missing_coords(gdf)
    reglas.append(r1)
    gdf, r2 = check_lat_lon_swap(gdf)
    reglas.append(r2)
    gdf, r3 = check_peru_bbox(gdf)
    reglas.append(r3)
    gdf, r4 = check_point_in_declared_polygon(
        gdf, polygon_gdf, point_district_col or "", polygon_district_col="ubigeo"
    )
    reglas.append(r4)
    gdf, r5 = check_duplicate_ids(gdf, id_col, updated_at_col)
    reglas.append(r5)
    gdf, r6 = check_encoding(gdf, text_cols)
    reglas.append(r6)

    report = {
        "dataset": dataset_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_registros_inicial": n_start,
        "n_registros_final": len(gdf),
        "reglas": reglas,
    }
    return gdf, report


def write_quality_report(reports: list[dict[str, Any]], path: Path | None = None) -> Path:
    path = path or config.ruta("quality_report")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": reports,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
