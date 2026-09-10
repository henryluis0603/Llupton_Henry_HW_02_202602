"""Fase 3 — Construcción de métricas. Funciones puras DataFrame -> DataFrame
(o -> escalar/Series), sin efectos secundarios: export.py decide qué guardar.

Todo lo ponderado usa `poblacion` (config.md: ponderacion) — un promedio sin
pesar no es aceptable según el issue.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj

from src import config


# ---------------------------------------------------------------------- #
# 1) t_min ya viene de routing.nearest_facility (una fila por demand point)
# ---------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
# 2) Bandas de cobertura
# ---------------------------------------------------------------------- #
def coverage_bands(
    df: pd.DataFrame, time_col: str = "t_min", weight_col: str = "poblacion", thresholds: list[int] | None = None
) -> pd.DataFrame:
    """`pd.cut` deja NaN sin banda (no encaja en ningún intervalo) y
    `groupby(observed=True)` los elimina en silencio del total — con red
    real, un punto de demanda puede no tener NINGUNA facility alcanzable
    (componente vial desconectado, ver [[routing.nearest_facility]]), y esa
    población quedaría descontada de golpe del denominador de %. Se agrega
    una banda explícita 'Sin ruta (red desconectada)' para esos casos en vez
    de dejarlos fuera del reporte."""
    thresholds = thresholds or config.thresholds_minutos()
    edges = [0] + sorted(thresholds) + [np.inf]
    labels = [f"<= {edges[i+1]} min" if edges[i + 1] != np.inf else f"> {edges[-2]} min" for i in range(len(edges) - 1)]
    sin_ruta_label = "Sin ruta (red desconectada)"
    band = pd.cut(df[time_col], bins=edges, labels=labels, right=True, include_lowest=True).astype("object")
    band = band.where(df[time_col].notna(), sin_ruta_label)
    out = df.assign(_band=band).groupby("_band", observed=True)[weight_col].sum().reset_index()
    out.columns = ["banda", "poblacion"]
    out["pct_poblacion"] = out["poblacion"] / out["poblacion"].sum() * 100
    order = {label: i for i, label in enumerate(labels + [sin_ruta_label])}
    return out.sort_values("banda", key=lambda s: s.map(order)).reset_index(drop=True)


# ---------------------------------------------------------------------- #
# 3) Tiempo promedio de acceso ponderado por población, por nivel geográfico
# ---------------------------------------------------------------------- #
def weighted_mean_access(
    df: pd.DataFrame, group_col: str, time_col: str = "t_min", weight_col: str = "poblacion"
) -> pd.DataFrame:
    """Promedio ponderado SOLO entre población con ruta (t_min no-NaN):
    sumar `g[time_col] * w` con NaN en `time_col` y luego dividir por
    `w.sum()` de TODO el grupo mezclaría numerador (sin los NaN, por el
    skipna por defecto de `.sum()`) con un denominador que sí los incluye
    — subestimaría el tiempo promedio real al tratar a la población sin
    ruta como si tardara 0 minutos. Se excluye esa población del promedio
    y se reporta aparte en `pct_sin_ruta`."""
    def _wmean(g: pd.DataFrame) -> pd.Series:
        g_valid = g.dropna(subset=[time_col])
        w = g_valid[weight_col]
        t_pond = float((g_valid[time_col] * w).sum() / w.sum()) if w.sum() > 0 else np.nan
        w_total = g[weight_col].sum()
        pct_sin_ruta = float(100 * (w_total - w.sum()) / w_total) if w_total > 0 else np.nan
        return pd.Series({"t_min_ponderado": t_pond, "pct_sin_ruta": pct_sin_ruta})

    out = df.groupby(group_col).apply(_wmean, include_groups=False).reset_index()
    return out.sort_values("t_min_ponderado", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------- #
# 4) Distritos críticos
# ---------------------------------------------------------------------- #
def critical_districts(
    df: pd.DataFrame, group_col: str, time_col: str = "t_min", weight_col: str = "poblacion", top_n: int = 10
) -> pd.DataFrame:
    ranked = weighted_mean_access(df, group_col, time_col, weight_col)
    return ranked.head(top_n)


# ---------------------------------------------------------------------- #
# 5) Desigualdad — Gini ponderado por población, vía curva de Lorenz
# ---------------------------------------------------------------------- #
def gini_weighted(values: np.ndarray, weights: np.ndarray) -> float:
    """Gini ponderado (no el Gini simple): usamos la fórmula basada en la
    curva de Lorenz porque t_min no es ingreso, así que la fórmula
    pair-difference clásica normalizada por la media no aplica igual de
    directo con pesos poblacionales desiguales por punto de demanda."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values)
    values, weights = values[mask], weights[mask]
    if len(values) == 0 or weights.sum() == 0:
        return float("nan")

    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum_w = np.concatenate([[0], np.cumsum(weights) / weights.sum()])
    cum_xw = np.concatenate([[0], np.cumsum(values * weights) / (values * weights).sum()])
    gini = 1 - np.sum((cum_xw[1:] + cum_xw[:-1]) * (cum_w[1:] - cum_w[:-1]))
    return float(gini)


# ---------------------------------------------------------------------- #
# 6) Urbano/rural — regla explícita: umbral poblacional (config.md)
# ---------------------------------------------------------------------- #
def classify_urban_rural(df: pd.DataFrame, pop_col: str = "poblacion") -> pd.DataFrame:
    df = df.copy()
    umbral = config.poblacion_minima_urbana()
    df["es_urbano"] = df[pop_col] >= umbral
    return df


def urban_rural_contrast(
    df: pd.DataFrame, time_col: str = "t_min", weight_col: str = "poblacion"
) -> pd.DataFrame:
    df = classify_urban_rural(df, weight_col)
    return weighted_mean_access(df, "es_urbano", time_col, weight_col)


# ---------------------------------------------------------------------- #
# Cross-analysis: altitud vs. tiempo de acceso
# ---------------------------------------------------------------------- #
def altitude_access_cross(
    df: pd.DataFrame, alt_col: str = "altitud_m", time_col: str = "t_min", weight_col: str = "poblacion"
) -> dict[str, float | str]:
    """Correlación ponderada + pendiente de regresión ponderada simple.
    Nota obligatoria: correlación, no causalidad — altitud correlaciona con
    ruralidad y densidad vial, variables que aquí no se aíslan."""
    sub = df[[alt_col, time_col, weight_col]].dropna()
    if len(sub) < 2:
        return {
            "r_ponderado": float("nan"),
            "pendiente_min_por_metro": float("nan"),
            "n": int(len(sub)),
            "interpretacion": (
                "Sin datos de altitud suficientes para esta corrida "
                f"(n={len(sub)}) — no se integró una fuente de altitud real (DEM)."
            ),
        }
    w = sub[weight_col].to_numpy()
    x = sub[alt_col].to_numpy()
    y = sub[time_col].to_numpy()

    w_mean_x = np.average(x, weights=w)
    w_mean_y = np.average(y, weights=w)
    cov = np.average((x - w_mean_x) * (y - w_mean_y), weights=w)
    var_x = np.average((x - w_mean_x) ** 2, weights=w)
    var_y = np.average((y - w_mean_y) ** 2, weights=w)
    r = cov / np.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else np.nan
    slope = cov / var_x if var_x > 0 else np.nan

    return {
        "r_ponderado": float(r),
        "pendiente_min_por_metro": float(slope),
        "n": int(len(sub)),
        "interpretacion": (
            "Correlacional, no causal: la altitud correlaciona con ruralidad y "
            "densidad vial, variables que este análisis no aísla."
        ),
    }


# ---------------------------------------------------------------------- #
# Innovación 2 — 2SFCA (Two-Step Floating Catchment Area)
# ---------------------------------------------------------------------- #
def two_step_floating_catchment_area(
    matrix: pd.DataFrame,
    demand_df: pd.DataFrame,
    facility_df: pd.DataFrame,
    demand_id_col: str,
    facility_id_col: str,
    demand_pop_col: str = "poblacion",
) -> pd.DataFrame:
    """2SFCA: accesibilidad = oferta disponible por habitante dentro del
    umbral de tiempo, sumada sobre todas las facilities alcanzables.

    Paso 1 — por facility j: R_j = capacidad_j / población total dentro del
    catchment (demanda a <= threshold_min de j).
    Paso 2 — por punto de demanda i: A_i = suma de R_j para toda facility j
    dentro del catchment de i.

    Reutiliza `matrix` (Fase 2), no dispara ningún cómputo de ruteo nuevo.
    """
    cfg = config.dos_sfca_cfg()
    threshold_min = cfg["threshold_min"]
    capacity_col = cfg["capacidad_col"]

    df = matrix.reset_index()
    df = df.merge(demand_df[[demand_id_col, demand_pop_col]], on=demand_id_col, how="left")
    df = df.merge(facility_df[[facility_id_col, capacity_col]], on=facility_id_col, how="left")
    within = df[df["t_min"] <= threshold_min].copy()

    # Paso 1: R_j
    pop_per_facility = within.groupby(facility_id_col)[demand_pop_col].sum()
    capacity_per_facility = facility_df.set_index(facility_id_col)[capacity_col]
    r_j = (capacity_per_facility / pop_per_facility.reindex(capacity_per_facility.index)).rename("R_j")
    r_j = r_j.replace([np.inf, -np.inf], np.nan)

    # Paso 2: A_i
    within = within.merge(r_j.reset_index(), on=facility_id_col, how="left")
    a_i = within.groupby(demand_id_col)["R_j"].sum(min_count=1).rename("accesibilidad_2sfca")

    out = demand_df[[demand_id_col]].merge(a_i.reset_index(), on=demand_id_col, how="left")
    out["accesibilidad_2sfca"] = out["accesibilidad_2sfca"].fillna(0.0)
    return out


# ---------------------------------------------------------------------- #
# Discusión — comparativa línea recta vs. red vial (issue #186, Fase 5)
# ---------------------------------------------------------------------- #
def straight_line_vs_network(
    matrix: pd.DataFrame,
    demand_gdf: gpd.GeoDataFrame,
    facility_gdf: gpd.GeoDataFrame,
    demand_id_col: str,
    facility_id_col: str,
) -> pd.DataFrame:
    """Para cada punto de demanda, compara la facility más cercana en línea
    recta (distancia euclidiana en la zona UTM del departamento) contra la
    facility óptima según la matriz de red (t_min). Cuantifica el error de
    un análisis naive que ignore la red vial: cuándo elige una facility
    distinta, y cuántos minutos de más le tomaría llegar a la que la línea
    recta sugiere en vez de a la óptima real."""
    lon = pd.concat([demand_gdf.geometry.x, facility_gdf.geometry.x])
    lat = pd.concat([demand_gdf.geometry.y, facility_gdf.geometry.y])
    utm_epsg = config.utm_epsg_for_lonlat(float(lon.mean()), float(lat.mean()))
    transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)

    dx, dy = transformer.transform(demand_gdf.geometry.x.to_numpy(), demand_gdf.geometry.y.to_numpy())
    fx, fy = transformer.transform(facility_gdf.geometry.x.to_numpy(), facility_gdf.geometry.y.to_numpy())
    demand_ids = demand_gdf[demand_id_col].to_numpy()
    facility_ids = facility_gdf[facility_id_col].to_numpy()

    dist_m = np.hypot(dx[:, None] - fx[None, :], dy[:, None] - fy[None, :])
    nearest_idx = dist_m.argmin(axis=1)
    straight = pd.DataFrame({
        demand_id_col: demand_ids,
        "facility_recta": facility_ids[nearest_idx],
        "dist_recta_m": dist_m[np.arange(len(demand_ids)), nearest_idx],
    })

    net = matrix.reset_index()
    net_valid = net.dropna(subset=["t_min"])
    network_best = net.loc[net_valid.groupby(demand_id_col)["t_min"].idxmin()]
    network_best = network_best.rename(columns={facility_id_col: "facility_red", "t_min": "t_min_red"})
    network_best = network_best[[demand_id_col, "facility_red", "t_min_red"]]

    comp = straight.merge(network_best, on=demand_id_col, how="inner")
    comp["coincide"] = comp["facility_recta"] == comp["facility_red"]

    t_min_lookup = net.set_index([demand_id_col, facility_id_col])["t_min"]
    comp["t_min_si_recta"] = [
        t_min_lookup.get((d, f), np.nan) for d, f in zip(comp[demand_id_col], comp["facility_recta"])
    ]
    comp["penalidad_min"] = comp["t_min_si_recta"] - comp["t_min_red"]
    return comp
