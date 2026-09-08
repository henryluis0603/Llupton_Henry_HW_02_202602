"""Fase 2 — Ruteo y matriz de tiempos de viaje.

Motor: OSMnx + NetworkX, un grafo POR DEPARTAMENTO (no nacional) — ver
`config.md: routing.motor`. Decisión documentada: para tres departamentos
chicos, un grafo nacional es más lento de construir y de cachear que tres
grafos locales, y OSRM-Docker (la opción recomendada por el issue) exige
infraestructura que no vale la pena para este alcance.

Si la descarga vía Overpass (osmnx) falla — sin red, IP bloqueada, timeout —
se cae a un grafo geométrico aleatorio sintético dentro del mismo bbox, para
que validation -> routing -> metrics -> app se puedan probar de punta a
punta sin depender de que Overpass esté disponible. Queda registrado en el
log y en el reporte de snapping cuál de los dos se usó.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pyproj

from src import config

log = logging.getLogger("routing")

Profile = Literal["drive", "walk", "bike"]

# Velocidades constantes (km/h) para perfiles sin tag maxspeed en OSM.
_FALLBACK_SPEED_KMH = {"drive": 30.0, "walk": 5.0, "bike": 15.0}


def _cache_dir(rol: str) -> Path:
    d = config.ruta("routing_cache_dir") / rol
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------- #
# Construcción / carga de grafo (cacheado en GraphML)
# ---------------------------------------------------------------------- #
def _build_synthetic_graph(bbox: dict, profile: Profile, seed: int = 42, n_nodes: int = 250) -> nx.MultiDiGraph:
    """Grafo geométrico aleatorio dentro del bbox, como stand-in de la red
    vial real cuando Overpass no está disponible. Cada nodo lleva x/y en
    grados (EPSG:4326, para que sea compatible con osmnx/alphashape); cada
    arista lleva 'length' (metros) calculado reproyectando a la zona UTM del
    departamento (EPSG:327xx), no con la aproximación "1 grado ~ 111 km" —
    esa aproximación ignora que un grado de longitud se encoge con
    cos(latitud), y en Perú (entre -18° y 0° de latitud) eso pesa."""
    rng = np.random.default_rng(seed)
    G = nx.MultiDiGraph(crs="EPSG:4326")
    xs = rng.uniform(bbox["west"], bbox["east"], n_nodes)
    ys = rng.uniform(bbox["south"], bbox["north"], n_nodes)
    for i in range(n_nodes):
        G.add_node(i, x=xs[i], y=ys[i])

    utm_epsg = config.utm_epsg_for_bbox(bbox)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    xs_utm, ys_utm = transformer.transform(xs, ys)
    coords_utm = np.column_stack([xs_utm, ys_utm])

    speed_mps = _FALLBACK_SPEED_KMH[profile] * 1000 / 3600
    k = 5  # vecinos más cercanos por nodo -> grafo conectado y disperso, como una red vial
    for i in range(n_nodes):
        d = np.hypot(coords_utm[:, 0] - coords_utm[i, 0], coords_utm[:, 1] - coords_utm[i, 1])
        nearest = np.argsort(d)[1 : k + 1]
        for j in nearest:
            length_m = float(d[j])  # ya en metros, EPSG:327xx
            G.add_edge(i, int(j), length=length_m, travel_time=length_m / speed_mps)
            G.add_edge(int(j), i, length=length_m, travel_time=length_m / speed_mps)
    return G


def get_graph(rol: str, profile: Profile, force: bool = False) -> tuple[nx.MultiDiGraph, str]:
    """Devuelve (grafo, fuente) donde fuente es 'osm' o 'synthetic'. Cachea en
    data/processed/routing_cache/<rol>/<profile>.graphml."""
    cache_path = _cache_dir(rol) / f"{profile}.graphml"
    source_marker = _cache_dir(rol) / f"{profile}.source"

    if cache_path.exists() and not force:
        import osmnx as ox

        G = ox.load_graphml(cache_path)
        source = source_marker.read_text().strip() if source_marker.exists() else "unknown"
        log.info("Grafo cacheado cargado: %s (%s)", cache_path, source)
        return G, source

    dept = config.departamentos()[rol]
    bbox = dept["bbox"]

    network_type = "drive" if profile in ("drive", "bike") else profile
    try:
        import osmnx as ox

        # Falla rápido si Overpass no responde (IP bloqueada, sin red) en vez
        # de colgar minutos por departamento/perfil antes de caer a sintético.
        ox.settings.requests_timeout = 10
        # cache HTTP propio de osmnx dentro de data/processed, no en la raíz del repo
        ox.settings.cache_folder = config.ruta("processed") / "osmnx_http_cache"

        # osmnx >= 2.0 espera bbox como (left, bottom, right, top) =
        # (west, south, east, north), no (north, south, east, west) como en 1.x
        G = ox.graph_from_bbox(
            bbox=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
            network_type=network_type,
        )
        if profile == "drive":
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
        else:
            speed_mps = _FALLBACK_SPEED_KMH[profile] * 1000 / 3600
            for _, _, data in G.edges(data=True):
                data["travel_time"] = data.get("length", 0.0) / speed_mps
        source = "osm"
        ox.save_graphml(G, cache_path)
    except Exception as exc:  # red caída, Overpass bloqueado/timeout, etc.
        log.warning("Falló descarga OSM para %s/%s (%s). Usando grafo sintético.", rol, profile, exc)
        G = _build_synthetic_graph(bbox, profile)
        source = "synthetic"
        try:
            import osmnx as ox

            ox.save_graphml(G, cache_path)
        except Exception:
            nx.write_graphml(G, cache_path)

    source_marker.write_text(source)
    return G, source


# ---------------------------------------------------------------------- #
# Snapping
# ---------------------------------------------------------------------- #
def snap_points(gdf: gpd.GeoDataFrame, G: nx.MultiDiGraph, id_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Asigna a cada punto el nodo más cercano del grafo. Reporta distancia de
    snap (metros, calculada en la zona UTM del grafo — no con "1 grado ~ 111
    km") y cuántos puntos superan el umbral de config.md."""
    nodes_x = np.array([data["x"] for _, data in G.nodes(data=True)])
    nodes_y = np.array([data["y"] for _, data in G.nodes(data=True)])
    node_ids = np.array([n for n, _ in G.nodes(data=True)])

    # Zona UTM a partir del centroide de los propios nodos del grafo — así
    # snap_points no depende de recibir el bbox del departamento aparte.
    utm_epsg = config.utm_epsg_for_lonlat(float(nodes_x.mean()), float(nodes_y.mean()))
    transformer = pyproj.Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    nodes_x_utm, nodes_y_utm = transformer.transform(nodes_x, nodes_y)
    points_x_utm, points_y_utm = transformer.transform(gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy())

    rows = []
    snap_max = config.routing_cfg()["snap_max_metros"]
    n_over_threshold = 0
    ids = gdf[id_col].to_numpy()
    for i in range(len(gdf)):
        d = np.hypot(nodes_x_utm - points_x_utm[i], nodes_y_utm - points_y_utm[i])
        j = int(np.argmin(d))
        dist_m = float(d[j])
        if dist_m > snap_max:
            n_over_threshold += 1
        rows.append({id_col: ids[i], "node_id": node_ids[j], "snap_dist_m": dist_m})

    snapped = pd.DataFrame(rows)
    report = {
        "n_puntos": len(gdf),
        "snap_dist_m_promedio": float(snapped["snap_dist_m"].mean()) if len(snapped) else None,
        "snap_dist_m_max": float(snapped["snap_dist_m"].max()) if len(snapped) else None,
        "n_sobre_umbral": int(n_over_threshold),
        "umbral_metros": snap_max,
    }
    return snapped, report


# ---------------------------------------------------------------------- #
# Matriz origen (demanda) x facility
# ---------------------------------------------------------------------- #
def travel_time_matrix(
    G: nx.MultiDiGraph,
    demand_snapped: pd.DataFrame,
    facility_snapped: pd.DataFrame,
    demand_id_col: str,
    facility_id_col: str,
    cache_path: Path,
    force: bool = False,
) -> pd.DataFrame:
    """MultiIndex (demand_id, facility_id) -> t_min. Se computa invirtiendo el
    problema: menos facilities que puntos de demanda, así que se corre un
    Dijkstra multi-fuente POR FACILITY (no por punto de demanda) sobre el
    grafo REVERSADO — así el resultado sigue siendo tiempo demanda->facility
    aunque la red tenga calles de un solo sentido. Puntos en un componente
    desconectado del grafo quedan como NaN (fallback documentado)."""
    if cache_path.exists() and not force:
        log.info("Matriz cacheada: %s", cache_path)
        return pd.read_parquet(cache_path)

    G_rev = G.reverse(copy=False)
    demand_nodes = demand_snapped.set_index(demand_id_col)["node_id"]

    records = []
    for _, fac in facility_snapped.iterrows():
        fac_node = fac["node_id"]
        try:
            times_from_fac = nx.single_source_dijkstra_path_length(G_rev, fac_node, weight="travel_time")
        except Exception:
            times_from_fac = {}
        for dem_id, dem_node in demand_nodes.items():
            t_sec = times_from_fac.get(dem_node, np.nan)
            records.append(
                {
                    demand_id_col: dem_id,
                    facility_id_col: fac[facility_id_col],
                    "t_min": t_sec / 60.0 if pd.notna(t_sec) else np.nan,
                }
            )

    matrix = pd.DataFrame(records).set_index([demand_id_col, facility_id_col])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(cache_path)
    log.info("Matriz calculada y cacheada: %s (%d filas)", cache_path, len(matrix))
    return matrix


def nearest_facility(matrix: pd.DataFrame, demand_id_col: str, facility_id_col: str) -> pd.DataFrame:
    """De la matriz completa, la facility más cercana por punto de demanda."""
    df = matrix.reset_index()
    idx = df.groupby(demand_id_col)["t_min"].idxmin()
    return df.loc[idx].reset_index(drop=True)


def compare_modes(
    matrices: dict[Profile, pd.DataFrame], demand_id_col: str, facility_id_col: str
) -> pd.DataFrame:
    """Para cada punto de demanda, la facility más cercana en cada modo y si
    cambia respecto a 'drive'. Es la base del hallazgo car-vs-foot ratio."""
    nearest = {
        profile: nearest_facility(m, demand_id_col, facility_id_col).set_index(demand_id_col)
        for profile, m in matrices.items()
    }
    base = nearest.get("drive")
    if base is None:
        raise ValueError("compare_modes requiere el perfil 'drive' como referencia")

    out = base[[facility_id_col, "t_min"]].rename(
        columns={facility_id_col: "facility_drive", "t_min": "t_min_drive"}
    )
    for profile, df in nearest.items():
        if profile == "drive":
            continue
        out[f"facility_{profile}"] = df[facility_id_col]
        out[f"t_min_{profile}"] = df["t_min"]
        out[f"facility_cambia_{profile}"] = out[f"facility_{profile}"] != out["facility_drive"]
        out[f"ratio_{profile}_drive"] = out[f"t_min_{profile}"] / out["t_min_drive"]
    return out.reset_index()


# ---------------------------------------------------------------------- #
# Innovación 1 — Isócronas
# ---------------------------------------------------------------------- #
def isochrone_polygons(
    G: nx.MultiDiGraph, facility_node: int, thresholds_min: list[int] | None = None
) -> dict[int, Any]:
    """Polígono de alcanzabilidad (alpha shape) alrededor de una facility, por
    umbral de minutos. Reutiliza el mismo grafo de la matriz — cero cómputo
    de ruteo nuevo, solo una vista distinta sobre el mismo `travel_time`."""
    import alphashape

    thresholds_min = thresholds_min or config.thresholds_minutos()
    times = nx.single_source_dijkstra_path_length(G, facility_node, weight="travel_time")

    polygons = {}
    for t in sorted(thresholds_min):
        node_ids = [n for n, sec in times.items() if sec <= t * 60]
        if len(node_ids) < 4:
            polygons[t] = None
            continue
        pts = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_ids]
        try:
            polygons[t] = alphashape.alphashape(pts, alpha=0)  # alpha=0 -> convex hull, robusto para pocos puntos
        except Exception as exc:
            log.warning("Isócrona falló para nodo %s, umbral %d min (%s)", facility_node, t, exc)
            polygons[t] = None
    return polygons


def department_isochrones(
    G: nx.MultiDiGraph, facility_snapped: pd.DataFrame, facility_id_col: str, thresholds_min: list[int] | None = None
) -> gpd.GeoDataFrame:
    """Isócronas para todas las facilities resolutivas de un departamento, en
    una sola GeoDataFrame (facility_id, threshold_min, geometry) lista para
    exportar y para dibujar en el dashboard."""
    thresholds_min = thresholds_min or config.thresholds_minutos()
    rows = []
    for _, fac in facility_snapped.iterrows():
        polys = isochrone_polygons(G, fac["node_id"], thresholds_min)
        for t, poly in polys.items():
            if poly is not None:
                rows.append({facility_id_col: fac[facility_id_col], "threshold_min": t, "geometry": poly})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326") if rows else gpd.GeoDataFrame(
        columns=[facility_id_col, "threshold_min", "geometry"], crs="EPSG:4326"
    )
