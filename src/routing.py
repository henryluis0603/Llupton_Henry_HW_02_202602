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

# Velocidad por tipo de vía (km/h) cuando no hay tag maxspeed — la mayoría de
# vías en los departamentos analizados no lo tienen. Tabla estándar usada por
# routers open-source (OSRM car.lua / OSMnx internamente), no inventada.
_HIGHWAY_SPEED_KMH = {
    "motorway": 100, "motorway_link": 60,
    "trunk": 80, "trunk_link": 50,
    "primary": 60, "primary_link": 40,
    "secondary": 50, "secondary_link": 35,
    "tertiary": 40, "tertiary_link": 30,
    "unclassified": 30, "residential": 25, "living_street": 15,
    "service": 20, "track": 20, "road": 30,
}
_DEFAULT_DRIVE_SPEED_KMH = 30.0


def _parse_maxspeed_kmh(raw: object) -> float | None:
    """'50', '50 km/h', '30 mph' -> km/h. None si no se puede parsear."""
    import re

    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip().lower()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    value = float(match.group(1))
    return value * 1.60934 if "mph" in text else value


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


_PYROSM_NETWORK_TYPE = {"drive": "driving", "walk": "walking", "bike": "cycling"}

# Dos bugs distintos, encontrados y corregidos en cadena para Madre de Dios
# (bbox ~400km de lado):
#
# 1) get_network(bounding_box=<bbox grande>) sobre el .pbf NACIONAL completo
#    (256MB) devolvía una red cortada a media altura del departamento —
#    parecía depender de la RAM libre del proceso (menos determinista
#    corriendo dentro del pipeline completo que aislado). Fix: preextraer
#    con `pyosmium` (streaming en C++, memoria acotada) un .pbf chico
#    reference-complete SOLO para el bbox del departamento (~20MB), y que
#    pyrosm parsee ese archivo chico en vez del país completo. Con esto
#    get_network(network_type='all') sí trae los 665k nodos completos, de
#    forma reproducible.
#
# 2) Con la cobertura de nodos ya completa, `to_graph(...)` seguía
#    devolviendo solo 278k nodos (cortados en el mismo punto) — esta vez NO
#    por bbox/memoria sino porque `to_graph` por defecto usa
#    `retain_all=False`: se queda solo con el componente conexo más grande
#    y descarta el resto. La red vial filtrada por `highway` resultó estar
#    fragmentada en cientos de miles de componentes (confirmado con
#    `nx.weakly_connected_components`) — el más grande cubre Puerto
#    Maldonado hacia el sur/oeste, y hay componentes separados más al norte
#    y este (hacia Iñapari/Iberia) que `to_graph` descartaba por completo.
#    Esto — no la extracción — era la causa real de que las facilities/
#    demanda del norte del departamento terminaran snapeadas a ~130km de
#    distancia: no faltaba el nodo más cercano, pyrosm lo había podado.
#    Fix: `retain_all=True` en `to_graph`, y dejar que sea el snapping +
#    matriz de tiempos quien refleje honestamente cuándo dos puntos NO
#    están conectados por red vial mapeada (tiempo infinito/NaN), en vez de
#    que pyrosm decida en silencio qué componente "cuenta".
_DRIVE_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "road",
}
_EXCLUDE_WALK_HIGHWAYS = {"motorway", "motorway_link"}
_EXCLUDE_BIKE_HIGHWAYS = {"motorway", "motorway_link"}


def _extract_pbf_for_bbox(rol: str, bbox: dict) -> Path:
    """Preextrae, con pyosmium (streaming, memoria acotada), un .pbf chico
    y reference-complete (vías con todos sus nodos, aunque algún nodo caiga
    fuera del bbox) para el bbox del departamento. Cacheado en disco — se
    reusa entre profiles (drive/walk/bike) del mismo rol. No confundir con
    el .graphml cacheado por profile: este extracto es un insumo intermedio,
    un nivel más abajo."""
    import osmium

    out_path = _cache_dir(rol) / "extract.osm.pbf"
    if out_path.exists():
        return out_path

    pbf_path = config.ruta("raw") / "peru-latest.osm.pbf"
    if not pbf_path.exists():
        raise FileNotFoundError(f"No existe {pbf_path} — descargar con curl (ver README) antes de usar --pbf")

    bounds = (bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    # tmp_path DEBE terminar en .pbf: osmium detecta el formato de salida por
    # extensión, y un ".tmp" al final rompe esa detección (bug real, ya
    # atrapado: hacía que todo el flujo cayera silenciosamente a sintético).
    tmp_path = out_path.with_name(out_path.stem + ".tmp.pbf")
    with osmium.ForwardReferenceWriter(
        str(tmp_path), ref_src=str(pbf_path), overwrite=True, back_references=True, remove_tags=False
    ) as writer:
        fp = osmium.FileProcessor(str(pbf_path)).with_filter(osmium.filter.EntityFilter(osmium.osm.NODE))
        for node in fp:
            if bounds[0] <= node.location.lon <= bounds[2] and bounds[1] <= node.location.lat <= bounds[3]:
                writer.add_node(node)
    tmp_path.rename(out_path)
    return out_path


def _get_graph_from_pbf(rol: str, profile: Profile, bbox: dict) -> nx.MultiDiGraph:
    """Construye el grafo desde el extracto local `peru-latest.osm.pbf`
    (issue #186: fuente OSM Perú literal) en vez de Overpass en vivo. Todo
    el trabajo es local — sin llamadas de red — así que no puede toparse con
    el corte de conexión a ~60s que sí afecta a Overpass para departamentos
    grandes (ver README, sección de decisiones de clase)."""
    from pyrosm import OSM

    extract_path = _extract_pbf_for_bbox(rol, bbox)

    network_type = _PYROSM_NETWORK_TYPE[profile]
    osm = OSM(str(extract_path), bounding_box=[bbox["west"], bbox["south"], bbox["east"], bbox["north"]])
    nodes, edges = osm.get_network(network_type="all", nodes=True)
    if edges is None or len(edges) == 0:
        raise ValueError(f"pyrosm no encontró ninguna vía en el bbox de {rol}")

    if profile == "drive":
        edges = edges[edges["highway"].isin(_DRIVE_HIGHWAYS)]
    elif profile == "walk":
        edges = edges[~edges["highway"].isin(_EXCLUDE_WALK_HIGHWAYS)]
    elif profile == "bike":
        edges = edges[~edges["highway"].isin(_EXCLUDE_BIKE_HIGHWAYS)]
    if len(edges) == 0:
        raise ValueError(f"Sin vías tipo '{profile}' tras filtrar por highway en el bbox de {rol}")

    G_raw = osm.to_graph(
        nodes, edges, graph_type="networkx", network_type=network_type, osmnx_compatible=True, retain_all=True
    )

    # Grafo limpio con solo los atributos que routing.py usa: pyrosm deja
    # columnas OSM crudas (oneway, access, bridge, ...) con NaN cuando el tag
    # no existe, y el (de)serializador GraphML de osmnx espera tipos fijos
    # para esos nombres de atributo conocidos (p.ej. 'oneway' -> bool) — un
    # NaN ahí rompe `ox.load_graphml` al recargar desde caché. Más simple y
    # robusto no arrastrar esas columnas en vez de sanearlas una por una.
    G = nx.MultiDiGraph(crs="EPSG:4326")
    for n, data in G_raw.nodes(data=True):
        G.add_node(n, x=data["x"], y=data["y"])

    speed_mps_const = _FALLBACK_SPEED_KMH[profile] * 1000 / 3600
    for u, v, data in G_raw.edges(data=True):
        length_m = float(data.get("length", 0.0))
        if profile == "drive":
            speed_kmh = _parse_maxspeed_kmh(data.get("maxspeed"))
            if speed_kmh is None:
                speed_kmh = _HIGHWAY_SPEED_KMH.get(str(data.get("highway")), _DEFAULT_DRIVE_SPEED_KMH)
            travel_time = length_m / (speed_kmh * 1000 / 3600)
        else:
            travel_time = length_m / speed_mps_const
        G.add_edge(u, v, length=length_m, travel_time=travel_time)

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
    G = None
    source = None

    # 1) Extracto local peru-latest.osm.pbf (issue #186: fuente OSM Perú
    # literal) si está descargado — todo el trabajo es local, así que no
    # puede toparse con el corte de conexión a ~60s que sí afecta a
    # Overpass en vivo para departamentos grandes (ver README).
    pbf_path = config.ruta("raw") / "peru-latest.osm.pbf"
    if pbf_path.exists():
        try:
            G = _get_graph_from_pbf(rol, profile, bbox)
            source = "osm-pbf"
        except Exception as exc:
            log.warning("Falló parseo del .pbf para %s/%s (%s). Intentando Overpass en vivo.", rol, profile, exc)

    # 2) Overpass en vivo, si no hay .pbf o falló el parseo.
    if G is None:
        try:
            import osmnx as ox

            # 240s, no 10s: un timeout corto falla rápido cuando el servidor
            # está inalcanzable (conexión rechazada/reset falla casi
            # instantáneo pase lo que pase), pero mata subconsultas de
            # departamentos grandes que están respondiendo, solo que lento
            # — confirmado empíricamente: con 10s, Huancavelica (13x el área
            # máxima de consulta, se subdivide en muchas subconsultas)
            # perdía perfiles completos que sí terminaban en ~3 minutos
            # cuando se les daba tiempo.
            ox.settings.requests_timeout = 240
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
            source = "osm-overpass"
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
    """De la matriz completa, la facility más cercana por punto de demanda.
    Puntos de demanda sin NINGUNA facility alcanzable por red (todo NaN —
    p.ej. quedaron en un componente vial desconectado de las 3 facilities,
    algo real y posible ahora que el grafo ya no se poda al componente más
    grande, ver comentario sobre retain_all en _get_graph_from_pbf) se
    excluyen del resultado en vez de hacer fallar idxmin."""
    df = matrix.reset_index()
    valid = df.dropna(subset=["t_min"])
    idx = valid.groupby(demand_id_col)["t_min"].idxmin()
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
