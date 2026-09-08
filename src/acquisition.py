"""Fase 1a — Adquisición de datos.

Idempotente: si el archivo destino ya existe, no vuelve a descargar (a menos
que `force=True`). Cada descarga real está envuelta en try/except: si la
fuente no responde (bloqueo de IP, mantenimiento, cambio de URL), se registra
en logs/acquisition.log y el pipeline puede seguir con `--synthetic` en vez
de morir a mitad de camino.

Uso:
    python -m src.acquisition --synthetic          # genera datos falsos, los 3 deptos
    python -m src.acquisition --real                # descarga y parsea RENIPRESS/centros poblados IGN/población RENIEC reales
"""
from __future__ import annotations

import argparse
import logging
import unicodedata
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.ruta("logs") / "acquisition.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("acquisition")

# Portales gob.pe devuelven 403/418 al User-Agent por defecto de requests
# (bloqueo de WAF a tráfico no-navegador) — confirmado empíricamente: la
# misma URL responde 200 con este header y sin él no.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

# Fuentes reales, verificadas manualmente (URLs resueltas vía la API CKAN de
# cada portal, no adivinadas) el 2026-09-08. Si el portal cambia de URL o
# bloquea la IP del ejecutor, ver logs/acquisition.log y usar --synthetic
# para no bloquear el resto del pipeline mientras se resuelve.
SOURCES = {
    # RENIPRESS — SUSALUD, registro nacional de IPRESS. 26,850 registros a
    # nivel nacional, actualizado mensualmente.
    "renipress": "http://datos.susalud.gob.pe/sites/default/files/RENIPRESS_2026_v6.csv",
    # Sustitución documentada de "MINEDU/SIGMED" (issue #186): SIGMED-MINEDU
    # es un geoportal interactivo sin endpoint de descarga directa
    # scripteable. Se usa en su lugar el shapefile oficial de Centros
    # Poblados del IGN (Instituto Geográfico Nacional), publicado en el
    # mismo portal nacional de datos abiertos — misma finalidad (ubicación
    # de centros poblados), fuente oficial distinta.
    "centros_poblados_ign": "https://www.datosabiertos.gob.pe/sites/default/files/CCPP_0.zip",
    # Proxy de población por distrito: no existe un CSV oficial de población
    # por CENTRO POBLADO descargable en bloque (el censo INEI 2017 solo se
    # consulta vía REDATAM, interactivo). RENIEC publica trimestralmente
    # población identificada con DNI por distrito (UBIGEO_INEI), que se usa
    # como proxy oficial y se reparte hacia centros poblados individuales
    # por tipo de asentamiento — ver `distribute_population`.
    "poblacion_reniec": "https://www.datosabiertos.gob.pe/sites/default/files/BD_DAbiertos_16_OPP_2026_06.01.csv",
}


def _idempotent_download(url: str, dest: Path, force: bool = False) -> bool:
    """Descarga a `dest` si no existe. Devuelve True si el archivo quedó disponible."""
    if dest.exists() and not force:
        log.info("SKIP (ya existe): %s", dest)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        log.info("OK descarga: %s -> %s (%d bytes)", url, dest, len(resp.content))
        return True
    except requests.RequestException as exc:
        log.warning("FALLÓ descarga %s (%s). Usar --synthetic o reintentar luego.", url, exc)
        return False


def download_renipress(force: bool = False) -> bool:
    dest = config.ruta("raw") / "renipress.csv"
    return _idempotent_download(SOURCES["renipress"], dest, force=force)


def download_centros_poblados(force: bool = False) -> bool:
    dest = config.ruta("raw") / "centros_poblados_ign.zip"
    ok = _idempotent_download(SOURCES["centros_poblados_ign"], dest, force=force)
    if ok:
        extract_dir = config.ruta("raw") / "centros_poblados_ign"
        shp = extract_dir / "CCPP_IGN100K.shp"
        if not shp.exists() or force:
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(extract_dir)
            log.info("Descomprimido: %s", extract_dir)
    return ok


def download_poblacion_reniec(force: bool = False) -> bool:
    dest = config.ruta("raw") / "poblacion_reniec.csv"
    return _idempotent_download(SOURCES["poblacion_reniec"], dest, force=force)


def download_real(force: bool = False) -> dict[str, bool]:
    return {
        "renipress": download_renipress(force=force),
        "centros_poblados_ign": download_centros_poblados(force=force),
        "poblacion_reniec": download_poblacion_reniec(force=force),
    }


# --------------------------------------------------------------------------
# Parsing de fuentes reales a un esquema común (mismas columnas que produce
# generate_synthetic_dataset) para que validation.py / routing.py / metrics.py
# no distingan sintético de real.
# --------------------------------------------------------------------------

# Capacidad de resolución (camas/quirófanos) no viene en el CSV abierto de
# RENIPRESS — solo categoría. Proxy ordinal documentado por nivel de
# categoría, usado únicamente por la innovación 2SFCA; no es un dato oficial
# de infraestructura real.
_CAPACIDAD_PROXY_POR_CATEGORIA = {
    "I-1": 2, "I-2": 3, "I-3": 5, "I-4": 8,
    "II-1": 15, "II-2": 25, "II-E": 20,
    "III-1": 40, "III-2": 60, "III-E": 50,
}

# Peso relativo de población por tipo de asentamiento (CAT_POBLAD del IGN),
# usado solo para repartir la población distrital de RENIEC entre los
# centros poblados de ese distrito — ver `distribute_population`. Es una
# heurística documentada (capital/ciudad concentra más población que un
# caserío o anexo), no una fuente oficial de población por punto.
_PESO_POBLACION_POR_TIPO = {
    "CIUDAD": 20, "VILLA": 10, "PUEBLO": 8, "PUEBLO JOVEN": 8,
    "BARRIO O CUARTEL": 6, "URBANIZACION": 6,
    "ASOCIACION DE VIVIENDA": 5, "CONJTO.HABITACIONAL": 5, "COOPERATIVA DE VIVIENDA": 5,
    "COOP. AGRARIA": 3, "COMUNIDAD": 3,
    "CASERÍO": 2, "ANEXO": 2, "CAMPO MINERO": 2,
}
_PESO_POBLACION_DEFAULT = 1
_BONUS_CAPITAL_DISTRITO = 3  # multiplicador si CATEGORIA == "Capital de Distrito"


def _norm_name(s: object) -> str:
    """Mayúsculas + sin tildes/diéresis. Necesario porque IGN y RENIEC no
    acentúan igual el mismo nombre de distrito (p.ej. 'CORDOVA' vs
    'CÓRDOVA') — confirmado inspeccionando los nombres que no cruzaban."""
    if pd.isna(s):
        return ""
    normalized = unicodedata.normalize("NFKD", str(s).strip().upper())
    return "".join(c for c in normalized if not unicodedata.combining(c))


def load_renipress(rol: str) -> gpd.GeoDataFrame:
    """Parsea RENIPRESS crudo a facilities del departamento `rol` (esquema
    común con generate_synthetic_dataset). NORTE/ESTE del CSV son en
    realidad latitud/longitud en grados decimales, no coordenadas UTM pese
    al nombre — confirmado inspeccionando los valores (rango -18 a 0)."""
    dept = config.departamentos()[rol]
    path = config.ruta("raw") / "renipress.csv"
    df = pd.read_csv(path, sep=";", encoding="utf-8", dtype=str)
    df = df[df["DEPARTAMENTO"].apply(_norm_name) == _norm_name(dept["nombre"])].copy()

    lat = pd.to_numeric(df["NORTE"], errors="coerce")
    lon = pd.to_numeric(df["ESTE"], errors="coerce")
    geometry = [Point(x, y) if pd.notna(x) and pd.notna(y) else None for x, y in zip(lon, lat)]

    out = gpd.GeoDataFrame(
        {
            "facility_id": df["COD_IPRESS"],
            "nombre": df["NOMBRE"],
            "categoria": df["CATEGORIA"],
            "institucion": df["INSTITUCION"],
            "activo": df["ESTADO"].apply(_norm_name) == "ACTIVO",
            "departamento": df["DEPARTAMENTO"],
            "provincia": df["PROVINCIA"],
            "distrito": df["DISTRITO"],
            "ubigeo": df["UBIGEO"],
            "capacidad": df["CATEGORIA"].map(_CAPACIDAD_PROXY_POR_CATEGORIA).fillna(1),
            "geometry": geometry,
        },
        crs="EPSG:4326",
    )
    log.info("RENIPRESS real cargado para %s: %d facilities", rol, len(out))
    return out


def load_centros_poblados(rol: str) -> gpd.GeoDataFrame:
    """Parsea el shapefile IGN a centros poblados del departamento `rol`,
    sin población todavía (ver `distribute_population`)."""
    dept = config.departamentos()[rol]
    shp = config.ruta("raw") / "centros_poblados_ign" / "CCPP_IGN100K.shp"
    g = gpd.read_file(shp)
    g = g[g["DEP"].apply(_norm_name) == _norm_name(dept["nombre"])].copy()

    # cp_id = OBJECTID, no CÓDIGO: 72,388/136,587 (53%) de los centros
    # poblados del shapefile nacional del IGN NO tienen CÓDIGO oficial
    # (columna nula). Usar CÓDIGO como id colapsaba todos los "nan" en un
    # solo string y la regla de duplicados los tumbaba como si fueran el
    # mismo registro. OBJECTID es el id de fila de ArcGIS: completo y único
    # en las 136,587 filas nacionales. CÓDIGO se conserva aparte, nulo donde
    # falte, para trazabilidad.
    out = gpd.GeoDataFrame(
        {
            "cp_id": "IGN-" + g["OBJECTID"].astype(str),
            "codigo_ign": g["CÓDIGO"],
            "nombre": g["NOM_POBLAD"],
            "departamento": g["DEP"],
            "provincia": g["PROV"],
            "distrito": g["DIST"],
            "cat_poblad": g["CAT_POBLAD"],
            "es_capital_distrito": g["CATEGORIA"].apply(_norm_name) == "CAPITAL DE DISTRITO",
            "geometry": g.geometry,
        },
        crs=g.crs or "EPSG:4326",
    ).to_crs("EPSG:4326")
    log.info("Centros poblados IGN cargados para %s: %d puntos", rol, len(out))
    return out


def load_poblacion_distrital(deptos: list[str]) -> pd.DataFrame:
    """Agrega población RENIEC (todas las edades/sexos) a nivel distrito para
    los departamentos dados. Lee en chunks: el CSV nacional pesa ~65MB, no
    hace falta cargarlo entero en memoria para 3 departamentos."""
    path = config.ruta("raw") / "poblacion_reniec.csv"
    deptos_norm = {_norm_name(d) for d in deptos}
    chunks = []
    for chunk in pd.read_csv(path, chunksize=200_000):
        sub = chunk[chunk["Departamento"].apply(_norm_name).isin(deptos_norm)]
        if not sub.empty:
            chunks.append(sub)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["UBIGEO_INEI", "Departamento", "Provincia", "Distrito", "Cantidad"]
    )
    agg = (
        df.groupby(["UBIGEO_INEI", "Departamento", "Provincia", "Distrito"], as_index=False)["Cantidad"]
        .sum()
        .rename(columns={"Cantidad": "poblacion_distrital"})
    )
    agg["ubigeo"] = agg["UBIGEO_INEI"].astype(str).str.zfill(6)
    return agg


def distribute_population(centros: gpd.GeoDataFrame, poblacion_distrital: pd.DataFrame) -> gpd.GeoDataFrame:
    """Reparte la población distrital de RENIEC entre los centros poblados de
    cada distrito, ponderando por tipo de asentamiento (`cat_poblad`) — ver
    `_PESO_POBLACION_POR_TIPO`. Aproximación documentada: no existe fuente
    oficial de población por centro poblado descargable en bloque."""
    centros = centros.copy()
    peso = centros["cat_poblad"].apply(_norm_name).map(_PESO_POBLACION_POR_TIPO).fillna(_PESO_POBLACION_DEFAULT)
    peso = peso * np.where(centros["es_capital_distrito"], _BONUS_CAPITAL_DISTRITO, 1)
    centros["_peso"] = peso

    pob = poblacion_distrital.copy()
    pob["_dep_n"] = pob["Departamento"].apply(_norm_name)
    pob["_prov_n"] = pob["Provincia"].apply(_norm_name)
    pob["_dist_n"] = pob["Distrito"].apply(_norm_name)
    centros["_dep_n"] = centros["departamento"].apply(_norm_name)
    centros["_prov_n"] = centros["provincia"].apply(_norm_name)
    centros["_dist_n"] = centros["distrito"].apply(_norm_name)

    peso_total_distrito = centros.groupby(["_dep_n", "_prov_n", "_dist_n"])["_peso"].transform("sum")
    centros["_peso_share"] = centros["_peso"] / peso_total_distrito.replace(0, np.nan)

    merged = centros.merge(
        pob[["_dep_n", "_prov_n", "_dist_n", "poblacion_distrital"]],
        on=["_dep_n", "_prov_n", "_dist_n"], how="left",
    )
    merged["poblacion"] = (merged["_peso_share"] * merged["poblacion_distrital"]).round().fillna(0).astype(int)
    n_sin_match = merged["poblacion_distrital"].isna().sum()
    if n_sin_match:
        log.warning(
            "%d/%d centros poblados sin match de distrito en RENIEC (nombre no coincide) -> poblacion=0",
            n_sin_match, len(merged),
        )
    return merged.drop(columns=["_peso", "_dep_n", "_prov_n", "_dist_n", "_peso_share", "poblacion_distrital"])


def sample_demand_if_needed(
    gdf: gpd.GeoDataFrame, max_points: int, seed: int, strata_col: str = "distrito"
) -> tuple[gpd.GeoDataFrame, dict]:
    """Muestreo estratificado por distrito si se excede el tope de puntos de
    demanda del issue (config.md: demanda_max_puntos). Preserva
    representación proporcional de cada distrito en vez de truncar."""
    if len(gdf) <= max_points:
        return gdf, {"muestreado": False, "n_original": len(gdf), "n_final": len(gdf)}
    frac = max_points / len(gdf)
    rng = np.random.default_rng(seed)
    parts = []
    for _, sub in gdf.groupby(strata_col):
        n = min(len(sub), max(1, round(len(sub) * frac)))
        idx = rng.choice(sub.index.to_numpy(), size=n, replace=False)
        parts.append(sub.loc[idx])
    sampled = pd.concat(parts)
    report = {
        "muestreado": True, "n_original": len(gdf), "n_final": len(sampled),
        "estrategia": f"estratificado por {strata_col}, fracción {frac:.3f}, seed {seed}",
    }
    log.info("Muestreo aplicado: %d -> %d puntos (%s)", len(gdf), len(sampled), report["estrategia"])
    return sampled, report


def build_real_dataset(rol: str, force: bool = False) -> tuple[Path, Path]:
    """Orquesta la carga real (facilities + demanda con población repartida
    + muestreo si aplica) y la deja en el mismo esquema/ubicación que
    generate_synthetic_dataset, lista para validation.py."""
    out_dir = config.ruta("raw") / "real" / rol
    demanda_path = out_dir / "demanda.parquet"
    facilities_path = out_dir / "facilities.parquet"
    if demanda_path.exists() and facilities_path.exists() and not force:
        log.info("SKIP (real ya cargado) para %s", rol)
        return demanda_path, facilities_path

    facilities = load_renipress(rol)
    centros = load_centros_poblados(rol)
    poblacion = load_poblacion_distrital([config.departamentos()[rol]["nombre"]])
    demanda = distribute_population(centros, poblacion)
    # Sin fuente de altitud integrada (requeriría un DEM) — se deja NaN de
    # forma explícita; metrics.altitude_access_cross reporta n=0 en vez de
    # fallar o inventar un valor.
    demanda["altitud_m"] = np.nan
    demanda, sample_report = sample_demand_if_needed(
        demanda, config.demanda_max_puntos() // len(config.departamentos()), config.sintetico_cfg()["seed"]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    demanda.to_parquet(demanda_path)
    facilities.to_parquet(facilities_path)
    log.info(
        "Real cargado para %s: %d demanda (%s), %d facilities",
        rol, len(demanda), sample_report, len(facilities),
    )
    return demanda_path, facilities_path


# --------------------------------------------------------------------------
# Datos sintéticos — validan acquisition -> validation -> routing -> metrics
# -> app de punta a punta mientras las fuentes reales bajan o se resuelven
# bloqueos de red.
# --------------------------------------------------------------------------

def _random_points_in_bbox(bbox: dict, n: int, rng: np.random.Generator) -> gpd.GeoSeries:
    lons = rng.uniform(bbox["west"], bbox["east"], n)
    lats = rng.uniform(bbox["south"], bbox["north"], n)
    return gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lons, lats)], crs="EPSG:4326")


def generate_synthetic_dataset(rol: str, force: bool = False) -> tuple[Path, Path]:
    """Genera demanda y facilities falsas dentro del bbox del departamento `rol`.

    Incluye deliberadamente algunos registros "sucios" (nulos, lat/lon
    invertidos, duplicados, fuera de bbox) para que validation.py tenga algo
    real que limpiar incluso en modo sintético.
    """
    dept = config.departamentos()[rol]
    sint = config.sintetico_cfg()
    out_dir = config.ruta("raw") / "synthetic" / rol
    demanda_path = out_dir / "demanda.parquet"
    facilities_path = out_dir / "facilities.parquet"

    if demanda_path.exists() and facilities_path.exists() and not force:
        log.info("SKIP (sintético ya existe) para %s", rol)
        return demanda_path, facilities_path

    rng = np.random.default_rng(sint["seed"])
    bbox = dept["bbox"]
    n_dem = sint["n_demanda"]
    n_fac = sint["n_facilities"]

    # --- demanda (centros poblados) ---
    demanda_pts = _random_points_in_bbox(bbox, n_dem, rng)
    demanda = gpd.GeoDataFrame(
        {
            "cp_id": [f"{rol[:3].upper()}-CP-{i:04d}" for i in range(n_dem)],
            "nombre": [f"Centro Poblado {i}" for i in range(n_dem)],
            "poblacion": rng.integers(20, 5000, n_dem),
            "departamento": dept["nombre"],
            "geometry": demanda_pts,
        },
        crs="EPSG:4326",
    )
    # altitud sintética: más alta cuanto más al sur/oeste dentro del bbox (proxy
    # burdo, solo para poder probar el cross-analysis altitud vs. acceso)
    demanda["altitud_m"] = rng.uniform(50, 4500, n_dem).round(0)
    # distrito sintético: binning espacial 3x3 sobre el bbox, solo para poder
    # probar agregaciones a nivel distrito en metrics.py sin polígonos reales
    lon_bin = pd.cut(demanda.geometry.x, bins=3, labels=["A", "B", "C"])
    lat_bin = pd.cut(demanda.geometry.y, bins=3, labels=["1", "2", "3"])
    demanda["distrito"] = dept["nombre"][:3].upper() + "-" + lat_bin.astype(str) + lon_bin.astype(str)
    demanda["provincia"] = dept["nombre"][:3].upper() + "-PROV-" + lat_bin.astype(str)

    # --- facilities de salud ---
    categorias = ["I-1", "I-2", "I-3", "I-4"] + sorted(config.categorias_resolutivas())
    fac_pts = _random_points_in_bbox(bbox, n_fac, rng)
    facilities = gpd.GeoDataFrame(
        {
            "facility_id": [f"{rol[:3].upper()}-EESS-{i:04d}" for i in range(n_fac)],
            "nombre": [f"Establecimiento {i}" for i in range(n_fac)],
            "categoria": rng.choice(categorias, n_fac),
            "institucion": rng.choice(["MINSA", "ESSALUD", "PRIVADO"], n_fac, p=[0.7, 0.2, 0.1]),
            "activo": rng.choice([True, True, True, False], n_fac),
            "capacidad": rng.integers(1, 20, n_fac),  # camas/quirófanos, usado por 2SFCA
            "departamento": dept["nombre"],
            "geometry": fac_pts,
        },
        crs="EPSG:4326",
    )

    demanda = _inject_dirty_records(demanda, rng)
    facilities = _inject_dirty_records(facilities, rng)

    out_dir.mkdir(parents=True, exist_ok=True)
    demanda.to_parquet(demanda_path)
    facilities.to_parquet(facilities_path)
    log.info("Sintético generado para %s: %d demanda, %d facilities", rol, len(demanda), len(facilities))
    return demanda_path, facilities_path


def _inject_dirty_records(gdf: gpd.GeoDataFrame, rng: np.random.Generator) -> gpd.GeoDataFrame:
    """Ensucia ~8% de las filas para que validation.py tenga trabajo real que hacer."""
    gdf = gdf.copy()
    n = len(gdf)
    if n < 5:
        return gdf
    idx = rng.choice(gdf.index, size=max(1, n // 12), replace=False)
    chunks = np.array_split(idx, 4)

    # 1) coordenada nula/cero
    for i in chunks[0]:
        gdf.at[i, "geometry"] = Point(0.0, 0.0)
    # 2) lat/lon invertidos (swap dentro de bbox real -> sale del bbox tal cual)
    for i in chunks[1]:
        pt = gdf.at[i, "geometry"]
        gdf.at[i, "geometry"] = Point(pt.y, pt.x)
    # 3) duplicado exacto de la fila anterior (mismo id, para dedup)
    for i in chunks[2]:
        pos = gdf.index.get_loc(i)
        if pos > 0:
            gdf.iloc[pos, gdf.columns.get_loc("geometry")] = gdf.iloc[pos - 1]["geometry"]
    # 4) fuera del bbox de Perú (para probar el filtro de bbox nacional)
    for i in chunks[3]:
        gdf.at[i, "geometry"] = Point(10.0, 10.0)

    return gdf


def generate_synthetic_all(force: bool = False) -> None:
    for rol in config.departamentos():
        generate_synthetic_dataset(rol, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="genera datos falsos para los 3 departamentos")
    parser.add_argument("--real", action="store_true", help="descarga y parsea RENIPRESS/centros poblados IGN/población RENIEC reales")
    parser.add_argument("--force", action="store_true", help="ignora idempotencia y vuelve a generar/descargar")
    args = parser.parse_args()

    if not args.synthetic and not args.real:
        parser.error("especificar --synthetic y/o --real")

    if args.real:
        download_real(force=args.force)
        for rol in config.departamentos():
            build_real_dataset(rol, force=args.force)
    if args.synthetic:
        generate_synthetic_all(force=args.force)


if __name__ == "__main__":
    main()
