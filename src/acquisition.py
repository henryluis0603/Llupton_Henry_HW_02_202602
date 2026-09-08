"""Fase 1a — Adquisición de datos.

Idempotente: si el archivo destino ya existe, no vuelve a descargar (a menos
que `force=True`). Cada descarga real está envuelta en try/except: si la
fuente no responde (bloqueo de IP, mantenimiento, cambio de URL), se registra
en logs/acquisition.log y el pipeline puede seguir con `--synthetic` en vez
de morir a mitad de camino.

Uso:
    python -m src.acquisition --synthetic          # genera datos falsos, los 3 deptos

TODO: acquisition real (RENIPRESS, centros poblados, población) — mientras se
resuelven las fuentes exactas y el acceso de red, el pipeline se prueba de
punta a punta con datos sintéticos.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
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
    parser.add_argument("--force", action="store_true", help="ignora idempotencia y vuelve a generar")
    args = parser.parse_args()

    if not args.synthetic:
        parser.error("especificar --synthetic")

    generate_synthetic_all(force=args.force)


if __name__ == "__main__":
    main()
