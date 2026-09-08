"""Single source of truth para el pipeline: parsea el bloque YAML de config.md.

Ningún otro módulo debe hardcodear departamentos, thresholds, categorías o
rutas — todos importan `CONFIG` desde aquí.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_MD_PATH = REPO_ROOT / "config.md"

_YAML_BLOCK_RE = re.compile(r"```yaml\n(.*?)\n```", re.DOTALL)


def _extract_yaml_block(md_text: str) -> str:
    match = _YAML_BLOCK_RE.search(md_text)
    if not match:
        raise ValueError(
            f"No se encontró bloque ```yaml``` en {CONFIG_MD_PATH}. "
            "config.md debe seguir siendo la fuente de verdad machine-readable."
        )
    return match.group(1)


@lru_cache(maxsize=1)
def load_config(config_path: Path = CONFIG_MD_PATH) -> dict[str, Any]:
    text = Path(config_path).read_text(encoding="utf-8")
    yaml_block = _extract_yaml_block(text)
    return yaml.safe_load(yaml_block)


CONFIG = load_config()


def departamentos() -> dict[str, dict[str, Any]]:
    return CONFIG["departamentos"]


def departamento_nombres() -> list[str]:
    return [d["nombre"] for d in CONFIG["departamentos"].values()]


def bbox_for(rol: str) -> dict[str, float]:
    """rol: 'costa' | 'andino' | 'amazonico'."""
    return CONFIG["departamentos"][rol]["bbox"]


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """EPSG UTM WGS84 correspondiente a un punto. Perú cae en las zonas
    17S/18S/19S (EPSG 32717/32718/32719) — se usa para cualquier cálculo de
    distancia en metros, en vez de la aproximación "1 grado ~ 111 km" (que
    ignora que un grado de longitud se encoge con cos(latitud))."""
    zone = int((lon + 180) / 6) + 1
    return (32700 if lat < 0 else 32600) + zone


def utm_epsg_for_bbox(bbox: dict[str, float]) -> int:
    lon = (bbox["west"] + bbox["east"]) / 2
    lat = (bbox["south"] + bbox["north"]) / 2
    return utm_epsg_for_lonlat(lon, lat)


def categorias_resolutivas() -> set[str]:
    return set(CONFIG["categorias_resolutivas"])


def normalizacion_categorias() -> dict[str, str]:
    return CONFIG.get("normalizacion_categorias", {})


def peru_bbox() -> dict[str, float]:
    return CONFIG["peru_bbox"]


def thresholds_minutos() -> list[int]:
    return CONFIG["thresholds_minutos"]


def ponderacion_col() -> str:
    return CONFIG["ponderacion"]


def routing_cfg() -> dict[str, Any]:
    return CONFIG["routing"]


def ruta(clave: str) -> Path:
    """Devuelve una ruta declarada en config.md, resuelta contra REPO_ROOT."""
    rutas = CONFIG["rutas"]
    if clave in rutas:
        return REPO_ROOT / rutas[clave]
    raise KeyError(f"'{clave}' no está declarado en rutas: dentro de config.md")


def sintetico_cfg() -> dict[str, Any]:
    return CONFIG["sintetico"]


def demanda_max_puntos() -> int:
    return CONFIG["demanda_max_puntos"]


def poblacion_minima_urbana() -> int:
    return CONFIG["clasificacion_urbano_rural"]["poblacion_minima"]


def dos_sfca_cfg() -> dict[str, Any]:
    return CONFIG["dos_sfca"]
