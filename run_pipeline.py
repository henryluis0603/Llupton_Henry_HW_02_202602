"""Orquestador: Fase 1 -> 2 -> 3 -> exports para data/outputs/, que app.py
(Fase 4) lee sin volver a calcular nada. Módulos en src/ se importan; este
script solo encadena llamadas y decide qué guardar.

Uso:
    python run_pipeline.py --synthetic          # pipeline completo con datos falsos
    python run_pipeline.py --synthetic --force  # ignora cachés y recalcula todo
"""
from __future__ import annotations

import argparse
import logging

import geopandas as gpd
import pandas as pd

from src import acquisition, config, export, metrics, routing, validation

log = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROFILES = config.routing_cfg()["perfiles"]


def run_department(rol: str, force: bool = False) -> dict:
    log.info("=== Departamento: %s ===", rol)
    dem_path, fac_path = acquisition.generate_synthetic_dataset(rol, force=force)
    demanda = gpd.read_parquet(dem_path)
    facilities = gpd.read_parquet(fac_path)

    demanda, rep_dem = validation.validate_dataset(
        demanda,
        dataset_name=f"{rol}_demanda",
        id_col="cp_id",
        text_cols=["nombre", "departamento", "distrito", "provincia"],
    )
    facilities, rep_fac = validation.validate_dataset(
        facilities,
        dataset_name=f"{rol}_facilities",
        id_col="facility_id",
        text_cols=["nombre", "categoria", "institucion", "departamento"],
    )
    facilities = validation.normalize_categoria(facilities)

    facilities_resolutivas = facilities[(facilities["resolutivo"]) & (facilities["activo"])].copy()
    log.info(
        "%s: %d demanda validos, %d facilities validas, %d resolutivas activas",
        rol, len(demanda), len(facilities), len(facilities_resolutivas),
    )

    if facilities_resolutivas.empty:
        log.warning("%s: 0 facilities resolutivas activas tras validación — se omite ruteo", rol)
        return {
            "rol": rol, "demanda": demanda, "facilities": facilities,
            "reports": [rep_dem, rep_fac], "matrices": {}, "snap_reports": {},
        }

    matrices: dict[str, pd.DataFrame] = {}
    snap_reports: dict[str, dict] = {}
    graph_sources: dict[str, str] = {}
    G_drive = None
    fac_snapped_drive = None

    for profile in PROFILES:
        G, source = routing.get_graph(rol, profile, force=force)
        graph_sources[profile] = source

        dem_snapped, dem_snap_report = routing.snap_points(demanda, G, "cp_id")
        fac_snapped, fac_snap_report = routing.snap_points(facilities_resolutivas, G, "facility_id")
        snap_reports[profile] = {"demanda": dem_snap_report, "facilities": fac_snap_report}

        cache_path = config.ruta("routing_cache_dir") / rol / f"matrix_{profile}.parquet"
        matrix = routing.travel_time_matrix(
            G, dem_snapped, fac_snapped, "cp_id", "facility_id", cache_path, force=force
        )
        matrices[profile] = matrix
        log.info("%s/%s: grafo=%s, matriz=%d filas", rol, profile, source, len(matrix))

        if profile == "drive":
            G_drive, fac_snapped_drive = G, fac_snapped

    isochrones = routing.department_isochrones(G_drive, fac_snapped_drive, "facility_id")

    nearest_drive = routing.nearest_facility(matrices["drive"], "cp_id", "facility_id")
    demanda_metrics = demanda.merge(
        nearest_drive.rename(columns={"facility_id": "facility_mas_cercana", "t_min": "t_min"}),
        on="cp_id", how="left",
    )

    mode_comparison = routing.compare_modes(matrices, "cp_id", "facility_id")

    return {
        "rol": rol,
        "demanda": demanda,
        "facilities": facilities,
        "demanda_metrics": demanda_metrics,
        "matrices": matrices,
        "mode_comparison": mode_comparison,
        "isochrones": isochrones,
        "snap_reports": snap_reports,
        "graph_sources": graph_sources,
        "reports": [rep_dem, rep_fac],
    }


def compute_and_export_metrics(dep_results: list[dict]) -> None:
    out_dir = config.ruta("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_demanda_metrics = pd.concat(
        [r["demanda_metrics"] for r in dep_results if "demanda_metrics" in r], ignore_index=True
    )
    all_demanda_metrics.to_parquet(out_dir / "demanda_metrics.parquet")

    # Facilities YA validadas/normalizadas (con columna 'resolutivo') — el
    # dashboard debe leer esto, nunca data/raw directo.
    all_facilities = pd.concat([r["facilities"] for r in dep_results], ignore_index=True)
    all_facilities.to_parquet(out_dir / "facilities_validated.parquet")

    coverage = metrics.coverage_bands(all_demanda_metrics)
    coverage.to_csv(out_dir / "coverage_bands.csv", index=False)
    export.to_latex_table(
        coverage, "coverage_bands",
        caption="Población por banda de tiempo de acceso (ponderado)",
        label="coverage_bands",
    )

    for group_col, fname in [("distrito", "access_by_distrito"), ("provincia", "access_by_provincia"), ("departamento", "access_by_departamento")]:
        wmean = metrics.weighted_mean_access(all_demanda_metrics, group_col)
        wmean.to_csv(out_dir / f"{fname}.csv", index=False)

    critical = metrics.critical_districts(all_demanda_metrics, "distrito", top_n=10)
    critical.to_csv(out_dir / "critical_districts.csv", index=False)
    export.to_latex_table(
        critical, "critical_districts",
        caption="Distritos con peor tiempo de acceso ponderado por población",
        label="critical_districts",
    )

    gini = metrics.gini_weighted(all_demanda_metrics["t_min"].to_numpy(), all_demanda_metrics["poblacion"].to_numpy())
    urban_rural = metrics.urban_rural_contrast(all_demanda_metrics)
    urban_rural.to_csv(out_dir / "urban_rural_contrast.csv", index=False)

    altitude_cross = metrics.altitude_access_cross(all_demanda_metrics)

    summary = {
        "gini_t_min_ponderado": gini,
        "n_demanda_total": int(len(all_demanda_metrics)),
        "poblacion_total": int(all_demanda_metrics["poblacion"].sum()),
        "cross_analysis_altitud": altitude_cross,
    }
    pd.Series(summary).to_json(out_dir / "summary.json", indent=2, force_ascii=False)

    mode_comparisons = pd.concat(
        [r["mode_comparison"] for r in dep_results if "mode_comparison" in r], ignore_index=True
    )
    mode_comparisons.to_parquet(out_dir / "mode_comparison.parquet")
    ratio_summary = mode_comparisons[[c for c in mode_comparisons.columns if c.startswith("ratio_")]].describe().T.reset_index()
    ratio_summary = ratio_summary.rename(columns={"index": "perfil"})
    export.to_latex_table(
        ratio_summary[["perfil", "mean", "50%", "max"]], "mode_comparison",
        caption="Ratio de tiempo de viaje (modo / auto) por perfil",
        label="mode_comparison",
    )

    # Innovación 2 — 2SFCA, por departamento (facilities no se comparten entre deptos)
    sfca_frames = []
    for r in dep_results:
        if "matrices" not in r or "drive" not in r.get("matrices", {}):
            continue
        facilities_resolutivas = r["facilities"][(r["facilities"]["resolutivo"]) & (r["facilities"]["activo"])]
        sfca = metrics.two_step_floating_catchment_area(
            r["matrices"]["drive"], r["demanda"], facilities_resolutivas, "cp_id", "facility_id",
        )
        sfca["departamento"] = r["rol"]
        sfca_frames.append(sfca)
    if sfca_frames:
        pd.concat(sfca_frames, ignore_index=True).to_parquet(out_dir / "sfca_2step.parquet")

    # Innovación 1 — isócronas, por departamento (perfil drive)
    iso_frames = [r["isochrones"] for r in dep_results if "isochrones" in r and not r["isochrones"].empty]
    if iso_frames:
        all_isochrones = gpd.GeoDataFrame(pd.concat(iso_frames, ignore_index=True), crs=iso_frames[0].crs)
        all_isochrones.to_parquet(out_dir / "isochrones.parquet")

    log.info("Métricas exportadas a %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.synthetic:
        parser.error("por ahora solo --synthetic está implementado end-to-end en este script")

    dep_results = [run_department(rol, force=args.force) for rol in config.departamentos()]

    all_reports = [rep for r in dep_results for rep in r["reports"]]
    quality_path = validation.write_quality_report(all_reports)
    log.info("Quality report: %s", quality_path)

    snap_summary = {r["rol"]: r.get("snap_reports", {}) for r in dep_results}
    graph_sources = {r["rol"]: r.get("graph_sources", {}) for r in dep_results}
    (config.ruta("outputs") / "snap_report.json").write_text(
        pd.Series({"snap": snap_summary, "graph_sources": graph_sources}).to_json(indent=2, force_ascii=False),
        encoding="utf-8",
    )

    compute_and_export_metrics(dep_results)
    log.info("Pipeline completo.")


if __name__ == "__main__":
    main()
