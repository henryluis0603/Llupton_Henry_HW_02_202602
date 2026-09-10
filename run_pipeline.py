"""Orquestador: Fase 1 -> 2 -> 3 -> exports para data/outputs/, que app.py
(Fase 4) lee sin volver a calcular nada. Módulos en src/ se importan; este
script solo encadena llamadas y decide qué guardar.

Uso:
    python run_pipeline.py --synthetic          # pipeline completo con datos falsos
    python run_pipeline.py --real               # pipeline completo con RENIPRESS/IGN/RENIEC reales
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


def run_department(rol: str, force: bool = False, real: bool = False) -> dict:
    log.info("=== Departamento: %s ===", rol)
    distritos = None
    if real:
        dem_path, fac_path, dist_path = acquisition.build_real_dataset(rol, force=force)
        distritos = gpd.read_parquet(dist_path)
    else:
        dem_path, fac_path = acquisition.generate_synthetic_dataset(rol, force=force)
    demanda = gpd.read_parquet(dem_path)
    facilities = gpd.read_parquet(fac_path)

    # Regla de validación 4 (punto fuera de su polígono distrital declarado)
    # solo corre de verdad en modo real: los datos sintéticos no tienen
    # UBIGEO propio ni polígonos distritales que cruzar.
    demanda, rep_dem = validation.validate_dataset(
        demanda,
        dataset_name=f"{rol}_demanda",
        id_col="cp_id",
        text_cols=["nombre", "departamento", "distrito", "provincia"],
        point_district_col="ubigeo" if real else None,
        polygon_gdf=distritos,
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
            "rol": rol, "demanda": demanda, "facilities": facilities, "distritos": distritos,
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

        # El grafo (data/processed/routing_cache/<rol>/<perfil>.graphml) se
        # comparte entre sintético y real: es la misma red vial. La MATRIZ no
        # — sus índices son los IDs de demanda/facility de cada fuente, así
        # que va en un subdirectorio separado para no leer una matriz
        # sintética al correr --real (o viceversa) sin --force.
        fuente_dir = "real" if real else "synthetic"
        cache_path = config.ruta("routing_cache_dir") / rol / fuente_dir / f"matrix_{profile}.parquet"
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

    # Discusión (issue #186, Fase 5): línea recta vs. red vial real. Se
    # calcula ANTES del fallback de abajo porque el factor de desvío se
    # deriva empíricamente de los puntos que sí tienen ruta.
    straight_vs_network = metrics.straight_line_vs_network(
        matrices["drive"], demanda, facilities_resolutivas, "cp_id", "facility_id"
    )

    # Fallback para puntos no ruteables (issue #186, Fase 2: "fallback
    # documentado" + "factor de desvío empíricamente justificado") — en vez
    # de dejar t_min indefinido para un punto en un componente vial
    # desconectado de toda facility, se estima vía distancia recta x factor
    # de desvío mediano observado en los puntos SÍ ruteados de este mismo
    # departamento (la curvatura de la red varía demasiado por geografía
    # para usar un factor nacional único, ver \S5.1 del reporte).
    factor_desvio = metrics.circuity_factor(straight_vs_network)
    demanda_metrics["t_min_estimado"] = demanda_metrics["t_min"].isna()
    n_faltantes = int(demanda_metrics["t_min_estimado"].sum())
    if n_faltantes > 0:
        faltantes = demanda_metrics[demanda_metrics["t_min_estimado"]]
        fallback = metrics.estimate_fallback_time(
            faltantes, facilities_resolutivas, "cp_id", "facility_id", factor_desvio
        )
        demanda_metrics = demanda_metrics.merge(
            fallback[["cp_id", "facility_id_fallback", "t_min_fallback"]], on="cp_id", how="left"
        )
        mask = demanda_metrics["t_min_estimado"]
        demanda_metrics.loc[mask, "t_min"] = demanda_metrics.loc[mask, "t_min_fallback"]
        demanda_metrics.loc[mask, "facility_mas_cercana"] = demanda_metrics.loc[mask, "facility_id_fallback"]
        demanda_metrics = demanda_metrics.drop(columns=["facility_id_fallback", "t_min_fallback"])
        log.info(
            "%s: %d/%d puntos sin ruta real -> t_min estimado (factor de desvío = %.2f)",
            rol, n_faltantes, len(demanda_metrics), factor_desvio,
        )

    return {
        "rol": rol,
        "demanda": demanda,
        "facilities": facilities,
        "distritos": distritos,
        "demanda_metrics": demanda_metrics,
        "matrices": matrices,
        "mode_comparison": mode_comparison,
        "straight_vs_network": straight_vs_network,
        "factor_desvio": factor_desvio,
        "isochrones": isochrones,
        "snap_reports": snap_reports,
        "graph_sources": graph_sources,
        "reports": [rep_dem, rep_fac],
    }


def generate_report_figures(
    all_demanda_metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    mode_comparisons: pd.DataFrame,
    dep_results: list[dict],
) -> None:
    """Figuras vectoriales (PDF) para report/main.tex — generadas por el
    pipeline en cada corrida, no capturas de pantalla."""
    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    # Fig 1 — bandas de cobertura
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(coverage["banda"], coverage["poblacion"], color="#4c78a8")
    ax.set_ylabel("Población")
    ax.set_title("Población por banda de tiempo de acceso")
    for i, v in enumerate(coverage["pct_poblacion"]):
        ax.text(i, coverage["poblacion"].iloc[i], f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    export.save_figure(fig, "coverage_bands", fmt="pdf")
    plt.close(fig)

    # Fig 2 — distribución de t_min por departamento
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for depto, grupo in all_demanda_metrics.groupby("departamento"):
        ax.hist(grupo["t_min"].dropna(), bins=30, alpha=0.5, label=depto)
    ax.set_xlabel("t_min (minutos)")
    ax.set_ylabel("N. de puntos de demanda")
    ax.set_title("Distribución de tiempo de acceso por departamento")
    ax.legend(fontsize=8)
    plt.tight_layout()
    export.save_figure(fig, "distribucion_t_min", fmt="pdf")
    plt.close(fig)

    # Fig 3 — ratio modo/auto por departamento (evidencia del artefacto
    # sintético: constante en deptos 100% sintéticos, con variación real
    # solo donde el grafo es OSM real — ver Discusión/Limitaciones)
    mc = mode_comparisons.merge(all_demanda_metrics[["cp_id", "departamento"]], on="cp_id", how="left")
    resumen_ratio = mc.groupby("departamento")[["ratio_walk_drive", "ratio_bike_drive"]].median()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    resumen_ratio.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518"])
    ax.set_ylabel("Ratio mediano (modo / auto)")
    ax.set_title("Ratio de tiempo modo/auto, por departamento")
    plt.xticks(rotation=0)
    plt.tight_layout()
    export.save_figure(fig, "ratio_modos", fmt="pdf")
    plt.close(fig)

    # Fig 4 — isócronas de un establecimiento resolutivo de ejemplo
    iso_frames = [r["isochrones"] for r in dep_results if "isochrones" in r and not r["isochrones"].empty]
    if iso_frames:
        ejemplo = iso_frames[0]
        facility_id_col = [c for c in ejemplo.columns if c not in ("threshold_min", "geometry")][0]
        ejemplo_id = ejemplo[facility_id_col].iloc[0]
        sub = ejemplo[ejemplo[facility_id_col] == ejemplo_id].sort_values("threshold_min", ascending=False)
        rol_ejemplo = next(r["rol"] for r in dep_results if r.get("isochrones") is ejemplo)
        colors = {30: "#31a354", 60: "#fdae61", 120: "#d7191c"}

        fig, ax = plt.subplots(figsize=(6, 6))
        demanda_dept = all_demanda_metrics[all_demanda_metrics["departamento"] == config.departamentos()[rol_ejemplo]["nombre"]]
        gpd.GeoSeries(demanda_dept.geometry).plot(ax=ax, markersize=6, color="lightgray", zorder=1)
        for _, row in sub.iterrows():
            gpd.GeoSeries([row.geometry]).plot(
                ax=ax, alpha=0.35, color=colors.get(row["threshold_min"], "gray"),
                edgecolor="black", linewidth=0.5, zorder=2, label=f"{row['threshold_min']} min",
            )
        ax.set_title(f"Isócronas — {ejemplo_id} ({rol_ejemplo})")
        ax.legend()
        plt.tight_layout()
        export.save_figure(fig, "isocronas_ejemplo", fmt="pdf")
        plt.close(fig)

    log.info("Figuras exportadas a %s", config.ruta("report_figures"))


def compute_and_export_metrics(dep_results: list[dict], real: bool = False) -> None:
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

    # Choropleth real por distrito (sesión 6 del curso: Geopandas1_clean.ipynb
    # usa este mismo shapefile INEI para un choropleth de COVID por distrito)
    # — solo en modo real, donde demanda trae 'ubigeo' propio del cruce con
    # el shapefile de distritos.
    distritos_frames = [r["distritos"] for r in dep_results if r.get("distritos") is not None]
    if real and distritos_frames and "ubigeo" in all_demanda_metrics.columns:
        all_distritos = gpd.GeoDataFrame(pd.concat(distritos_frames, ignore_index=True), crs=distritos_frames[0].crs)
        acceso_por_ubigeo = metrics.weighted_mean_access(all_demanda_metrics, "ubigeo").rename(columns={"ubigeo": "ubigeo_join"})
        poblacion_por_ubigeo = all_demanda_metrics.groupby("ubigeo", as_index=False)["poblacion"].sum()
        choropleth = all_distritos.merge(
            acceso_por_ubigeo, left_on="ubigeo", right_on="ubigeo_join", how="left"
        ).drop(columns="ubigeo_join").merge(poblacion_por_ubigeo, on="ubigeo", how="left")
        choropleth.to_parquet(out_dir / "distritos_choropleth.parquet")

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

    n_estimado = int(all_demanda_metrics["t_min_estimado"].sum()) if "t_min_estimado" in all_demanda_metrics.columns else 0
    factor_desvio_por_depto = {r["rol"]: r["factor_desvio"] for r in dep_results if "factor_desvio" in r}

    summary = {
        "fuente": "real" if real else "synthetic",
        "gini_t_min_ponderado": gini,
        "n_demanda_total": int(len(all_demanda_metrics)),
        "poblacion_total": int(all_demanda_metrics["poblacion"].sum()),
        "n_t_min_estimado_por_factor_desvio": n_estimado,
        "factor_desvio_por_departamento": factor_desvio_por_depto,
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

    # Discusión: línea recta vs. red vial real, por departamento.
    svn_frames = [r["straight_vs_network"].assign(departamento=r["rol"]) for r in dep_results if "straight_vs_network" in r]
    if svn_frames:
        all_svn = pd.concat(svn_frames, ignore_index=True)
        all_svn.to_parquet(out_dir / "straight_line_vs_network.parquet")
        svn_summary = (
            all_svn.groupby("departamento")
            .apply(lambda g: pd.Series({
                "pct_facility_distinta": 100 * (~g["coincide"]).mean(),
                "penalidad_min_mediana": g.loc[~g["coincide"], "penalidad_min"].median(),
            }), include_groups=False)
            .reset_index()
        )
        svn_summary.to_csv(out_dir / "straight_line_vs_network_summary.csv", index=False)
        export.to_latex_table(
            svn_summary, "straight_line_vs_network",
            caption="Línea recta vs. red: \\% de puntos donde la facility más cercana cambia, y minutos de más si se usara la sugerida por línea recta",
            label="straight_line_vs_network",
        )

    generate_report_figures(all_demanda_metrics, coverage, mode_comparisons, dep_results)

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
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.synthetic and not args.real:
        parser.error("especificar --synthetic o --real")
    if args.synthetic and args.real:
        parser.error("usar --synthetic o --real, no ambos en la misma corrida")

    if args.real:
        acquisition.download_real(force=args.force)
    dep_results = [run_department(rol, force=args.force, real=args.real) for rol in config.departamentos()]

    all_reports = [rep for r in dep_results for rep in r["reports"]]
    quality_path = validation.write_quality_report(all_reports)
    log.info("Quality report: %s", quality_path)

    snap_summary = {r["rol"]: r.get("snap_reports", {}) for r in dep_results}
    graph_sources = {r["rol"]: r.get("graph_sources", {}) for r in dep_results}
    (config.ruta("outputs") / "snap_report.json").write_text(
        pd.Series({"snap": snap_summary, "graph_sources": graph_sources}).to_json(indent=2, force_ascii=False),
        encoding="utf-8",
    )

    compute_and_export_metrics(dep_results, real=args.real)
    log.info("Pipeline completo.")


if __name__ == "__main__":
    main()
