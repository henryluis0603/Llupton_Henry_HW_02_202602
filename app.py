"""Fase 4 — Dashboard Streamlit. Solo LEE data/outputs/ precomputado por
run_pipeline.py: cero llamadas a motores de ruteo ni a APIs externas al
cargar. `streamlit run app.py` después de `pip install -r requirements.txt`.
"""
from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config

st.set_page_config(page_title="Acceso a salud resolutiva — Perú", layout="wide")

OUT = config.ruta("outputs")

# Paleta: secuencial de un solo hue (Viridis, colorblind-safe) para magnitud
# de tiempo de acceso; categórica de orden fijo para institución/categoría.
SEQ_SCALE = "Viridis"
CATEGORICAL = px.colors.qualitative.Set2


def _wmean_safe(g: pd.DataFrame) -> float:
    """t_min ponderado por población, SOLO entre puntos con ruta. `t_min`
    puede ser NaN cuando un punto queda en un componente vial desconectado
    de toda facility resolutiva (ver src/routing.nearest_facility) — sumar
    `t_min * poblacion` con NaN en el numerador (`.sum()` los ignora por
    defecto) pero dividir por la población TOTAL del grupo subestimaría el
    tiempo, tratando a la población sin ruta como si tardara 0 minutos.
    NaN (no división por cero) si nadie en el grupo tiene ruta."""
    g_valid = g.dropna(subset=["t_min"])
    w = g_valid["poblacion"].sum()
    return float((g_valid["t_min"] * g_valid["poblacion"]).sum() / w) if w > 0 else float("nan")


# --------------------------------------------------------------------- #
# Carga cacheada
# --------------------------------------------------------------------- #
@st.cache_data
def load_demanda_metrics() -> gpd.GeoDataFrame:
    return gpd.read_parquet(OUT / "demanda_metrics.parquet")


@st.cache_data
def load_facilities() -> gpd.GeoDataFrame:
    """Facilities YA validadas y normalizadas por Fase 1 (columna
    'resolutivo' incluida) — nunca se lee data/raw directo desde el dashboard."""
    path = OUT / "facilities_validated.parquet"
    return gpd.read_parquet(path) if path.exists() else gpd.GeoDataFrame()


@st.cache_data
def load_matrix_all() -> pd.DataFrame:
    """Matriz completa origen×facility (perfil drive), concatenada entre
    departamentos — la usa el simulador de escenario para recalcular sin
    tocar el motor de ruteo.

    Incluye tanto `matrix_drive.parquet` (facilities YA resolutivas, usada
    para el resto del dashboard) como `matrix_drive_candidatas.parquet`
    (facilities I-3/I-4 activas, calculada solo para que el simulador de
    upgrade de §6 tenga t_min hacia ellas — si no se concatenara, cualquier
    establecimiento candidato a mejorar quedaría fuera de la matriz y el
    simulador siempre mostraría 0 puntos mejorados).

    La matriz cacheada vive en un subdirectorio por fuente (real/synthetic)
    porque sus índices son los IDs de esa fuente — leer la fuente equivocada
    mezclaría IDs de demanda/facility sin dar error. `summary.json` dice cuál
    fue la fuente de la última corrida del pipeline.
    """
    summary_path = OUT / "summary.json"
    fuente = "synthetic"
    if summary_path.exists():
        fuente = json.loads(summary_path.read_text(encoding="utf-8")).get("fuente", "synthetic")

    frames = []
    for rol in config.departamentos():
        for fname in ("matrix_drive.parquet", "matrix_drive_candidatas.parquet"):
            p = config.ruta("routing_cache_dir") / rol / fuente / fname
            if p.exists():
                m = pd.read_parquet(p).reset_index()
                m["departamento_rol"] = rol
                frames.append(m)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data
def load_quality_report() -> dict:
    path = config.ruta("quality_report")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_mode_comparison() -> pd.DataFrame:
    p = OUT / "mode_comparison.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_sfca() -> pd.DataFrame:
    p = OUT / "sfca_2step.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_isochrones() -> gpd.GeoDataFrame:
    p = OUT / "isochrones.parquet"
    return gpd.read_parquet(p) if p.exists() else gpd.GeoDataFrame()


@st.cache_data
def load_choropleth() -> gpd.GeoDataFrame:
    """Polígonos distritales (INEI, vía sesión 6 del curso) con t_min
    ponderado ya unido — solo existe en modo real. En modo sintético no hay
    polígonos y el mapa cae al proxy de puntos. Se comprueba `fuente` en
    summary.json (igual que load_matrix_all): si la última corrida fue
    sintética, un distritos_choropleth.parquet de una corrida real anterior
    quedaría huérfano — mostraría distritos que no corresponden a la
    demanda actual — así que se ignora."""
    p = OUT / "distritos_choropleth.parquet"
    if not p.exists():
        return gpd.GeoDataFrame()
    summary_path = OUT / "summary.json"
    if summary_path.exists():
        fuente = json.loads(summary_path.read_text(encoding="utf-8")).get("fuente", "synthetic")
        if fuente != "real":
            return gpd.GeoDataFrame()
    return gpd.read_parquet(p)


demanda = load_demanda_metrics()
facilities = load_facilities()
quality_report = load_quality_report()
mode_comparison = load_mode_comparison()
sfca = load_sfca()
isochrones = load_isochrones()
choropleth = load_choropleth()

if demanda.empty:
    st.error(
        "No hay datos precomputados en data/outputs/. Corre primero:\n\n"
        "`python run_pipeline.py --synthetic`"
    )
    st.stop()

# --------------------------------------------------------------------- #
# Sidebar — filtros
# --------------------------------------------------------------------- #
st.sidebar.header("Filtros")
departamentos_disp = sorted(demanda["departamento"].dropna().unique())
sel_departamentos = st.sidebar.multiselect("Departamento", departamentos_disp, default=departamentos_disp)

provincias_disp = sorted(demanda.loc[demanda["departamento"].isin(sel_departamentos), "provincia"].dropna().unique())
sel_provincias = st.sidebar.multiselect("Provincia", provincias_disp, default=provincias_disp)

categorias_disp = sorted(facilities["categoria"].dropna().unique()) if not facilities.empty else []
sel_categorias = st.sidebar.multiselect("Categoría de establecimiento", categorias_disp, default=categorias_disp)

instituciones_disp = sorted(facilities["institucion"].dropna().unique()) if not facilities.empty else []
sel_instituciones = st.sidebar.multiselect("Institución", instituciones_disp, default=instituciones_disp)

threshold = st.sidebar.slider("Umbral de acceso (min)", min_value=10, max_value=180, value=60, step=10)

# Selección vacía no debe crashear: si el usuario deselecciona todo, mostramos
# un aviso y detenemos el render en vez de operar sobre DataFrames vacíos.
if not sel_departamentos or not sel_provincias:
    st.warning("Selecciona al menos un departamento y una provincia en la barra lateral para ver el dashboard.")
    st.stop()

demanda_f = demanda[demanda["departamento"].isin(sel_departamentos) & demanda["provincia"].isin(sel_provincias)]
facilities_f = (
    facilities[facilities["categoria"].isin(sel_categorias) & facilities["institucion"].isin(sel_instituciones)]
    if not facilities.empty
    else facilities
)

if demanda_f.empty:
    st.warning("No hay puntos de demanda para esta combinación de filtros.")
    st.stop()

# --------------------------------------------------------------------- #
# 1) KPI header
# --------------------------------------------------------------------- #
st.title("Acceso a establecimientos de salud resolutivos")
st.caption("Tumbes (costa) · Huancavelica (andino) · Madre de Dios (amazónico) — HW_02_202602")

pob_total = demanda_f["poblacion"].sum()
pob_cubierta = demanda_f.loc[demanda_f["t_min"] <= threshold, "poblacion"].sum()
# t_min NaN (sin ninguna ruta vial mapeada a una facility, ver
# routing.nearest_facility) cuenta como "> 60 min" — es estrictamente peor
# que cualquier tiempo finito, no "no aplica". `t_min > 60` por sí solo
# evalúa a False en NaN y los excluiría en silencio. En una corrida normal
# esto ya casi no dispara: esos puntos llevan un t_min ESTIMADO vía factor
# de desvío (t_min_estimado=True, ver run_pipeline.run_department, issue
# #186 Fase 2) en vez de NaN — el fallback de acá cubre el caso residual
# (p.ej. un departamento sin ninguna facility resolutiva).
pob_mas_60 = demanda_f.loc[(demanda_f["t_min"] > 60) | demanda_f["t_min"].isna(), "poblacion"].sum()
col_estimado = "t_min_estimado" if "t_min_estimado" in demanda_f.columns else None
pob_estimada = (
    demanda_f.loc[demanda_f[col_estimado].fillna(False), "poblacion"].sum() if col_estimado else 0
) + demanda_f.loc[demanda_f["t_min"].isna(), "poblacion"].sum()
_acceso_por_distrito = demanda_f.groupby("distrito").apply(_wmean_safe, include_groups=False).dropna()
peor_distrito = _acceso_por_distrito.idxmax() if not _acceso_por_distrito.empty else "—"
mediana_acceso = demanda_f["t_min"].median()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"Población cubierta (≤{threshold} min)", f"{pob_cubierta:,.0f}", f"{pob_cubierta / pob_total * 100:.1f}% del total")
c2.metric("Población a > 60 min", f"{pob_mas_60:,.0f}", f"{pob_mas_60 / pob_total * 100:.1f}% del total")
c3.metric("t_min estimado (sin ruta real, factor de desvío)", f"{pob_estimada:,.0f}", f"{pob_estimada / pob_total * 100:.1f}% del total" if pob_total else None)
c4.metric("Distrito con peor acceso", str(peor_distrito))
c5.metric("Mediana de acceso", f"{mediana_acceso:.1f} min")

st.divider()

# --------------------------------------------------------------------- #
# 2) Choropleth por distrito (polígonos INEI, sesión 6 del curso) cuando hay
# datos reales; si no (modo sintético, sin polígonos), cae al proxy de
# puntos coloreados por t_min.
# --------------------------------------------------------------------- #
choropleth_f = (
    choropleth[choropleth["departamento"].isin(sel_departamentos)]
    if not choropleth.empty
    else choropleth
)

left, right = st.columns([2, 1])
with left:
    if not choropleth_f.empty and choropleth_f["t_min_ponderado"].notna().any():
        st.subheader("Tiempo de acceso ponderado por distrito")
        bounds = choropleth_f.total_bounds  # minx, miny, maxx, maxy
        center = {"lat": (bounds[1] + bounds[3]) / 2, "lon": (bounds[0] + bounds[2]) / 2}
        fig_map = px.choropleth_map(
            choropleth_f,
            geojson=json.loads(choropleth_f.to_json()),
            locations="ubigeo",
            featureidkey="properties.ubigeo",
            color="t_min_ponderado",
            color_continuous_scale=SEQ_SCALE,
            hover_data=["distrito", "provincia", "poblacion", "t_min_ponderado"],
            center=center,
            zoom=6,
            opacity=0.75,
            height=500,
        )
    else:
        st.subheader("Tiempo de acceso por centro poblado")
        st.caption("Proxy de choropleth: sin polígonos distritales reales en modo sintético, se colorea cada punto de demanda.")
        fig_map = px.scatter_map(
            demanda_f,
            lat=demanda_f.geometry.y,
            lon=demanda_f.geometry.x,
            color="t_min",
            size="poblacion",
            color_continuous_scale=SEQ_SCALE,
            hover_data=["distrito", "poblacion", "t_min"],
            zoom=6,
            height=500,
        )
    if not facilities_f.empty:
        fac_show = st.checkbox("Mostrar establecimientos de salud", value=True)
        if fac_show:
            fig_fac = px.scatter_map(
                facilities_f,
                lat=facilities_f.geometry.y,
                lon=facilities_f.geometry.x,
                color="resolutivo",
                color_discrete_sequence=["#d62728", "#2ca02c"],
                hover_data=["nombre", "categoria", "institucion", "activo"],
            )
            for trace in fig_fac.data:
                fig_map.add_trace(trace)

    # Innovación 1 — isócronas: reutilizan el mismo grafo/matriz de Fase 2,
    # solo se dibujan como polígonos sobre el mapa ya construido.
    iso_disp = isochrones[isochrones["facility_id"].isin(facilities_f["facility_id"])] if not isochrones.empty else isochrones
    if not iso_disp.empty:
        fac_for_iso = st.selectbox(
            "Ver isócronas de un establecimiento resolutivo",
            ["(ninguno)"] + sorted(iso_disp["facility_id"].unique().tolist()),
        )
        if fac_for_iso != "(ninguno)":
            sub = iso_disp[iso_disp["facility_id"] == fac_for_iso].sort_values("threshold_min", ascending=False)
            iso_colors = {30: "rgba(49,163,84,0.35)", 60: "rgba(254,178,76,0.30)", 120: "rgba(222,45,38,0.25)"}
            for _, row in sub.iterrows():
                x, y = row.geometry.exterior.coords.xy
                fig_map.add_trace(go.Scattermap(
                    lon=list(x), lat=list(y), mode="lines", fill="toself",
                    fillcolor=iso_colors.get(row["threshold_min"], "rgba(100,100,100,0.2)"),
                    line=dict(width=1, color="rgba(80,80,80,0.6)"),
                    name=f"{row['threshold_min']} min", showlegend=True,
                ))

    fig_map.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, width='stretch')

with right:
    st.subheader("Distribución de acceso")
    seg_by = st.radio("Segmentar por", ["departamento", "es_urbano"], horizontal=True)
    demanda_seg = demanda_f.copy()
    if "es_urbano" not in demanda_seg.columns:
        demanda_seg["es_urbano"] = demanda_seg["poblacion"] >= config.poblacion_minima_urbana()
    fig_hist = px.histogram(
        demanda_seg, x="t_min", color=seg_by, barmode="overlay", opacity=0.6,
        color_discrete_sequence=CATEGORICAL, nbins=30,
    )
    st.plotly_chart(fig_hist, width='stretch')

st.divider()

# --------------------------------------------------------------------- #
# 5) Tabla de distritos críticos, descargable
# --------------------------------------------------------------------- #
st.subheader("Distritos con peor acceso (ponderado por población)")
def _distrito_row(g: pd.DataFrame) -> pd.Series:
    pob_total_g = g["poblacion"].sum()
    if "t_min_estimado" in g.columns:
        pob_estimada_g = g.loc[g["t_min_estimado"].fillna(False), "poblacion"].sum()
    else:
        pob_estimada_g = g.loc[g["t_min"].isna(), "poblacion"].sum()
    return pd.Series({
        "t_min_ponderado": _wmean_safe(g),
        "poblacion": pob_total_g,
        "pct_estimado_por_desvio": 100 * pob_estimada_g / pob_total_g if pob_total_g > 0 else float("nan"),
    })


critico = (
    demanda_f.groupby("distrito")
    .apply(_distrito_row, include_groups=False)
    .reset_index()
    .sort_values("t_min_ponderado", ascending=False)
)
st.dataframe(critico, width='stretch')
st.download_button(
    "Descargar tabla (CSV)", critico.to_csv(index=False).encode("utf-8"),
    file_name="distritos_criticos.csv", mime="text/csv",
)

st.divider()

# --------------------------------------------------------------------- #
# 6) Simulador de escenario — upgrade I-3/I-4 a resolutivas
# --------------------------------------------------------------------- #
st.subheader("Simulador: upgrade de establecimientos I-3/I-4 a resolutivos")
st.caption("Recalcula t_min SOLO para puntos que mejoran, reutilizando la matriz origen×facility ya precomputada — sin llamar al motor de ruteo.")

upgradables = facilities[facilities["categoria"].isin(["I-3", "I-4"])] if not facilities.empty else pd.DataFrame()
matrix_all = load_matrix_all()

if upgradables.empty or matrix_all.empty:
    st.info("No hay establecimientos I-3/I-4 candidatos, o falta la matriz precomputada.")
else:
    sel_upgrade = st.multiselect(
        "Establecimientos a mejorar a categoría resolutiva",
        upgradables["facility_id"].tolist(),
        format_func=lambda fid: f"{fid} — {upgradables.set_index('facility_id').loc[fid, 'nombre']}",
    )
    if sel_upgrade:
        candidatos = matrix_all[matrix_all["facility_id"].isin(sel_upgrade)]
        t_min_actual = demanda_f.set_index("cp_id")["t_min"]
        mejora = candidatos.merge(t_min_actual.rename("t_min_actual"), left_on="cp_id", right_index=True, how="inner")
        mejora = mejora.reset_index(drop=True)  # el merge deja el índice de t_min_actual (nombrado 'cp_id'), ambiguo con la columna homónima
        mejora = mejora[mejora["t_min"] < mejora["t_min_actual"]]
        mejor_por_punto = mejora.loc[mejora.groupby("cp_id")["t_min"].idxmin()]

        pob_ganada = demanda_f.set_index("cp_id").loc[mejor_por_punto["cp_id"], "poblacion"].sum()
        st.metric(
            "Población que mejora su acceso",
            f"{pob_ganada:,.0f}",
            f"{len(mejor_por_punto)} centros poblados con nueva facility más cercana",
        )
        st.dataframe(
            mejor_por_punto[["cp_id", "facility_id", "t_min_actual", "t_min"]].rename(
                columns={"t_min_actual": "t_min_antes", "t_min": "t_min_despues"}
            ),
            width='stretch',
        )
    else:
        st.caption("Selecciona uno o más establecimientos arriba para ver el efecto del upgrade.")

st.divider()

# --------------------------------------------------------------------- #
# Comparativa de modos + 2SFCA (innovaciones)
# --------------------------------------------------------------------- #
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Auto vs. caminar vs. bicicleta")
    if not mode_comparison.empty:
        mc = mode_comparison[mode_comparison["cp_id"].isin(demanda_f["cp_id"])]
        ratios = mc[[c for c in mc.columns if c.startswith("ratio_")]].melt(var_name="perfil", value_name="ratio")
        fig_ratio = px.box(ratios, x="perfil", y="ratio", color="perfil", color_discrete_sequence=CATEGORICAL)
        fig_ratio.update_layout(showlegend=False)
        st.plotly_chart(fig_ratio, width='stretch')
        st.caption("Ratio = tiempo_modo / tiempo_auto. Solo puntos con ruta real por ese perfil (el fallback de §3.3 no aplica por modo). La diferencia entre departamentos es menor de lo esperado: Madre de Dios tiene el ratio walk/drive más bajo de los tres (~9.4), no el más alto.")
    else:
        st.info("Sin datos de comparación de modos.")

with col_b:
    st.subheader("Accesibilidad 2SFCA (oferta por habitante)")
    if not sfca.empty:
        sfca_f = sfca[sfca["cp_id"].isin(demanda_f["cp_id"])]
        fig_sfca = px.histogram(sfca_f, x="accesibilidad_2sfca", color_discrete_sequence=[CATEGORICAL[0]])
        st.plotly_chart(fig_sfca, width='stretch')
    else:
        st.info("Sin datos de 2SFCA.")

st.divider()

# --------------------------------------------------------------------- #
# 7) Panel de calidad de datos
# --------------------------------------------------------------------- #
st.subheader("Calidad de datos (Fase 1)")
if quality_report:
    for ds in quality_report.get("datasets", []):
        with st.expander(f"{ds['dataset']} — {ds['n_registros_inicial']} → {ds['n_registros_final']} registros"):
            st.dataframe(pd.DataFrame(ds["reglas"]), width='stretch')
else:
    st.info("No se encontró data/outputs/quality_report.json — corre run_pipeline.py primero.")
