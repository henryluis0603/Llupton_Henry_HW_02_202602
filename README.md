# Llupton_Henry_HW_02_202602

Acceso a establecimientos de salud resolutivos en Perú por carretera —
Tumbes (costa), Huancavelica (andino) y Madre de Dios (amazónico).
[Issue #186](https://github.com/d2cml-ai/Data-Science-Python/issues/186).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fase 1-3: genera datos sintéticos, valida, rutea y calcula métricas
python run_pipeline.py --synthetic

# Fase 4: dashboard (lee solo data/outputs/, sin llamadas a motores de ruteo)
streamlit run app.py

# guion de exposición para el video (lee data/outputs/, no recalcula nada)
jupyter notebook walkthrough.ipynb

# alternativa con datos reales (RENIPRESS, centros poblados IGN, población RENIEC)
python run_pipeline.py --real

# compilar report/main.pdf (motor LaTeX autocontenido, no requiere instalar MacTeX)
curl -L -o /tmp/tectonic.tar.gz "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@0.17.0/tectonic-0.17.0-$(uname -m | sed s/arm64/aarch64/)-apple-darwin.tar.gz"
mkdir -p .tools && tar -xzf /tmp/tectonic.tar.gz -C .tools && chmod +x .tools/tectonic
(cd report && ../.tools/tectonic main.tex)
```

`config.md` es la única fuente de verdad: departamentos, categorías
resolutivas, thresholds, rutas y motor de ruteo se declaran ahí, no en el
código.

`walkthrough.ipynb` es un notebook único de exposición, no el pipeline: solo
lee `data/outputs/` ya calculado y narra la historia (calidad de datos,
decisiones de ruteo, métricas, isócronas, 2SFCA, y una nota honesta sobre por
qué el ratio caminar/auto sale constante mientras el motor de ruteo cae al
grafo sintético). Toda la lógica reusable sigue viviendo en `src/`.

## Estado actual

- [x] Arquitectura base (`config.md` + `src/`) y pipeline sintético
      corriendo de punta a punta: `acquisition -> validation -> routing ->
      metrics -> app.py`.
- [x] Grafo de ruteo con fallback automático a un grafo sintético cuando
      Overpass/OSM no responde (ver `src/routing.py`), documentado en
      `data/outputs/snap_report.json`.
- [x] Innovaciones implementadas: isócronas (`routing.isochrone_polygons`) y
      2SFCA (`metrics.two_step_floating_catchment_area`).
- [x] `walkthrough.ipynb` — notebook único de exposición para el video,
      ejecutado y verificado sin errores (`jupyter nbconvert --execute`).
- [x] Datos reales: RENIPRESS (SUSALUD), centros poblados (IGN, sustituyendo
      el SIGMED-MINEDU del issue, que es un geoportal sin descarga
      scripteable — documentado en `SOURCES` en `src/acquisition.py`) y
      población distrital (RENIEC, repartida a centros poblados por tipo de
      asentamiento, ver `distribute_population`). `python run_pipeline.py --real`.
- [x] Todo cálculo de distancia en metros usa la zona UTM del departamento
      (`config.utm_epsg_for_bbox`, EPSG 32717/32718/32719 según corresponda)
      vía `pyproj`, no la aproximación "1 grado ~ 111 km" (que ignora que un
      grado de longitud se encoge con cos(latitud)).
- [x] Polígonos administrativos: shapefile de 1,873 distritos (INEI, columna
      `FUENTE`) — el mismo que usa la sesión 6 del curso
      (`Geopandas1_clean.ipynb`) para su choropleth de COVID, descargado vía
      `huggingface_hub` desde el dataset del curso (`acquisition.download_distritos`).
      Cierra dos huecos a la vez: la regla de validación 4 (punto fuera de su
      polígono distrital) ahora corre de verdad contra datos reales, y el
      dashboard pinta un **choropleth real por distrito** en vez del proxy de
      puntos (`app.py` cae al proxy solo si no hay polígonos, p.ej. en modo
      sintético).
- [ ] Motor de ruteo sobre red vial real: Overpass no respondió en ninguna
      corrida reciente desde este entorno — el pipeline corre igual gracias
      al fallback documentado, pero los tiempos de viaje siguen siendo sobre
      el grafo sintético hasta correr esto desde una red sin ese bloqueo.
- [ ] Altitud — sin DEM integrado; `cross_analysis_altitud` reporta n=0
      honestamente en vez de inventar un valor. La sesión 7 del curso
      (`raster_aplicado.ipynb`) enseña la técnica concreta para cerrar esto
      —`rasterio` + `rasterstats.zonal_stats` sobre un DEM— pendiente de
      aplicar aquí con un raster de elevación real.
- [x] Reporte LaTeX (`report/main.pdf`, 10 páginas): las 9 secciones
      obligatorias con cifras reales, 4 figuras vectoriales generadas por el
      pipeline (`run_pipeline.generate_report_figures`, no capturas de
      pantalla), 4 tablas regeneradas en cada corrida, comparativa línea
      recta vs. red real (`metrics.straight_line_vs_network`, issue #186
      Fase 5) y sección de limitaciones. Compilado con
      [Tectonic](https://tectonic-typesetting.github.io/) (motor LaTeX
      autocontenido, no requiere instalar MacTeX/TeX Live) — este entorno no
      tenía `pdflatex` instalado, ver comando de compilación arriba.
- [ ] Video de presentación.

## Decisiones frente al material de clase (sesiones 6 y 7)

- **Polígonos distritales**: mismo shapefile INEI que la sesión 6
  (`sessions/06-geoespacial/lecture-geopandas/Geopandas1_clean.ipynb`), mismo
  caso de uso (choropleth + cruce punto-en-polígono). Se adaptó el *loader*
  para descargarlo vía `huggingface_hub` en vez de una ruta local fija, para
  que sea reproducible desde cero (`acquisition.download_distritos`).
- **CRS para distancias/áreas — se optó por NO usar EPSG:24891.** La clase
  (sesión 6 y `raster_aplicado.ipynb` de la sesión 7) usa `EPSG:24891`
  (PSAD56 / Peru west zone) para reproyectar y calcular áreas/centroides a
  nivel nacional. Es una elección razonable ahí: el caso de uso es una
  variable de control de área para un panel nacional, donde un CRS "lo
  bastante bueno" alcanza. Pero `EPSG:24891` solo es válido al oeste de 79°O
  (franja costera norte) — verificado con `pyproj.CRS.from_epsg(24891).area_of_use`
  — y este proyecto necesita distancias precisas de ruteo en Huancavelica
  (sierra) y Madre de Dios (selva), fuera de esa franja. Se usa en su lugar
  la zona UTM real de cada departamento (`config.utm_epsg_for_bbox`, EPSG
  32717/32718/32719), calculada dinámicamente — más correcto para distancias
  punto-a-punto que un único CRS nacional, al costo de no poder sumar áreas
  directamente entre departamentos de zonas distintas sin reproyectar antes
  (no es un problema aquí: cada departamento se procesa por separado).
- **Zonal statistics** (sesión 7, `raster_aplicado.ipynb`): la técnica
  (`rasterio.open` + `rasterstats.zonal_stats(polígonos, raster, stats=[...],
  nodata=..., all_touched=True)`) es la ruta identificada para cerrar el
  hueco de altitud — pendiente de aplicar con un DEM real, ver checklist
  arriba.

## Estructura

```
config.md              # única fuente de verdad
requirements.txt
run_pipeline.py         # orquesta Fase 1 -> 2 -> 3 y exporta a data/outputs/
app.py                  # Fase 4 — Streamlit, solo lee precomputado
src/
  config.py              # parsea config.md
  acquisition.py          # Fase 1a — descarga idempotente + generador sintético
  validation.py           # Fase 1b — 6 reglas + quality report
  routing.py               # Fase 2 — grafo, snapping, matriz, isócronas
  metrics.py                # Fase 3 — métricas ponderadas + 2SFCA
  export.py                  # Fase 5 — tablas .tex y figuras
data/{raw,processed,outputs}
report/{main.tex,figures,tables}
logs/
```
