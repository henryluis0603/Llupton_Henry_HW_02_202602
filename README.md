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
- [ ] Polígonos administrativos — no se descargaron todavía. Sin ellos, la
      regla de validación 4 (punto fuera del polígono distrital) no se
      ejecuta contra datos reales, y el "choropleth" del dashboard sigue
      siendo un proxy de puntos, no un choropleth real por distrito.
- [ ] Motor de ruteo sobre red vial real: Overpass no respondió en ninguna
      corrida reciente desde este entorno — el pipeline corre igual gracias
      al fallback documentado, pero los tiempos de viaje siguen siendo sobre
      el grafo sintético hasta correr esto desde una red sin ese bloqueo.
- [ ] Altitud — sin DEM integrado; `cross_analysis_altitud` reporta n=0
      honestamente en vez de inventar un valor.
- [ ] Reporte LaTeX (`report/main.tex`) — estructura y tablas conectadas al
      pipeline, contenido narrativo pendiente de redactar. Nota: este
      entorno no tiene `pdflatex`/`xelatex` instalado, falta resolver el
      toolchain para compilar el PDF.
- [ ] Video de presentación.

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
