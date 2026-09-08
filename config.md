# Configuración del proyecto

Este archivo es la **única fuente de verdad** para el pipeline. Ningún módulo en
`src/` hardcodea departamentos, thresholds, categorías o rutas: todos importan
`src/config.py`, que parsea el bloque YAML de abajo. Cambiar un departamento o
un umbral implica editar solo este archivo.

## Departamentos de análisis

Un departamento por tipo de geografía, exigido por el issue #186:

| Rol       | Departamento   | Motivo de selección                                                        |
|-----------|----------------|-----------------------------------------------------------------------------|
| Costa     | Tumbes         | Chico, red vial densa y conectada — caso base de referencia.               |
| Andino    | Huancavelica   | Altitud alta y variable — habilita el cross-analysis altitud vs. acceso.   |
| Amazónico | Madre de Dios  | Red vial dispersa — esperamos el mayor ratio caminar/auto del estudio.     |

## Categorías resolutivas

Un establecimiento es "resolutivo" si está **activo** y su categoría está en
la whitelist. Todas las categorías `I-*` son no resolutivas por definición
del issue.

## Innovaciones habilitadas

De las 7 ideas de innovación del issue, este proyecto implementa **2**,
elegidas por su relación costo/impacto dado el pipeline obligatorio:

1. **Isócronas** — reutilizan el mismo grafo y la misma matriz de Fase 2, sin
   cómputo nuevo más allá de un `ego_graph` + alpha shape por facility.
2. **2SFCA** — reutiliza la matriz origen×facility de Fase 2 y agrega una
   segunda métrica de accesibilidad ponderada por capacidad de oferta.

Las otras 5 (optimización de siting, bootstrapping de incertidumbre, escala
nacional, comparativa temporal RENIPRESS, regeneración automática del PDF) se
excluyen deliberadamente: cada una es un submódulo nuevo que compite por
horas contra la solidez de las 5 fases obligatorias.

```yaml
# ---- BLOQUE MÁQUINA-LEGIBLE: src/config.py parsea esto, no el resto del archivo ----

departamentos:
  costa:
    nombre: "TUMBES"
    # bbox aproximado (north, south, east, west) — WGS84.
    # TODO: reemplazar por el bbox exacto derivado del polígono oficial (INEI)
    # una vez descargado en Fase 1; este es un rango de partida conservador.
    bbox: { north: -3.35, south: -4.05, east: -80.10, west: -80.65 }
  andino:
    nombre: "HUANCAVELICA"
    bbox: { north: -11.80, south: -13.75, east: -74.35, west: -75.80 }
  amazonico:
    nombre: "MADRE DE DIOS"
    bbox: { north: -9.90,  south: -13.50, east: -68.65, west: -72.50 }

peru_bbox: { lon_min: -81.4, lon_max: -68.6, lat_min: -18.4, lat_max: -0.04 }

categorias_resolutivas:
  - "II-1"
  - "II-2"
  - "II-E"
  - "III-1"
  - "III-2"
  - "III-E"

# Variantes de string observadas en RENIPRESS que deben normalizarse a las
# claves de arriba antes de comparar contra la whitelist. Se completa en
# Fase 1 conforme se van descubriendo en los datos reales.
normalizacion_categorias:
  "II-1 ": "II-1"
  "ii-1": "II-1"
  "II–1": "II-1"   # guion en-dash en vez de guion normal
  "II_1": "II-1"

thresholds_minutos: [30, 60, 120]

ponderacion: "poblacion"   # toda métrica de acceso se pondera por esta columna

clasificacion_urbano_rural:
  # regla explícita exigida por Fase 3: centro poblado urbano si
  # poblacion >= poblacion_minima (convención INEI para centros poblados)
  poblacion_minima: 2000

dos_sfca:
  # Innovación 2: capacidad de oferta usada en el paso 1 del 2SFCA.
  # En datos reales, reemplazar por camas/quirófanos de RENIPRESS si está
  # disponible; en datos sintéticos, acquisition.py genera esta columna.
  capacidad_col: "capacidad"
  threshold_min: 60

routing:
  motor: "osmnx_networkx"   # osrm_docker | osmnx_networkx | ors_api
  perfiles: ["drive", "walk", "bike"]
  # snapping: distancia máxima aceptable (metros) antes de flaggear el punto
  # como "snap dudoso" en el reporte de Fase 2
  snap_max_metros: 500

demanda_max_puntos: 5000   # límite duro del issue; si se excede, muestrear y documentar

rutas:
  raw: "data/raw"
  processed: "data/processed"
  outputs: "data/outputs"
  logs: "logs"
  quality_report: "data/outputs/quality_report.json"
  routing_cache_dir: "data/processed/routing_cache"
  report_figures: "report/figures"
  report_tables: "report/tables"

sintetico:
  # usado solo por acquisition.py cuando --synthetic está activo, para probar
  # el pipeline de punta a punta mientras las fuentes reales descargan
  n_demanda: 200
  n_facilities: 20
  seed: 42
```
