# TerraLens v2.1: Guía Maestra de Arquitectura, Código Fuente y Paper Científico de Referencia

> **Proyecto:** TerraLens v2.1 — *Multi-Scale Earth System Trend & Explainable ML Detective*
> **Reto Oficial:** NASA Space Apps Challenge 2026 — *Be An Earth System Trend Detective!*
> **Equipo:** Proyecto Hail Mary (`@xavierpena` — Xavier Sebastián Peña Avilés, Cuenca, Ecuador)
> **Garantía de Integridad Empírica (Ley #0):** El 100% de las métricas, pendientes, valores $p$, rupturas estructurales y pesos de Machine Learning provienen de datos satelitales reales y ejecuciones verificables en código. Cero valores inventados o simulados.

---

## Índice General

1. [Resumen Ejecutivo (Abstract Estilo Paper)](#1-resumen-ejecutivo-abstract-estilo-paper)
2. [Mapa Completo del Proyecto: Archivo por Archivo y Qué Hace Cada Módulo](#2-mapa-completo-del-proyecto-archivo-por-archivo-y-qué-hace-cada-módulo)
3. [Datos Satelitales y Climáticos Reales Utilizados (Fuentes, Resoluciones y Archivos)](#3-datos-satelitales-y-climáticos-reales-utilizados-fuentes-resoluciones-y-archivos)
4. [Fundamento Matemático y Estadístico (Ecuaciones Exactas en el Código)](#4-fundamento-matemático-y-estadístico-ecuaciones-exactas-en-el-código)
5. [Arquitectura de Machine Learning (Partición 3-Way Train / Val / Test sobre 55,697 Observaciones)](#5-arquitectura-de-machine-learning-partición-3-way-train--val--test-sobre-55697-observaciones)
6. [Guía Página por Página de la Interfaz Web (`static/index.html`)](#6-guía-página-por-página-de-la-interfaz-web-staticindexhtml)
7. [Los Casos de Estudio Científicos Explicados a Fondo (Hipótesis, Datos, Pruebas y Resultados)](#7-los-casos-de-estudio-científicos-explicados-a-fondo-hipótesis-datos-pruebas-y-resultados)
8. [Auditoría de Verificación Automática (`tests/test_scientific_validity.py`)](#8-auditoría-de-verificación-automática-teststest_scientific_validitypy)
9. [Manual Visual: Cómo Interpretar Cada Gráfica, Línea, Color y Tabla en la Interfaz](#9-manual-visual-cómo-interpretar-cada-gráfica-línea-color-y-tabla-en-la-interfaz)
10. [Preguntas Clave de Defensa Científica: Referencias Modernas, Predicción ML y Tiempo Real vs. Caché](#10-preguntas-clave-de-defensa-científicareferencias-modernas-predicción-ml-y-tiempo-real-vs-caché)

---

## 1. Resumen Ejecutivo (Abstract Estilo Paper)

Detectar tendencias reales de largo plazo en series temporales del sistema terrestre presenta cuatro trampas metodológicas clásicas que suelen producir **falsos descubrimientos** en análisis convencionales:

1. **Estacionalidad dominante:** El ciclo anual oculta las anomalías interanuales si no se desestacionaliza correctamente.
2. **Autocorrelación temporal (persistencia mes a mes $r_1 > 0$):** Viola el supuesto de independencia de la regresión lineal ordinaria (OLS) y de la prueba clásica de Mann-Kendall, inflando artificialmente la significancia estadística (Santer et al., 2000; Hamed & Rao, 1998).
3. **Sesgo por eventos extremos aislados (Outlier Leverage):** Un único superevento de El Niño (como 1982–83 o 1997–98) cerca del inicio o final de la ventana temporal distorsiona la pendiente paramétrica OLS (Acker et al., 2009).
4. **Discontinuidades instrumentales satelitales:** El empalme de distintos sensores espaciales (por ejemplo, la transición en enero de 2001 entre los radiómetros ERBE/SRB y los sensores modernos **CERES** en el producto `ALLSKY_SFC_SW_DWN` de NASA POWER) genera saltos de escalón artificiales que parecen tendencias climáticas pero son artefactos de calibración.

**TerraLens v2.1** resuelve estos cuatro problemas integrando un motor estadístico no paramétrico corregido por autocorrelación con una suite de Machine Learning entrenada bajo partición estricta sin fuga de datos (**Train / Validation `1981–2018` + Test Futuro Ciego `2019–2024/2025`**) sobre **55,697 observaciones espacio-temporales** y una arquitectura multiescala (`~55 km` NASA POWER MERRA-2/CERES $\rightarrow$ `1 km` Meteomatics $\rightarrow$ `250 m` NASA Terra MODIS `MOD13Q1 v061` + Microsoft Planetary Computer STAC).

---

## 2. Mapa Completo del Proyecto: Archivo por Archivo y Qué Hace Cada Módulo

Todo el código ejecutable vive dentro de `c:\Users\xavit\Desktop\Sebas\Nasa\terralens\`:

| Archivo | Rol Principal y Funciones Clave |
| :--- | :--- |
| [api.py](file:///c:/Users/xavit/Desktop/Sebas/Nasa/terralens/api.py) | **Servidor Backend FastAPI.** Expone los endpoints REST (`/api/catalog`, `/api/full-dashboard/{region_id}`, `/api/dipole`, `/api/multiscale`, `/api/live-probe`) y sirve la interfaz web en `/`. |
| [core/catalog.py](file:///c:/Users/xavit/Desktop/Sebas/Nasa/terralens/core/catalog.py) | **Catálogo Bilingüe (`EN / ES`).** Define las 11 regiones globales (`BENCHMARK_REGIONS`), las 6 variables climáticas (`CORE_VARIABLES`) y los 5 casos de estudio guiados (`GUIDED_CASES`), incluyendo los nombres traducidos al español. |
| [core/connectors.py](file:///c:/Users/xavit/Desktop/Sebas/Nasa/terralens/core/connectors.py) | **Conectores de Datos Satelitales.** Descarga y cachea en lectura directa (*read-through cache*) NASA POWER (1D y malla 2D de 96 celdas), NOAA ONI, NASA ORNL DAAC MODIS `MOD13Q1` (250 m), Meteomatics (1 km) y Microsoft Planetary Computer STAC v1. |
| [core/stats_engine.py](file:///c:/Users/xavit/Desktop/Sebas/Nasa/terralens/core/stats_engine.py) | **Motor Estadístico de Grado Publicación.** Implementa: `deseasonalize_acker()`, `ols_trend_santer()`, `mann_kendall_hamed_rao()`, `pettitt_change_point()`, `benjamini_hochberg_fdr()`, `compute_annual_trends()`, `compute_climatology_and_seasonal_trends()` y `synthesize_scientific_verdict()`. |
| [core/ml_engine.py](file:///c:/Users/xavit/Desktop/Sebas/Nasa/terralens/core/ml_engine.py) | **Motor de Inferencia ML Explicable y Experimento de Acker.** Carga el artefacto pre-entrenado `models/terralens_ml_suite.joblib` (o ejecuta en vivo para sondas nuevas) para generar: consenso de anomalías de 3 modelos (`IsolationForest` + `LocalOutlierFactor` + `Mahalanobis`), proyección PCA 2D + regímenes K-Means, atribución física rezagada (`RandomForest`) y `run_acker_ml_outlier_experiment()`. |
| [train_models.py](file:///c:/Users/xavit/Desktop/Sebas/Nasa/terralens/train_models.py) | **Pipeline de Entrenamiento Offline Sin Fuga de Datos (3-Way Split).** Estandariza exclusivamente en `1981–2018`, ejecuta `5-Fold TimeSeriesSplit` en `1981–2018` (`Train + Validation`), evalúa en el conjunto futuro bloqueado `2019–2024/2025` (`Locked Test`), calcula importancia por permutación en el Test futuro y guarda `models/terralens_ml_suite.joblib` (`26.68 MB`) + `models/training_manifest.json`. |
| [static/index.html](file:///c:/Users/xavit/Desktop/Sebas/Nasa/terralens/static/index.html) | **Observatorio Web Interactivo (Frontend Single-Page).** Sistema de diseño 100% bilingüe (`EN / ES`), modo oscuro/claro (`Dark / Light`), 3 tarjetas de síntesis, 4 tarjetas KPI con sparkline vectorial, 7 vistas interactivas, gráficas Chart.js y mapa Leaflet de 96 celdas + 11 nodos globales. |
| [static/favicon.svg](file:///c:/Users/xavit/Desktop/Sebas/Nasa/terralens/static/favicon.svg) | **Ícono Vectorial Personalizado.** Representa el globo terrestre, la retícula orbital y la curva de tendencia ascendente. |
| [tests/test_scientific_validity.py](file:///c:/Users/xavit/Desktop/Sebas/Nasa/terralens/tests/test_scientific_validity.py) | **Suite de 6 Pruebas Unitarias Científicas.** Verifica matemáticamente Theil-Sen vs OLS ante outliers, Hamed-Rao vs Mann-Kendall bajo autocorrelación $r_1$, detección exacta de Pettitt, control BH-FDR, consenso ML y reproducibilidad empírica sobre Paute. |

---

## 3. Datos Satelitales y Climáticos Reales Utilizados (Fuentes, Resoluciones y Archivos)

Todos los archivos de datos reales están almacenados en `c:\Users\xavit\Desktop\Sebas\Nasa\terralens\data\cache\` para garantizar respuesta instantánea (`< 50 ms`) y reproducibilidad total offline:

1. **NASA POWER (`MERRA-2` + `CERES SYN1deg` / `SRB`) — Series Mensuales 1981–2025 (45 años, 540 meses):**
   - **Archivos:** `nasa_power_{REGION_ID}.json` (11 regiones globales: 5 en Ecuador + 2 en Brasil + 1 en Perú + 2 en EE. UU. incluyendo Alaska y Delta del Misisipi + 1 en España).
   - **Variables (6):** `T2M` (°C), `PRECTOTCORR` (mm/day), `GWETROOT` (humedad radicular 0–1), `ALLSKY_SFC_SW_DWN` (irradiancia solar MJ/m²/day, disponible 1984–2025), `RH2M` (%) y `WS10M` (m/s).
   - **Variable física derivada (en `connectors.py`):** **Déficit de Presión de Vapor (`VPD`, en `kPa`)**, calculado mediante la ecuación de Tetens a partir de `T2M` y `RH2M`:
     $$e_s = 0.61078 \exp\left(\frac{17.27 \cdot T}{T + 237.3}\right), \quad \text{VPD} = e_s \cdot \left(1 - \frac{\text{RH}}{100}\right)$$
2. **NASA POWER Regional 2D Data Cube (`0.5° × 0.625°`, ~55 km) sobre Ecuador:**
   - **Archivo:** `nasa_power_ecuador_grid.json` (**51,840 registros nodo-mes**).
   - **Cobertura:** Malla completa de **96 celdas** ($8 \times 12$, Costa, Andes y Amazonía) durante 540 meses (1981–2025).
3. **NOAA CPC Oceanic Niño Index (`ONI`, ERSSTv5 Niño 3.4):** `noaa_oni_monthly.json` (540 meses, 1981–2025).
4. **NASA Terra MODIS `MOD13Q1 v061` (`250 m`, `16 días`):** 115 pases satelitales por región entre 2021 y 2025 extraídos de NASA ORNL DAAC (575 pases en total).
5. **Meteomatics High-Resolution Downscaling (`1 km`, 2023–2024):** `meteomatics_ecuador_series.json` (308 registros quincenales sobre Costa, Sierra Norte, Sierra Sur Paute y Amazonía).
6. **Microsoft Planetary Computer STAC API v1 (`modis-13Q1-061`):** `planetary_computer_stac.json` (metadatos reales verificables de gránulos satelitales `h10v09` de Terra MODIS).

---

## 4. Fundamento Matemático y Estadístico (Ecuaciones Exactas en el Código)

En [core/stats_engine.py](file:///c:/Users/xavit/Desktop/Sebas/Nasa/terralens/core/stats_engine.py), cada serie temporal pasa por 5 etapas matemáticas rigurosas:

### 4.1 Desestacionalización Climática (Acker et al., 2009 — Paso 2)
Para cada mes calendario $m \in \{1, \dots, 12\}$, se calcula la media climatológica $\mu_m$ y la desviación estándar $\sigma_m$:
$$X'_{y,m} = X_{y,m} - \mu_m, \qquad Z_{y,m} = \frac{X_{y,m} - \mu_m}{\sigma_m}$$

### 4.2 Corrección de Grados de Libertad Efectivos (Santer et al., 2000)
Sobre los residuos $e_t$ de la regresión OLS se calcula la autocorrelación lag-1 ($r_1$). El tamaño de muestra efectivo es:
$$n_{\text{eff}} = N \cdot \frac{1 - r_1}{1 + r_1}$$
En la temperatura (`T2M`) de Paute ($N = 540$ meses), $r_1 = 0.628 \Rightarrow n_{\text{eff}} = 123.2$. El error estándar de OLS se infla por $\sqrt{(N - 2)/(n_{\text{eff}} - 2)}$.

### 4.3 Prueba de Mann-Kendall Modificada de Hamed & Rao (1998) y Pendiente de Theil-Sen (1968)
- **Estadístico $S$:** $S = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \text{sgn}(x_j - x_i)$.
- **Inflación de varianza de Hamed & Rao (1998):** $\text{Var}^*(S) = \text{Var}(S) \cdot \left[1 + \frac{2}{n(n-1)(n-2)} \sum_{k=1}^{n-1} (n-k)(n-k-1)(n-k-2)\rho_k\right]$.
- **Pendiente de Theil-Sen:** $\beta_{\text{TS}} = \text{mediana}_{i < j} \left(\frac{x_j - x_i}{t_j - t_i}\right)$, con punto de ruptura del $29.3\%$ frente a valores atípicos extremos.

### 4.4 Detección No Paramétrica de Cambio de Régimen de Pettitt (1979)
Localiza el mes exacto $K_n = \max_\tau |U_{\tau, n}|$ donde la media de la serie da un salto estructural $\Delta = \bar{X}_{\text{después}} - \bar{X}_{\text{antes}}$, con $p \approx 2 \exp\left(\frac{-6 K_n^2}{n^3 + n^2}\right)$.

### 4.5 Control de Falsos Descubrimientos de Benjamini-Hochberg (BH-FDR, 1995)
Al evaluar simultáneamente $m = 6$ parámetros en una región, calcula $q_{(i)} = \min_{k \ge i} \left(\frac{m}{k} p_{(k)}\right)$ para eliminar falsos descubrimientos por comparaciones múltiples.

---

## 5. Arquitectura de Machine Learning (Partición 3-Way Train / Val / Test sobre 55,697 Observaciones)

El script [train_models.py](file:///c:/Users/xavit/Desktop/Sebas/Nasa/terralens/train_models.py) entrena dos niveles de modelos con **estricto aislamiento cronológico sin fuga de datos (*zero data leakage*)** y guarda el resultado en `models/terralens_ml_suite.joblib` (`26.68 MB`) + `models/training_manifest.json` (`205.47 s` de ejecución total):

### Nivel A: Benchmark Supervisado sobre el Cubo Espacio-Temporal (`50,208` observaciones de malla + `5,489` regionales = `55,697` muestras)

- **Objetivo ($y$):** Predecir la anomalía estandarizada de humedad del suelo en zona radicular (`GWETROOT_z`) a partir de 16 predictores físicos y topográficos (`lat`, `lon`, `elevation_m`, mes senoidal/cosenoidal, precipitación actual y acumulada de 3/6 meses, temperatura, radiación solar y rezagos `lag1`, `lag2`, `lag3`).
- **Partición Estricta de 3 Etapas (*Train / Validation / Locked Future Test Split*):**
  1. **Estandarización sin fuga de datos:** Las medias y desviaciones estándar climatológicas ($\mu_m, \sigma_m$) se calculan **exclusivamente sobre `1981–2018`** (nunca tocando el futuro `2019–2025`).
  2. **Entrenamiento + Validación Cruzada Temporal (`1981–2018`, 43,296 muestras):** Validación `TimeSeriesSplit(n_splits=5)` entrenando siempre en el pasado y evaluando en bloques futuros dentro de `1981–2018`.
  3. **Test Futuro Ciego (*Locked Unseen Future Test Set*, `2019–2024`, 6,912 muestras):** Los años `2019–2024` permanecen totalmente ocultos hasta evaluar la generalización final y la importancia por permutación (*out-of-sample permutation importance*).
- **Resultados Reales Verificados (`training_manifest.json`):**
  1. **`HistGradientBoostingRegressor`:** Val CV (1981–2018) $R^2 = \mathbf{0.9214 \pm 0.0169}$ | **Test Futuro Ciego (2019–2024) $R^2 = \mathbf{0.9191}$**, $\text{RMSE} = 0.2308\sigma$, $\text{MAE} = 0.1700\sigma$.
  2. **`RandomForestRegressor` (200 árboles):** Val CV $R^2 = \mathbf{0.9073 \pm 0.0241}$ | **Test Futuro $R^2 = \mathbf{0.9019}$**, $\text{RMSE} = 0.2542\sigma$.
  3. **`GradientBoostingRegressor`:** Val CV $R^2 = \mathbf{0.9151 \pm 0.0196}$ | **Test Futuro $R^2 = \mathbf{0.8989}$**, $\text{RMSE} = 0.2579\sigma$.
  4. **`RidgeBaseline` (Comparador Lineal):** Val CV $R^2 = \mathbf{0.9147 \pm 0.0224}$ | **Test Futuro $R^2 = \mathbf{0.8907}$**, $\text{RMSE} = 0.2682\sigma$ (demostrando cuantitativamente que el modelado no lineal basado en árboles supera al baseline lineal en el futuro no visto).

### Nivel B: Suite ML Regional por Nodo (`11 Regiones` $\times$ `499 meses rezagados` = `5,489` observaciones)

Para cada una de las 11 regiones (entrenando en `1981–2018` [415 meses] y validando en test futuro `2019–2025` [84 meses]):
1. **Consenso No Supervisado de 3 Detectores (`contaminación = 5%`):**
   - `IsolationForest(n_estimators=500)` + `LocalOutlierFactor(n_neighbors=20)` + Distancia de `Mahalanobis` ($\chi^2_{6, 0.95}$).
   - Un mes se marca como anomalía extrema cuando $\ge 2/3$ modelos votan positivo.
2. **PCA 2D + K-Means con barrido de Silueta ($k \in \{2,3,4,5\}$):** Proyecta los meses en `PC1 × PC2` e identifica regímenes hidroclimáticos físicos.
3. **Atribución Física con Rezagos (`RandomForestRegressor`):** Cuantifica qué porcentaje de la humedad del suelo radicular es controlado por precipitación acumulada de 3/6 meses, humedad relativa, VPD, radiación o ENSO (`Test R² = 0.899` en Napo Amazon, `0.841` en Pantanal, `0.823` en Mississippi Delta, `0.806` en Manaus, `0.786` en Paute Andes).

---

## 6. Guía Página por Página de la Interfaz Web (`static/index.html`)

### Barra Superior de Comandos (`.command-bar`)
- **`Location / Region` (`#regionSelect`):** Cambia entre las 11 regiones globales (en español o inglés).
- **`Climate Variable` (`#variableSelect`):** Cambia entre las 6 variables satelitales.
- **`Analysis Period` (`#startYearInput` $\rightarrow$ `#endYearInput`):** Selectores de año inicial y final (`1981–2025`).
- **Switch 1 — `2001+ Sensor Filter` (`#btnCeresGuard`):** Restringe `ALLSKY_SFC_SW_DWN` a la era homogénea CERES (`2001–2025`) para evitar el salto instrumental de 2001.
- **Switch 2 — `Filter ML Extremes` (`#btnExcludeExtremes`):** Ejecuta en vivo el experimento de Acker et al. (2009), omitiendo el 5% de meses extremos ML y recalculando todas las pendientes y pruebas de significancia.
- **Utilidades (`.utility-group`):** `? Guide / Guía`, `🌐 ES / EN` (traducción completa al 100% de toda la aplicación, tablas, gráficas, regiones y casos), `☀ Light / ☾ Dark` y `⬇ CSV`.

### Las 7 Vistas del Espacio de Trabajo
1. **Vista 1 — `Long-Term Trends` (`#tab-trends`):** Anomalía mensual (`#monthlyTrendChart`), verificación anual (`#annualTrendChart`), auditoría simultánea de las 6 variables con `BH-FDR q` (`#fdrScanTableBody`), climatología de 12 meses (`#climatologyChart`) y desglose estacional (`#seasonalTrendsBody`).
2. **Vista 2 — `Extreme Events & Drivers` (`#tab-ml`):** Espacio `PC1 × PC2` (`#pcaScatterChart`), atribución de rezagos con Random Forest (`#rfImportanceChart`) y tabla de anomalías compuestas con votos individuales y $\pm\sigma$ (`#mlAnomaliesTableBody`).
3. **Vista 3 — `Model Performance` (`#tab-telemetry`):** Leaderboard de los 4 modelos con **Val CV R² (1981–2018)** y **Unseen Test R² (2019–2024)** (`#cvLeaderboardBody`), importancia por permutación en el Test futuro (`#cubePermChart`) y telemetría de las 11 regiones (`#regionalTelemetryBody`).
4. **Vista 4 — `Outlier Sensitivity` (`#tab-acker`):** Compara lado a lado pendientes con todos los meses vs. sin extremos ML en las 6 variables (`#ackerBarChart` y `#ackerTableBody`).
5. **Vista 5 — `Regional Comparison` (`#tab-dipole`):** Compara anomalías estandarizadas `Z-score` entre cualquier par de regiones frente al índice `NOAA ONI` (`#dipoleChart`).
6. **Vista 6 — `Regional Map & NDVI` (`#tab-multiscale`):** Mapa Leaflet con las **96 celdas de Ecuador (55 km)** + **11 nodos globales de referencia interactivos** + botones de zoom rápido (`Ecuador Grid` / `11 Global Nodes`) y leyenda visual de colores; además de MODIS 250 m (`#modisChart`), Sonda Global en Vivo `(lat, lon)`, Meteomatics 1 km (`#meteomaticsChart`) y tabla STAC (`#stacTableBody`).
7. **Vista 7 — `How to Use & Guide` (`#tab-guide`):** Manual completo bilingüe en inglés y español.

---

## 7. Los Casos de Estudio Científicos Explicados a Fondo

1. **Caso Base Inicial (`EC_Andes_Paute · T2M`):** Calentamiento andino estructural de **`+0.197 °C/década`** ($r_1 = 0.628, n_{\text{eff}} = 123.2, p < 0.0001$), salto de régimen Pettitt en `2002-11` (`+0.53 °C`). Al quitar los 29 extremos ML, la pendiente apenas cambia `-6.3%` (`p < 0.0001`).
2. **Caso Guiado 1 (`case_1_enso_dipole`):** Dipolo ENSO Sudamericano entre Costa de Ecuador (`EC_Coast_Guayas`, $r_{\text{ONI}} = +0.209$) y Amazonía Central (`BR_Amazon_Manaus`, $r_{\text{ONI}} = -0.329$, tendencia seca `-0.35 mm/day/década, p = 0.0002`).
3. **Caso Guiado 2 (`case_2_multiscale_andes`):** En Paute `GWETROOT` (`55 km`), OLS ingenuo da falso positivo ($p_{\text{naive}} = 0.0300$), pero $r_1 = 0.941$ corrige el valor real a $p_{\text{Santer}} = 0.7275$. Al bajar a `1 km` Meteomatics, el déficit hídrico salta de `10.5 mm` a `43.2 mm` en 2024, validado por `250 m` Terra MODIS NDVI.
4. **Caso Guiado 3 (`case_3_satellite_splice`):** Captura el falso enfriamiento solar en Guayas (`p = 1.10e-16` en el salto `1999-05` / transición GEWEX $\rightarrow$ CERES 2001). Al activar el filtro CERES 2001+, la serie `2001–2025` es completamente plana (`p = 0.99`).
5. **Caso Guiado 4 (`case_4_pantanal_compound`):** Tipping point compuesto en el Pantanal Brasileño (`T2M +0.53 °C/década`, `PRECTOTCORR -0.50 mm/day/década`, `RH2M -3.07%/década`, `p < 0.0001`) que persiste intacto tras excluir los extremos ML.
6. **Caso Guiado 5 (`case_5_global_arctic_acker`):** Escalabilidad global comparando la amplificación polar ártica en **Utqiaġvik, Alaska (`71.29°N`)** (`+0.579 °C/década, p < 0.0001`, salto Pettitt `2002-02` de `+1.68 °C`) contra el sitio oficial de **Acker et al. (2009) en el Delta del Misisipi, EE. UU.** Demuestra además que en el Ártico el test futuro `2019–2025` arroja $R^2 < 0$ por **cambio de régimen no estacionario del permafrost** fuera de la envolvente `1981–2018`.

---

## 8. Auditoría de Verificación Automática (`tests/test_scientific_validity.py`)

Comando de ejecución:
```powershell
python "c:\Users\xavit\Desktop\Sebas\Nasa\terralens\tests\test_scientific_validity.py"
```
Las **6 pruebas deterministas** verifican: robustez de Theil-Sen vs OLS (`+25σ`), corrección de Hamed-Rao bajo $\text{AR}(1)$, detección exacta de Pettitt, monotonicidad de `BH-FDR`, precisión 100% del consenso ML de 3 modelos y reproducibilidad empírica sobre `EC_Andes_Paute`.

---

## 9. Manual Visual: Cómo Interpretar Cada Gráfica, Línea, Color y Tabla en la Interfaz

- **Tarjetas de Síntesis Superiores:** Muestran `Tendencia Theil-Sen` (velocidad real por década), `Cambio de Régimen Pettitt` (mes exacto de ruptura), `Persistencia r₁ → GL efectivos`, `Valores p ajustados` (`< 0.05` = real, `> 0.05` = ruido) y `Anomalía Pico ML`.
- **Pestaña 1 (`Trends`):** Si la línea naranja punteada (`OLS`) se desvía de la verde sólida (`Theil-Sen`), los picos extremos están sesgando la regresión clásica. En la tabla `BH-FDR`, solo las filas con punto verde pasaron tanto Hamed-Rao como Benjamini-Hochberg.
- **Pestaña 2 (`ML`):** Puntos rojos alejados del centro en `PC1 × PC2` son desastres compuestos multivariados. Las barras horizontales suman el peso físico (`%`) de lluvia rezagada, VPD y temperatura sobre la humedad del suelo.
- **Pestaña 3 (`CV / Model Performance`):** `Val CV R² (1981–18)` mide consistencia histórica por pliegues y `Unseen Test R² (2019–24)` mide generalización ciega en años futuros nunca vistos (`0.9191` en `HistGradientBoostingRegressor`).
- **Pestaña 4 (`Acker Test`):** Si la barra azul y la verde son casi iguales (`Δ% < 15%`), la tendencia es estructural; si la barra colapsa al quitar los extremos, la tendencia era un espejismo de El Niño.
- **Pestaña 5 (`Dipole`):** Curva azul subiendo y curva roja bajando cuando la amarilla (`ONI`) sube = dipolo hidroclimático opuesto confirmado.
- **Pestaña 6 (`Mapa & NDVI`):** Rectángulos de colores = malla de 96 celdas en Ecuador (borde sólido = `p < 0.05`); círculos brillantes = los **11 nodos globales** (haz clic para cargar cualquiera o usa los botones de zoom Ecuador/Global). Cualquier clic sobre tierra o mar ejecuta la sonda en vivo.

---

## 10. Preguntas Clave de Defensa Científica: Referencias Modernas, Predicción ML y Tiempo Real vs. Caché

1. **Referencias Clásicas y Modernas (2019–2025):** Se cita `Acker et al. (2009)` porque es el paper metodológico oficial del reto NASA *Earth System Trend Detective*; se citan `Theil-Sen`, `Hamed-Rao`, `Santer`, `Pettitt` y `BH-FDR` por ser los estimadores estándar exigidos en el **IPCC AR6 WGI (2021/2023, Cap. 11)**; y el diseño de ML y extremos compuestos se respalda en **Zscheischler et al. (2020, *Nature Reviews Earth & Environment*)**, **Reichstein et al. (2019, *Nature*)**, **Camps-Valls et al. (2021)** y **Stackhouse et al. (2024 — NASA POWER 2.4)**.
2. **Justificación del ML y Predicción:** Combinamos **predicción supervisada espaciotemporal** (generalizando `GWETROOT_z` al test ciego `2019–2024` con $R^2 = 0.9191$ frente a $0.8907$ del modelo lineal Ridge) y **detección no supervisada de consenso** para alimentar la prueba de exclusión de extremos de Acker.
3. **Tiempo Real vs. Caché Read-Through:** Las 11 regiones y 96 celdas se descargaron desde las APIs reales de la NASA/NOAA/Meteomatics/Planetary Computer y se sirven desde caché local para responder en `< 30 ms` (si falta un archivo, `connectors.py` lo descarga de la API en vivo). Y la **Sonda Global en Vivo** de la Pestaña 6 llama en tiempo real a `power.larc.nasa.gov` para cualquier `(lat, lon)` del planeta.
