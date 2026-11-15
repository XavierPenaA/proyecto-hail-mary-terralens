# TerraLens: guía maestra de arquitectura, método y defensa científica

> **Proyecto:** TerraLens — detective de tendencias del sistema Tierra
> **Reto:** NASA Space Apps Challenge 2026 — *Be An Earth System Trend Detective!*
> **Equipo:** Proyecto Hail Mary (Ecuador)
> **Regla de esta guía:** cada número citado sale del motor. Los valores de ejemplo se recalcularon el 14 de noviembre de 2026; si cambian los datos, `pytest tests/test_case_findings.py` avisa qué afirmación dejó de cumplirse.

---

## Índice

1. [Resumen](#1-resumen)
2. [Mapa del proyecto, archivo por archivo](#2-mapa-del-proyecto-archivo-por-archivo)
3. [Datos usados](#3-datos-usados)
4. [Fundamento estadístico](#4-fundamento-estadístico)
5. [Machine learning y su línea base](#5-machine-learning-y-su-línea-base)
6. [La interfaz, vista por vista](#6-la-interfaz-vista-por-vista)
7. [Los seis casos guiados](#7-los-seis-casos-guiados)
8. [Pruebas automáticas](#8-pruebas-automáticas)
9. [Limitaciones](#9-limitaciones)
10. [Preguntas de defensa ante el jurado](#10-preguntas-de-defensa-ante-el-jurado)

---

## 1. Resumen

Una recta ajustada a una serie satelital o de reanálisis puede reportar una tendencia que no existe. TerraLens prueba cada tendencia contra tres trampas:

1. **Memoria (autocorrelación).** Si cada mes se parece al anterior, la regresión y el Mann-Kendall clásicos sobrestiman la significancia (Santer et al., 2000; Hamed y Rao, 1998).
2. **Cambios en los datos.** Un sensor nuevo o nuevos datos de entrada crean un salto que parece cambio climático. En NASA POWER, la irradiancia solar pasa de GEWEX SRB (hasta 2000) a CERES SYN1deg (desde enero de 2001).
3. **Extremos.** Un El Niño o una sequía enormes cerca de un extremo del registro inclinan la recta (prueba de exclusión de extremos de Acker et al., 2009).

Para 11 regiones y seis variables de NASA POWER (1981–2025), y para cualquier coordenada mediante la sonda en vivo, la app responde en una frase si hay una tendencia real y muestra la evidencia.

---

## 2. Mapa del proyecto, archivo por archivo

| Archivo | Qué hace |
| :--- | :--- |
| `api.py` | Servidor FastAPI. Valida entradas (404/422), traduce fallas externas a 502, limita la sonda a 20 consultas por minuto, comprime con GZip y precalcula todas las regiones al arrancar. Endpoints: `/api/catalog`, `/api/ml-manifest`, `/api/full-dashboard/{region}`, `/api/dipole`, `/api/multiscale`, `/api/live-probe`. |
| `core/catalog.py` | Periodo de datos, seis variables (nombres y unidades en EN/ES, fuente, empalme documentado, escala del mapa) y 11 regiones con descripciones cualitativas. No contiene cifras escritas a mano. |
| `core/stats_engine.py` | Desestacionalización, Theil-Sen, Mann-Kendall con Hamed-Rao, OLS con Santer, Pettitt con bootstrap por bloques, prueba del empalme documentado, comparación salto-vs-tendencia (BIC), correlación con ENSO con tamaño efectivo, BH-FDR y veredictos con códigos estables. |
| `core/ml_engine.py` | Detectores de anomalías (Isolation Forest, LOF, Mahalanobis con χ²), atribución en σ, PCA y K-Means con k elegido por silueta, Random Forest de factores con línea base de persistencia, filas de la prueba de extremos. |
| `core/analysis_service.py` | Capa con caché (`lru_cache`) que comparten la API y los casos; normaliza los argumentos para que consultas equivalentes compartan resultado. Detecta saltos simultáneos en varias variables. |
| `core/case_studies.py` | Los seis casos guiados: configuración, pasos del recorrido y el texto del hallazgo, escrito con los números en vivo. |
| `core/connectors.py` | Lectura de las cachés, consulta en vivo a NASA POWER (con TLS verificado y caché limitada en `data/cache/live/`) y funciones de descarga de MODIS y STAC. |
| `train_models.py` | Benchmark offline de humedad del suelo con persistencia, nowcast y pronóstico; escribe `models/training_manifest.json`. `--save-model` guarda además el modelo. |
| `scripts/refresh_satellite_caches.py` | Vuelve a descargar NDVI de MODIS (ventana de píxeles con filtro de calidad) y metadatos STAC. |
| `static/index.html` | Solo el marcado de la página, con claves de traducción (`data-i18n`) y roles ARIA. |
| `static/css/app.css` | Estilos y tokens de color para tema claro y oscuro (contraste AA), responsive y movimiento reducido. |
| `static/js/` | Módulos ES: `main.js` (orquestador), `state.js` (estado y URL compartible), `api.js` (peticiones cancelables), `i18n.js`, `theme.js`, `charts.js`, `labels.js`, `ui.js`, `export.js` y una carpeta `views/` con un módulo por vista. |
| `static/i18n/en.json`, `es.json` | Todos los textos de la interfaz, con las mismas claves en ambos idiomas. |
| `static/vendor/` | Chart.js 4.4.1 y Leaflet 1.9.4 copiados para que la demo funcione sin internet. |
| `tests/` | Suite de `pytest` (sección 8). |

---

## 3. Datos usados

Todo se lee desde `data/cache/`, así la app funciona sin conexión.

| Archivo | Contenido |
| :--- | :--- |
| `{REGION}.json` (11 archivos) | Respuesta de NASA POWER: 540 meses (1981–2025) de `T2M`, `PRECTOTCORR`, `GWETROOT`, `GWETTOP`, `ALLSKY_SFC_SW_DWN` (desde 1984) y `RH2M`. Fuentes según la propia API: MERRA-2, GEWEX SRB, CERES SYN1deg. |
| `ecuador_regional_grid.json` | Malla de 96 celdas (~55 km) sobre Ecuador para `T2M`, `PRECTOTCORR` y `GWETROOT`. |
| `oni.txt` | Índice Oceánico de El Niño de NOAA CPC. Cada trimestre se asigna a su mes central (DEF → enero). |
| `nasa_modis_250m_ndvi.json` | NDVI MOD13Q1 (250 m, 16 días), 2021–2025, para cinco regiones; un solo píxel en el punto de cada región. |
| `meteomatics_1km_ecuador_2023_2024.json` | 77 fotos por región (un día cada 10 días) de un modelo comercial a 1 km, cuatro regiones de Ecuador. |
| `planetary_computer_stac.json` | Metadatos de gránulos MODIS que contienen cada punto (procedencia). |

`PRECTOTCORR` es la precipitación de MERRA-2 corregida por sesgo, según la definición de la API de NASA POWER.

---

## 4. Fundamento estadístico

### 4.1 Desestacionalización (Acker et al., 2009)
Para cada mes calendario *m* se calcula la media μ_m y la desviación σ_m:

$$X'_{y,m} = X_{y,m} - \mu_m, \qquad Z_{y,m} = \frac{X_{y,m} - \mu_m}{\sigma_m}$$

Las anomalías anuales son el promedio de las mensuales de cada año (exigiendo al menos 9 meses). Las estaciones usan años estacionales: diciembre cuenta con el enero–febrero siguiente, y solo entran estaciones completas.

### 4.2 Santer et al. (2000)
Con la autocorrelación lag-1 *r₁* de los residuos de OLS:

$$n_{\text{eff}} = N \cdot \frac{1 - r_1}{1 + r_1}$$

El error estándar se infla por √((N − 2)/(n_eff − 2)).

### 4.3 Mann-Kendall con Hamed y Rao (1998) y Theil-Sen
$$S = \sum_{i<j} \operatorname{sgn}(x_j - x_i), \qquad \operatorname{Var}^*(S) = \operatorname{Var}(S)\left[1 + \frac{2}{n(n-1)(n-2)}\sum_k (n-k)(n-k-1)(n-k-2)\,\rho_k\right]$$

Solo se suman los rezagos con autocorrelación significativa de los rangos de la serie sin tendencia. La pendiente de Theil-Sen es la mediana de todas las pendientes entre pares.

**Veredicto.** Una tendencia vale lo que la más débil de las dos pruebas corregidas: p = máx(Hamed-Rao, Santer). p < 0,05 → significativa; entre 0,05 y 0,10 → débil; si no → sin tendencia. Si Hamed-Rao ve una tendencia (p < 0,10), Santer no, y r₁ > 0,5, el código es `autocorr_rejected` (la memoria explica la pendiente). Esta regla sale de la calibración Monte Carlo de la sección 4.8: Hamed-Rao solo declaraba significativas entre 11 % y 28 % de las series sin tendencia cuando r₁ ≥ 0,5.

**Intervalo de la pendiente.** El intervalo de Theil-Sen de SciPy supone puntos independientes (excluía el cero hasta en el 74 % de las series sin tendencia). Se ensancha alrededor de la pendiente por √(N/n_eff), con n_eff de Santer.

### 4.4 Pettitt (1979) con bootstrap por bloques
Pettitt ubica el mes con el mayor cambio de nivel (estadístico K). La fórmula asintótica del p-valor supone meses independientes y produce valores absurdos con series persistentes (p = 7,9 × 10⁻⁴⁵ en Napo). TerraLens calcula el p-valor con un **bootstrap circular por bloques**: remuestrea bloques de longitud (1 + r₁)/(1 − r₁) —el cociente N/n_eff de Santer— para conservar la memoria, 999 veces con semilla fija. El mínimo posible es p = 0,001.

### 4.5 ¿Salto o tendencia?
Pettitt no distingue un escalón de una tendencia gradual. Se comparan con BIC un modelo lineal (2 parámetros) y un modelo de dos medias separadas en la fecha de Pettitt (3 parámetros). El BIC usa el tamaño efectivo n_eff (con la persistencia de los residuos del modelo que mejor ajusta) en lugar de N, porque con memoria alta N puntos aportan mucha menos información. Si la diferencia supera 10 (evidencia "muy fuerte" según Kass y Raftery, 1995), se prefiere ese modelo; si no, el resultado es "no concluyente". Con N en vez de n_eff, el BIC declaraba "salto" en el 72 % de las series sin salto con r₁ = 0,85. Una tendencia significativa que se explica mejor como un salto recibe la advertencia `step_dominated`.

Además:
- **Empalme documentado:** para la irradiancia solar se prueba directamente la diferencia de medias en 2001-01, con el mismo bootstrap.
- **Saltos simultáneos:** si tres o más variables tienen su quiebre significativo a menos de dos meses entre sí, la app lo señala, porque suele indicar un cambio en los datos de entrada.

### 4.6 Correlaciones con ENSO
Se correlaciona la anomalía con el ONI adelantado 0 a 6 meses. El p-valor usa el tamaño efectivo de Bretherton et al. (1999), n_eff = n(1 − r₁ₐ r₁ᵦ)/(1 + r₁ₐ r₁ᵦ), y los siete desfases se corrigen con BH-FDR. El dipolo se declara "opuesto" solo si ambas regiones tienen correlación significativa y de signo contrario.

### 4.7 Benjamini-Hochberg (1995)
$$q_{(i)} = \min_{k \ge i} \frac{m}{k}\, p_{(k)}$$
Se aplica a las seis variables de una región y a las 96 celdas del mapa (significancia de campo, Wilks 2016).

### 4.8 Calibración Monte Carlo
`scripts/calibration_study.py` ejecuta las funciones de la app sobre 1000 series sintéticas de 540 meses por escenario (ruido AR(1) con φ de 0 a 0,95; resultados en `models/calibration_report.json`). Falsos positivos sin tendencia, nivel nominal 5 %:

| φ | Mann-Kendall clásico | Hamed-Rao | Santer | Veredicto actual |
| :--- | :--- | :--- | :--- | :--- |
| 0,0 | 4,7 % | 4,2 % | 4,4 % | 3,3 % |
| 0,5 | 24,7 % | 11,3 % | 4,7 % | 4,5 % |
| 0,7 | 42,1 % | 17,7 % | 5,3 % | 5,1 % |
| 0,85 | 57,7 % | 22,6 % | 5,5 % | 5,4 % |
| 0,95 | 73,9 % | 29,9 % | 5,6 % | 5,6 % |

El costo es potencia: con φ = 0,7, una tendencia de 0,2σ por década se detecta el 69 % de las veces (Hamed-Rao solo: 85 %), y con 0,4σ el 99,9 %. El BIC con n_eff casi no genera saltos falsos (≤ 0,2 %) pero también detecta menos saltos reales cuando hay mucha memoria: un salto de 2σ se encuentra en el 82 % de los casos con φ = 0,7 y en el 7 % con φ = 0,95. Con la regla corregida, 11 de las 66 series región × variable (casi todas de humedad del suelo, r₁ ≈ 0,83–0,94) pasaron de significativas a `autocorr_rejected` o débiles.

### 4.9 Contraste independiente con ERA5-Land
`scripts/fetch_era5land.py` descarga ERA5-Land (Open-Meteo, sin cuenta) y `scripts/validate_era5land.py` lo compara con MERRA-2 mes a mes en las 11 regiones (`models/era5land_validation.json`). ERA5-Land es otro modelo de suelo con otro forzamiento, pero sigue siendo un reanálisis; mide acuerdo entre dos productos, no error contra la realidad.

| Variable | Regiones | r mediana de anomalías | Regiones con r < 0,5 | Significativa en MERRA-2 / ERA5-Land | Ambas significativas y mismo signo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Temperatura | 11 | 0,89 | 1 | 9 / 8 | 7 de 7 |
| Humedad radicular | 10 | 0,41 | 6 | 2 / 3 | 1 de 2 |
| Humedad superficial | 10 | 0,51 | 5 | 2 / 3 | 1 de 2 |

- La temperatura concuerda, aunque en tres regiones el veredicto difiere (Guayas y Paute: solo MERRA-2 ve tendencia; Napo: solo ERA5-Land).
- La humedad del suelo concuerda poco en la variabilidad mensual y no confirma las tendencias. El caso más claro es Napo: MERRA-2 sube (+0,49 desviaciones por década, quiebre en 2007-08) y ERA5-Land baja, sin ese quiebre. Es coherente con un cambio en los datos de entrada de MERRA-2, no con una señal climática.
- Pantanal es el único caso donde ambos productos ven la misma bajada significativa de humedad radicular (y de la superficial).
- Límites: ERA5-Land entrega humedad volumétrica (m³/m³) y MERRA-2 una fracción de saturación, por eso se comparan anomalías; la celda del Altiplano es una constante (0,021) en ERA5-Land y se excluye; la precipitación de ERA5-Land no se obtuvo, así que esa variable queda sin contrastar. SMAP (verdad satelital, desde 2015) sigue pendiente porque exige credenciales de Earthdata.

---

## 5. Machine learning y su línea base

### 5.1 En la app (por región, al momento)
- **Anomalías:** Isolation Forest (`contamination="auto"`), LOF (`"auto"`) y Mahalanobis con umbral χ² de 6 grados de libertad y p < 0,01. Un mes es anómalo si al menos dos coinciden. Así una región estable puede tener pocas anomalías; antes cada detector marcaba siempre el 5 % por construcción.
- **Regímenes:** PCA para dibujar y K-Means con k ∈ {2, 3, 4} elegido por silueta; los nombres (húmedo, cálido y seco, etc.) salen de las medias en σ, sin atribuir causas.
- **Factores de la humedad del suelo.** Random Forest con lluvia, temperatura, humedad, radiación y ONI del mes y rezagados. La climatología se ajusta dentro de cada pliegue de entrenamiento. En Quito, el modelo obtiene R² = −0,02 frente a 0,86 de la línea base de persistencia.

### 5.2 Benchmark offline (`train_models.py`)
El desarrollo de 1981–2018 usa cinco pliegues de meses completos, con climatología estimada solo en el entrenamiento de cada pliegue. La prueba independiente cubre 2019–2025. En el pronóstico a tres meses empieza en marzo para respetar la fecha de emisión más temprana con el modelo ajustado hasta diciembre de 2018.

| Benchmark | Qué usa | Resultado (test R²) |
| :--- | :--- | :--- |
| Persistencia | Solo el mes anterior | 0,758 |
| Nowcast (Ridge, elegido por validación cruzada) | Clima del mismo mes + rezagos | 0,892 |
| Pronóstico a 1 mes (Ridge) | Solo datos conocidos 1 mes antes | 0,779 (+0,022 sobre persistencia) |
| Pronóstico a 3 meses (Ridge) | Solo datos conocidos 3 meses antes | 0,349 (+0,006 sobre persistencia) |

El nowcast puntúa alto porque usa la lluvia del mismo mes y la humedad del mes anterior (la importancia por permutación de `GWET_lag1` es la mayor). No es un pronóstico. Los modelos regionales fallan en el Ártico (R² negativo) porque el suelo congelado casi no varía.

---

## 6. La interfaz, vista por vista

- **Barra superior:** región, variable, periodo (mínimo 10 años, sin rangos invertidos), filtro de sensor 2001, quitar meses extremos, modo experto, idioma, tema y CSV con metadatos. La dirección de la página guarda todo para compartir.
- **Tarjeta de respuesta:** una frase con el veredicto, advertencias (salto, empalme, extremos quitados) y una explicación de la variable y la región.
- **Tendencias:** serie mensual con Theil-Sen, OLS, escalón de Pettitt y meses anómalos; promedios anuales; las seis variables con q de FDR (columnas extra en modo experto); año normal y tendencias por estación.
- **Meses extremos:** PCA coloreado por régimen, factores de la humedad del suelo frente a persistencia y tabla de meses anómalos con qué fue inusual.
- **Sensibilidad a extremos:** tendencias con y sin los meses anómalos.
- **Comparar dos regiones:** anomalías estandarizadas con el ONI en un segundo eje, correlación con n efectivo y diagnóstico de ENSO y de tendencias.
- **Mapa y vegetación:** malla con escala divergente centrada en cero y celdas sólidas solo si q < 0,05; NDVI de MODIS (con aviso cuando la región no tiene serie propia); fotos de Meteomatics; procedencia STAC; sonda en vivo (el clic en el mapa consulta solo con el modo sonda activo).
- **Rendimiento de modelos:** nowcast, pronóstico y modelos regionales, todos frente a persistencia.
- **Cómo usarlo** y **Acerca de y limitaciones.**

Accesibilidad: navegación con teclado entre vistas, roles ARIA, descripciones en los gráficos, `lang` del documento según el idioma, contraste AA en ambos temas, respeto de `prefers-reduced-motion` y diseño adaptable a tablet y teléfono.

---

## 7. Los seis casos guiados

El texto de cada caso lo genera `core/case_studies.py` con los números del motor. Valores al 14-nov-2026:

1. **Mismo El Niño, lluvia opuesta (Guayas vs. Manaos, precipitación).** La lluvia costera sube con El Niño (r = +0,22, ONI adelantado 1 mes, q = 0,031) y la de Manaos baja (r = −0,33, q ≤ 0,001). Las tendencias no son opuestas: la costa no tiene tendencia significativa (p = 0,81) y Manaos baja 0,356 mm/día por década, aunque se describe mejor como un salto hacia 2002-10.
2. **La trampa de la autocorrelación (Paute, humedad radicular).** OLS ingenuo da p = 0,030; con r₁ = 0,94 solo hay unos 16 valores independientes; Santer da p = 0,73 y la prueba anual p = 0,22.
3. **Una tendencia falsa por cambio de sensor (irradiancia solar).** En 1984–2025 la costa parece bajar 0,251 y el Pantanal subir 0,411 MJ/m²/día por década. Ambas series saltan en enero de 2001 (−0,69 y +0,92; p ≤ 0,001); Pettitt marca 1999-05 y 1998-10. En la era CERES la tendencia costera desaparece (p = 0,83) y la del Pantanal se mantiene más pequeña (+0,207, p = 0,046).
4. **Pantanal: más cálido y más seco.** Temperatura +0,546 °C/década, precipitación −0,437 mm/día/década, humedad superficial −0,034/década y humedad relativa −3,19 %/década; las cuatro sobreviven sin los 27 meses anómalos.
5. **Calentamiento ártico vs. delta del Misisipi.** Utqiaġvik +0,58 °C/década, 2,2 veces el delta (+0,26). El mayor quiebre ártico es 2002-02 (+1,68 °C), pero salto y tendencia se ajustan casi igual.
6. **Un salto que parece tendencia (Napo).** Cuatro variables cambian de nivel a menos de dos meses de 2007-08 (precipitación +1,92 mm/día); probablemente un cambio en los datos de entrada. La tendencia de lluvia debe contrastarse con estaciones.

---

## 8. Pruebas automáticas

```bash
pip install -r requirements-dev.txt
pytest
```

| Archivo | Qué comprueba |
| :--- | :--- |
| `tests/test_stats_engine.py` | Cada método sobre series sintéticas con respuesta conocida: desestacionalización, Mann-Kendall, Hamed-Rao, Santer, Pettitt con bootstrap, empalme, salto-vs-tendencia, n efectivo, DJF y BH-FDR. |
| `tests/test_api.py` | Contrato de la API, validación (404/422), filtro CERES, compresión y manifiesto con persistencia. |
| `tests/test_data_integrity.py` | Las cachés están completas y coinciden con el catálogo. |
| `tests/test_case_findings.py` | Cada afirmación de los casos se cumple con los datos actuales, y ambos idiomas citan los mismos números. |
| `tests/test_frontend_i18n.py` | Las traducciones tienen las mismas claves y marcadores, y toda clave usada existe. |

---

## 9. Limitaciones

- Celdas de ~55 km (solar ~110 km): un valor mezcla relieves muy distintos en los Andes.
- MERRA-2 es un reanálisis, no estaciones; los cambios en sus datos de entrada pueden crear saltos.
- Cada región es un punto. Los de Guayas y Manaos caen sobre ciudades y ríos, así que su NDVI describe agua y zona urbana. Para corregirlo: definir `modis_point` en `core/catalog.py` y ejecutar `scripts/refresh_satellite_caches.py`.
- Meteomatics son fotos de modelo, no promedios ni observaciones, y solo 2023–2024.
- Las pruebas describen el periodo analizado; no atribuyen causas.

---

## 10. Preguntas de defensa ante el jurado

1. **"Si la autocorrelación infla los p-valores, ¿por qué confiar en Pettitt?"** Porque su p-valor sale de un bootstrap por bloques que conserva la memoria de la serie, no de la fórmula asintótica.
2. **"¿Qué aporta el R² de 0,89?"** El nowcast usa el clima del mismo mes. Lo comparamos con persistencia (0,76). La ganancia del pronóstico ronda 0,02 a un mes y queda por debajo de 0,01 a tres meses.
3. **"¿Por qué Alaska tiene R² negativo?"** El suelo congelado casi no cambia; el modelo no sirve ahí y la app lo muestra sin maquillaje.
4. **"¿El punto de la Amazonía central es selva?"** No: cae sobre Manaos y el río Negro. Lo declaramos y dejamos el script para muestrear cobertura representativa.
5. **"¿La subida de lluvia en Napo es clima?"** Cuatro variables saltan el mismo mes de 2007; la app lo marca como posible cambio en los datos y pide contrastar con estaciones.
6. **"¿Cómo sé que los números de la presentación salen de la app?"** Los textos de los casos se generan desde el motor y una prueba automática falla si alguna afirmación deja de cumplirse.
7. **"¿Por qué Acker et al. (2009)?"** Su análisis de series de clorofila SeaWiFS en costas con influencia de ríos usa anomalías mensuales y anuales y excluye un evento extremo (la crecida del Po de 2000) para probar si la tendencia persiste; TerraLens sigue ese esquema y lo complementa con correcciones por memoria y por cambios en los datos.
