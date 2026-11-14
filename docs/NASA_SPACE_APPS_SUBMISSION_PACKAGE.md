# Official NASA Space Apps Submission Package — Proyecto Hail Mary (`TerraLens v2.1`)

> **Ready-to-paste English text for the official NASA Space Apps Challenge Project Page (`spaceappschallenge.org`), plus the strict 30-Second Video / 7-Slide Deck Storyboard.**

---

## PART 1: Official Project Page Fields (Copy & Paste in English)

### 1. Project Name
`TerraLens: Multi-Scale Earth System Trend, Breakpoint & Anomaly Detective`

### 2. High-Level Project Summary (Max ~150 words)
Earth observation time series are full of statistical traps: serial autocorrelation inflates false-positive trends by over 75%, satellite sensor handoffs create fake climate shifts, and coarse 55 km reanalysis grids smooth out severe mountain droughts. **TerraLens v2.1** is an interactive statistical and explainable machine learning workstation built on NASA GES DISC’s foundational methodology (*Acker et al., 2009*). Across 45 years of monthly NASA observations (1981–2025, 55,697 spatio-temporal records across 11 global benchmark regions and 96 regional grid cells), TerraLens combines autocorrelation-corrected inference (Hamed-Rao Modified Mann-Kendall, Santer $n_{\text{eff}}$, Theil-Sen slopes, Pettitt breakpoints, and Benjamini-Hochberg FDR) with an offline-trained 3-model anomaly ensemble (`IsolationForest`, `LOF`, `Mahalanobis`) and a zero-leakage Train/Val (`1981–2018`) + Locked Future Test (`2019–2024`) hydroclimatic predictor (Val CV $R^2 = 0.9214$, Unseen Test $R^2 = 0.9191$). Users can zoom from 55 km NASA `MERRA-2`/`CERES` grids down to 1 km Meteomatics topography and 250 m NASA Terra `MODIS` NDVI, or probe any coordinate on Earth live.

### 3. Link to Project Demo (30-Second Video OR 7-Slide Presentation)
* **Rule Reminder:** Must be **$\le 30$ seconds** (if video) or **$\le 7$ slides** (if slide deck), in **English**, and **100% publicly accessible without login** (YouTube Unlisted/Public or Google Drive/Slides with *"Anyone with the link can view"*).

### 4. Link to Final Project (Live Web App / Repository)
* **GitHub Repository (Apache 2.0):** `https://github.com/XavierPenaA/proyecto-hail-mary-terralens`
* **License & Full Third-Party Citations:** Included at repo root (`LICENSE` — Apache License 2.0, plus `SOURCES_AND_LICENSES.md` citing NASA, NOAA, Meteomatics, Microsoft Planetary Computer STAC, peer-reviewed papers, and all OSI-licensed Python/JS packages).
* **Team Members (Proyecto Hail Mary · Ecuador):**
  - Xavier Sebastián Peña Avilés (`@xavierpena` · Team Owner)
  - Bryam Astudillo (`@bryamaa`)
  - Juan Jose Coronel Crespo (`@jcoronel`)
  - David Sebastián Célleri Bojorque (`@celleri`)

### 5. Detailed Project Description
#### The Problem We Solved
When analyzing multi-decadal Earth science series across the atmosphere, hydrosphere, and biosphere, standard linear regression fails in four critical ways that we empirically proved across 107 spatial nodes:
1. **Serial Autocorrelation False Positives:** Monthly soil moisture (`GWETROOT`) and temperature (`T2M`) exhibit lag-1 autocorrelation up to $r_1 = 0.94$, reducing 540 monthly observations to as few as $n_{\text{eff}} = 16$ independent degrees of freedom. Naive OLS flags 38 of 48 regional series as significant ($p < 0.05$), many of which are statistical illusions.
2. **Satellite Calibration Splices:** NASA POWER's surface shortwave irradiance (`ALLSKY_SFC_SW_DWN`) transitions from GEWEX SRB (1984–2000) to CERES SYN1deg (January 2001–present). Without breakpoint detection, this sensor splice creates an artificial dimming trend ($p < 10^{-14}$), whereas the homogeneous CERES era (2001–2025) is completely flat ($p = 0.99$).
3. **Topographic Resolution Blindness:** During the 2023–2024 Andean hydroelectric drought in Ecuador's Paute/Mazar basin, 55 km `MERRA-2` grid cells averaged high-altitude paramo with humid Amazonian slopes, masking the valley drought.
4. **Opposite Regional Teleconnections (Dipoles):** A single global forcing (El Niño / ENSO) drives opposite-sign extremes simultaneously across South America.

#### Who Uses TerraLens (Target Users & Beneficiaries)
* **Hydropower & Reservoir Operators:** Avoid misallocating water reserves caused by 55 km grid averaging (e.g., Paute–Mazar hydroelectric complex) or statistical noise in soil wetness.
* **Agronomists & Agricultural Ministries:** Inspect 250 m Terra MODIS NDVI canopy stress alongside 1 km soil moisture deficits during ENSO shocks.
* **Climate Researchers & Data Journalists:** Audit any global coordinate against persistence bias ($r_1$), satellite sensor switches (ERBE→CERES 2001), and isolated El Niño leverage before publishing multi-decadal trend claims.
* **Educators & Citizen Scientists:** Explore 45 years of real NASA/NOAA data through an open Apache 2.0 bilingual (`EN`/`ES`) interface.

#### How TerraLens v2.1 Works
* **Long-Term Trends & Breakpoints (`Acker et al., 2009` Steps 1–3):** Computes monthly climatological normals ($\bar{X}_m \pm s_m$), deseasonalized monthly anomalies, annual mean verification, and seasonal sub-series (`DJF`, `MAM`, `JJA`, `SON`). Reports Theil-Sen median slopes alongside Santer $n_{\text{eff}}$, Hamed-Rao $p$-values, Benjamini-Hochberg FDR $q$-values, and Pettitt structural step-changes.
* **3-Model Unsupervised Anomaly Detective & Lagged Driver Attribution:** Flags compound multi-sphere extremes where at least 2 of 3 detectors (`IsolationForest`, `LocalOutlierFactor`, `Mahalanobis Distance`) agree, decomposing each event into exact $\pm\sigma$ contributions. A supervised `HistGradientBoostingRegressor` & `RandomForest` pipeline—trained with a zero-leakage chronological split (5-fold `TimeSeriesSplit` Validation on `1981–2018` achieving $R^2 = 0.9214$, and Locked Unseen Future Test on `2019–2024` achieving $R^2 = 0.9191$, outperforming `RidgeBaseline` Test $R^2 = 0.8907$)—identifies the exact lagged atmospheric drivers of root-zone soil moisture across 55,697 observations.
* **Acker Outlier Sensitivity Experiment:** Lets users remove ML-flagged extreme months in one click and re-evaluate all slopes to prove whether a 45-year trend is structural or driven by isolated El Niño shocks.
* **Multi-Scale Zoom (55 km $\rightarrow$ 1 km $\rightarrow$ 250 m) & Live Global Coordinate Probe:** Connects the 96-cell NASA POWER 2D Ecuador grid with 1 km Meteomatics downscaling (capturing the 2024 Paute soil moisture deficit jump from `10.5 mm` to `43.2 mm`), 250 m NASA Terra `MODIS` (`MOD13Q1` v061) NDVI passes, and Microsoft Planetary Computer `STAC API v1` granule verification.

### 6. Space Agency Data & Global Collaborator Tools Used
1. **NASA POWER API v2 (`MERRA-2` Assimilation & `CERES SYN1deg`):** 45-year monthly point series (`1981–2025`) for `T2M`, `PRECTOTCORR`, `GWETROOT`, `GWETTOP`, `ALLSKY_SFC_SW_DWN`, and `RH2M` across 11 global benchmark regions + 96-cell 2D regional grid over Ecuador.
2. **NASA ORNL DAAC MODIS REST API (`MOD13Q1` v061):** 250 m 16-day Terra MODIS Normalized Difference Vegetation Index (`NDVI`) series (575 satellite passes, `2021–2025`).
3. **NASA GES DISC Reference Methodology:** Acker et al. (2009), *Using Time Series for Data Discovery and Analysis in Earth Science* (`NASA/TM-2009-215858`).
4. **NOAA CPC Oceanic Niño Index (`ONI`):** Official monthly ENSO SST anomaly index (`1981–2025`).
5. **Meteomatics Weather API (NASA Space Apps Collaborator):** 1 km downscaled hydroclimatic series (`soil_moisture_deficit:mm`, `soil_moisture_index_-15cm:idx`, `evapotranspiration_24h:mm`, `t_2m:C`, `precip_24h:mm`).
6. **Microsoft Planetary Computer STAC API v1 (NASA Space Apps Collaborator):** SpatioTemporal Asset Catalog verification for `modis-13Q1-061` satellite granules.

### 7. Super Intelligence (SI) / AI & ML Disclosure
All statistical inference and machine learning models (`scikit-learn`: `IsolationForest`, `LocalOutlierFactor`, `HistGradientBoostingRegressor`, `RandomForestRegressor`, `PCA`, `KMeans`) execute deterministically on real satellite and reanalysis datasets. AI coding assistance (Google DeepMind Antigravity) was used to accelerate software engineering, UI design, and automated testing. Zero synthetic data or NASA logos are used.

---

## PART 2: Strict 30-Second Video Storyboard (OR 7-Slide Deck)

### Option A: 30-Second Video Script (Record Screen on `http://127.0.0.1:8085`)
* **0:00 – 0:05 (Hook & Problem):** Show the main **Long-Term Trends** view on `Paute Basin`. *"Earth observation series hide statistical traps: autocorrelation inflates false trends, sensor splices fake climate shifts, and 55 km grids miss mountain droughts. Meet TerraLens."*
* **0:05 – 0:12 (Trap #1 & #2 — Autocorrelation & 2001 Sensor Splice):** Click **Case 3 (2001 Sensor Splice)** on the left sidebar, then toggle **CERES 2001+ Guard**. *"TerraLens implements NASA GES DISC's Acker methodology with Hamed-Rao autocorrelation correction and Pettitt breakpoint detection—catching the January 2001 GEWEX-to-CERES satellite splice instantly."*
* **0:12 – 0:20 (Explainable ML & 5-Fold CV):** Click **Extreme Events & Drivers**, then **Model Performance (5-Fold CV)**. *"Our 3-model anomaly ensemble and 5-fold cross-validated gradient boosting suite—trained on 55,697 observations with 0.92 R²—attribute every extreme month in standard deviations."*
* **0:20 – 0:30 (Multi-Scale Zoom 55km $\rightarrow$ 1km $\rightarrow$ 250m):** Click **Regional Map & NDVI**. *"And when 55 km grids smooth out the Andes, TerraLens zooms into 1 km Meteomatics soil moisture deficit and 250 m NASA MODIS NDVI—turning raw satellite data into verified Earth science."*

### Option B: 7-Slide Presentation Structure (If Submitting Slide Deck)
1. **Slide 1 — Title & Mission:** *TerraLens v2.1: Multi-Scale Earth System Trend, Breakpoint & Anomaly Detective* (Team Proyecto Hail Mary — Universal Event).
2. **Slide 2 — The 3 Traps in Earth Observation Time Series:** Empirical proof of (1) Serial autocorrelation false positives ($r_1 = 0.94 \Rightarrow n_{\text{eff}} = 16$), (2) The Jan 2001 `GEWEX` $\rightarrow$ `CERES` satellite splice ($p < 10^{-14}$), and (3) 55 km Andean topographic smoothing.
3. **Slide 3 — Statistical Engine (`Acker et al., 2009` + `Hamed-Rao` + `Pettitt` + `BH-FDR`):** Screenshot of View 1 (`Long-Term Trends`) showing deseasonalized anomalies, Theil-Sen vs. OLS, Pettitt step-change line, and 6-variable FDR table.
4. **Slide 4 — Zero-Leakage 3-Way Train / Val / Future Test ML Pipeline (`55,697` Obs):** Screenshot of View 3 (`Model Performance`) showing `HistGradientBoostingRegressor` (Val CV `1981–2018`: $R^2 = 0.9214 \pm 0.0169$; Unseen Future Test `2019–2024`: $R^2 = 0.9191, \text{RMSE} = 0.2308\sigma$) and out-of-sample permutation importances.
5. **Slide 5 — Explainable 3-Model Anomaly Detective & Acker Outlier Test:** Screenshot of View 2 (`Extreme Events & Drivers`) & View 4 (`Outlier Sensitivity`) showing `IsolationForest + LOF + Mahalanobis` $\pm\sigma$ attribution and slope robustness when extremes are removed.
6. **Slide 6 — Multi-Scale Resolution Zoom (`55 km` $\rightarrow$ `1 km` $\rightarrow$ `250 m`) & Global 11-Node Network:** Screenshot of View 6 (`Regional Map & NDVI`) showing the 96-cell Ecuador trend map + all 11 Global Benchmark Nodes (including Arctic Polar Amplification in Utqiaġvik, Alaska `+0.579 °C/decade` vs. Acker et al.'s Mississippi Delta site), 250 m NASA Terra `MODIS` NDVI, 1 km `Meteomatics` Paute drought spike (`10.5` $\rightarrow$ `43.2 mm`), and `Microsoft Planetary Computer STAC` granules.
7. **Slide 7 — Impact, Live Global Probe & Modern Successor to NASA Hydrology Data Rods:** How TerraLens replaces the retiring NASA GES DISC Hydrology Data Rods workflow for researchers, hydropower operators, and climate analysts worldwide.
