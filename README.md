# TerraLens v2.1 — Multi-Scale Earth System Trend, Breakpoint & Anomaly Detective

**2026 NASA International Space Apps Challenge**  
**Challenge:** [Be An Earth System Trend Detective!](https://www.spaceappschallenge.org/2026/challenges/earth-system-trend-detective/) (Advanced · Earth Science · Software)  
**Team:** [Proyecto Hail Mary](https://www.spaceappschallenge.org/2026/find-a-team/proyecto-hail-mary/) · **Location:** Ecuador (Universal Event)  
**Team Members:** Xavier Sebastián Peña Avilés (`@xavierpena`), Bryam Astudillo (`@bryamaa`), Juan Jose Coronel Crespo (`@jcoronel`), David Sebastián Célleri Bojorque (`@celleri`)  
**License:** [Apache License 2.0](LICENSE) · **Third-Party & Data Licenses:** [SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md)

---

## Overview

**TerraLens v2.1** is a multi-scale Earth observation statistical and machine learning workstation built to answer the core challenge of *Be An Earth System Trend Detective!*: distinguishing genuine multi-decadal climate trends from serial autocorrelation noise, satellite calibration splices, coarse-grid topographic smoothing, and isolated ENSO shocks.

Following NASA GES DISC's foundational methodology (**Acker et al., 2009**, NASA/TM-2009-215858), TerraLens combines deterministic statistical inference with explainable multi-model machine learning across **55,697 monthly observations** spanning **107 spatial nodes** (1981–2025).

---

## Core Scientific & Machine Learning Capabilities

1. **Acker et al. (2009) 3-Step Deseasonalization & Trend Suite**:
   - **Step 1 (Monthly Deseasonalization):** Removes the 12-month climatological normal ($\bar{X}_m \pm s_m$) to compute monthly anomalies $A_{y,m}$ and standardized $Z_{y,m}$ scores.
   - **Step 2 (Annual Mean Verification):** Aggregates monthly anomalies into annual series to eliminate intra-annual seasonal persistence.
   - **Step 3 (Seasonal Sub-Series Breakdown):** Isolates decadal trends across `DJF`, `MAM`, `JJA`, and `SON`.
2. **Autocorrelation-Corrected Statistical Inference**:
   - **Santer et al. (2000) Effective Sample Size ($n_{\text{eff}}$):** Adjusts Ordinary Least Squares (OLS) degrees of freedom for lag-1 serial persistence $r_1$.
   - **Hamed & Rao (1998) Modified Mann-Kendall & Theil-Sen Estimator:** Variance-corrected non-parametric significance testing and outlier-resistant median slopes.
   - **Benjamini-Hochberg (1995) False Discovery Rate ($q$-value):** Controls false positives across simultaneous 6-parameter scans.
3. **Pettitt (1979) Non-Parametric Change-Point & Satellite Splice Detector**:
   - Automatically identifies abrupt structural breakpoints and sensor handoffs—such as the **January 2001 GEWEX SRB $\rightarrow$ CERES SYN1deg** solar irradiance splice ($p < 10^{-14}$)—and provides a one-click **CERES 2001+ Homogeneous Era Guard**.
4. **Offline-Trained & Cross-Validated Spatio-Temporal Machine Learning Suite (`train_models.py`)**:
   - **3-Model Unsupervised Anomaly Consensus:** Combines `IsolationForest` (500 trees), `LocalOutlierFactor` ($k=24$), and Robust `Mahalanobis` Distance ($D_M$) with quantitative $\pm\sigma$ per-variable attribution.
   - **Zero-Leakage 3-Way Train / Validation / Locked Future Test Hydroclimatic Benchmark:** Monthly climatologies and `5-Fold TimeSeriesSplit` CV run strictly on `1981–2018` (`43,296` rows), and final generalization is evaluated on an unseen locked future test set (`2019–2024`, `6,912` rows), achieving **Val CV $R^2 = 0.9214 \ (\pm 0.0169)$** and **Unseen Future Test $R^2 = 0.9191$ ($\text{RMSE} = 0.2308\sigma$)** with `HistGradientBoostingRegressor` (outperforming `RidgeBaseline` Test $R^2 = 0.8907$).
   - **PCA & K-Means Hydroclimatic Regime Discovery:** Validated via Silhouette Score scans across $k \in \{2, 3, 4, 5\}$.
5. **Multi-Scale Resolution Zoom (55 km $\rightarrow$ 1 km $\rightarrow$ 250 m) & Live Global Coordinate Probe**:
   - Integrates **NASA POWER (`MERRA-2` / `CERES`)** 0.5° grids, **Meteomatics 1 km** topographic downscaling (exposing the 2024 Paute/Mazar soil moisture deficit spike from `10.5 mm` to `43.2 mm`), **NASA ORNL DAAC Terra MODIS 250 m (`MOD13Q1` v061)** NDVI series, and **Microsoft Planetary Computer STAC API v1** granule verification.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Re-run the full Spatio-Temporal ML training & 5-fold CV pipeline
python train_models.py

# 3. Run the automated scientific verification test suite
pytest tests/test_scientific_validity.py

# 4. Launch the FastAPI + Interactive Web Observatory on http://127.0.0.1:8085
python -m uvicorn api:app --host 127.0.0.1 --port 8085
```

---

## Data Sources, Licenses & References

Every original code, documentation, and visual artifact in TerraLens is open-source under the **Apache License 2.0** (`LICENSE`). Full license metadata for all external datasets and OSI open-source libraries is documented in [`SOURCES_AND_LICENSES.md`](SOURCES_AND_LICENSES.md).

- **NASA POWER Project (MERRA-2 & CERES SYN1deg):** NASA Langley Research Center (LaRC) POWER API v2 (`https://power.larc.nasa.gov/`) — NASA Earthdata Open Data Policy.
- **NASA Terra MODIS 250m Vegetation Indices (MOD13Q1 v061):** NASA ORNL DAAC MODIS Global Subsets REST API (`https://modis.ornl.gov/`) — NASA Earthdata Open Data Policy.
- **NASA GES DISC Foundational Methodology:** Acker, J., Williams, R., Chiu, L., et al. (2009). *Using Time Series for Data Discovery and Analysis in Earth Science*. NASA/TM-2009-215858 (`20090042747`).
- **NOAA Climate Prediction Center (CPC):** Oceanic Niño Index (ONI v5, `https://www.cpc.ncep.noaa.gov/`) — U.S. Government Public Domain.
- **Meteomatics High-Resolution Downscaling API:** 1 km hydroclimatic parameters (`https://www.meteomatics.com/`, NASA Space Apps Partner Offer).
- **Microsoft Planetary Computer STAC API v1:** SpatioTemporal Asset Catalog (`https://planetarycomputer.microsoft.com/api/stac/v1`).
- **Statistical Citations:** Sen (1968), Pettitt (1979), Benjamini & Hochberg (1995), Hamed & Rao (1998), Santer et al. (2000).
- **Open-Source Software Stack:** FastAPI (`MIT`), Uvicorn (`BSD-3`), NumPy/Pandas/SciPy/Scikit-Learn/Joblib (`BSD-3`), Pytest (`MIT`), Leaflet (`BSD-2`), Chart.js (`MIT`), CARTO/OpenStreetMap Basemaps (`ODbL / CC BY 4.0`).

---

## Super Intelligence (SI) & AI/ML Disclosure

In accordance with NASA Space Apps Challenge guidelines:
- **Deterministic Statistical & ML Engine:** All statistical equations, $p$-values, Theil-Sen slopes, Pettitt change-points, and Scikit-Learn models (`IsolationForest`, `LocalOutlierFactor`, `HistGradientBoostingRegressor`, `RandomForestRegressor`, `PCA`, `KMeans`) run deterministically on empirical satellite and reanalysis observations. Zero synthetic or generative numbers are used.
- **Development Assistance:** AI coding assistance (Google DeepMind Antigravity) was used for software architecture, UI layout implementation, and automated unit testing. No official NASA logos or restricted agency insignia are used in this application.
