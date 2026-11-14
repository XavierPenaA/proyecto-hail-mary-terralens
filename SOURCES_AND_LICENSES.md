# TerraLens — Third-Party Data Sources, Open-Source Libraries & Scientific Citations

In strict compliance with the **2026 NASA Space Apps Challenge Participant Terms and Conditions**, every original artifact in TerraLens is licensed under the **Apache License 2.0** (`LICENSE`), and all NASA, interagency, and open-source third-party materials are documented below with their respective licenses and references.

---

## 1. Space Agency & Climate Datasets

| Source / Dataset | Provider | Coverage Used in TerraLens | License / Access Terms | Reference URL |
| :--- | :--- | :--- | :--- | :--- |
| **NASA POWER** (MERRA-2 / CERES SYN1deg / GPM IMERG) | NASA Langley Research Center (LaRC) | Monthly & regional climate cubes (`1981–2025`), 11 global benchmark regions, 96-cell Ecuador grid (`~55 km`), live `(lat, lon)` coordinate probe | Open Public Data — NASA Earthdata Open Data Policy | https://power.larc.nasa.gov/ |
| **NASA Terra MODIS MOD13Q1 v061 NDVI** | NASA EOSDIS LP DAAC (via APPEEARS / ORNL) | 250 m 16-day vegetation greenness stress passes (`115` passes, `2021–2025`, Paute Basin) | Open Public Data — NASA Earthdata Open Data Policy | https://modis.gsfc.nasa.gov/ |
| **NASA Earthdata CMR / STAC Catalog** | NASA Earthdata | Multi-scale granule catalog discovery metadata (`MOD13Q1`, `MERRA-2`, `CERES`) | Open Public API — NASA Earthdata | https://cmr.earthdata.nasa.gov/ |
| **NOAA Oceanic Niño Index (ONI)** | NOAA Climate Prediction Center (CPC) | Monthly ENSO SST anomaly index (`1981–2025`) for South American teleconnection dipole verification | U.S. Government Public Domain (No restrictions) | https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php |
| **Meteomatics Weather API** | Meteomatics AG | Topographically downscaled `1 km` soil moisture deficit, `-15 cm` moisture index, evapotranspiration, temperature, and precipitation (`2023–2024`) | Free / Academic Open API Tier (Public educational use cited) | https://www.meteomatics.com/en/weather-api/ |

---

## 2. Open-Source Software Libraries (OSI-Approved Licenses)

| Package / Library | Role in TerraLens | OSI License | Source / URL |
| :--- | :--- | :--- | :--- |
| **FastAPI** | High-speed REST API & static UI host | MIT License | https://fastapi.tiangolo.com/ |
| **Uvicorn** | ASGI web server | BSD 3-Clause License | https://www.uvicorn.org/ |
| **NumPy** | Numerical array transformations & deseasonalization | BSD 3-Clause License | https://numpy.org/ |
| **Pandas** | Multi-sensor time-series alignment & CSV export | BSD 3-Clause License | https://pandas.pydata.org/ |
| **SciPy** | Theil-Sen slopes, Mann-Kendall variance & statistical tests | BSD 3-Clause License | https://scipy.org/ |
| **Scikit-Learn** | `HistGradientBoostingRegressor`, `RandomForestRegressor`, `IsolationForest`, `LocalOutlierFactor`, `EllipticEnvelope`, `PCA`, `TimeSeriesSplit` | BSD 3-Clause License | https://scikit-learn.org/ |
| **Joblib** | Model persistence (`terralens_ml_suite.joblib`) | BSD 3-Clause License | https://joblib.readthedocs.io/ |
| **Pytest** | Automated scientific validation harness | MIT License | https://docs.pytest.org/ |
| **Leaflet.js** | Interactive regional + global trend & coordinate map | BSD 2-Clause License | https://leafletjs.com/ |
| **Chart.js** | Interactive time-series, PCA phase space & sensitivity charts | MIT License | https://www.chartjs.org/ |
| **CARTO / OpenStreetMap Tiles** | Dark & light cartographic reference basemaps | ODbL / CC BY 4.0 (with attribution rendered on map) | https://carto.com/attributions |
| **Plus Jakarta Sans & JetBrains Mono** | Observatory typography (Google Fonts) | SIL Open Font License 1.1 | https://fonts.google.com/ |

---

## 3. Peer-Reviewed Scientific & Statistical Methods Cited

1. **Acker, J., et al. (2009).** *Use of NASA Giovanni to Examine Regional Time-Series Trends and Anomaly Signatures.* NASA Goddard Space Flight Center (`NASA_Reference_Paper_20090042747.pdf`). Used as scientific motivation for satellite trend verification and extreme-event sensitivity analysis.
2. **Hamed, K. H., & Rao, A. R. (1998).** *A modified Mann-Kendall trend test for autocorrelated data.* Journal of Hydrology, 204(1–4), 182–196. Used to prevent positive lag-1 autocorrelation from inflating false climate trend significance.
3. **Santer, B. D., et al. (2000).** *Statistical significance of trends and trend differences in layer-average atmospheric temperature time series.* Journal of Geophysical Research, 105(D6), 7337–7356. Used to compute effective degrees of freedom (`n_eff`).
4. **Sen, P. K. (1968).** *Estimates of the regression coefficient based on Kendall's tau.* JASA, 63(324), 1379–1389. Non-parametric median slope estimator resistant to ENSO/drought spikes.
5. **Pettitt, A. N. (1979).** *A non-parametric approach to the change-point problem.* Applied Statistics, 28(2), 126–135. Identifies exact instrument boundaries (ERBE→CERES 2001) and abrupt structural climate shifts.
6. **Benjamini, Y., & Hochberg, Y. (1995).** *Controlling the false discovery rate: a practical and powerful approach to multiple testing.* JRSS-B, 57(1), 289–300. Controls family-wise false discoveries across multi-variable tables.

---

## 4. Team Attribution & AI Disclosure

- **Team Name:** Proyecto Hail Mary (Ecuador)
- **Members:**
  - **Xavier Sebastián Peña Avilés** (`@xavierpena` · Team Owner)
  - **Bryam Astudillo** (`@bryamaa`)
  - **Juan Jose Coronel Crespo** (`@jcoronel`)
  - **David Sebastián Célleri Bojorque** (`@celleri`)
- **AI Tool Disclosure:** AI coding assistants were used during prototyping, bilingual UI engineering (`EN`/`ES`), visualization scripting, and documentation formatting. All satellite time series (`1981–2025`), statistical hypotheses, chronological 3-way ML training splits, and unit tests were executed deterministically and verified directly against real NASA/NOAA data.
