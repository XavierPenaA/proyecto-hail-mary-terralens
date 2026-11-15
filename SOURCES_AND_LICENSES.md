# TerraLens — Data Sources, Libraries and Citations

TerraLens's own code, documentation and visuals are licensed under the **Apache License 2.0** (`LICENSE`). Third-party data and software keep their own terms, listed below.

---

## 1. Data

| Dataset | Provider | Use in TerraLens | Terms | Reference |
| :--- | :--- | :--- | :--- | :--- |
| **NASA POWER** monthly point and regional API (sources reported by the API: MERRA-2, GEWEX SRB, CERES SYN1deg) | NASA Langley Research Center | Monthly 1981–2025 series for 11 regions, a 96-cell grid over Ecuador, and the live coordinate probe. `PRECTOTCORR` is MERRA-2 bias-corrected precipitation; solar irradiance is GEWEX SRB to 2000 and CERES SYN1deg from 2001 | NASA Earthdata open data policy | https://power.larc.nasa.gov/ |
| **MODIS MOD13Q1 v061 NDVI** | NASA ORNL DAAC (MODIS/VIIRS subsets web service) | 250 m, 16-day NDVI, 2021–2025, for five regions | NASA Earthdata open data policy | https://modis.ornl.gov/data/modis_webservice.html |
| **Planetary Computer STAC** (`modis-13Q1-061` collection) | Microsoft | Granule metadata shown as provenance | Planetary Computer terms of use | https://planetarycomputer.microsoft.com/ |
| **Oceanic Niño Index (ONI)** | NOAA Climate Prediction Center | Monthly ENSO index for correlations and event context | U.S. Government public domain | https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt |
| **ERA5-Land** (ECMWF reanalysis, Copernicus Climate Change Service) via the **Open-Meteo** historical API | ECMWF / Copernicus C3S; Open-Meteo | Independent check of the MERRA-2 temperature and soil-moisture anomalies (`scripts/fetch_era5land.py`, cached monthly in `data/cache/era5land_monthly.json`). Precipitation was not returned and the Altiplano soil-moisture cell is constant, so both are excluded | Open-Meteo data: CC BY 4.0, attribution to Open-Meteo.com required; ERA5-Land contains modified Copernicus Climate Change Service information (license: Copernicus products licence) | https://open-meteo.com/en/docs/historical-weather-api · https://doi.org/10.24381/cds.e2161bac |
| **Meteomatics Weather API** | Meteomatics AG (NASA Space Apps partner offer) | 1 km downscaled snapshots for four Ecuadorian regions, 2023–2024 | NASA Space Apps Global Collaborator Offer (non-commercial hackathon evaluation use; proprietary Meteomatics data excluded from the Apache 2.0 code license; support: support@meteomatics.com) | https://www.meteomatics.com/en/meteomatics-and-nasa-space-apps-challenge/ |

## 2. Software

| Package | Role | License | Source |
| :--- | :--- | :--- | :--- |
| FastAPI | REST API and static file server | MIT | https://fastapi.tiangolo.com/ |
| Uvicorn | ASGI server | BSD-3-Clause | https://www.uvicorn.org/ |
| NumPy | Numerical arrays | BSD-3-Clause | https://numpy.org/ |
| pandas | Time-series tables | BSD-3-Clause | https://pandas.pydata.org/ |
| SciPy | Theil-Sen slopes, distributions, ranks | BSD-3-Clause | https://scipy.org/ |
| scikit-learn | Isolation Forest, Local Outlier Factor, PCA, K-Means, Random Forest, gradient boosting, Ridge, TimeSeriesSplit | BSD-3-Clause | https://scikit-learn.org/ |
| joblib | Optional export of the benchmark model (`train_models.py --save-model`) | BSD-3-Clause | https://joblib.readthedocs.io/ |
| pytest, httpx | Test suite (development only) | MIT, BSD-3-Clause | https://docs.pytest.org/ |
| Chart.js 4.4.1 | Charts (vendored in `static/vendor/`) | MIT | https://www.chartjs.org/ |
| Leaflet 1.9.4 | Map (vendored in `static/vendor/leaflet/`) | BSD-2-Clause | https://leafletjs.com/ |
| Esri World Light/Dark Gray Canvas basemaps | Map tiles (loaded online) | Esri terms of use; attribution "Esri, HERE, Garmin, © OpenStreetMap contributors" shown on the map | https://www.esri.com/en-us/legal/terms/full-master-agreement |
| Instrument Sans, IBM Plex Mono | Observatory typefaces, served locally | SIL Open Font License 1.1 ([Instrument Sans](static/fonts/OFL-Instrument-Sans.txt), [IBM Plex Mono](static/fonts/OFL-IBM-Plex-Mono.txt)) | https://fonts.google.com/specimen/Instrument+Sans · https://fonts.google.com/specimen/IBM+Plex+Mono |
| Plus Jakarta Sans, JetBrains Mono | HTML pitch typefaces (Google Fonts) | SIL Open Font License 1.1 | https://fonts.google.com/ |

## 3. Methods cited

1. **Acker, J. G., McMahon, E., Shen, S., Hearty, T., & Casey, N. (2009).** *Time-series analysis of remotely-sensed SeaWiFS chlorophyll in river-influenced coastal regions.* Preprint, NASA NTRS 20090042747. Monthly and annual anomaly analysis and the exclusion of an extreme event to test a trend (their Po River example).
2. **Hamed, K. H., & Rao, A. R. (1998).** A modified Mann-Kendall trend test for autocorrelated data. *Journal of Hydrology*, 204(1–4), 182–196.
3. **Santer, B. D., et al. (2000).** Statistical significance of trends and trend differences in layer-average atmospheric temperature time series. *JGR*, 105(D6), 7337–7356.
4. **Sen, P. K. (1968).** Estimates of the regression coefficient based on Kendall's tau. *JASA*, 63(324), 1379–1389.
5. **Pettitt, A. N. (1979).** A non-parametric approach to the change-point problem. *Applied Statistics*, 28(2), 126–135. Used with a block-bootstrap p-value.
6. **Benjamini, Y., & Hochberg, Y. (1995).** Controlling the false discovery rate: a practical and powerful approach to multiple testing. *JRSS B*, 57(1), 289–300.
7. **Bretherton, C. S., et al. (1999).** The effective number of spatial degrees of freedom of a time-varying field. *Journal of Climate*, 12(7), 1990–2009. Effective sample size for correlations.
8. **Kass, R. E., & Raftery, A. E. (1995).** Bayes factors. *JASA*, 90(430), 773–795. BIC evidence thresholds for the step-versus-trend check.
9. **Wilks, D. S. (2016).** "The stippling shows statistically significant grid points". *BAMS*, 97(12), 2263–2273. Field significance on the map.
10. **Muñoz-Sabater, J., et al. (2021).** ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13(9), 4349–4383. Independent reanalysis used to check MERRA-2.

## 4. Team and AI disclosure

- **Team:** Proyecto Hail Mary (Ecuador) — Xavier Sebastián Peña Avilés (team owner), Bryam Astudillo, Juan Jose Coronel Crespo, David Sebastián Célleri Bojorque.
- **AI disclosure:** AI coding assistants were used for software engineering, the bilingual interface, visualization code, tests and documentation. All statistics run deterministically on the cached NASA and NOAA data with fixed random seeds.
