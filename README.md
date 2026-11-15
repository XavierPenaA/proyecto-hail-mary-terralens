# TerraLens — Earth System Trend Detective

**2026 NASA International Space Apps Challenge** · Challenge: [Be An Earth System Trend Detective!](https://www.spaceappschallenge.org/2026/challenges/earth-system-trend-detective/)
**Team:** [Proyecto Hail Mary](https://www.spaceappschallenge.org/2026/find-a-team/proyecto-hail-mary/) (Ecuador) — Xavier Sebastián Peña Avilés, Bryam Astudillo, Juan Jose Coronel Crespo, David Sebastián Célleri Bojorque
**License:** [Apache 2.0](LICENSE) · data and library licenses in [SOURCES_AND_LICENSES.md](SOURCES_AND_LICENSES.md)

---

## What it does

A straight line fitted to a satellite or reanalysis series can report a trend that is not there. TerraLens tests every trend against the three traps that cause most false trends:

1. **Memory.** Persistent variables (soil moisture, for example) make weak trends look significant.
2. **Data changes.** A new sensor or new input data creates a step that looks like climate change.
3. **Extremes.** One huge El Niño or drought near the end of the record tilts the line.

For any of 11 regions (or any coordinate through the live probe) and six NASA POWER variables, the observatory answers in one sentence whether there is a real long-term trend, then shows the evidence. Six guided cases walk through each trap on real data; their text is written from the numbers the engine computes at start-up, so it cannot drift from the results.

## Quick start

```bash
pip install -r requirements.txt
python run_terralens.py            # opens http://127.0.0.1:8085
```

Or run the server directly: `python -m uvicorn api:app --host 127.0.0.1 --port 8085`. The API documentation is at `/docs`.
At start-up the server precomputes every region in the background (about 40 s); later clicks answer in a few milliseconds.

Tests:

```bash
pip install -r requirements-dev.txt
pytest
```

The suite checks the statistics on synthetic series with known answers, the API contract and validation, the cached data, the translation files, and every data-dependent claim made by the guided cases.

## Methods

| Step | Method | Reference |
| --- | --- | --- |
| Deseasonalize | Monthly anomalies, annual means, seasonal subsets (DJF uses the previous December) | Acker et al. (2009) |
| Trend size | Theil-Sen slope | Sen (1968) |
| Trend significance | Modified Mann-Kendall with autocorrelation correction; OLS with lag-1 effective sample size; annual-mean test as a second check | Hamed & Rao (1998); Santer et al. (2000) |
| Change points | Pettitt location with a circular block-bootstrap p-value | Pettitt (1979) |
| Step or trend? | BIC comparison of one level shift versus a linear trend; direct test of documented source changes (solar, January 2001); warning when several variables shift within two months | Kass & Raftery (1995) |
| Many tests | Benjamini-Hochberg q-values across the six variables and across the 96 map cells | Benjamini & Hochberg (1995); Wilks (2016) |
| ENSO links | Correlation with the ONI at 0–6 month leads, effective sample size for persistent series | Bretherton et al. (1999) |
| Anomalous months | Isolation Forest, Local Outlier Factor and Mahalanobis distance (χ², p < 0.01), each with a data-driven threshold; a month counts when two agree | — |
| Outlier test | Every trend recomputed without the anomalous months | Acker et al. (2009) |
| Soil-moisture drivers | Random Forest on concurrent and lagged drivers, reported with out-of-sample R² next to a persistence baseline | — |

### Machine-learning benchmark

`python train_models.py` evaluates soil-moisture models over the Ecuador grid and the 11 regions and writes `models/training_manifest.json` for the *Model performance* view. Five chronological folds group entire months and fit climatology on the training portion of each fold. Model selection uses cross-validation on 1981–2018, followed by evaluation in 2019–2025. The three-month holdout starts in March 2019 so its earliest prediction can use the December 2018 training cutoff.

* **Nowcast** (uses same-month weather). The model selected by cross-validation scores approximately 0.89 on the test period versus 0.76 for persistence.
* **Forecast** (only data known in advance). The selected models gain about 0.02 R² over persistence one month ahead and less than 0.01 three months ahead.
* **Regional driver models:** fail in the Arctic (negative R²) and lose to persistence in several regions; the app says so.

`--save-model` also writes the best nowcast model with `joblib` (not committed).

## Project layout

```
api.py                     FastAPI app: validation, caching, JSON API and the static observatory
run_terralens.py           Starts the server and opens the browser
train_models.py            Offline ML benchmark -> models/training_manifest.json
core/
  catalog.py               Data period, variables, regions (qualitative text only)
  stats_engine.py          Trend, change-point, homogeneity and ENSO statistics
  ml_engine.py             Anomaly detectors, regimes, driver model, outlier test rows
  analysis_service.py      Memoized analyses shared by the API and the cases
  case_studies.py          Guided cases and their findings, written from live numbers
  connectors.py            Cached data access, live NASA POWER probe, MODIS/STAC fetchers
scripts/refresh_satellite_caches.py   Re-downloads MODIS NDVI (quality-filtered window) and STAC metadata
static/                    index.html, css/, js/ (ES modules per view), i18n/ (en.json, es.json), vendor/
tests/                     pytest suite
data/cache/                Cached NASA POWER, NOAA ONI, MODIS, Meteomatics and STAC data
```

Adding a language means adding `static/i18n/<lang>.json` with the same keys (a test checks parity) and listing it in `static/js/i18n.js`.

## Data

* **NASA POWER** monthly point and grid data, 1981–2025: MERRA-2 temperature, relative humidity, soil wetness and bias-corrected precipitation; GEWEX SRB (to 2000) and CERES SYN1deg (from 2001) solar irradiance.
* **NOAA CPC Oceanic Niño Index.**
* **NASA ORNL DAAC MODIS MOD13Q1 v061** NDVI, 250 m, 16-day, 2021–2025.
* **Meteomatics** 1 km downscaled snapshots, one day every 10 days, 2023–2024 (Space Apps partner offer).
* **Microsoft Planetary Computer STAC** metadata for the MODIS granules (provenance only).

The observatory reads only the caches in `data/cache/`, so it works offline. The live probe writes to `data/cache/live/` (ignored by git, capped at 200 files, 20 requests per minute).

## Limitations

* NASA POWER cells are about 55 km (solar about 110 km); one value mixes very different terrain in the Andes.
* MERRA-2 is a reanalysis, not station data; changes in its inputs can create steps (see the Napo case).
* Each region is a single point. The Guayas and Manaus points fall on cities and rivers, so their MODIS NDVI describes water and urban land. Set `modis_point` for a region in `core/catalog.py` and run `scripts/refresh_satellite_caches.py` to sample representative land cover.
* Meteomatics values are model snapshots, not monthly means or observations.
* Trend tests describe the period analysed; they do not attribute causes.

## References

* Acker, J. G., McMahon, E., Shen, S., Hearty, T., & Casey, N. (2009). *Time-series analysis of remotely-sensed SeaWiFS chlorophyll in river-influenced coastal regions.* Preprint, NASA NTRS 20090042747.
* Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*, 57(1), 289–300.
* Bretherton, C. S., Widmann, M., Dymnikov, V. P., Wallace, J. M., & Bladé, I. (1999). The effective number of spatial degrees of freedom of a time-varying field. *Journal of Climate*, 12(7), 1990–2009.
* Hamed, K. H., & Rao, A. R. (1998). A modified Mann-Kendall trend test for autocorrelated data. *Journal of Hydrology*, 204, 182–196.
* Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *JASA*, 90(430), 773–795.
* Pettitt, A. N. (1979). A non-parametric approach to the change-point problem. *Applied Statistics*, 28(2), 126–135.
* Santer, B. D., et al. (2000). Statistical significance of trends and trend differences in layer-average atmospheric temperature time series. *JGR*, 105(D6), 7337–7356.
* Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *JASA*, 63(324), 1379–1389.
* Wilks, D. S. (2016). "The stippling shows statistically significant grid points": how research results are routinely overstated and overinterpreted, and what to do about it. *BAMS*, 97(12), 2263–2273.

## AI disclosure

All statistics and models run deterministically on the cached data with fixed random seeds. AI coding assistants were used for software engineering, interface work, testing and documentation. No official NASA logos are used.
