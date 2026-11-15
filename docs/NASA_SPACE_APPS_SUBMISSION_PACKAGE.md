# NASA Space Apps Submission Package — Proyecto Hail Mary (TerraLens)

> English text for the Space Apps project page, plus the 30-second video and 7-slide storyboard.
> Every number quoted here is produced by the app; re-check them in the guided cases after any data refresh
> (`pytest tests/test_case_findings.py` fails if a claim stops holding).

---

## Part 1: Project page fields

### 1. Project name
TerraLens: Earth System Trend Detective

### 2. High-level summary (≈150 words)
A straight line fitted to a satellite or reanalysis series can report a trend that is not there. Three traps cause most false trends: month-to-month memory (autocorrelation) that inflates significance, changes in the data source that create steps, and single extreme events that tilt the line. TerraLens is an interactive observatory that tests every trend against those traps. It deseasonalizes 45 years of monthly NASA POWER data (1981–2025) for 11 regions and a 96-cell grid over Ecuador, following Acker et al. (2009), then applies Theil-Sen slopes with Hamed-Rao and Santer corrections, Pettitt change points with a block bootstrap, a step-versus-trend check, false discovery rate control, ENSO correlations with an effective sample size and a three-detector anomaly test. Six guided cases, written from live results, show each trap on real data. The interface is bilingual (English/Spanish).

### 3. Link to project demo
* 30-second video or 7-slide deck, in English, publicly accessible without login.

### 4. Link to final project
* **Live demo:** add the public URL here before submitting.
* **Repository (Apache 2.0):** `https://github.com/XavierPenaA/proyecto-hail-mary-terralens`
* **Team (Proyecto Hail Mary · Ecuador):** Xavier Sebastián Peña Avilés (team owner), Bryam Astudillo, Juan Jose Coronel Crespo, David Sebastián Célleri Bojorque.

### 5. Detailed description

#### The problem
1. **Memory.** Root-zone soil moisture in the Paute basin has a lag-1 autocorrelation of 0.94, so 540 months carry roughly 16 independent values. A plain regression reports a trend with p = 0.030; after the Santer correction p = 0.73 and the annual-mean test gives p = 0.22.
2. **Data changes.** NASA POWER solar irradiance comes from GEWEX SRB until 2000 and from CERES SYN1deg from January 2001. Over 1984–2025 the coast of Ecuador seems to dim by 0.25 MJ/m²/day per decade; a direct test finds a −0.69 MJ/m²/day step at January 2001, and within 2001–2025 the trend disappears (p = 0.83). In the Pantanal the same splice goes the other way, and a smaller increase remains after 2001.
3. **Steps that look like trends.** In the Napo foothills four variables shift level within two months of August 2007, a typical sign of a change in the data inputs; their "significant" trends need checking against stations.
4. **Teleconnections.** El Niño raises rainfall on Ecuador's coast and lowers it in the central Amazon, but their long-term trends do not mirror each other.

#### Who it is for
Environmental analysts, hydrologists, water and energy planners, and students who must decide whether a long-term signal is real before acting on it. Ecuador is the focus because within 400 km it spans an El Niño coast, narrow Andean hydropower basins and Amazon headwaters; reference sites in Brazil, Peru, the USA (Arctic and the Mississippi Delta studied by Acker et al.) and Spain test the method elsewhere.

#### How it works
* **Trends:** monthly anomalies, annual means and seasonal subsets (December counted with the following January–February); Theil-Sen slope; Hamed-Rao modified Mann-Kendall and Santer-corrected OLS; Benjamini-Hochberg q-values across variables and across map cells.
* **Change points:** Pettitt location with a circular block-bootstrap p-value; BIC comparison of a single step versus a gradual trend; a direct test of documented source changes; a warning when several variables shift together.
* **Anomalies:** Isolation Forest, Local Outlier Factor and Mahalanobis distance, each with a data-driven threshold; a month counts when two agree. The outlier test of Acker et al. (2009) recomputes every trend without those months.
* **Machine learning.** We select models using chronological validation by whole month, fit each fold's climatology on its training data, and compare the selected models with persistence on held-out years. Test R² is about 0.89 versus 0.76 for the spatial nowcast (combining concurrent climate with antecedent soil memory). The forecast gains about 0.02 one month ahead and less than 0.01 three months ahead. In the live observatory, regional Random Forest models explicitly isolate purely external atmospheric forcing (excluding soil autoregression) to evaluate true climatic drivers without persistent memory bias.
* **Calibration.** A Monte Carlo study (1000 trend-free AR(1) series per persistence level, `scripts/calibration_study.py`) showed that Hamed-Rao alone called 11–28% of them significant when month-to-month memory was r₁ ≥ 0.5. The verdict now requires both Hamed-Rao and Santer to agree (false positives 3–6%), the Theil-Sen interval is widened by the effective sample size, and the step-versus-trend BIC uses it too.
* **Independent check.** Monthly anomalies of MERRA-2 are compared with ERA5-Land in all 11 regions (`scripts/validate_era5land.py`). Temperature agrees (median r = 0.89). Soil moisture agrees poorly (median r = 0.41) and its trends are not reproduced, e.g. Napo rises in MERRA-2 and falls in ERA5-Land. This is agreement between two reanalyses, not error against observations; SMAP is pending.
* **Map and probe:** trend map over Ecuador with a diverging scale and FDR-controlled significance, MODIS NDVI, Meteomatics 1 km snapshots and a live NASA POWER probe for any coordinate.

#### Limitations (stated in the app)
~55 km cells (solar ~110 km); reanalysis rather than stations; one point per region (the Guayas and Manaus points fall on cities and rivers); Meteomatics values are 2023–2024 model snapshots; the soil-moisture models fail in the Arctic (negative test R²) for a reason we have not isolated; and ERA5-Land, an independent reanalysis, confirms MERRA-2 temperature trends but not its soil-moisture trends (see the validation below).

### 6. Data and tools
1. **NASA POWER** monthly point and regional API: MERRA-2 temperature, humidity, soil wetness and bias-corrected precipitation; GEWEX SRB and CERES SYN1deg solar irradiance (1981–2025).
2. **NASA ORNL DAAC MODIS subsets (MOD13Q1 v061):** 250 m, 16-day NDVI (2021–2025).
3. **NOAA CPC Oceanic Niño Index.**
4. **Meteomatics** (Space Apps partner): 1 km downscaled snapshots, 2023–2024.
5. **Microsoft Planetary Computer STAC:** MODIS granule metadata for provenance.
6. **ERA5-Land** (Copernicus Climate Change Service / ECMWF) through the Open-Meteo historical API (CC BY 4.0): monthly temperature and soil moisture used as an independent check of MERRA-2.
7. **Reference method:** Acker, J. G., McMahon, E., Shen, S., Hearty, T., & Casey, N. (2009). *Time-series analysis of remotely-sensed SeaWiFS chlorophyll in river-influenced coastal regions.* NASA NTRS 20090042747.

### 7. AI disclosure
All statistics and models run deterministically on the cached data with fixed random seeds. AI coding assistants were used for software engineering, interface work, testing and documentation. No NASA logos are used.

---

## Part 2: 30-second video or 7-slide deck

### Option A: 30-second video (record the running app)
* **0:00–0:06 Hook.** Long-term trends view, answer card visible. *"A straight line can show a climate trend that isn't there. TerraLens checks every trend against three traps."*
* **0:06–0:14 Trap 1, memory.** Guided case 2. *"Soil moisture remembers last month. Treat 540 months as independent and you get p = 0.03; correct for memory and the trend is gone."*
* **0:14–0:22 Trap 2, data change.** Guided case 3, then turn on the 2001 filter. *"The coast seems to dim, but the data source changed in January 2001. Within the homogeneous CERES era the trend disappears."*
* **0:22–0:30 Steps and extremes.** Guided case 6, then the map. *"Four variables jump in the same month: a data change, not climate. Every view in English and Spanish, for any coordinate on Earth."*

### Option B: 7-slide deck
1. **Title:** TerraLens: Earth System Trend Detective — Proyecto Hail Mary.
2. **The three traps:** memory, data changes, extremes (one sentence and one number each, from the cases).
3. **Method:** Acker et al. (2009) anomalies + Hamed-Rao/Santer + Pettitt with bootstrap + step-vs-trend + FDR.
4. **Trap 1 on real data:** guided case 2 (Paute soil moisture).
5. **Trap 2 on real data:** guided case 3 (2001 solar splice), before and after the filter.
6. **Machine learning, compared with persistence:** nowcast vs. forecast vs. persistence table.
7. **Impact and limits:** who uses it, bilingual access, live probe, and the stated limitations.
