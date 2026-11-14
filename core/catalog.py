"""
Catalog of Empirically Verified Regions, Satellite Variables, Instrument Boundaries,
and Guided Detective Cases for TerraLens v2.1.
"""

from typing import Dict, Any, List

VARIABLES: Dict[str, Dict[str, Any]] = {
    "T2M": {
        "code": "T2M",
        "name": "Air Temperature at 2 Meters",
        "name_es": "Temperatura del Aire a 2 Metros",
        "short_name": "2m Air Temp",
        "short_name_es": "Temp. Aire 2m",
        "units": "°C",
        "decade_units": "°C / decade",
        "sphere": "Atmosphere (Thermal)",
        "sensor_source": "NASA MERRA-2 Reanalysis (M2TMNXSLV v5.12.4)",
        "resolution": "0.5° × 0.625° (~55 km)",
        "temporal_coverage": "1981–2025 (Monthly)",
        "known_splice": None,
        "student_explainer": (
            "Measures average air temperature 2 meters above the ground. "
            "Persistent positive anomalies indicate atmospheric warming or El Niño thermal pulses."
        ),
    },
    "PRECTOTCORR": {
        "code": "PRECTOTCORR",
        "name": "Corrected Total Precipitation",
        "name_es": "Precipitación Total Corregida",
        "short_name": "Precipitation",
        "short_name_es": "Precipitación",
        "units": "mm/day",
        "decade_units": "mm/day / decade",
        "sphere": "Hydrosphere (Atmospheric Water)",
        "sensor_source": "NASA MERRA-2 + GPCP Satellite-Gauge Corrected",
        "resolution": "0.5° × 0.625° (~55 km)",
        "temporal_coverage": "1981–2025 (Monthly)",
        "known_splice": None,
        "student_explainer": (
            "Measures daily average rainfall and snowfall corrected with satellite and rain-gauge observations. "
            "Reveals long-term drying, wetting, and extreme flood/drought pulses."
        ),
    },
    "GWETROOT": {
        "code": "GWETROOT",
        "name": "Root Zone Soil Wetness",
        "name_es": "Humedad del Suelo en Zona Radicular",
        "short_name": "Root Soil Moisture",
        "short_name_es": "Humedad Suelo Raíz",
        "units": "fraction (0–1)",
        "decade_units": "fraction / decade",
        "sphere": "Hydrosphere / Pedosphere",
        "sensor_source": "NASA MERRA-2 Land Surface Model (GWETROOT)",
        "resolution": "0.5° × 0.625° (~55 km)",
        "temporal_coverage": "1981–2025 (Monthly)",
        "known_splice": None,
        "student_explainer": (
            "Fraction of water saturation in the root zone (top ~100 cm of soil) available to plants and crops. "
            "Has high month-to-month memory (serial autocorrelation), requiring Hamed-Rao / Santer statistical correction."
        ),
    },
    "GWETTOP": {
        "code": "GWETTOP",
        "name": "Surface Soil Wetness (0–5 cm)",
        "name_es": "Humedad del Suelo Superficial (0–5 cm)",
        "short_name": "Surface Soil Moisture",
        "short_name_es": "Humedad Suelo Sup.",
        "units": "fraction (0–1)",
        "decade_units": "fraction / decade",
        "sphere": "Hydrosphere / Land Surface",
        "sensor_source": "NASA MERRA-2 Land Surface Model (GWETTOP)",
        "resolution": "0.5° × 0.625° (~55 km)",
        "temporal_coverage": "1981–2025 (Monthly)",
        "known_splice": None,
        "student_explainer": (
            "Water saturation in the top 5 cm of soil. Responds rapidly to evaporative demand, heatwaves, and rainfall deficits."
        ),
    },
    "ALLSKY_SFC_SW_DWN": {
        "code": "ALLSKY_SFC_SW_DWN",
        "name": "All-Sky Surface Shortwave Downward Irradiance",
        "name_es": "Irradiancia Solar de Onda Corta en Superficie",
        "short_name": "Solar Irradiance (Cloud Proxy)",
        "short_name_es": "Irradiancia Solar",
        "units": "MJ/m²/day",
        "decade_units": "MJ/m²/day / decade",
        "sphere": "Atmosphere (Radiation & Clouds)",
        "sensor_source": "NASA GEWEX SRB (1984–2000) + NASA CERES SYN1deg (2001–2025)",
        "resolution": "1.0° × 1.0° (~110 km)",
        "temporal_coverage": "1984–2025 (1981–1983 missing)",
        "known_splice": {
            "date": "2001-01",
            "year": 2001,
            "pre_sensor": "NASA GEWEX SRB v4 (1984–2000)",
            "post_sensor": "NASA CERES SYN1deg (2001–present)",
            "warning": (
                "INSTRUMENT SPLICE WARNING: NASA POWER transitions from GEWEX SRB to CERES SYN1deg in January 2001. "
                "Analyzing 1984–2025 as a single homogeneous series introduces an artificial step jump (p < 1e-8). "
                "Enable the Homogeneous CERES Era Filter (2001–2025) to measure true physical radiation trends."
            ),
        },
        "student_explainer": (
            "Solar energy reaching Earth's surface through clouds and aerosols. "
            "Also serves as a textbook case study in satellite data hygiene due to the 2001 GEWEX-to-CERES sensor transition."
        ),
    },
    "RH2M": {
        "code": "RH2M",
        "name": "Relative Humidity at 2 Meters",
        "name_es": "Humedad Relativa a 2 Metros",
        "short_name": "Relative Humidity",
        "short_name_es": "Humedad Relativa",
        "units": "%",
        "decade_units": "% / decade",
        "sphere": "Atmosphere (Moisture)",
        "sensor_source": "NASA MERRA-2 Reanalysis",
        "resolution": "0.5° × 0.625° (~55 km)",
        "temporal_coverage": "1981–2025 (Monthly)",
        "known_splice": None,
        "student_explainer": (
            "Ratio of current air water vapor to saturation at 2m height. "
            "Falling relative humidity combined with rising temperature accelerates evapotranspiration and wildfire risk."
        ),
    },
}

REGIONS: Dict[str, Dict[str, Any]] = {
    "EC_Coast_Guayas": {
        "id": "EC_Coast_Guayas",
        "name": "Ecuador Pacific Coast (Guayas / El Oro)",
        "name_es": "Costa del Pacífico de Ecuador (Guayas / El Oro)",
        "country": "Ecuador",
        "zone": "Ecuador & Andes Focus",
        "lat": -2.19,
        "lon": -79.89,
        "biome": "Tropical Pacific Coastal Plain",
        "has_modis_250m": True,
        "has_meteomatics_1km": True,
        "summary": (
            "Primary Pacific entry zone for El Niño (ENSO) warm SST anomalies. "
            "Exhibits extreme positive rainfall pulses (+4.8σ in 1997–98 El Niño) and strong positive correlation with NOAA ONI."
        ),
    },
    "EC_Andes_Paute": {
        "id": "EC_Andes_Paute",
        "name": "Southern Ecuadorian Andes (Paute / Mazar Basin)",
        "name_es": "Andes del Sur de Ecuador (Cuenca Paute / Mazar)",
        "country": "Ecuador",
        "zone": "Ecuador & Andes Focus",
        "lat": -2.61,
        "lon": -78.63,
        "biome": "Tropical Andean Hydroelectric Canyon (~2,844m grid avg)",
        "has_modis_250m": True,
        "has_meteomatics_1km": True,
        "summary": (
            "Ecuador's critical hydroelectric corridor (Mazar-Paute). Demonstrates why coarse 55 km MERRA-2 grids smooth narrow mountain valleys, "
            "requiring 1 km Meteomatics soil moisture deficit and 250 m NASA MODIS NDVI to resolve the 2023–2024 drought crisis."
        ),
    },
    "EC_Andes_Quito": {
        "id": "EC_Andes_Quito",
        "name": "Northern Ecuadorian Andes (Quito / Pichincha)",
        "name_es": "Andes del Norte de Ecuador (Quito / Pichincha)",
        "country": "Ecuador",
        "zone": "Ecuador & Andes Focus",
        "lat": -0.18,
        "lon": -78.47,
        "biome": "High-Altitude Inter-Andean Valley",
        "has_modis_250m": True,
        "has_meteomatics_1km": True,
        "summary": (
            "Exhibits statistically significant 45-year warming (+0.27 °C/decade, Hamed-Rao p < 0.0001) and drying "
            "(PRECTOTCORR -0.15 mm/day/decade, p = 0.044; RH2M -0.30%/decade, p = 0.0051)."
        ),
    },
    "EC_Amazon_Napo": {
        "id": "EC_Amazon_Napo",
        "name": "Ecuadorian Amazon Foothills (Sucumbíos / Napo)",
        "name_es": "Piedemonte Amazónico Ecuatoriano (Sucumbíos / Napo)",
        "country": "Ecuador",
        "zone": "Ecuador & Andes Focus",
        "lat": -0.47,
        "lon": -76.99,
        "biome": "Western Amazonian Piedmont",
        "has_modis_250m": False,
        "has_meteomatics_1km": True,
        "summary": (
            "Andean-Amazonian transition zone. Features a verified August 2007 reanalysis assimilation regime shift "
            "detected by the Pettitt change-point test (p < 1e-19)."
        ),
    },
    "EC_Galapagos": {
        "id": "EC_Galapagos",
        "name": "Galápagos Islands (Santa Cruz)",
        "name_es": "Islas Galápagos (Santa Cruz, Ecuador)",
        "country": "Ecuador",
        "zone": "Ecuador & Andes Focus",
        "lat": -0.63,
        "lon": -90.36,
        "biome": "Equatorial Pacific Marine-Terrestrial Ecosystem",
        "has_modis_250m": False,
        "has_meteomatics_1km": False,
        "summary": (
            "Direct oceanic thermometer for Niño 3 / 3.4 anomalies (T2M correlation with ONI r = +0.52, p < 0.0001) "
            "with significant long-term root-zone wetting (+0.011/decade, p = 0.0029)."
        ),
    },
    "BR_Amazon_Manaus": {
        "id": "BR_Amazon_Manaus",
        "name": "Central Amazon Basin (Manaus, Brazil)",
        "name_es": "Cuenca Amazónica Central (Manaos, Brasil)",
        "country": "Brazil",
        "zone": "South American Benchmark",
        "lat": -3.12,
        "lon": -60.02,
        "biome": "Central Lowland Tropical Rainforest",
        "has_modis_250m": True,
        "has_meteomatics_1km": False,
        "summary": (
            "Opposite pole of the South American ENSO dipole: while El Niño wets coastal Ecuador, it strongly dries Central Amazonia "
            "(GWETROOT r_ONI = -0.42, p < 0.0001), compounding a severe 45-year drying trend (-0.35 mm/day/decade, p = 0.0002)."
        ),
    },
    "BR_Pantanal": {
        "id": "BR_Pantanal",
        "name": "Brazilian Pantanal Tropical Wetland",
        "name_es": "Humedal Tropical del Pantanal (Brasil)",
        "country": "Brazil",
        "zone": "South American Benchmark",
        "lat": -17.75,
        "lon": -56.90,
        "biome": "World's Largest Tropical Flooded Savanna",
        "has_modis_250m": True,
        "has_meteomatics_1km": False,
        "summary": (
            "Hotspot of the most severe compound hydroclimatic trend in South America (1981–2025): warming at +0.53 °C/decade, "
            "rainfall declining by -0.50 mm/day/decade, and relative humidity dropping -3.07%/decade (all Hamed-Rao p < 0.0001)."
        ),
    },
    "PE_Altiplano": {
        "id": "PE_Altiplano",
        "name": "Peruvian-Bolivian Altiplano (Lake Titicaca)",
        "name_es": "Altiplano Peruano-Boliviano (Lago Titicaca)",
        "country": "Peru / Bolivia",
        "zone": "South American Benchmark",
        "lat": -15.84,
        "lon": -69.50,
        "biome": "High-Andean Endorheic Basin (~3,810m)",
        "has_modis_250m": False,
        "has_meteomatics_1km": False,
        "summary": (
            "High-altitude plateau showing sustained warming (+0.13 °C/decade, p = 0.037) and strong thermal coupling to El Niño (r_ONI = +0.38)."
        ),
    },
    "US_Mississippi_Delta": {
        "id": "US_Mississippi_Delta",
        "name": "Mississippi River Delta (Louisiana, USA — Acker Ref.)",
        "name_es": "Delta del Río Misisipi (EE. UU. — Ref. Acker et al.)",
        "country": "United States",
        "zone": "Global SME Reference Nodes",
        "lat": 29.15,
        "lon": -89.25,
        "biome": "River-Influenced Coastal Margin",
        "has_modis_250m": False,
        "has_meteomatics_1km": False,
        "summary": (
            "Primary reference region analyzed in the official challenge benchmark paper by Acker et al. (2009, NASA GES DISC NTRS 20090042747)."
        ),
    },
    "US_Arctic_Barrow": {
        "id": "US_Arctic_Barrow",
        "name": "Arctic Coastal Plain (Utqiaġvik / Barrow, Alaska)",
        "name_es": "Llanura Costera Ártica (Utqiaġvik / Barrow, Alaska)",
        "country": "United States",
        "zone": "Global SME Reference Nodes",
        "lat": 71.29,
        "lon": -156.79,
        "biome": "Arctic Tundra / Cryosphere Margin",
        "has_modis_250m": False,
        "has_meteomatics_1km": False,
        "summary": (
            "Global benchmark for polar amplification, exhibiting extreme multi-decadal warming and cryospheric regime shift."
        ),
    },
    "ES_Med_Almeria": {
        "id": "ES_Med_Almeria",
        "name": "Western Mediterranean Basin (Almería, Spain)",
        "name_es": "Cuenca Mediterránea Occidental (Almería, España)",
        "country": "Spain",
        "zone": "Global SME Reference Nodes",
        "lat": 36.84,
        "lon": -2.46,
        "biome": "Semi-Arid Mediterranean Coastal Basin",
        "has_modis_250m": False,
        "has_meteomatics_1km": False,
        "summary": (
            "European benchmark for Mediterranean aridification and compound summer heat-drought anomalies."
        ),
    },
}

GUIDED_CASES: List[Dict[str, Any]] = [
    {
        "id": "case_1_enso_dipole",
        "title": "Case 1 — The South American ENSO Dipole & Opposite Trends",
        "title_es": "Caso 1 — Dipolo ENSO Sudamericano y Tendencias Opuestas",
        "badge": "Challenge Core Prompt: Opposite Trends",
        "primary_region": "EC_Coast_Guayas",
        "secondary_region": "BR_Amazon_Manaus",
        "variable": "PRECTOTCORR",
        "question": "Why does the same global climate driver (El Niño / ENSO) trigger extreme flooding on Ecuador's Pacific Coast while simultaneously driving severe drought in the Central Amazon?",
        "key_finding": (
            "Empirical cross-correlation with NOAA's Oceanic Niño Index (ONI, 1981–2025) proves an unmistakable opposite-sign teleconnection: "
            "during warm El Niño phases, Ecuador's Pacific Coast (EC_Coast_Guayas) experiences intense positive rainfall anomalies "
            "(r_ONI = +0.21, p < 0.0001; +4.8σ peak in 1997–98), whereas the Central Amazon (BR_Amazon_Manaus) suffers severe rainfall "
            "and soil moisture deficits (PRECTOTCORR r_ONI = -0.33, GWETROOT r_ONI = -0.42, p < 0.0001) alongside a 45-year secular drying trend (-0.35 mm/day/decade)."
        ),
    },
    {
        "id": "case_2_multiscale_andes",
        "title": "Case 2 — Multi-Scale Resolution Detective: Why 55 km Misses Andean Valley Droughts",
        "title_es": "Caso 2 — Detective Multiescala: Por qué 55 km no ve Sequías Andinas",
        "badge": "NASA MERRA-2 (55km) + MODIS (250m) + Meteomatics (1km)",
        "primary_region": "EC_Andes_Paute",
        "secondary_region": "EC_Andes_Quito",
        "variable": "GWETROOT",
        "question": "Why did Ecuador's Paute/Mazar hydroelectric basin suffer catastrophic drought in 2023–2024 even though coarse 55 km global reanalysis shows a slight long-term wetting trend?",
        "key_finding": (
            "1) Autocorrelation Trap Caught: Naive OLS on monthly Paute root soil moisture gives p = 0.030 (false positive), "
            "but lag-1 autocorrelation is r1 = 0.941; after Santer et al. (2000) and Hamed-Rao (1998) correction, p_adj = 0.728 (no secular trend). "
            "2) Spatial Resolution Trap Solved: A 55 km MERRA-2 grid cell averages the Eastern Cordillera with Pacific/Amazonian moisture, "
            "whereas 1 km Meteomatics downscaling captures Paute's soil moisture deficit quadrupling from 10.5 mm to 43.2 mm in late 2024, "
            "corroborated by 250 m NASA Terra MODIS NDVI canopy stress."
        ),
    },
    {
        "id": "case_3_satellite_splice",
        "title": "Case 3 — Instrument Breakpoint Detective: Catching a Fake Climate Trend (2001 Sensor Splice)",
        "title_es": "Caso 3 — Quiebre Instrumental: Falsa Tendencia por Cambio de Satélite (2001)",
        "badge": "Scientific Validity & Pettitt Change-Point Test",
        "primary_region": "EC_Coast_Guayas",
        "secondary_region": "BR_Pantanal",
        "variable": "ALLSKY_SFC_SW_DWN",
        "question": "Did surface solar radiation across South America really plummet by -0.24 MJ/m²/day/decade over the last 40 years, or is it a satellite calibration artifact?",
        "key_finding": (
            "TerraLens's Pettitt Non-Parametric Change-Point Detector flags an extreme step discontinuity in January 2001 (p = 4.0e-15 in Guayas, "
            "p = 2.6e-15 in Pantanal). That exact date marks NASA POWER's splice between GEWEX SRB (1984–2000) and CERES SYN1deg (2001–2025). "
            "When restricted to the homogeneous CERES satellite era (2001–2025), the Guayas trend is +0.00 MJ/m²/day/decade (p = 0.99), "
            "preventing a major false scientific conclusion."
        ),
    },
    {
        "id": "case_4_pantanal_compound",
        "title": "Case 4 — Compound Hydroclimatic Tipping Point in the Brazilian Pantanal",
        "title_es": "Caso 4 — Punto de Inflexión Compuesto en el Pantanal Brasileño",
        "badge": "Explainable ML (Isolation Forest + PCA + Acker Outlier Switch)",
        "primary_region": "BR_Pantanal",
        "secondary_region": "EC_Galapagos",
        "variable": "T2M",
        "question": "Which multivariate drivers explain the post-2019 compound megadrought and heatwave regime in the Pantanal, and does the trend persist if extreme months are excluded?",
        "key_finding": (
            "Every hydroclimatic variable in the Pantanal shows a highly significant 45-year trend even after Hamed-Rao autocorrelation correction: "
            "T2M +0.53 °C/decade, PRECTOTCORR -0.50 mm/day/decade, GWETTOP -0.031/decade, and RH2M -3.07%/decade (all p < 0.0001). "
            "Using the Acker et al. (2009) Outlier Switch to remove ML-detected extreme months (Isolation Forest top 5%) proves the warming and drying "
            "is a structural background shift rather than an artifact of isolated outlier months."
        ),
    },
    {
        "id": "case_5_global_arctic_acker",
        "title": "Case 5 — Global Benchmark: Arctic Polar Amplification (Alaska) vs. Mississippi Delta",
        "title_es": "Caso 5 — Referencia Global: Amplificación Ártica (Alaska) vs. Delta del Misisipi",
        "badge": "Global Reach: Arctic Cryosphere + Acker et al. (2009) USA Site",
        "primary_region": "US_Arctic_Barrow",
        "secondary_region": "US_Mississippi_Delta",
        "variable": "T2M",
        "question": "How does extreme high-latitude Arctic warming in Utqiaġvik (Barrow, Alaska) compare against the subtropical Mississippi River Delta reference site from Acker et al. (2009)?",
        "key_finding": (
            "Demonstrates TerraLens's global scalability across North America: Utqiaġvik, Alaska (71.29°N) exhibits extreme polar amplification "
            "(+0.579 °C/decade, Hamed-Rao p < 0.0001, Pettitt regime step in 2002-02 of +1.68 °C), nearly 3× faster than Andean nodes (+0.197 °C/decade), "
            "while simultaneously benchmarking against the Mississippi River Delta (29.15°N) studied in Acker et al. (2009)."
        ),
    },
]
