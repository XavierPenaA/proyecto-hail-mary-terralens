"""
TerraLens catalog: data period, variables and benchmark regions.

Descriptions are qualitative on purpose. Every number shown to users is computed from the data
at run time (see core/case_studies.py for the guided-case findings).
"""

from typing import Any, Dict, Optional

DATA_START_YEAR = 1981
DATA_END_YEAR = 2025
# The machine-learning matrix needs all six variables; solar irradiance starts in 1984.
ML_START_YEAR = 1984
MIN_WINDOW_YEARS = 10

VARIABLES: Dict[str, Dict[str, Any]] = {
    "T2M": {
        "code": "T2M",
        "name": "Air Temperature at 2 Meters",
        "name_es": "Temperatura del aire a 2 metros",
        "short_name": "2 m air temperature",
        "short_name_es": "Temperatura del aire 2 m",
        "units": "°C",
        "decade_units": "°C / decade",
        "units_es": "°C",
        "decade_units_es": "°C / década",
        "source": "NASA MERRA-2 reanalysis (via NASA POWER)",
        "resolution": "0.5° × 0.625° (~55 km)",
        "first_year": 1981,
        "known_splice": None,
        "map": {"scale_limit": 0.4, "palette": "temperature"},
        "explainer": {
            "en": "Average air temperature 2 m above the ground.",
            "es": "Temperatura media del aire a 2 m sobre el suelo.",
        },
    },
    "PRECTOTCORR": {
        "code": "PRECTOTCORR",
        "name": "Precipitation (bias-corrected)",
        "name_es": "Precipitación (corregida por sesgo)",
        "short_name": "Precipitation",
        "short_name_es": "Precipitación",
        "units": "mm/day",
        "decade_units": "mm/day / decade",
        "units_es": "mm/día",
        "decade_units_es": "mm/día / década",
        "source": "NASA MERRA-2 bias-corrected precipitation (via NASA POWER)",
        "resolution": "0.5° × 0.625° (~55 km)",
        "first_year": 1981,
        "known_splice": None,
        "map": {"scale_limit": 0.5, "palette": "moisture"},
        "explainer": {
            "en": "Average daily precipitation, corrected with observation-based products.",
            "es": "Precipitación media diaria, corregida con productos basados en observaciones.",
        },
    },
    "GWETROOT": {
        "code": "GWETROOT",
        "name": "Root Zone Soil Wetness",
        "name_es": "Humedad del suelo en la zona radicular",
        "short_name": "Root-zone soil wetness",
        "short_name_es": "Humedad del suelo radicular",
        "units": "fraction (0–1)",
        "decade_units": "fraction / decade",
        "units_es": "fracción (0–1)",
        "decade_units_es": "fracción / década",
        "source": "NASA MERRA-2 land surface model (via NASA POWER)",
        "resolution": "0.5° × 0.625° (~55 km)",
        "first_year": 1981,
        "known_splice": None,
        "map": {"scale_limit": 0.03, "palette": "moisture"},
        "explainer": {
            "en": "Share of water saturation in the upper ~1 m of soil. It changes slowly, so consecutive months are strongly correlated.",
            "es": "Fracción de saturación de agua en el primer metro de suelo. Cambia lento, así que los meses consecutivos están muy correlacionados.",
        },
    },
    "GWETTOP": {
        "code": "GWETTOP",
        "name": "Surface Soil Wetness (0–5 cm)",
        "name_es": "Humedad del suelo superficial (0–5 cm)",
        "short_name": "Surface soil wetness",
        "short_name_es": "Humedad del suelo superficial",
        "units": "fraction (0–1)",
        "decade_units": "fraction / decade",
        "units_es": "fracción (0–1)",
        "decade_units_es": "fracción / década",
        "source": "NASA MERRA-2 land surface model (via NASA POWER)",
        "resolution": "0.5° × 0.625° (~55 km)",
        "first_year": 1981,
        "known_splice": None,
        "map": None,
        "explainer": {
            "en": "Share of water saturation in the top 5 cm of soil. Reacts quickly to rain and heat.",
            "es": "Fracción de saturación de agua en los primeros 5 cm del suelo. Reacciona rápido a la lluvia y al calor.",
        },
    },
    "ALLSKY_SFC_SW_DWN": {
        "code": "ALLSKY_SFC_SW_DWN",
        "name": "All-Sky Surface Shortwave Downward Irradiance",
        "name_es": "Irradiancia solar de onda corta en superficie",
        "short_name": "Surface solar irradiance",
        "short_name_es": "Irradiancia solar",
        "units": "MJ/m²/day",
        "decade_units": "MJ/m²/day / decade",
        "units_es": "MJ/m²/día",
        "decade_units_es": "MJ/m²/día / década",
        "source": "NASA GEWEX SRB (to 2000) + NASA CERES SYN1deg (2001 on), via NASA POWER",
        "resolution": "1° × 1° (~110 km)",
        "first_year": 1984,
        "known_splice": {"date": "2001-01", "pre_sensor": "GEWEX SRB", "post_sensor": "CERES SYN1deg"},
        "map": None,
        "explainer": {
            "en": "Solar energy reaching the ground through clouds and aerosols. NASA POWER switches source in January 2001.",
            "es": "Energía solar que llega al suelo a través de nubes y aerosoles. NASA POWER cambia de fuente en enero de 2001.",
        },
    },
    "RH2M": {
        "code": "RH2M",
        "name": "Relative Humidity at 2 Meters",
        "name_es": "Humedad relativa a 2 metros",
        "short_name": "Relative humidity",
        "short_name_es": "Humedad relativa",
        "units": "%",
        "decade_units": "% / decade",
        "units_es": "%",
        "decade_units_es": "% / década",
        "source": "NASA MERRA-2 reanalysis (via NASA POWER)",
        "resolution": "0.5° × 0.625° (~55 km)",
        "first_year": 1981,
        "known_splice": None,
        "map": None,
        "explainer": {
            "en": "Water vapour in the air as a share of saturation, 2 m above the ground.",
            "es": "Vapor de agua del aire como porcentaje de la saturación, a 2 m del suelo.",
        },
    },
}

# zone codes: ecuador, south_america, global_reference (labels in the frontend locales)
REGIONS: Dict[str, Dict[str, Any]] = {
    "EC_Coast_Guayas": {
        "name": "Ecuador Pacific Coast (Guayas)",
        "name_es": "Costa del Pacífico de Ecuador (Guayas)",
        "zone": "ecuador",
        "lat": -2.19,
        "lon": -79.89,
        "has_modis_250m": True,
        "has_meteomatics_1km": True,
        "summary": {
            "en": "Coastal plain where El Niño events bring heavy rain. The point lies on Guayaquil and the Guayas river.",
            "es": "Llanura costera donde El Niño trae lluvias fuertes. El punto cae sobre Guayaquil y el río Guayas.",
        },
    },
    "EC_Andes_Paute": {
        "name": "Southern Ecuadorian Andes (Paute / Mazar)",
        "name_es": "Andes del sur de Ecuador (Paute / Mazar)",
        "zone": "ecuador",
        "lat": -2.61,
        "lon": -78.63,
        "has_modis_250m": True,
        "has_meteomatics_1km": True,
        "summary": {
            "en": "Hydropower basin in steep terrain, where a ~55 km grid cell mixes very different slopes.",
            "es": "Cuenca hidroeléctrica en relieve abrupto, donde una celda de ~55 km mezcla laderas muy distintas.",
        },
    },
    "EC_Andes_Quito": {
        "name": "Northern Ecuadorian Andes (Quito)",
        "name_es": "Andes del norte de Ecuador (Quito)",
        "zone": "ecuador",
        "lat": -0.18,
        "lon": -78.47,
        "has_modis_250m": True,
        "has_meteomatics_1km": True,
        "summary": {
            "en": "High inter-Andean valley around the capital.",
            "es": "Valle interandino alto alrededor de la capital.",
        },
    },
    "EC_Amazon_Napo": {
        "name": "Ecuadorian Amazon Foothills (Napo)",
        "name_es": "Piedemonte amazónico de Ecuador (Napo)",
        "zone": "ecuador",
        "lat": -0.47,
        "lon": -76.99,
        "has_modis_250m": False,
        "has_meteomatics_1km": True,
        "summary": {
            "en": "Transition between the Andes and the Amazon lowlands.",
            "es": "Transición entre los Andes y la llanura amazónica.",
        },
    },
    "EC_Galapagos": {
        "name": "Galápagos Islands (Santa Cruz)",
        "name_es": "Islas Galápagos (Santa Cruz)",
        "zone": "ecuador",
        "lat": -0.63,
        "lon": -90.36,
        "has_modis_250m": False,
        "has_meteomatics_1km": False,
        "summary": {
            "en": "Equatorial Pacific islands sitting on the ocean surface that defines El Niño.",
            "es": "Islas del Pacífico ecuatorial, sobre el océano que define a El Niño.",
        },
    },
    "BR_Amazon_Manaus": {
        "name": "Central Amazon (Manaus, Brazil)",
        "name_es": "Amazonía central (Manaos, Brasil)",
        "zone": "south_america",
        "lat": -3.12,
        "lon": -60.02,
        "has_modis_250m": True,
        "has_meteomatics_1km": False,
        "summary": {
            "en": "Central Amazon basin. The point lies on the city of Manaus and the Rio Negro.",
            "es": "Cuenca amazónica central. El punto cae sobre la ciudad de Manaos y el río Negro.",
        },
    },
    "BR_Pantanal": {
        "name": "Pantanal Wetland (Brazil)",
        "name_es": "Humedal del Pantanal (Brasil)",
        "zone": "south_america",
        "lat": -17.75,
        "lon": -56.90,
        "has_modis_250m": True,
        "has_meteomatics_1km": False,
        "summary": {
            "en": "The world's largest tropical wetland, a seasonal flooded savanna.",
            "es": "El mayor humedal tropical del mundo, una sabana inundable estacional.",
        },
    },
    "PE_Altiplano": {
        "name": "Peru-Bolivia Altiplano (Lake Titicaca)",
        "name_es": "Altiplano peruano-boliviano (lago Titicaca)",
        "zone": "south_america",
        "lat": -15.84,
        "lon": -69.50,
        "has_modis_250m": False,
        "has_meteomatics_1km": False,
        "summary": {
            "en": "High plateau around 3,800 m.",
            "es": "Meseta alta, alrededor de 3.800 m.",
        },
    },
    "US_Mississippi_Delta": {
        "name": "Mississippi River Delta (Louisiana, USA)",
        "name_es": "Delta del río Misisipi (Luisiana, EE. UU.)",
        "zone": "global_reference",
        "lat": 29.15,
        "lon": -89.25,
        "has_modis_250m": False,
        "has_meteomatics_1km": False,
        "summary": {
            "en": "River-influenced coast also studied by Acker et al. (2009), the reference paper of this method.",
            "es": "Costa con influencia fluvial, también estudiada por Acker et al. (2009), el artículo de referencia del método.",
        },
    },
    "US_Arctic_Barrow": {
        "name": "Arctic Coastal Plain (Utqiaġvik, Alaska)",
        "name_es": "Llanura costera ártica (Utqiaġvik, Alaska)",
        "zone": "global_reference",
        "lat": 71.29,
        "lon": -156.79,
        "has_modis_250m": False,
        "has_meteomatics_1km": False,
        "summary": {
            "en": "Arctic tundra with seasonally frozen ground and permafrost.",
            "es": "Tundra ártica con suelo estacionalmente congelado y permafrost.",
        },
    },
    "ES_Med_Almeria": {
        "name": "Western Mediterranean (Almería, Spain)",
        "name_es": "Mediterráneo occidental (Almería, España)",
        "zone": "global_reference",
        "lat": 36.84,
        "lon": -2.46,
        "has_modis_250m": False,
        "has_meteomatics_1km": False,
        "summary": {
            "en": "Semi-arid Mediterranean coast.",
            "es": "Costa mediterránea semiárida.",
        },
    },
}


def analysis_start_year(variable: str, requested_start: int, homogeneous_splice: bool) -> int:
    """
    First year actually analysed: never before the variable's first year, and after the documented
    instrument splice when the homogeneity guard is on.
    """
    meta = VARIABLES[variable]
    start = max(requested_start, meta["first_year"])
    splice = meta["known_splice"]
    if homogeneous_splice and splice:
        start = max(start, int(splice["date"][:4]))
    return start


def known_break_date(variable: str) -> Optional[str]:
    """Documented break date of a variable (YYYY-MM) or None."""
    splice = VARIABLES[variable]["known_splice"]
    return splice["date"] if splice else None
