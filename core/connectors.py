"""
TerraLens data access.

Runtime reads only local caches (NASA POWER points and grid, NOAA ONI, MODIS NDVI, Meteomatics,
Planetary Computer STAC), so the observatory works offline. Network access happens in three
places: a region whose POWER cache is missing, the live coordinate probe, and the cache refresh
script (scripts/refresh_satellite_caches.py), which uses the fetch_* functions below.
"""

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.catalog import DATA_END_YEAR, DATA_START_YEAR, REGIONS, VARIABLES
from core.stats_engine import SIGNIFICANCE_ALPHA, benjamini_hochberg_fdr, deseasonalize_acker, mann_kendall_hamed_rao

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
LIVE_CACHE_DIR = os.path.join(CACHE_DIR, "live")
LIVE_CACHE_MAX_FILES = 200
HTTP_TIMEOUT_S = 30
USER_AGENT = "TerraLens/2.2 (NASA Space Apps Challenge)"
SSL_CONTEXT = ssl.create_default_context()

POWER_POINT_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
MODIS_API_URL = "https://modis.ornl.gov/rst/api/v1/MOD13Q1"
STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

MODIS_NDVI_BAND = "250m_16_days_NDVI"
MODIS_RELIABILITY_BAND = "250m_16_days_pixel_reliability"
MODIS_GOOD_PIXELS = (0, 1)  # 0 = good data, 1 = marginal data; 2 = snow/ice, 3 = cloudy
MODIS_DATES_PER_REQUEST = 10
MODIS_TIMEOUT_S = 120  # the ORNL subset service often takes 30 s or more per call

ONI_SEASON_CENTER_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


class UpstreamDataError(RuntimeError):
    """A remote data service failed or returned an unusable payload."""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _http_get(url: str, body: Optional[bytes] = None, headers: Optional[Dict[str, str]] = None,
              timeout: int = HTTP_TIMEOUT_S) -> str:
    request = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(request, context=SSL_CONTEXT, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpstreamDataError(f"Request to {urllib.parse.urlsplit(url).netloc} failed: {exc}") from exc


def _http_get_json(url: str, body: Optional[Dict[str, Any]] = None, timeout: int = HTTP_TIMEOUT_S) -> Dict[str, Any]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    text = _http_get(url, payload, headers, timeout)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise UpstreamDataError(f"Invalid JSON from {urllib.parse.urlsplit(url).netloc}") from exc


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# NASA POWER
# ---------------------------------------------------------------------------
def parse_nasa_power_json(raw_json: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parses a NASA POWER monthly point payload into [date, year, month, decimal_year, <variables>].
    Fill values (-999) become NaN and the annual summary (month 13) is dropped.
    """
    param_dict = raw_json.get("properties", {}).get("parameter", {})
    coords = raw_json.get("geometry", {}).get("coordinates", [0.0, 0.0, 0.0])
    meta = {
        "lon": float(coords[0]) if len(coords) > 0 else 0.0,
        "lat": float(coords[1]) if len(coords) > 1 else 0.0,
        "elevation_m": float(coords[2]) if len(coords) > 2 else 0.0,
        "api_version": raw_json.get("header", {}).get("api", {}).get("version", "v2"),
        "sources": raw_json.get("header", {}).get("sources", []),
    }

    rows: Dict[str, Dict[str, Any]] = {}
    for var_name, ym_map in param_dict.items():
        for ym_str, val in ym_map.items():
            if len(ym_str) != 6 or not 1 <= int(ym_str[4:6]) <= 12:
                continue
            year, month = int(ym_str[:4]), int(ym_str[4:6])
            date_str = f"{year:04d}-{month:02d}"
            rec = rows.setdefault(
                date_str,
                {"date": date_str, "year": year, "month": month, "decimal_year": year + (month - 0.5) / 12.0},
            )
            fval = float(val)
            rec[var_name] = np.nan if fval <= -900.0 else fval

    df = pd.DataFrame(list(rows.values()))
    if not df.empty:
        df = df.sort_values(["year", "month"]).reset_index(drop=True)
    return df, meta


def fetch_nasa_power_point(lat: float, lon: float, start_year: int = DATA_START_YEAR,
                           end_year: int = DATA_END_YEAR) -> Dict[str, Any]:
    """Raw NASA POWER monthly point payload for every catalog variable."""
    query = urllib.parse.urlencode({
        "parameters": ",".join(VARIABLES.keys()),
        "community": "AG",
        "longitude": round(float(lon), 3),
        "latitude": round(float(lat), 3),
        "start": start_year,
        "end": end_year,
        "format": "JSON",
    })
    raw = _http_get_json(f"{POWER_POINT_URL}?{query}")
    if not raw.get("properties", {}).get("parameter"):
        raise UpstreamDataError("NASA POWER returned no parameters for this coordinate")
    return raw


def load_region_nasa_power(region_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads a catalog region from its cache file, downloading and caching it if missing."""
    if region_id not in REGIONS:
        raise ValueError(f"Unknown region_id: {region_id}")
    reg = REGIONS[region_id]
    cache_file = os.path.join(CACHE_DIR, f"{region_id}.json")
    if os.path.exists(cache_file):
        raw = _read_json(cache_file)
    else:
        raw = fetch_nasa_power_point(reg["lat"], reg["lon"])
        _write_json(cache_file, raw)
    df, meta = parse_nasa_power_json(raw)
    meta.update({"source_mode": "cache", "region_id": region_id})
    return df, meta


def _prune_live_cache() -> None:
    """Keeps only the LIVE_CACHE_MAX_FILES most recently written probe files."""
    files = [os.path.join(LIVE_CACHE_DIR, name) for name in os.listdir(LIVE_CACHE_DIR) if name.endswith(".json")]
    for path in sorted(files, key=os.path.getmtime, reverse=True)[LIVE_CACHE_MAX_FILES:]:
        os.remove(path)


def fetch_live_nasa_power(lat: float, lon: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Live NASA POWER query for any coordinate, cached under data/cache/live/ (bounded size)."""
    lat_r, lon_r = round(float(lat), 3), round(float(lon), 3)
    cache_key = f"live_{lat_r}_{lon_r}".replace("-", "m").replace(".", "p")
    cache_file = os.path.join(LIVE_CACHE_DIR, f"{cache_key}.json")

    if os.path.exists(cache_file):
        raw, mode = _read_json(cache_file), "cache"
    else:
        raw, mode = fetch_nasa_power_point(lat_r, lon_r), "live"
        _write_json(cache_file, raw)
        _prune_live_cache()
    df, meta = parse_nasa_power_json(raw)
    meta["source_mode"] = mode
    return df, meta


# ---------------------------------------------------------------------------
# NOAA ONI
# ---------------------------------------------------------------------------
def parse_oni_text(text: str) -> pd.DataFrame:
    """Maps NOAA's 3-month seasons to their center month (DJF -> January, ..., NDJ -> December)."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] in ONI_SEASON_CENTER_MONTH:
            try:
                year, month, anom = int(parts[1]), ONI_SEASON_CENTER_MONTH[parts[0]], float(parts[3])
            except ValueError:
                continue
            rows.append({"year": year, "month": month, "date": f"{year:04d}-{month:02d}", "oni": anom})
    return pd.DataFrame(rows)


def load_noaa_oni() -> pd.DataFrame:
    """NOAA CPC Oceanic Niño Index, read from cache (downloaded once if missing)."""
    oni_file = os.path.join(CACHE_DIR, "oni.txt")
    if os.path.exists(oni_file):
        with open(oni_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = _http_get(ONI_URL)
        with open(oni_file, "w", encoding="utf-8") as f:
            f.write(text)
    return parse_oni_text(text)


# ---------------------------------------------------------------------------
# Ecuador regional grid
# ---------------------------------------------------------------------------
def load_ecuador_regional_grid_trends(variable: str = "T2M") -> pd.DataFrame:
    """
    Theil-Sen slope and Hamed-Rao p-value for every NASA POWER grid cell over Ecuador.
    Significance uses Benjamini-Hochberg q-values across all cells (field significance).
    """
    grid_file = os.path.join(CACHE_DIR, "ecuador_regional_grid.json")
    if not os.path.exists(grid_file):
        return pd.DataFrame()

    rows = []
    for feat in _read_json(grid_file).get(variable, []):
        lon, lat, elev = (float(c) for c in feat.get("geometry", {}).get("coordinates", [0.0, 0.0, 0.0]))
        df_cell, _ = parse_nasa_power_json(feat)
        if variable not in df_cell or df_cell[variable].count() < 60:
            continue
        des = deseasonalize_acker(df_cell, variable)
        mk = mann_kendall_hamed_rao(des["decimal_year"].to_numpy(), des["anomaly"].to_numpy())
        rows.append(
            {
                "lat": lat,
                "lon": lon,
                "elevation_m": elev,
                "theil_sen_decade": round(mk["theil_sen_decade"], 4),
                "p_hamed_rao": round(mk["p_hamed_rao"], 5),
                "mean_value": round(float(df_cell[variable].mean()), 2),
            }
        )
    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid
    grid["q_fdr"] = np.round(benjamini_hochberg_fdr(grid["p_hamed_rao"].tolist()), 5)
    grid["significant"] = grid["q_fdr"] < SIGNIFICANCE_ALPHA
    return grid


# ---------------------------------------------------------------------------
# MODIS NDVI (ORNL DAAC)
# ---------------------------------------------------------------------------
def load_modis_ndvi(region_id: str) -> pd.DataFrame:
    """Cached MOD13Q1 NDVI series for a region (empty when the region has none)."""
    modis_file = os.path.join(CACHE_DIR, "nasa_modis_250m_ndvi.json")
    if not os.path.exists(modis_file):
        return pd.DataFrame()
    records = _read_json(modis_file).get(region_id) or []
    return pd.DataFrame(records)


def fetch_modis_ndvi(lat: float, lon: float, start_date: str, end_date: str, km_window: int = 1) -> List[Dict[str, Any]]:
    """
    MOD13Q1 NDVI around (lat, lon) from the ORNL DAAC subset API. km_window extends the box
    that many km on each side; pixels flagged snow/ice, cloudy or fill are dropped and the
    remaining NDVI values are summarized by their median.
    """
    coords = {"latitude": lat, "longitude": lon}
    available = _http_get_json(f"{MODIS_API_URL}/dates?{urllib.parse.urlencode(coords)}", timeout=MODIS_TIMEOUT_S).get("dates", [])
    dates = [d for d in available if start_date <= d["calendar_date"] <= end_date]

    records = []
    for i in range(0, len(dates), MODIS_DATES_PER_REQUEST):
        chunk = dates[i:i + MODIS_DATES_PER_REQUEST]
        window = {**coords, "startDate": chunk[0]["modis_date"], "endDate": chunk[-1]["modis_date"],
                  "kmAboveBelow": km_window, "kmLeftRight": km_window}
        ndvi = _http_get_json(
            f"{MODIS_API_URL}/subset?{urllib.parse.urlencode({**window, 'band': MODIS_NDVI_BAND})}", timeout=MODIS_TIMEOUT_S
        )
        quality = _http_get_json(
            f"{MODIS_API_URL}/subset?{urllib.parse.urlencode({**window, 'band': MODIS_RELIABILITY_BAND})}",
            timeout=MODIS_TIMEOUT_S,
        )
        scale = float(ndvi.get("scale") or 0.0001)
        reliability = {item["modis_date"]: item["data"] for item in quality.get("subset", [])}
        for item in ndvi.get("subset", []):
            values = np.asarray(item["data"], dtype=float)
            flags = np.asarray(reliability.get(item["modis_date"], [-1] * len(values)))
            good = np.isin(flags, MODIS_GOOD_PIXELS) & (values > -2000)
            if good.any():
                records.append(
                    {
                        "modis_date": item["modis_date"],
                        "calendar_date": item["calendar_date"],
                        "ndvi": round(float(np.median(values[good])) * scale, 4),
                        "n_valid_pixels": int(good.sum()),
                        "n_pixels": int(len(values)),
                    }
                )
    return records


# ---------------------------------------------------------------------------
# Meteomatics and Planetary Computer STAC (cached)
# ---------------------------------------------------------------------------
def load_meteomatics_1km_cache() -> Dict[str, pd.DataFrame]:
    """Cached Meteomatics 1 km snapshots (one day every 10 days, 2023-2024) per region."""
    mm_file = os.path.join(CACHE_DIR, "meteomatics_1km_ecuador_2023_2024.json")
    if not os.path.exists(mm_file):
        return {}
    return {
        region: pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        for region, rows in _read_json(mm_file).items()
        if rows
    }


def load_planetary_computer_stac(region_id: str) -> List[Dict[str, Any]]:
    """Cached MODIS granule metadata from the Planetary Computer STAC API (provenance only)."""
    stac_file = os.path.join(CACHE_DIR, "planetary_computer_stac.json")
    if not os.path.exists(stac_file):
        return []
    return _read_json(stac_file).get(region_id, [])


def fetch_planetary_computer_stac(lat: float, lon: float, limit: int = 6) -> List[Dict[str, Any]]:
    """Searches the modis-13Q1-061 collection for granules that contain (lat, lon)."""
    result = _http_get_json(
        STAC_SEARCH_URL,
        body={
            "collections": ["modis-13Q1-061"],
            "intersects": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "datetime": f"{DATA_END_YEAR - 3}-01-01T00:00:00Z/{DATA_END_YEAR}-12-31T23:59:59Z",
            "limit": limit,
        },
    )
    items = []
    for item in result.get("features", []):
        props = item.get("properties", {})
        items.append(
            {
                "stac_id": item.get("id", ""),
                "collection": item.get("collection", "modis-13Q1-061"),
                "datetime": (props.get("datetime") or props.get("start_datetime") or "")[:10],
                "platform": props.get("platform", ""),
                "tile_grid": f"h{int(props.get('modis:horizontal-tile', 0)):02d}v{int(props.get('modis:vertical-tile', 0)):02d}",
                "bbox": [round(float(x), 2) for x in (item.get("bbox") or [])],
            }
        )
    return items
