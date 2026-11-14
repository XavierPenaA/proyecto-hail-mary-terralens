"""
TerraLens v2.1 Real Data Connectors & Verified Cache Manager.

STRICT ZERO-HALLUCINATION POLICY (Rule #0):
- Never generates synthetic or random data.
- Serves empirically verified NASA POWER (MERRA-2 / CERES), NOAA CPC ONI,
  NASA ORNL DAAC Terra MODIS 250m (MOD13Q1), and Meteomatics 1km downscaled series.
- Supports live queries for any global coordinate (lat, lon) via NASA POWER REST API.
"""

import os
import json
import ssl
import base64
import urllib.request
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from core.catalog import VARIABLES, REGIONS
from core.stats_engine import mann_kendall_hamed_rao, deseasonalize_acker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

SSL_CTX = ssl._create_unverified_context()


def _load_dotenv_if_present() -> None:
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#") and "=" in s:
                    k, v = s.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv_if_present()


def parse_nasa_power_json(raw_json: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parses a raw NASA POWER monthly point JSON payload into a clean DataFrame
    with columns: [date, year, month, decimal_year, T2M, PRECTOTCORR, ...].
    Replaces -999 / -999.0 fill values with NaN.
    """
    param_dict = raw_json.get("properties", {}).get("parameter", {})
    coords = raw_json.get("geometry", {}).get("coordinates", [0.0, 0.0, 0.0])
    header = raw_json.get("header", {})

    meta = {
        "lon": float(coords[0]) if len(coords) > 0 else 0.0,
        "lat": float(coords[1]) if len(coords) > 1 else 0.0,
        "elevation_m": float(coords[2]) if len(coords) > 2 else 0.0,
        "title": header.get("title", "NASA POWER Climate Data"),
        "api_version": header.get("api", {}).get("version", "v2"),
    }

    rows: Dict[str, Dict[str, Any]] = {}
    for var_name, ym_map in param_dict.items():
        for ym_str, val in ym_map.items():
            if len(ym_str) != 6:
                continue
            month_num = int(ym_str[4:6])
            if month_num < 1 or month_num > 12:
                # Skip month 13 (annual summary in NASA POWER)
                continue
            year_num = int(ym_str[0:4])
            date_str = f"{year_num:04d}-{month_num:02d}"
            rec = rows.setdefault(
                date_str,
                {
                    "date": date_str,
                    "year": year_num,
                    "month": month_num,
                    "decimal_year": year_num + (month_num - 0.5) / 12.0,
                },
            )
            fval = float(val)
            rec[var_name] = np.nan if fval <= -900.0 else fval

    df = pd.DataFrame(list(rows.values()))
    if not df.empty:
        df = df.sort_values(["year", "month"]).reset_index(drop=True)
    return df, meta


def load_region_nasa_power(region_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads a pre-cached region from disk or fetches it live from NASA POWER if not yet cached.
    """
    if region_id not in REGIONS:
        raise ValueError(f"Unknown region_id: {region_id}")
    reg = REGIONS[region_id]
    cache_file = os.path.join(CACHE_DIR, f"{region_id}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        df, meta = parse_nasa_power_json(raw)
        meta["source_mode"] = "Verified Local Cache (NASA POWER MERRA-2/CERES)"
        meta["region_id"] = region_id
        meta["region_name"] = reg["name"]
        return df, meta

    return fetch_live_nasa_power(reg["lat"], reg["lon"], save_as_region=region_id)


def fetch_live_nasa_power(
    lat: float,
    lon: float,
    start_year: int = 1981,
    end_year: int = 2025,
    save_as_region: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Queries the official NASA POWER Monthly Point API in real time for any (lat, lon) on Earth.
    """
    lat_r = round(float(lat), 3)
    lon_r = round(float(lon), 3)
    cache_key = save_as_region or f"live_{lat_r}_{lon_r}".replace("-", "m").replace(".", "p")
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        df, meta = parse_nasa_power_json(raw)
        meta["source_mode"] = "Cached NASA POWER Coordinate Probe"
        return df, meta

    params = ",".join(VARIABLES.keys())
    url = (
        f"https://power.larc.nasa.gov/api/temporal/monthly/point?"
        f"parameters={params}&community=AG&longitude={lon_r}&latitude={lat_r}"
        f"&start={start_year}&end={end_year}&format=JSON"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "TerraLens-NASA-SpaceApps/2.1"})
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=30) as r:
        payload = r.read().decode("utf-8")
        raw = json.loads(payload)

    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(payload)

    df, meta = parse_nasa_power_json(raw)
    meta["source_mode"] = "Live NASA POWER API Query (200 OK)"
    return df, meta


def load_noaa_oni() -> pd.DataFrame:
    """
    Loads the official NOAA Climate Prediction Center Oceanic Niño Index (ONI) series.
    Maps rolling 3-month seasons (DJF=1, JFM=2, ..., NDJ=12) to calendar year and month.
    """
    oni_file = os.path.join(CACHE_DIR, "oni.txt")
    if not os.path.exists(oni_file):
        url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
        req = urllib.request.Request(url, headers={"User-Agent": "TerraLens/2.1"})
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=20) as r:
            txt = r.read().decode("utf-8")
        with open(oni_file, "w", encoding="utf-8") as f:
            f.write(txt)
    else:
        with open(oni_file, "r", encoding="utf-8") as f:
            txt = f.read()

    season_to_month = {
        "DJF": 1,
        "JFM": 2,
        "FMA": 3,
        "MAM": 4,
        "AMJ": 5,
        "MJJ": 6,
        "JJA": 7,
        "JAS": 8,
        "ASO": 9,
        "SON": 10,
        "OND": 11,
        "NDJ": 12,
    }
    rows = []
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] in season_to_month:
            try:
                yr = int(parts[1])
                mo = season_to_month[parts[0]]
                anom = float(parts[3])
                rows.append(
                    {
                        "year": yr,
                        "month": mo,
                        "date": f"{yr:04d}-{mo:02d}",
                        "oni": anom,
                    }
                )
            except ValueError:
                continue
    return pd.DataFrame(rows)


def load_ecuador_regional_grid_trends(variable: str = "T2M") -> pd.DataFrame:
    """
    Computes real 45-year Theil-Sen trend slopes and Hamed-Rao p-values across the 96-cell
    NASA POWER 2D Regional Grid over Ecuador & adjacent Andean/Pacific/Amazonian sectors.
    """
    grid_file = os.path.join(CACHE_DIR, "ecuador_regional_grid.json")
    if not os.path.exists(grid_file):
        return pd.DataFrame()

    with open(grid_file, "r", encoding="utf-8") as f:
        grid_data = json.load(f)

    features = grid_data.get(variable, [])
    rows = []
    for feat in features:
        coords = feat.get("geometry", {}).get("coordinates", [0.0, 0.0, 0.0])
        lon, lat, elev = float(coords[0]), float(coords[1]), float(coords[2])
        ym_map = feat.get("properties", {}).get("parameter", {}).get(variable, {})
        pts = []
        for ym_str, val in ym_map.items():
            if len(ym_str) == 6:
                mo = int(ym_str[4:6])
                if 1 <= mo <= 12 and float(val) > -900:
                    yr = int(ym_str[0:4])
                    pts.append(
                        {
                            "year": yr,
                            "month": mo,
                            "date": f"{yr:04d}-{mo:02d}",
                            "decimal_year": yr + (mo - 0.5) / 12.0,
                            variable: float(val),
                        }
                    )
        if len(pts) < 60:
            continue
        df_cell = pd.DataFrame(pts)
        des = deseasonalize_acker(df_cell, variable)
        mk = mann_kendall_hamed_rao(des["decimal_year"].to_numpy(), des["anomaly"].to_numpy())
        rows.append(
            {
                "lat": lat,
                "lon": lon,
                "elevation_m": elev,
                "variable": variable,
                "theil_sen_decade": round(mk["theil_sen_decade"], 4),
                "p_hamed_rao": round(mk["p_hamed_rao"], 5),
                "tau": round(mk["tau"], 3),
                "significant": bool(mk["p_hamed_rao"] < 0.05),
                "mean_value": round(float(df_cell[variable].mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def load_or_fetch_modis_250m(region_id: str, lat: float, lon: float) -> pd.DataFrame:
    """
    Loads cached NASA ORNL DAAC Terra MODIS 250m (MOD13Q1) NDVI observations or fetches
    recent observations live from https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset.
    """
    modis_file = os.path.join(CACHE_DIR, "nasa_modis_250m_ndvi.json")
    if os.path.exists(modis_file):
        with open(modis_file, "r", encoding="utf-8") as f:
            all_modis = json.load(f)
        if region_id in all_modis and len(all_modis[region_id]) > 0:
            df = pd.DataFrame(all_modis[region_id])
            df["year"] = df["calendar_date"].str.slice(0, 4).astype(int)
            df["month"] = df["calendar_date"].str.slice(5, 7).astype(int)
            return df

    # Live fallback for recent 10-date windows (2023 and 2024 dry seasons)
    records = []
    windows = [("A2023209", "A2023353"), ("A2024209", "A2024353")]
    for s_d, e_d in windows:
        url = (
            f"https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset?"
            f"latitude={lat}&longitude={lon}&band=250m_16_days_NDVI"
            f"&startDate={s_d}&endDate={e_d}&kmAboveBelow=0&kmLeftRight=0"
        )
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "TerraLens/2.1"})
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as r:
                d = json.loads(r.read().decode("utf-8"))
                scale = float(d.get("scale", 0.0001))
                for item in d.get("subset", []):
                    val = item["data"][0]
                    if val > -2000:
                        records.append(
                            {
                                "modis_date": item["modis_date"],
                                "calendar_date": item["calendar_date"],
                                "ndvi": round(val * scale, 4),
                            }
                        )
        except Exception:
            continue
    df = pd.DataFrame(records)
    if not df.empty:
        df["year"] = df["calendar_date"].str.slice(0, 4).astype(int)
        df["month"] = df["calendar_date"].str.slice(5, 7).astype(int)
    return df


def load_meteomatics_1km_cache() -> Dict[str, pd.DataFrame]:
    """
    Loads the pre-verified 1 km Meteomatics hydroclimatic downscaling series
    for Ecuador's regions (Guayas, Paute, Quito, Napo) across 2023–2024.
    """
    mm_file = os.path.join(CACHE_DIR, "meteomatics_1km_ecuador_2023_2024.json")
    if not os.path.exists(mm_file):
        return {}
    with open(mm_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, rows in raw.items():
        if rows:
            out[k] = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def load_or_fetch_planetary_computer_stac(region_id: str, lat: float, lon: float) -> List[Dict[str, Any]]:
    """
    Queries or loads cached Microsoft Planetary Computer STAC API v1 metadata
    for Terra/Aqua MODIS 250m (modis-13Q1-061) granules intersecting (lat, lon).
    """
    stac_file = os.path.join(CACHE_DIR, "planetary_computer_stac.json")
    all_stac: Dict[str, List[Dict[str, Any]]] = {}
    if os.path.exists(stac_file):
        try:
            with open(stac_file, "r", encoding="utf-8") as f:
                all_stac = json.load(f)
            if region_id in all_stac and len(all_stac[region_id]) > 0:
                return all_stac[region_id]
        except Exception:
            all_stac = {}

    body = json.dumps({
        "collections": ["modis-13Q1-061"],
        "intersects": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "datetime": "2022-01-01T00:00:00Z/2025-12-31T23:59:59Z",
        "limit": 6,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://planetarycomputer.microsoft.com/api/stac/v1/search",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "TerraLens-NASA-SpaceApps/2.1"},
    )
    feats: List[Dict[str, Any]] = []
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as r:
            res = json.loads(r.read().decode("utf-8"))
            for item in res.get("features", []):
                props = item.get("properties", {})
                h_tile = props.get("modis:horizontal-tile", 10)
                v_tile = props.get("modis:vertical-tile", 9)
                feats.append({
                    "stac_id": item.get("id", ""),
                    "collection": item.get("collection", "modis-13Q1-061"),
                    "datetime": (props.get("datetime") or props.get("start_datetime") or "")[:10],
                    "platform": props.get("platform", "terra,aqua"),
                    "tile_grid": f"h{int(h_tile):02d}v{int(v_tile):02d}",
                    "bbox": [round(float(x), 2) for x in (item.get("bbox") or [])],
                    "assets": list((item.get("assets") or {}).keys())[:6],
                })
        if feats:
            all_stac[region_id] = feats
            with open(stac_file, "w", encoding="utf-8") as f:
                json.dump(all_stac, f, indent=2)
    except Exception:
        pass
    return feats
