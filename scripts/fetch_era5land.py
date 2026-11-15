"""
Downloads ERA5-Land daily series for the 11 benchmark regions from the Open-Meteo historical API
(no account needed), aggregates them to months and writes data/cache/era5land_monthly.json.

ERA5-Land is a different land model with different forcing than MERRA-2, so it is an independent
check of the MERRA-2 values TerraLens uses (it is still a reanalysis, not an observation).

Only the public catalog coordinates are sent. Soil moisture is volumetric (m3/m3) in four layers;
the root-zone series is the depth-weighted mean of 0-7, 7-28 and 28-100 cm (0-100 cm).

Usage: python scripts/fetch_era5land.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.catalog import REGIONS  # noqa: E402
from core.connectors import CACHE_DIR, UpstreamDataError, _http_get_json  # noqa: E402

URL = (
    "https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
    "&start_date={start}-01-01&end_date={end}-12-31&models=era5_land&timezone=UTC"
    "&daily=precipitation_sum,temperature_2m_mean,soil_moisture_0_to_7cm_mean,"
    "soil_moisture_7_to_28cm_mean,soil_moisture_28_to_100cm_mean"
)
LAYER_WEIGHTS = {"soil_moisture_0_to_7cm_mean": 7, "soil_moisture_7_to_28cm_mean": 21, "soil_moisture_28_to_100cm_mean": 72}
# The free tier weighs a request by days x variables, so long ranges are fetched in 15-year chunks.
YEAR_CHUNKS = ((1981, 1995), (1996, 2010), (2011, 2025))
PAUSE_SECONDS = 20.0
RETRY_WAIT_SECONDS = 600.0  # the free tier allows a fixed request weight per hour
MAX_RETRIES = 10
MIN_DAYS_PER_MONTH = 25


def fetch_chunk(lat: float, lon: float, start: int, end: int) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            return _http_get_json(URL.format(lat=lat, lon=lon, start=start, end=end), timeout=120)
        except UpstreamDataError as exc:
            if "429" not in str(exc) or attempt == MAX_RETRIES - 1:
                raise
            print(f"   rate limited, waiting {RETRY_WAIT_SECONDS:.0f}s", flush=True)
            time.sleep(RETRY_WAIT_SECONDS)


def monthly_table(payloads: list) -> list:
    daily = pd.concat([pd.DataFrame(p["daily"]) for p in payloads], ignore_index=True)
    daily["date"] = daily.pop("time").str.slice(0, 7)
    daily["ROOT"] = sum(daily[c] * w for c, w in LAYER_WEIGHTS.items()) / sum(LAYER_WEIGHTS.values())
    grouped = daily.groupby("date")
    monthly = grouped.agg(
        PRECTOT=("precipitation_sum", "mean"), T2M=("temperature_2m_mean", "mean"),
        SM_TOP=("soil_moisture_0_to_7cm_mean", "mean"), SM_ROOT=("ROOT", "mean"), n_days=("ROOT", "count"),
    )
    monthly = monthly[monthly["n_days"] >= MIN_DAYS_PER_MONTH].drop(columns="n_days").round(5)
    return monthly.reset_index().to_dict("records")


def main() -> None:
    path = Path(CACHE_DIR) / "era5land_monthly.json"
    out = json.loads(path.read_text(encoding="utf-8"))["regions"] if path.exists() else {}
    for region_id, meta in REGIONS.items():
        if region_id in out:
            print(f"{region_id}: already cached", flush=True)
            continue
        print(f"{region_id} ({meta['lat']}, {meta['lon']})", flush=True)
        payloads = []
        for start, end in YEAR_CHUNKS:
            payloads.append(fetch_chunk(meta["lat"], meta["lon"], start, end))
            time.sleep(PAUSE_SECONDS)
        out[region_id] = {"lat": meta["lat"], "lon": meta["lon"], "grid_lat": payloads[0].get("latitude"),
                          "grid_lon": payloads[0].get("longitude"), "elevation_m": payloads[0].get("elevation"),
                          "monthly": monthly_table(payloads)}
        path.write_text(json.dumps({"source": "ERA5-Land via Open-Meteo archive API", "regions": out}), encoding="utf-8")
    print(f"Saved {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
