"""
Refreshes the cached MODIS NDVI series and Planetary Computer STAC metadata.

Examples
    python scripts/refresh_satellite_caches.py --regions EC_Andes_Paute --dry-run
    python scripts/refresh_satellite_caches.py --all-modis --start 2021-01-01 --end 2025-12-31 --km 1

NDVI is the median of the good-quality pixels (MOD13Q1 pixel reliability 0 or 1) in a box that
extends --km kilometres on each side of the sampling point. The sampling point is the region's
"modis_point" in core/catalog.py when present, else the region's point; set "modis_point" to a
spot whose land cover represents the region (the default Guayas and Manaus points fall on cities
and rivers).
"""

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.catalog import DATA_END_YEAR, REGIONS  # noqa: E402
from core.connectors import CACHE_DIR, fetch_modis_ndvi, fetch_planetary_computer_stac  # noqa: E402

MODIS_FILE = os.path.join(CACHE_DIR, "nasa_modis_250m_ndvi.json")
STAC_FILE = os.path.join(CACHE_DIR, "planetary_computer_stac.json")


def sampling_point(region_id: str) -> Dict[str, float]:
    region = REGIONS[region_id]
    return region.get("modis_point") or {"lat": region["lat"], "lon": region["lon"]}


def merge_into(path: str, updates: Dict[str, List[dict]]) -> None:
    data = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regions", nargs="*", default=[], help="region ids to refresh")
    parser.add_argument("--all-modis", action="store_true", help="refresh every region flagged has_modis_250m")
    parser.add_argument("--start", default=f"{DATA_END_YEAR - 4}-01-01", help="first calendar date (YYYY-MM-DD)")
    parser.add_argument("--end", default=f"{DATA_END_YEAR}-12-31", help="last calendar date (YYYY-MM-DD)")
    parser.add_argument("--km", type=int, default=1, help="kilometres around the point (0 = single pixel)")
    parser.add_argument("--dry-run", action="store_true", help="download and summarize without writing the caches")
    args = parser.parse_args()

    regions = list(args.regions)
    if args.all_modis:
        regions += [r for r, meta in REGIONS.items() if meta["has_modis_250m"] and r not in regions]
    unknown = [r for r in regions if r not in REGIONS]
    if unknown or not regions:
        parser.error(f"choose known regions (unknown: {unknown})" if unknown else "no region selected")

    modis_updates, stac_updates = {}, {}
    for region_id in regions:
        point = sampling_point(region_id)
        records = fetch_modis_ndvi(point["lat"], point["lon"], args.start, args.end, km_window=args.km)
        stac = fetch_planetary_computer_stac(point["lat"], point["lon"])
        valid = [r["n_valid_pixels"] / r["n_pixels"] for r in records]
        mean_ndvi = sum(r["ndvi"] for r in records) / len(records) if records else float("nan")
        print(f"{region_id:22s} {len(records):3d} NDVI dates, mean {mean_ndvi:.3f}, "
              f"good pixels {100 * sum(valid) / max(1, len(valid)):.0f}%, {len(stac)} STAC granules")
        modis_updates[region_id] = records
        stac_updates[region_id] = stac

    if args.dry_run:
        print("Dry run: caches not written.")
        return
    merge_into(MODIS_FILE, modis_updates)
    merge_into(STAC_FILE, stac_updates)
    print(f"Updated {MODIS_FILE} and {STAC_FILE}")


if __name__ == "__main__":
    main()
