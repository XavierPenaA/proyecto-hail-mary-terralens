"""
Independent check of the MERRA-2 series (via NASA POWER) against ERA5-Land.

For each region and each pair (MERRA-2 variable, ERA5-Land variable):
- correlation of monthly anomalies, with the Bretherton effective sample size;
- agreement of the TerraLens trend verdict (same direction, both significant, or one rejected);
- agreement of the slope (Theil-Sen per decade, in standard deviations of the anomalies so the two
  models' different units compare);
- whether the Pettitt change point falls in the same year.

Run scripts/fetch_era5land.py first. Writes models/era5land_validation.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.catalog import REGIONS  # noqa: E402
from core.connectors import CACHE_DIR, load_noaa_oni, load_region_nasa_power  # noqa: E402
from core.stats_engine import analyze_single_variable, correlation_with_neff  # noqa: E402

PAIRS = (("T2M", "T2M"), ("PRECTOTCORR", "PRECTOT"), ("GWETROOT", "SM_ROOT"), ("GWETTOP", "SM_TOP"))
START_YEAR, END_YEAR = 1981, 2025
MIN_VALID_MONTHS = 240
MIN_STD = 1e-6
OUT = Path(__file__).resolve().parents[1] / "models" / "era5land_validation.json"


def era5_frame(records: list, column: str) -> pd.DataFrame:
    df = pd.DataFrame(records)[["date", column]]
    year, month = df["date"].str.slice(0, 4).astype(int), df["date"].str.slice(5, 7).astype(int)
    return df.assign(year=year, month=month, decimal_year=year + (month - 0.5) / 12.0)


def summarize(res: dict) -> dict:
    sd = float(res["monthly_series"]["anomaly"].std(ddof=1))
    return {
        "verdict": res["verdict"]["code"], "direction": res["verdict"]["direction"],
        "slope_sd_per_decade": res["monthly_mk"]["theil_sen_decade"] / sd if sd > 0 else 0.0,
        "break_date": res["pettitt"]["change_date"], "break_significant": res["pettitt"]["is_significant_breakpoint"],
    }


def main() -> None:
    cache = json.loads((Path(CACHE_DIR) / "era5land_monthly.json").read_text(encoding="utf-8"))["regions"]
    oni = load_noaa_oni()
    rows = []
    for region_id in REGIONS:
        if region_id not in cache:
            continue
        merra, _ = load_region_nasa_power(region_id)
        for m_var, e_var in PAIRS:
            if m_var not in merra.columns:
                continue
            era5 = era5_frame(cache[region_id]["monthly"], e_var)
            if era5[e_var].notna().sum() < MIN_VALID_MONTHS:
                print(f"skip {region_id} {e_var}: only {int(era5[e_var].notna().sum())} valid months in the cache")
                continue
            if float(era5[e_var].std()) < MIN_STD:
                # e.g. the Altiplano cell is a constant 0.021 m3/m3 in ERA5-Land (water or ice surface).
                print(f"skip {region_id} {e_var}: constant series in ERA5-Land")
                continue
            res_m = analyze_single_variable(merra, m_var, oni, START_YEAR, END_YEAR)
            res_e = analyze_single_variable(era5, e_var, oni, START_YEAR, END_YEAR)
            merged = res_m["monthly_series"][["date", "anomaly"]].merge(
                res_e["monthly_series"][["date", "anomaly"]], on="date", suffixes=("_m", "_e")).dropna()
            corr = correlation_with_neff(merged["anomaly_m"].to_numpy(), merged["anomaly_e"].to_numpy())
            sm, se = summarize(res_m), summarize(res_e)
            both_sig = res_m["verdict"]["is_significant"] and res_e["verdict"]["is_significant"]
            rows.append({
                "region": region_id, "merra_var": m_var, "era5_var": e_var,
                "anomaly_r": round(corr["r"], 3), "n_eff": round(corr["n_eff"], 1),
                "merra": sm, "era5": se,
                "both_significant": bool(both_sig),
                "same_direction": sm["direction"] == se["direction"],
                "verdict_conflict": bool(res_m["verdict"]["is_significant"] != res_e["verdict"]["is_significant"]),
                "slope_gap_sd": round(abs(sm["slope_sd_per_decade"] - se["slope_sd_per_decade"]), 3),
                "break_year_gap": (abs(int(sm["break_date"][:4]) - int(se["break_date"][:4]))
                                   if sm["break_date"] and se["break_date"] else None),
            })
    frame = pd.DataFrame(rows)
    PAIRS_USED = tuple(p for p in PAIRS if (frame["merra_var"] == p[0]).any())
    summary = {
        pair[0]: {
            "regions": int((frame["merra_var"] == pair[0]).sum()),
            "median_anomaly_r": round(float(frame[frame["merra_var"] == pair[0]]["anomaly_r"].median()), 3),
            "min_anomaly_r": round(float(frame[frame["merra_var"] == pair[0]]["anomaly_r"].min()), 3),
            "verdict_conflicts": int(frame[frame["merra_var"] == pair[0]]["verdict_conflict"].sum()),
            "opposite_direction_when_both_significant": int(
                (frame[frame["merra_var"] == pair[0]]["both_significant"] & ~frame[frame["merra_var"] == pair[0]]["same_direction"]).sum()),
            "median_slope_gap_sd": round(float(frame[frame["merra_var"] == pair[0]]["slope_gap_sd"].median()), 3),
        }
        for pair in PAIRS_USED
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(frame[["region", "merra_var", "anomaly_r", "n_eff", "verdict_conflict", "same_direction", "slope_gap_sd"]]
          .assign(merra=[r["merra"]["verdict"] for r in rows], era5=[r["era5"]["verdict"] for r in rows]).to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
