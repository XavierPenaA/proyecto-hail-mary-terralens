"""
TerraLens v2.1 Full-Stack REST API & Interactive Web Observatory Server (FastAPI).
Serves both the REST API (/api/*) and the interactive Leaflet + Chart.js Web Observatory (/).
"""

import os
import json
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from core.catalog import REGIONS, VARIABLES, GUIDED_CASES
from core.connectors import (
    load_region_nasa_power,
    fetch_live_nasa_power,
    load_noaa_oni,
    load_ecuador_regional_grid_trends,
    load_or_fetch_modis_250m,
    load_meteomatics_1km_cache,
    load_or_fetch_planetary_computer_stac,
)
from core.stats_engine import (
    analyze_single_variable,
    compare_regional_dipole,
    benjamini_hochberg_fdr,
)
from core.ml_engine import (
    run_multivariate_anomaly_detective,
    run_acker_ml_outlier_experiment_all_vars,
    run_spatiotemporal_cube_ml,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(
    title="TerraLens v2.1 — Multi-Scale Earth System Trend, Breakpoint & Anomaly Detective",
    version="2.1.0",
    description=(
        "Peer-reviewed statistical trend inference (Acker et al. 2009, Hamed-Rao 1998, Santer et al. 2000, "
        "Pettitt 1979, Benjamini-Hochberg FDR) and 3-Model Explainable Machine Learning "
        "(Isolation Forest + LOF + Mahalanobis + Random Forest Lag Detective + PCA + KMeans) "
        "on real NASA MERRA-2/CERES, NASA Terra MODIS 250m, NOAA ONI, and Meteomatics 1km datasets."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_observatory_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "TerraLens v2.1 API is running. Visit /docs for OpenAPI documentation."}


def _load_ml_manifest() -> dict:
    manifest_path = os.path.join(BASE_DIR, "models", "training_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@app.get("/api/ml-manifest")
def get_ml_manifest():
    return _load_ml_manifest()


@app.get("/api/catalog")
def get_catalog():
    return {
        "regions": REGIONS,
        "variables": VARIABLES,
        "guided_cases": GUIDED_CASES,
        "ml_manifest": _load_ml_manifest(),
    }


@app.get("/api/full-dashboard/{region_id}")
def get_full_region_dashboard(
    region_id: str,
    variable: str = Query("T2M"),
    start_year: int = Query(1981),
    end_year: int = Query(2025),
    exclude_ml_extremes: bool = Query(False),
    homogeneous_ceres: bool = Query(True),
):
    if region_id not in REGIONS:
        raise HTTPException(status_code=404, detail=f"Unknown region_id: {region_id}")
    if variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"Unknown variable: {variable}")

    df, meta = load_region_nasa_power(region_id)
    oni_df = load_noaa_oni()

    eff_start = max(start_year, 2001) if (variable == "ALLSKY_SFC_SW_DWN" and homogeneous_ceres and start_year < 2001) else start_year

    # 1. Run 3-Model ML Ensemble + Lagged Driver RF + PCA + KMeans
    ml_res = run_multivariate_anomaly_detective(
        df, oni_df=oni_df, start_year=max(1984, start_year), end_year=end_year
    )
    excl_dates = ml_res["extreme_dates"] if exclude_ml_extremes else None

    # 2. Run baseline & active single-variable statistical analysis
    res_base = analyze_single_variable(
        df, variable, oni_df=oni_df, start_year=eff_start, end_year=end_year, exclude_dates=None
    )
    res_act = (
        analyze_single_variable(
            df, variable, oni_df=oni_df, start_year=eff_start, end_year=end_year, exclude_dates=excl_dates
        )
        if exclude_ml_extremes
        else res_base
    )

    # 3. Format monthly series for frontend interactive chart
    des = res_act["monthly_series"].copy()
    ols_m = res_act["monthly_ols"]
    mk_m = res_act["monthly_mk"]
    des["ols_line"] = np.round(ols_m["intercept"] + ols_m["slope_yr"] * des["decimal_year"], 4)
    des["ts_line"] = np.round(mk_m["theil_sen_intercept"] + mk_m["theil_sen_yr"] * des["decimal_year"], 4)
    des["raw_val"] = np.round(des[variable], 4)
    des["clim_mean"] = np.round(des["clim_mean"], 4)
    des["anomaly"] = np.round(des["anomaly"], 4)
    des["z_score"] = np.round(des["z_score"], 3)

    monthly_records = des[
        ["date", "year", "month", "raw_val", "clim_mean", "anomaly", "z_score", "ols_line", "ts_line"]
    ].to_dict(orient="records")

    ann = res_act["annual_series"].copy()
    ann_ols = res_act["annual_ols"]
    ann["ols_line"] = np.round(ann_ols["intercept"] + ann_ols["slope_yr"] * ann["year"], 4)
    ann["annual_anomaly"] = np.round(ann["annual_anomaly"], 4)
    ann["annual_raw"] = np.round(ann["annual_raw"], 4)
    annual_records = ann[["year", "annual_anomaly", "annual_raw", "ols_line"]].to_dict(orient="records")

    # 4. Multi-variable auto-scan with Benjamini-Hochberg FDR
    scan_rows = []
    p_vals = []
    for vc, vmeta in VARIABLES.items():
        sy = 2001 if (vc == "ALLSKY_SFC_SW_DWN" and homogeneous_ceres) else 1981
        r_s = analyze_single_variable(df, vc, oni_df=oni_df, start_year=sy, end_year=2025)
        p_hr = r_s["monthly_mk"]["p_hamed_rao"]
        p_vals.append(p_hr)
        scan_rows.append(
            {
                "variable": vc,
                "short_name": vmeta["short_name"],
                "units": vmeta["decade_units"],
                "period": f"{r_s['year_start']}–{r_s['year_end']}",
                "theil_sen_decade": round(r_s["monthly_mk"]["theil_sen_decade"], 4),
                "ols_decade": round(r_s["monthly_ols"]["slope_decade"], 4),
                "lag1_r": round(r_s["monthly_ols"]["lag1_r"], 3),
                "n_eff": round(r_s["monthly_ols"]["n_eff"], 1),
                "p_naive": round(r_s["monthly_ols"]["p_naive"], 5),
                "p_santer": round(r_s["monthly_ols"]["p_santer"], 5),
                "p_hamed_rao": round(p_hr, 5),
                "p_annual_mk": round(r_s["annual_mk"]["p_original"], 5),
                "pettitt_date": r_s["pettitt"]["change_date"],
                "pettitt_p": round(r_s["pettitt"]["p_value"], 6),
                "r_oni": round(r_s["enso"]["r_oni"], 3),
                "verdict": r_s["verdict"]["verdict"],
                "badge_color": r_s["verdict"]["badge_color"],
            }
        )
    q_vals = benjamini_hochberg_fdr(p_vals)
    for r_row, qv in zip(scan_rows, q_vals):
        r_row["q_fdr"] = round(qv, 5)

    # 5. Acker ML Outlier Experiment table
    acker_exp = run_acker_ml_outlier_experiment_all_vars(df, oni_df=oni_df, use_homogeneous_ceres=homogeneous_ceres)

    # 6. PCA scatter points
    z_mat = ml_res["matrix"]
    pca_points = []
    if not z_mat.empty:
        for _, row in z_mat.iterrows():
            pca_points.append(
                {
                    "date": str(row["date"]),
                    "pca_1": float(row["pca_1"]),
                    "pca_2": float(row["pca_2"]),
                    "is_anomaly": bool(row["is_anomaly"]),
                    "consensus_score_pct": float(row["consensus_score_pct"]),
                    "model_votes": int(row["model_votes"]),
                    "regime_name": str(row["regime_name"]),
                }
            )

    return {
        "meta": meta,
        "region": REGIONS[region_id],
        "variable_info": VARIABLES[variable],
        "year_start": res_act["year_start"],
        "year_end": res_act["year_end"],
        "n_months": res_act["n_months"],
        "n_years": res_act["n_years"],
        "monthly_ols": res_act["monthly_ols"],
        "monthly_mk": res_act["monthly_mk"],
        "baseline_mk": res_base["monthly_mk"],
        "annual_ols": res_act["annual_ols"],
        "annual_mk": res_act["annual_mk"],
        "pettitt": res_act["pettitt"],
        "enso": res_act["enso"],
        "verdict": res_act["verdict"],
        "monthly_climatology": res_act["monthly_climatology"],
        "seasonal_trends": res_act["seasonal_trends"],
        "monthly_series": monthly_records,
        "annual_series": annual_records,
        "auto_scan": scan_rows,
        "acker_experiment": acker_exp,
        "ml": {
            "anomalies": ml_res["anomalies"],
            "pca_variance_ratio": ml_res["pca_variance_ratio"],
            "pca_loadings": ml_res["pca_loadings"],
            "cluster_summary": ml_res["cluster_summary"],
            "lagged_driver_model": ml_res["lagged_driver_model"],
            "pca_points": pca_points,
        },
        "ml_manifest": _load_ml_manifest(),
    }


@app.get("/api/dipole")
def compare_dipole_endpoint(
    region_a: str = Query("EC_Coast_Guayas"),
    region_b: str = Query("BR_Amazon_Manaus"),
    variable: str = Query("PRECTOTCORR"),
    start_year: int = Query(1981),
    end_year: int = Query(2025),
):
    df_a, _ = load_region_nasa_power(region_a)
    df_b, _ = load_region_nasa_power(region_b)
    oni_df = load_noaa_oni()
    sy = 2001 if variable == "ALLSKY_SFC_SW_DWN" and start_year < 2001 else start_year
    res = compare_regional_dipole(df_a, df_b, variable, oni_df=oni_df, start_year=sy, end_year=end_year)
    merged = res["merged_series"].copy()
    merged["anom_a"] = np.round(merged["anom_a"], 4)
    merged["anom_b"] = np.round(merged["anom_b"], 4)
    merged["z_a"] = np.round(merged["z_a"], 3)
    merged["z_b"] = np.round(merged["z_b"], 3)
    return {
        "variable": variable,
        "variable_info": VARIABLES[variable],
        "region_a_info": REGIONS[region_a],
        "region_b_info": REGIONS[region_b],
        "stats_a": {
            "theil_sen_decade": res["region_a"]["monthly_mk"]["theil_sen_decade"],
            "p_hamed_rao": res["region_a"]["monthly_mk"]["p_hamed_rao"],
            "r_oni": res["region_a"]["enso"]["r_oni"],
            "p_oni": res["region_a"]["enso"]["p_oni"],
            "verdict": res["region_a"]["verdict"]["verdict"],
        },
        "stats_b": {
            "theil_sen_decade": res["region_b"]["monthly_mk"]["theil_sen_decade"],
            "p_hamed_rao": res["region_b"]["monthly_mk"]["p_hamed_rao"],
            "r_oni": res["region_b"]["enso"]["r_oni"],
            "p_oni": res["region_b"]["enso"]["p_oni"],
            "verdict": res["region_b"]["verdict"]["verdict"],
        },
        "pearson_r": res["pearson_r"],
        "pearson_p": res["pearson_p"],
        "spearman_rho": res["spearman_rho"],
        "r_during_strong_enso": res["r_during_strong_enso"],
        "has_opposite_secular_slopes": res["has_opposite_secular_slopes"],
        "has_opposite_enso_teleconnection": res["has_opposite_enso_teleconnection"],
        "series": merged[["date", "year", "month", "z_a", "z_b", "anom_a", "anom_b", "oni"]].fillna(0.0).to_dict(orient="records"),
    }


@app.get("/api/multiscale")
def get_multiscale_data(
    region_id: str = Query("EC_Andes_Paute"),
    grid_variable: str = Query("T2M"),
):
    gv = grid_variable if grid_variable in ["T2M", "PRECTOTCORR", "GWETROOT"] else "T2M"
    grid_df = load_ecuador_regional_grid_trends(gv)
    modis_reg = region_id if REGIONS.get(region_id, {}).get("has_modis_250m") else "EC_Andes_Paute"
    r_info = REGIONS[modis_reg]
    modis_df = load_or_fetch_modis_250m(modis_reg, r_info["lat"], r_info["lon"])
    stac_items = load_or_fetch_planetary_computer_stac(modis_reg, r_info["lat"], r_info["lon"])
    mm_dict = load_meteomatics_1km_cache()
    mm_out = {k: v.to_dict(orient="records") for k, v in mm_dict.items()}
    return {
        "grid_variable": gv,
        "grid_units": VARIABLES[gv]["decade_units"],
        "grid_cells": grid_df.to_dict(orient="records") if not grid_df.empty else [],
        "modis_region_id": modis_reg,
        "modis_region_name": r_info["name"],
        "modis_250m": modis_df.to_dict(orient="records") if not modis_df.empty else [],
        "planetary_computer_stac": stac_items,
        "meteomatics_1km": mm_out,
    }


@app.get("/api/live-probe")
def live_coordinate_probe(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    variable: str = Query("T2M"),
):
    df, meta = fetch_live_nasa_power(lat, lon)
    oni_df = load_noaa_oni()
    sy = 2001 if variable == "ALLSKY_SFC_SW_DWN" else 1981
    res = analyze_single_variable(df, variable, oni_df=oni_df, start_year=sy, end_year=2025)
    ml_res = run_multivariate_anomaly_detective(df, oni_df=oni_df, start_year=1984, end_year=2025)
    return {
        "meta": meta,
        "variable": variable,
        "variable_info": VARIABLES[variable],
        "monthly_ols": res["monthly_ols"],
        "monthly_mk": res["monthly_mk"],
        "pettitt": res["pettitt"],
        "enso": res["enso"],
        "verdict": res["verdict"],
        "top_anomalies": ml_res["anomalies"][:5],
        "lagged_driver_model": ml_res["lagged_driver_model"],
    }


@app.get("/api/cube-ml")
def get_spatiotemporal_cube_ml():
    oni_df = load_noaa_oni()
    region_dfs = {}
    for rid in REGIONS:
        df_r, _ = load_region_nasa_power(rid)
        region_dfs[rid] = df_r
    return run_spatiotemporal_cube_ml(region_dfs, oni_df=oni_df)


if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
