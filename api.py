"""
TerraLens REST API and web observatory server (FastAPI).

Serves the static observatory at / and a read-only JSON API under /api. Heavy computations are
memoized in core.analysis_service; the guided-case findings are warmed up at startup.
"""

import json
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core import analysis_service as svc
from core.case_studies import public_cases
from core.catalog import (
    DATA_END_YEAR,
    DATA_START_YEAR,
    MIN_WINDOW_YEARS,
    ML_START_YEAR,
    REGIONS,
    VARIABLES,
    analysis_start_year,
    known_break_date,
)
from core.connectors import (
    UpstreamDataError,
    fetch_live_nasa_power,
    load_meteomatics_1km_cache,
    load_modis_ndvi,
    load_planetary_computer_stac,
)
from core.ml_engine import run_multivariate_anomaly_detective
from core.stats_engine import analyze_single_variable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MANIFEST_PATH = os.path.join(BASE_DIR, "models", "training_manifest.json")
MODIS_FALLBACK_REGION = "EC_Andes_Paute"
MAP_VARIABLES = [code for code, meta in VARIABLES.items() if meta["map"]]
LIVE_PROBE_MAX_CALLS = 20
LIVE_PROBE_PERIOD_S = 60.0


class SlidingWindowLimiter:
    """Process-wide limit of max_calls per period (protects the upstream NASA POWER service)."""

    def __init__(self, max_calls: int, period_s: float) -> None:
        self.max_calls = max_calls
        self.period_s = period_s
        self._calls: deque = deque()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = time.monotonic()
        with self._lock:
            while self._calls and now - self._calls[0] > self.period_s:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                return False
            self._calls.append(now)
            return True


live_probe_limiter = SlidingWindowLimiter(LIVE_PROBE_MAX_CALLS, LIVE_PROBE_PERIOD_S)


def _warm_caches() -> None:
    public_cases()
    svc.warm_up()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Precompute the guided-case findings and every region in the background so clicks are instant.
    threading.Thread(target=_warm_caches, name="warm-caches", daemon=True).start()
    yield


app = FastAPI(
    title="TerraLens — Earth System Trend Detective",
    version="2.2.0",
    description=(
        "Autocorrelation-aware trend, change-point and anomaly analysis of NASA POWER (MERRA-2, "
        "GEWEX SRB, CERES), NOAA ONI, MODIS NDVI and Meteomatics data."
    ),
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.exception_handler(UpstreamDataError)
async def upstream_error_handler(_: Request, exc: UpstreamDataError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _check_region(region_id: str, status_code: int = 422) -> str:
    if region_id not in REGIONS:
        raise HTTPException(status_code=status_code, detail=f"Unknown region: {region_id}")
    return region_id


def _check_variable(variable: str, allowed: Optional[List[str]] = None) -> str:
    if variable not in (allowed or VARIABLES):
        raise HTTPException(status_code=422, detail=f"Unknown or unsupported variable: {variable}")
    return variable


def path_region(region_id: str = Path(...)) -> str:
    return _check_region(region_id, status_code=404)


def query_variable(variable: str = Query("T2M")) -> str:
    return _check_variable(variable)


class YearWindow:
    """start_year/end_year query parameters inside the data period, at least MIN_WINDOW_YEARS long."""

    def __init__(
        self,
        start_year: int = Query(DATA_START_YEAR, ge=DATA_START_YEAR, le=DATA_END_YEAR),
        end_year: int = Query(DATA_END_YEAR, ge=DATA_START_YEAR, le=DATA_END_YEAR),
    ) -> None:
        if end_year - start_year + 1 < MIN_WINDOW_YEARS:
            raise HTTPException(
                status_code=422,
                detail=f"The analysis window must span at least {MIN_WINDOW_YEARS} years (got {start_year}–{end_year}).",
            )
        self.start_year = start_year
        self.end_year = end_year


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _records(df: pd.DataFrame, columns: List[str], digits: int = 4) -> List[Dict[str, Any]]:
    """DataFrame rows as JSON-safe dicts (rounded floats, NaN -> None)."""
    if df.empty:
        return []
    out = df[columns].copy()
    float_cols = out.select_dtypes(include="number").columns
    out[float_cols] = out[float_cols].round(digits)
    return out.astype(object).where(pd.notna(out), None).to_dict(orient="records")


def _trend_payload(res: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("monthly_ols", "monthly_mk", "annual_ols", "annual_mk", "pettitt", "homogeneity", "known_break",
            "enso", "verdict", "warnings", "monthly_climatology", "seasonal_trends", "n_months", "n_years",
            "year_start", "year_end")
    return {k: res[k] for k in keys}


def _monthly_series(res: Dict[str, Any], variable: str) -> List[Dict[str, Any]]:
    ols, mk = res["monthly_ols"], res["monthly_mk"]
    des = res["monthly_series"].assign(
        raw_val=lambda d: d[variable],
        ols_line=lambda d: ols["intercept"] + ols["slope_yr"] * d["decimal_year"],
        ts_line=lambda d: mk["theil_sen_intercept"] + mk["theil_sen_yr"] * d["decimal_year"],
    )
    return _records(des, ["date", "raw_val", "clim_mean", "anomaly", "z_score", "ols_line", "ts_line"])


def _annual_series(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    mk = res["annual_mk"]
    ann = res["annual_series"].assign(ts_line=lambda d: mk["theil_sen_intercept"] + mk["theil_sen_yr"] * d["year"])
    return _records(ann, ["year", "annual_anomaly", "annual_raw", "ts_line"])


def _ml_payload(detective: Dict[str, Any]) -> Dict[str, Any]:
    matrix = detective["matrix"]
    points = (
        _records(matrix, ["date", "pca_1", "pca_2", "is_anomaly", "cluster_id", "consensus_score_pct"], 3)
        if "pca_1" in matrix
        else []
    )
    return {
        "n_months": detective["n_months"],
        "anomalies": detective["anomalies"],
        "pca": detective["pca"],
        "regimes": detective["regimes"],
        "lagged_driver_model": detective["lagged_driver_model"],
        "pca_points": points,
    }


def _region_payload(region_id: str) -> Dict[str, Any]:
    return {"id": region_id, **REGIONS[region_id]}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def serve_observatory_ui() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/catalog")
def get_catalog() -> Dict[str, Any]:
    return {
        "data_period": [DATA_START_YEAR, DATA_END_YEAR],
        "ml_start_year": ML_START_YEAR,
        "min_window_years": MIN_WINDOW_YEARS,
        "regions": REGIONS,
        "variables": VARIABLES,
        "map_variables": MAP_VARIABLES,
        "cases": public_cases(),
    }


@app.get("/api/ml-manifest")
def get_ml_manifest() -> Dict[str, Any]:
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/full-dashboard/{region_id}")
def get_full_region_dashboard(
    region_id: str = Depends(path_region),
    variable: str = Depends(query_variable),
    window: YearWindow = Depends(),
    exclude_ml_extremes: bool = Query(False),
    homogeneous_ceres: bool = Query(True),
) -> Dict[str, Any]:
    detective = svc.ml_detective(region_id, window.start_year, window.end_year)
    excluded = tuple(detective["extreme_dates"]) if exclude_ml_extremes else ()

    def analysis(excluded_dates: tuple) -> Dict[str, Any]:
        return svc.variable_analysis(region_id, variable, window.start_year, window.end_year, homogeneous_ceres, excluded_dates)

    res_active = analysis(excluded)
    res_baseline = analysis(()) if excluded else res_active

    return {
        "region": _region_payload(region_id),
        "variable_info": VARIABLES[variable],
        "window": {
            "requested": [window.start_year, window.end_year],
            "analysed": [res_active["year_start"], res_active["year_end"]],
            "start_moved_for_variable": analysis_start_year(variable, window.start_year, homogeneous_ceres) != window.start_year,
        },
        "filters": {"exclude_ml_extremes": exclude_ml_extremes, "homogeneous_ceres": homogeneous_ceres,
                    "n_excluded_months": len(excluded)},
        **_trend_payload(res_active),
        "baseline_mk": res_baseline["monthly_mk"],
        "monthly_series": _monthly_series(res_active, variable),
        "annual_series": _annual_series(res_active),
        "auto_scan": svc.auto_scan(region_id, homogeneous_ceres),
        "outlier_sensitivity": svc.outlier_sensitivity(region_id, homogeneous_ceres),
        "ml": _ml_payload(detective),
    }


@app.get("/api/dipole")
def compare_dipole_endpoint(
    region_a: str = Query("EC_Coast_Guayas"),
    region_b: str = Query("BR_Amazon_Manaus"),
    variable: str = Query("PRECTOTCORR"),
    window: YearWindow = Depends(),
) -> Dict[str, Any]:
    _check_region(region_a)
    _check_region(region_b)
    _check_variable(variable)
    if region_a == region_b:
        raise HTTPException(status_code=422, detail="Choose two different regions.")

    res = svc.dipole(region_a, region_b, variable, window.start_year, window.end_year)

    def _stats(r: Dict[str, Any]) -> Dict[str, Any]:
        enso = r["enso"]
        return {
            "theil_sen_decade": r["monthly_mk"]["theil_sen_decade"],
            "p_hamed_rao": r["monthly_mk"]["p_hamed_rao"],
            "verdict": r["verdict"],
            "warnings": r["warnings"],
            "enso": {k: enso[k] for k in ("r_oni", "p_oni", "best_lag", "r_best", "q_best")},
            "period": [r["year_start"], r["year_end"]],
        }

    return {
        "variable_info": VARIABLES[variable],
        "region_a_info": _region_payload(region_a),
        "region_b_info": _region_payload(region_b),
        "stats_a": _stats(res["region_a"]),
        "stats_b": _stats(res["region_b"]),
        "cross_correlation": res["cross_correlation"],
        "r_during_strong_enso": res["r_during_strong_enso"],
        "n_strong_enso_months": res["n_strong_enso_months"],
        "trend_relation": res["trend_relation"],
        "enso_relation": res["enso_relation"],
        "series": _records(res["merged_series"], ["date", "z_a", "z_b", "oni"], 3),
    }


@app.get("/api/multiscale")
def get_multiscale_data(
    region_id: str = Query("EC_Andes_Paute"),
    grid_variable: str = Query("T2M"),
) -> Dict[str, Any]:
    _check_region(region_id)
    _check_variable(grid_variable, allowed=MAP_VARIABLES)

    modis_region = region_id if REGIONS[region_id]["has_modis_250m"] else MODIS_FALLBACK_REGION
    grid = svc.grid_trends(grid_variable)
    return {
        "grid_variable": grid_variable,
        "grid_cells": _records(grid, list(grid.columns)) if not grid.empty else [],
        "modis": {
            "region_id": modis_region,
            "is_fallback": modis_region != region_id,
            "records": _records(load_modis_ndvi(modis_region), ["calendar_date", "ndvi"]),
        },
        "planetary_computer_stac": load_planetary_computer_stac(modis_region),
        "meteomatics_1km": {
            region: _records(df, list(df.columns), 3) for region, df in load_meteomatics_1km_cache().items()
        },
    }


@app.get("/api/live-probe")
def live_coordinate_probe(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    variable: str = Depends(query_variable),
) -> Dict[str, Any]:
    if not live_probe_limiter.allow():
        raise HTTPException(status_code=429, detail="Too many live probes; wait a minute and try again.")

    df, meta = fetch_live_nasa_power(lat, lon)
    if variable not in df.columns or df[variable].count() < 60:
        raise HTTPException(status_code=422, detail=f"NASA POWER has too little {variable} data at this coordinate.")

    res = analyze_single_variable(
        df, variable, oni_df=svc.oni(),
        start_year=analysis_start_year(variable, DATA_START_YEAR, homogeneous_splice=True),
        end_year=DATA_END_YEAR, known_break_date=known_break_date(variable),
    )
    detective = run_multivariate_anomaly_detective(df, oni_df=svc.oni(), start_year=ML_START_YEAR, end_year=DATA_END_YEAR)
    return {
        "meta": {**meta, "lat": float(np.round(meta["lat"], 3)), "lon": float(np.round(meta["lon"], 3))},
        "variable": variable,
        "variable_info": VARIABLES[variable],
        **_trend_payload(res),
        "top_anomalies": detective["anomalies"][:5],
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
