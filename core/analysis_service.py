"""
Cached analysis services shared by the API and the guided cases.

Arguments are normalized before they reach the memoized functions, so equivalent requests share
one cache entry (e.g. a 1981 start for solar irradiance is the same analysis as a 1984 start).
The data are static caches, so results never go stale while the process runs. Callers must treat
returned objects as read-only.
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.catalog import (
    DATA_END_YEAR,
    DATA_START_YEAR,
    ML_START_YEAR,
    REGIONS,
    VARIABLES,
    analysis_start_year,
    known_break_date,
)
from core.connectors import load_ecuador_regional_grid_trends, load_noaa_oni, load_region_nasa_power
from core.ml_engine import outlier_sensitivity_row, run_multivariate_anomaly_detective
from core.stats_engine import SIGNIFICANCE_ALPHA, analyze_single_variable, benjamini_hochberg_fdr, compare_regional_dipole

COINCIDENT_BREAK_TOLERANCE_MONTHS = 2
COINCIDENT_BREAK_MIN_VARIABLES = 3


@lru_cache(maxsize=None)
def region_data(region_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    return load_region_nasa_power(region_id)


@lru_cache(maxsize=1)
def oni() -> pd.DataFrame:
    return load_noaa_oni()


# ---------------------------------------------------------------------------
# Single-variable analysis
# ---------------------------------------------------------------------------
def variable_analysis(
    region_id: str,
    variable: str,
    start_year: int = DATA_START_YEAR,
    end_year: int = DATA_END_YEAR,
    homogeneous_splice: bool = True,
    excluded_dates: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """analyze_single_variable for a region, honouring the variable's first year and splice guard."""
    start = analysis_start_year(variable, start_year, homogeneous_splice)
    return _variable_analysis(region_id, variable, start, end_year, tuple(excluded_dates))


@lru_cache(maxsize=512)
def _variable_analysis(region_id: str, variable: str, start: int, end: int, excluded: Tuple[str, ...]) -> Dict[str, Any]:
    df, _ = region_data(region_id)
    return analyze_single_variable(
        df, variable, oni_df=oni(), start_year=start, end_year=end,
        exclude_dates=list(excluded) or None, known_break_date=known_break_date(variable),
    )


# ---------------------------------------------------------------------------
# Machine learning
# ---------------------------------------------------------------------------
def ml_detective(region_id: str, start_year: int = ML_START_YEAR, end_year: int = DATA_END_YEAR) -> Dict[str, Any]:
    return _ml_detective(region_id, max(ML_START_YEAR, start_year), end_year)


@lru_cache(maxsize=64)
def _ml_detective(region_id: str, start: int, end: int) -> Dict[str, Any]:
    df, _ = region_data(region_id)
    return run_multivariate_anomaly_detective(df, oni_df=oni(), start_year=start, end_year=end)


@lru_cache(maxsize=64)
def outlier_sensitivity(region_id: str, homogeneous_splice: bool = True) -> Dict[str, Any]:
    """Full-period trends with and without the multivariate anomalies (Acker et al., 2009)."""
    excluded = tuple(ml_detective(region_id)["extreme_dates"])
    rows = [
        outlier_sensitivity_row(
            variable,
            variable_analysis(region_id, variable, DATA_START_YEAR, DATA_END_YEAR, homogeneous_splice),
            variable_analysis(region_id, variable, DATA_START_YEAR, DATA_END_YEAR, homogeneous_splice, excluded),
        )
        for variable in VARIABLES
    ]
    return {"n_excluded_months": len(excluded), "excluded_dates": list(excluded), "comparisons": rows}


# ---------------------------------------------------------------------------
# Multi-variable scan
# ---------------------------------------------------------------------------
def _month_number(date: str) -> int:
    return int(date[:4]) * 12 + int(date[5:7])


def coincident_breaks(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Largest group of variables whose significant Pettitt breaks fall within
    COINCIDENT_BREAK_TOLERANCE_MONTHS of each other. A shift shared by several variables in the
    same month can come from a change in the data inputs, so it deserves a homogeneity check.
    """
    candidates = [r for r in rows if r["pettitt_significant"] and r["pettitt_date"]]
    best: List[Dict[str, Any]] = []
    for anchor in candidates:
        group = [
            r for r in candidates
            if abs(_month_number(r["pettitt_date"]) - _month_number(anchor["pettitt_date"])) <= COINCIDENT_BREAK_TOLERANCE_MONTHS
        ]
        if len(group) > len(best):
            best = group
    if len(best) < COINCIDENT_BREAK_MIN_VARIABLES:
        return None
    dates = sorted(r["pettitt_date"] for r in best)
    return {"date": dates[len(dates) // 2], "variables": [r["variable"] for r in best]}


@lru_cache(maxsize=64)
def auto_scan(region_id: str, homogeneous_splice: bool = True) -> Dict[str, Any]:
    """Full-period scan of every variable with Benjamini-Hochberg q-values and coincident breaks."""
    rows = []
    for variable in VARIABLES:
        res = variable_analysis(region_id, variable, DATA_START_YEAR, DATA_END_YEAR, homogeneous_splice)
        rows.append(
            {
                "variable": variable,
                "period": [res["year_start"], res["year_end"]],
                "theil_sen_decade": round(res["monthly_mk"]["theil_sen_decade"], 4),
                "ols_decade": round(res["monthly_ols"]["slope_decade"], 4),
                "lag1_r": round(res["monthly_ols"]["lag1_r"], 3),
                "n_eff": round(res["monthly_ols"]["n_eff"], 1),
                "p_naive": round(res["monthly_ols"]["p_naive"], 6),
                "p_hamed_rao": round(res["monthly_mk"]["p_hamed_rao"], 6),
                "p_annual_mk": round(res["annual_mk"]["p_original"], 6),
                "pettitt_date": res["pettitt"]["change_date"],
                "pettitt_p": round(res["pettitt"]["p_value"], 4),
                "pettitt_significant": res["pettitt"]["is_significant_breakpoint"],
                "r_oni": round(res["enso"]["r_oni"], 3),
                "verdict_code": res["verdict"]["code"],
                "warnings": res["warnings"],
            }
        )
    for row, q in zip(rows, benjamini_hochberg_fdr([r["p_hamed_rao"] for r in rows])):
        row["q_fdr"] = round(q, 6)
        row["significant_after_fdr"] = bool(q < SIGNIFICANCE_ALPHA and row["verdict_code"].startswith("sig_"))
    return {"rows": rows, "coincident_breaks": coincident_breaks(rows)}


# ---------------------------------------------------------------------------
# Spatial and two-region views
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def grid_trends(variable: str) -> pd.DataFrame:
    return load_ecuador_regional_grid_trends(variable)


def dipole(region_a: str, region_b: str, variable: str, start_year: int, end_year: int) -> Dict[str, Any]:
    return _dipole(region_a, region_b, variable, analysis_start_year(variable, start_year, True), end_year)


@lru_cache(maxsize=128)
def _dipole(region_a: str, region_b: str, variable: str, start: int, end: int) -> Dict[str, Any]:
    df_a, _ = region_data(region_a)
    df_b, _ = region_data(region_b)
    return compare_regional_dipole(df_a, df_b, variable, oni_df=oni(), start_year=start, end_year=end)


def warm_up(first_region: str = "EC_Andes_Paute") -> None:
    """Precomputes the map layers and the default region, then every other region, so later clicks are instant."""
    for variable in VARIABLES:
        if VARIABLES[variable]["map"]:
            grid_trends(variable)
    for region_id in [first_region, *[r for r in REGIONS if r != first_region]]:
        ml_detective(region_id)
        auto_scan(region_id)
        outlier_sensitivity(region_id)
