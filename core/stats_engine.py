"""
TerraLens statistical inference engine.

Every function is deterministic (bootstrap draws use a fixed seed) and returns plain
numbers or stable string codes; human-readable labels live in the frontend locales.

Methods and the references they follow:
- Monthly deseasonalization, annual anomalies and seasonal subsets
  (Acker et al., 2009, NASA NTRS 20090042747).
- OLS with the Santer et al. (2000) lag-1 effective sample size.
- Mann-Kendall with the Hamed & Rao (1998) variance correction and the Theil-Sen slope (Sen, 1968).
- Pettitt (1979) change-point location with a circular block-bootstrap p-value, so serial
  autocorrelation does not inflate its significance.
- Step-versus-trend model comparison (BIC) to flag trends that one level shift explains better.
- Lagged ENSO correlations with the Bretherton et al. (1999) effective sample size.
- Benjamini & Hochberg (1995) false discovery rate.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

SIGNIFICANCE_ALPHA = 0.05
MARGINAL_ALPHA = 0.10
BOOTSTRAP_SAMPLES = 999
BOOTSTRAP_SEED = 2009
# Kass & Raftery (1995): a BIC difference above 10 is "very strong" evidence for one model.
STEP_EVIDENCE_BIC = 10.0
# Lag-1 persistence above which "Hamed-Rao yes, Santer no" is reported as a memory artefact.
MEMORY_REJECTION_R1 = 0.5
MAX_ENSO_LAG_MONTHS = 6
MIN_MONTHS_EACH_SIDE_OF_BREAK = 24
# Seasons are labelled by the year of their last month, so DJF 1998 = Dec 1997 + Jan-Feb 1998.
SEASONS = (
    ("DJF", (12, 1, 2)),
    ("MAM", (3, 4, 5)),
    ("JJA", (6, 7, 8)),
    ("SON", (9, 10, 11)),
)


# ---------------------------------------------------------------------------
# Deseasonalization (Acker et al., 2009)
# ---------------------------------------------------------------------------
def deseasonalize_acker(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Step 1 of Acker et al. (2009): subtract each calendar month's mean to obtain monthly anomalies.
    Also returns standardized anomalies z = (x - mean_m) / std_m for multivariate comparisons.
    """
    sub = df[["year", "month", "date", "decimal_year", value_col]].dropna().copy()
    sub = sub.sort_values(["year", "month"]).reset_index(drop=True)

    clim = (
        sub.groupby("month")[value_col]
        .agg(clim_mean="mean", clim_std=lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 1.0)
        .reset_index()
    )
    sub = sub.merge(clim, on="month", how="left").sort_values(["year", "month"]).reset_index(drop=True)

    sub["anomaly"] = sub[value_col] - sub["clim_mean"]
    safe_std = np.where(sub["clim_std"] > 1e-12, sub["clim_std"], 1.0)
    sub["z_score"] = sub["anomaly"] / safe_std
    return sub


def compute_annual_anomalies(deseason_df: pd.DataFrame, value_col: str, min_months: int = 9) -> pd.DataFrame:
    """Step 2 of Acker et al. (2009): average the monthly anomalies of each calendar year."""
    grp = (
        deseason_df.groupby("year")
        .agg(
            annual_anomaly=("anomaly", "mean"),
            annual_raw=(value_col, "mean"),
            n_months=("anomaly", "count"),
        )
        .reset_index()
    )
    return grp[grp["n_months"] >= min_months].sort_values("year").reset_index(drop=True)


def season_year(year: pd.Series, month: pd.Series) -> pd.Series:
    """December belongs to the following year's DJF season."""
    return year + (month == 12).astype(int)


def compute_climatology_and_seasonal_trends(deseason_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Step 3 of Acker et al. (2009): the 12-month climatology and trends of each season's
    yearly mean anomaly. Only complete seasons (3 months) are used.
    """
    clim_rows = []
    for month in range(1, 13):
        m_sub = deseason_df[deseason_df["month"] == month]
        if m_sub.empty:
            continue
        mk_m = mann_kendall_hamed_rao(m_sub["year"].to_numpy(dtype=float), m_sub["anomaly"].to_numpy(dtype=float))
        clim_rows.append(
            {
                "month": month,
                "clim_mean": round(float(m_sub["clim_mean"].iloc[0]), 4),
                "clim_std": round(float(m_sub["clim_std"].iloc[0]), 4),
                "theil_sen_decade": round(mk_m["theil_sen_decade"], 4),
                "p_hamed_rao": round(mk_m["p_hamed_rao"], 5),
            }
        )

    work = deseason_df.assign(season_year=season_year(deseason_df["year"], deseason_df["month"]))
    seasonal_rows = []
    for code, months in SEASONS:
        s_sub = work[work["month"].isin(months)]
        s_ann = s_sub.groupby("season_year").agg(anomaly=("anomaly", "mean"), n=("anomaly", "count")).reset_index()
        s_ann = s_ann[s_ann["n"] == len(months)].sort_values("season_year")
        if len(s_ann) < 5:
            continue
        years = s_ann["season_year"].to_numpy(dtype=float)
        values = s_ann["anomaly"].to_numpy(dtype=float)
        mk_s = mann_kendall_hamed_rao(years, values)
        ols_s = ols_with_santer_correction(years, values)
        seasonal_rows.append(
            {
                "season": code,
                "n_years": int(len(s_ann)),
                "theil_sen_decade": round(mk_s["theil_sen_decade"], 4),
                "ols_decade": round(ols_s["slope_decade"], 4),
                "p_hamed_rao": round(mk_s["p_hamed_rao"], 5),
                "significant": bool(mk_s["p_hamed_rao"] < SIGNIFICANCE_ALPHA),
            }
        )

    return {"monthly_climatology": clim_rows, "seasonal_trends": seasonal_rows}


# ---------------------------------------------------------------------------
# Trend tests
# ---------------------------------------------------------------------------
def lag1_autocorrelation(values: np.ndarray) -> float:
    """Pearson correlation between consecutive values (0 when undefined)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 3 or np.std(v[:-1]) < 1e-12 or np.std(v[1:]) < 1e-12:
        return 0.0
    return float(np.corrcoef(v[:-1], v[1:])[0, 1])


def ols_with_santer_correction(t_years: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Linear regression y(t) = b0 + b1 t with the naive t-test and the Santer et al. (2000)
    correction n_eff = n (1 - r1) / (1 + r1), where r1 is the lag-1 autocorrelation of the residuals.
    """
    mask = np.isfinite(t_years) & np.isfinite(y)
    t = np.asarray(t_years[mask], dtype=float)
    v = np.asarray(y[mask], dtype=float)
    n = len(v)
    if n < 5:
        return {
            "n": n, "slope_yr": 0.0, "slope_decade": 0.0, "intercept": 0.0, "r_squared": 0.0,
            "p_naive": 1.0, "lag1_r": 0.0, "n_eff": float(n), "p_santer": 1.0,
            "ci95_santer_decade_low": 0.0, "ci95_santer_decade_high": 0.0,
        }

    res = stats.linregress(t, v)
    slope = float(res.slope)
    intercept = float(res.intercept)
    se_naive = float(res.stderr) if res.stderr and res.stderr > 0 else 1e-12

    r1 = lag1_autocorrelation(v - (intercept + slope * t))
    n_eff = max(4.0, n * (1.0 - r1) / (1.0 + r1)) if r1 > 0.0 else float(n)
    df_eff = max(2.0, n_eff - 2.0)
    se_santer = se_naive * np.sqrt((n - 2.0) / df_eff)
    t_santer = slope / se_santer if se_santer > 0 else 0.0
    p_santer = float(2.0 * stats.t.sf(abs(t_santer), df=df_eff))
    t_crit = float(stats.t.ppf(0.975, df=df_eff))

    return {
        "n": int(n),
        "slope_yr": slope,
        "slope_decade": slope * 10.0,
        "intercept": intercept,
        "r_squared": float(res.rvalue ** 2),
        "p_naive": float(res.pvalue),
        "lag1_r": r1,
        "n_eff": float(n_eff),
        "p_santer": p_santer,
        "ci95_santer_decade_low": float((slope - t_crit * se_santer) * 10.0),
        "ci95_santer_decade_high": float((slope + t_crit * se_santer) * 10.0),
    }


def mann_kendall_hamed_rao(t_years: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Mann-Kendall test (original and Hamed & Rao 1998 variance-corrected) plus Theil-Sen slope."""
    mask = np.isfinite(t_years) & np.isfinite(y)
    t = np.asarray(t_years[mask], dtype=float)
    x = np.asarray(y[mask], dtype=float)
    n = len(x)
    if n < 5:
        return {
            "n": n, "s_stat": 0.0, "tau": 0.0, "p_original": 1.0, "variance_inflation_factor": 1.0,
            "p_hamed_rao": 1.0, "theil_sen_yr": 0.0, "theil_sen_decade": 0.0, "theil_sen_intercept": 0.0,
            "theil_sen_ci_low_decade": 0.0, "theil_sen_ci_high_decade": 0.0,
        }

    diff = x[None, :] - x[:, None]
    s_stat = float(np.sum(np.sign(diff[np.triu_indices(n, k=1)])))

    _, counts = np.unique(x, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var_0 = max((n * (n - 1) * (2 * n + 5) - tie_term) / 18.0, 1e-12)

    def _z(variance: float) -> float:
        if s_stat > 0:
            return (s_stat - 1.0) / np.sqrt(variance)
        if s_stat < 0:
            return (s_stat + 1.0) / np.sqrt(variance)
        return 0.0

    ts_res = stats.theilslopes(x, t, 0.95)

    # Hamed & Rao (1998): autocorrelation of the ranks of the detrended series, significant lags only.
    detrended = x - float(stats.theilslopes(x)[0]) * np.arange(n, dtype=float)
    r_centered = stats.rankdata(detrended) - (n + 1) / 2.0
    denom = float(np.sum(r_centered ** 2))
    crit = float(stats.norm.ppf(0.975) / np.sqrt(n))
    corr_sum = 0.0
    if denom > 1e-12:
        for k in range(1, n):
            rk = float(np.sum(r_centered[:-k] * r_centered[k:]) / denom)
            if abs(rk) > crit:
                corr_sum += (n - k) * (n - k - 1) * (n - k - 2) * rk
    vif = max(1.0, 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * corr_sum)

    z_orig = _z(var_0)
    z_hr = _z(var_0 * vif)
    return {
        "n": int(n),
        "s_stat": s_stat,
        "tau": float(s_stat / (0.5 * n * (n - 1))),
        "p_original": float(2.0 * stats.norm.sf(abs(z_orig))),
        "variance_inflation_factor": float(vif),
        "p_hamed_rao": float(2.0 * stats.norm.sf(abs(z_hr))),
        "theil_sen_yr": float(ts_res.slope),
        "theil_sen_decade": float(ts_res.slope) * 10.0,
        "theil_sen_intercept": float(ts_res.intercept),
        "theil_sen_ci_low_decade": float(ts_res.low_slope) * 10.0,
        "theil_sen_ci_high_decade": float(ts_res.high_slope) * 10.0,
    }


# ---------------------------------------------------------------------------
# Change points and homogeneity
# ---------------------------------------------------------------------------
def bootstrap_block_length(values: np.ndarray) -> int:
    """
    Block length that keeps the serial memory of the series: (1 + r1) / (1 - r1), i.e. n / n_eff
    from the Santer et al. (2000) formula, capped at a tenth of the series.
    """
    n = len(values)
    r1 = lag1_autocorrelation(values)
    length = 1 if r1 <= 0 else int(np.ceil((1.0 + r1) / max(1e-6, 1.0 - r1)))
    return int(max(1, min(length, max(1, n // 10))))


def _circular_block_bootstrap(values: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Resamples whole blocks (wrapping around the end) to build series without a structural break."""
    n = len(values)
    block = bootstrap_block_length(values)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_samples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return values[idx.reshape(n_samples, -1)[:, :n]]


def _pettitt_statistic(values: np.ndarray) -> np.ndarray:
    """|U_t| for every split t (works row-wise on a 2-D array)."""
    arr = np.atleast_2d(values)
    n = arr.shape[1]
    ranks = stats.rankdata(arr, axis=1)
    u_t = 2.0 * np.cumsum(ranks, axis=1)[:, :-1] - np.arange(1, n) * (n + 1.0)
    return np.abs(u_t)


def pettitt_change_point_test(
    dates: Sequence[str],
    y: np.ndarray,
    n_boot: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """
    Pettitt (1979) change point. The location and K statistic are the classical ones; the p-value
    comes from a circular block bootstrap so that persistent series do not yield absurdly small p.
    Pettitt cannot tell a step from a gradual trend: see compare_step_vs_trend for that question.
    """
    mask = np.isfinite(y)
    v = np.asarray(y, dtype=float)[mask]
    valid_dates = [d for d, m in zip(dates, mask) if m]
    n = len(v)
    if n < 24:
        return {
            "change_index": 0, "change_date": valid_dates[0] if valid_dates else None, "k_stat": 0.0,
            "p_value": 1.0, "p_value_asymptotic": 1.0, "block_length": 1, "n_bootstrap": 0,
            "mean_before": 0.0, "mean_after": 0.0, "step_delta": 0.0, "is_significant_breakpoint": False,
        }

    abs_u = _pettitt_statistic(v)[0]
    k_idx = int(np.argmax(abs_u))
    k_stat = float(abs_u[k_idx])

    rng = np.random.default_rng(seed)
    boot_k = _pettitt_statistic(_circular_block_bootstrap(v, n_boot, rng)).max(axis=1)
    p_boot = float((1 + np.sum(boot_k >= k_stat)) / (n_boot + 1))
    p_asym = float(min(1.0, 2.0 * np.exp((-6.0 * k_stat ** 2) / (n ** 3 + n ** 2))))

    mean_before = float(np.mean(v[: k_idx + 1]))
    mean_after = float(np.mean(v[k_idx + 1:]))
    return {
        "change_index": k_idx,
        "change_date": valid_dates[k_idx],
        "k_stat": k_stat,
        "p_value": p_boot,
        "p_value_asymptotic": p_asym,
        "block_length": bootstrap_block_length(v),
        "n_bootstrap": int(n_boot),
        "mean_before": mean_before,
        "mean_after": mean_after,
        "step_delta": mean_after - mean_before,
        "is_significant_breakpoint": bool(p_boot < SIGNIFICANCE_ALPHA),
    }


def known_breakpoint_test(
    dates: Sequence[str],
    y: np.ndarray,
    break_date: str,
    n_boot: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Optional[Dict[str, Any]]:
    """
    Tests a documented break (e.g. an instrument change) at a fixed date: difference of means
    after minus before, with a circular block-bootstrap p-value. Returns None when the window
    does not hold enough months on both sides of the date.
    """
    mask = np.isfinite(y)
    v = np.asarray(y, dtype=float)[mask]
    if len(v) < 2 * MIN_MONTHS_EACH_SIDE_OF_BREAK:
        return None
    valid_dates = np.asarray([d for d, m in zip(dates, mask) if m])
    after = valid_dates >= break_date
    n_after = int(after.sum())
    n_before = int(len(v) - n_after)
    if min(n_after, n_before) < MIN_MONTHS_EACH_SIDE_OF_BREAK:
        return None

    delta = float(v[after].mean() - v[~after].mean())
    rng = np.random.default_rng(seed)
    boot = _circular_block_bootstrap(v, n_boot, rng)
    boot_delta = boot[:, after].mean(axis=1) - boot[:, ~after].mean(axis=1)
    p_boot = float((1 + np.sum(np.abs(boot_delta) >= abs(delta))) / (n_boot + 1))
    return {
        "break_date": break_date,
        "n_before": n_before,
        "n_after": n_after,
        "mean_before": float(v[~after].mean()),
        "mean_after": float(v[after].mean()),
        "step_delta": delta,
        "p_value": p_boot,
        "is_significant": bool(p_boot < SIGNIFICANCE_ALPHA),
    }


def compare_step_vs_trend(t_years: np.ndarray, y: np.ndarray, change_index: int) -> Dict[str, Any]:
    """
    Compares a linear trend (2 parameters) with a single level shift at the Pettitt date
    (3 parameters: two means and the break) using BIC. Because the break location tau was found
    by searching over n candidate points (Pettitt argmax), a standard Delta BIC > 2 threshold
    would suffer from post-selection inflation. We require Delta BIC > STEP_EVIDENCE_BIC (10.0),
    corresponding to Kass & Raftery (1995) 'very strong' evidence, to classify a preference.
    preferred_model is 'step' or 'trend' when |Delta BIC| exceeds STEP_EVIDENCE_BIC, else 'inconclusive'.
    """
    mask = np.isfinite(t_years) & np.isfinite(y)
    t = np.asarray(t_years, dtype=float)[mask]
    v = np.asarray(y, dtype=float)[mask]
    n = len(v)
    if n < 24 or not 0 <= change_index < n - 1:
        return {"bic_trend": None, "bic_step": None, "delta_bic": 0.0, "preferred_model": "inconclusive"}

    slope, intercept = np.polyfit(t, v, 1)
    resid_trend = v - (intercept + slope * t)
    before, after = v[: change_index + 1], v[change_index + 1:]
    resid_step = np.concatenate([before - before.mean(), after - after.mean()])
    rss_trend, rss_step = float(np.sum(resid_trend ** 2)), float(np.sum(resid_step ** 2))

    # Persistent noise carries less information than n independent points, so the likelihood term
    # uses n_eff. The persistence is read from the better-fitting model: a real step leaves white
    # residuals under the step model, whereas pure memory leaves them persistent under both.
    best_resid = resid_step if rss_step < rss_trend else resid_trend
    r1 = lag1_autocorrelation(best_resid)
    n_eff = float(np.clip(n * (1.0 - r1) / (1.0 + r1), 10.0, n)) if r1 > 0.0 else float(n)

    def _bic(rss: float, k: int) -> float:
        return float(n_eff * np.log(max(rss, 1e-12) / n) + k * np.log(n_eff))

    bic_trend, bic_step = _bic(rss_trend, 2), _bic(rss_step, 3)
    delta = bic_trend - bic_step
    if delta > STEP_EVIDENCE_BIC:
        preferred = "step"
    elif delta < -STEP_EVIDENCE_BIC:
        preferred = "trend"
    else:
        preferred = "inconclusive"
    return {"bic_trend": bic_trend, "bic_step": bic_step, "delta_bic": float(delta), "preferred_model": preferred,
            "n_eff": n_eff}


# ---------------------------------------------------------------------------
# Correlations with autocorrelation-aware significance
# ---------------------------------------------------------------------------
def correlation_with_neff(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Pearson correlation whose p-value uses the Bretherton et al. (1999) effective sample size
    n_eff = n (1 - r1x r1y) / (1 + r1x r1y).
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    n = len(a)
    if n < 12 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return {"r": 0.0, "p": 1.0, "n": n, "n_eff": float(n)}
    r = float(np.corrcoef(a, b)[0, 1])
    persistence = lag1_autocorrelation(a) * lag1_autocorrelation(b)
    n_eff = float(np.clip(n * (1.0 - persistence) / (1.0 + persistence), 3.0, n))
    dof = max(1.0, n_eff - 2.0)
    t_stat = r * np.sqrt(dof / max(1e-12, 1.0 - r ** 2))
    return {"r": r, "p": float(2.0 * stats.t.sf(abs(t_stat), df=dof)), "n": int(n), "n_eff": n_eff}


def _month_index(year: pd.Series, month: pd.Series) -> np.ndarray:
    return (year.astype(int) * 12 + month.astype(int) - 1).to_numpy()


def enso_teleconnection(des: pd.DataFrame, oni_df: pd.DataFrame, max_lag: int = MAX_ENSO_LAG_MONTHS) -> Dict[str, Any]:
    """
    Correlates monthly anomalies with the NOAA ONI leading by 0..max_lag months. p-values use the
    effective sample size; q-values control the false discovery rate across the tested lags.
    """
    empty = {"r_oni": 0.0, "p_oni": 1.0, "n_eff_oni": 0.0, "n_overlap": 0, "best_lag": 0, "r_best": 0.0,
             "q_best": 1.0, "lags": []}
    if oni_df is None or oni_df.empty or des.empty:
        return empty

    oni_by_month = pd.Series(oni_df["oni"].to_numpy(dtype=float), index=_month_index(oni_df["year"], oni_df["month"]))
    target_idx = _month_index(des["year"], des["month"])
    anomalies = des["anomaly"].to_numpy(dtype=float)

    rows = []
    for lag in range(max_lag + 1):
        oni_values = oni_by_month.reindex(target_idx - lag).to_numpy()
        ok = np.isfinite(oni_values) & np.isfinite(anomalies)
        if ok.sum() < 24:
            continue
        corr = correlation_with_neff(anomalies[ok], oni_values[ok])
        rows.append({"lag": lag, **corr})
    if not rows:
        return empty

    for row, q in zip(rows, benjamini_hochberg_fdr([r["p"] for r in rows])):
        row["q"] = q
    lag0 = next((r for r in rows if r["lag"] == 0), rows[0])
    best = max(rows, key=lambda r: abs(r["r"]))
    return {
        "r_oni": lag0["r"],
        "p_oni": lag0["p"],
        "n_eff_oni": lag0["n_eff"],
        "n_overlap": lag0["n"],
        "best_lag": best["lag"],
        "r_best": best["r"],
        "q_best": best["q"],
        "lags": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
    }


# ---------------------------------------------------------------------------
# Verdicts and multiple testing
# ---------------------------------------------------------------------------
def classify_trend_verdict(
    slope_decade: float,
    p_hamed_rao: float,
    p_santer: float = 0.0,
    lag1_r: float = 0.0,
) -> Dict[str, Any]:
    """
    Verdict code from two autocorrelation-corrected tests. Monte Carlo calibration
    (scripts/calibration_study.py) showed Hamed-Rao alone calls 11-28% of trend-free AR(1) series
    significant when r1 >= 0.5, while Santer-corrected OLS stays near the nominal 5%. A trend is
    therefore only as significant as the weaker of the two tests (p = max of both).
    Codes: sig_increase, sig_decrease, marginal_increase, marginal_decrease, not_significant,
    autocorr_rejected (Hamed-Rao alone sees a trend that Santer attributes to memory).
    """
    direction = "increase" if slope_decade > 0 else "decrease"
    if lag1_r > MEMORY_REJECTION_R1 and p_hamed_rao < MARGINAL_ALPHA <= p_santer:
        return {"code": "autocorr_rejected", "direction": direction, "is_significant": False}
    p_effective = max(p_hamed_rao, p_santer)
    if p_effective < SIGNIFICANCE_ALPHA:
        return {"code": f"sig_{direction}", "direction": direction, "is_significant": True}
    if p_effective < MARGINAL_ALPHA:
        return {"code": f"marginal_{direction}", "direction": direction, "is_significant": False}
    return {"code": "not_significant", "direction": direction, "is_significant": False}


def widen_theil_sen_ci(mk: Dict[str, Any], n_eff: float) -> Dict[str, Any]:
    """
    The Theil-Sen interval of scipy assumes independent points (it excluded zero in up to 74% of
    trend-free persistent series). Widening it around the slope by sqrt(n / n_eff) restores the
    coverage Santer et al. (2000) give OLS.
    """
    n = max(float(mk["n"]), 1.0)
    inflation = float(np.sqrt(n / np.clip(n_eff, 3.0, n)))
    slope = mk["theil_sen_decade"]
    return {
        **mk,
        "theil_sen_ci_low_decade": slope - (slope - mk["theil_sen_ci_low_decade"]) * inflation,
        "theil_sen_ci_high_decade": slope + (mk["theil_sen_ci_high_decade"] - slope) * inflation,
        "theil_sen_ci_inflation": inflation,
    }


def trend_warnings(verdict: Dict[str, Any], homogeneity: Dict[str, Any], known_break: Optional[Dict[str, Any]]) -> List[str]:
    """Warning codes that qualify a verdict: step_dominated, known_splice_in_window."""
    warnings = []
    trend_claimed = verdict["code"].startswith(("sig_", "marginal_"))
    if trend_claimed and homogeneity.get("preferred_model") == "step":
        warnings.append("step_dominated")
    if known_break is not None and known_break["is_significant"]:
        warnings.append("known_splice_in_window")
    return warnings


def benjamini_hochberg_fdr(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg (1995) q-values, returned in the input order."""
    m = len(p_values)
    if m == 0:
        return []
    arr = np.asarray(p_values, dtype=float)
    order = np.argsort(arr)
    ranked = arr[order] * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_out = np.empty(m, dtype=float)
    q_out[order] = np.minimum(q_ranked, 1.0)
    return [float(x) for x in q_out]


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def analyze_single_variable(
    df: pd.DataFrame,
    variable: str,
    oni_df: Optional[pd.DataFrame] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    exclude_dates: Optional[List[str]] = None,
    known_break_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Full trend, change-point, homogeneity and ENSO pipeline for one variable."""
    work = df
    if start_year is not None:
        work = work[work["year"] >= int(start_year)]
    if end_year is not None:
        work = work[work["year"] <= int(end_year)]
    if exclude_dates:
        work = work[~work["date"].isin(set(exclude_dates))]

    des = deseasonalize_acker(work, variable)
    ann = compute_annual_anomalies(des, variable)

    t_m = des["decimal_year"].to_numpy()
    a_m = des["anomaly"].to_numpy()
    dates_m = des["date"].tolist()

    ols_m = ols_with_santer_correction(t_m, a_m)
    mk_m = widen_theil_sen_ci(mann_kendall_hamed_rao(t_m, a_m), ols_m["n_eff"])
    pettitt_m = pettitt_change_point_test(dates_m, a_m)
    homogeneity = compare_step_vs_trend(t_m, a_m, pettitt_m["change_index"])
    known_break = known_breakpoint_test(dates_m, a_m, known_break_date) if known_break_date else None

    t_y = ann["year"].to_numpy(dtype=float)
    a_y = ann["annual_anomaly"].to_numpy(dtype=float)
    ols_y = ols_with_santer_correction(t_y, a_y)
    mk_y = widen_theil_sen_ci(mann_kendall_hamed_rao(t_y, a_y), ols_y["n_eff"])

    verdict = classify_trend_verdict(
        mk_m["theil_sen_decade"], mk_m["p_hamed_rao"],
        p_santer=ols_m["p_santer"], lag1_r=ols_m["lag1_r"],
    )

    return {
        "variable": variable,
        "n_months": int(len(des)),
        "n_years": int(len(ann)),
        "year_start": int(des["year"].min()) if not des.empty else 0,
        "year_end": int(des["year"].max()) if not des.empty else 0,
        "monthly_ols": ols_m,
        "monthly_mk": mk_m,
        "annual_ols": ols_y,
        "annual_mk": mk_y,
        "pettitt": pettitt_m,
        "homogeneity": homogeneity,
        "known_break": known_break,
        "enso": enso_teleconnection(des, oni_df),
        "verdict": verdict,
        "warnings": trend_warnings(verdict, homogeneity, known_break),
        **compute_climatology_and_seasonal_trends(des),
        "monthly_series": des,
        "annual_series": ann,
    }


def trend_relation(res_a: Dict[str, Any], res_b: Dict[str, Any]) -> str:
    """'opposite' or 'same_direction' when both trends are significant, else 'not_both_significant'."""
    if not (res_a["verdict"]["is_significant"] and res_b["verdict"]["is_significant"]):
        return "not_both_significant"
    same = res_a["verdict"]["direction"] == res_b["verdict"]["direction"]
    return "same_direction" if same else "opposite"


def enso_relation(enso_a: Dict[str, Any], enso_b: Dict[str, Any]) -> str:
    """
    'opposite' or 'same_sign' when both regions correlate with ENSO at their strongest lag
    (q < 0.05 across lags), else 'not_both_significant'.
    """
    if enso_a["q_best"] >= SIGNIFICANCE_ALPHA or enso_b["q_best"] >= SIGNIFICANCE_ALPHA:
        return "not_both_significant"
    return "opposite" if enso_a["r_best"] * enso_b["r_best"] < 0 else "same_sign"


def compare_regional_dipole(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    variable: str,
    oni_df: Optional[pd.DataFrame] = None,
    start_year: int = 1981,
    end_year: int = 2025,
    known_break_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Compares two regions: anomaly cross-correlation, ENSO polarity and trend directions."""
    res_a = analyze_single_variable(df_a, variable, oni_df=oni_df, start_year=start_year, end_year=end_year,
                                    known_break_date=known_break_date)
    res_b = analyze_single_variable(df_b, variable, oni_df=oni_df, start_year=start_year, end_year=end_year,
                                    known_break_date=known_break_date)

    ser_a = res_a["monthly_series"][["date", "year", "month", "anomaly", "z_score"]].rename(
        columns={"anomaly": "anom_a", "z_score": "z_a"}
    )
    ser_b = res_b["monthly_series"][["date", "anomaly", "z_score"]].rename(columns={"anomaly": "anom_b", "z_score": "z_b"})
    merged = ser_a.merge(ser_b, on="date", how="inner").dropna()
    if oni_df is not None and not oni_df.empty:
        merged = merged.merge(oni_df[["year", "month", "oni"]], on=["year", "month"], how="left")

    cross = correlation_with_neff(merged["anom_a"].to_numpy(), merged["anom_b"].to_numpy())
    strong = merged[merged["oni"].abs() >= 1.0].dropna() if "oni" in merged.columns else merged.iloc[0:0]
    r_strong = float(np.corrcoef(strong["z_a"], strong["z_b"])[0, 1]) if len(strong) >= 12 else None

    return {
        "variable": variable,
        "region_a": res_a,
        "region_b": res_b,
        "cross_correlation": cross,
        "r_during_strong_enso": r_strong,
        "n_strong_enso_months": int(len(strong)),
        "trend_relation": trend_relation(res_a, res_b),
        "enso_relation": enso_relation(res_a["enso"], res_b["enso"]),
        "merged_series": merged,
    }
