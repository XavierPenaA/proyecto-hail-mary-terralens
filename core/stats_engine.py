"""
TerraLens v2.1 Statistical Inference Engine (Zero-Hallucination, Peer-Reviewed Formulations).

Implements:
1. Acker et al. (2009, NASA GES DISC NTRS 20090042747) 3-Step Deseasonalization.
2. Ordinary Least Squares (OLS) with t-test, ANOVA F-test, 95% CI, and Santer et al. (2000)
   lag-1 autocorrelation adjustment (effective sample size n_eff).
3. Non-parametric Mann-Kendall Test (standard + Hamed & Rao 1998 variance-corrected
   Modified Mann-Kendall for autocorrelated series) and Theil-Sen Robust Estimator.
4. Pettitt (1979) Non-Parametric Change-Point & Satellite Splice Detector.
5. Benjamini-Hochberg (1995) False Discovery Rate (FDR q-value) for multi-variable scans.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def deseasonalize_acker(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Acker et al. (2009, p. 3) 3-Step Deseasonalization:
      - Computes monthly climatological mean X_bar_m and standard deviation s_m
      - Computes monthly absolute anomaly A_{y,m} = X_{y,m} - X_bar_m
      - Computes monthly standardized anomaly Z_{y,m} = (X_{y,m} - X_bar_m) / s_m
    """
    sub = df[["year", "month", "date", "decimal_year", value_col]].dropna().copy()
    sub = sub.sort_values(["year", "month"]).reset_index(drop=True)

    clim = (
        sub.groupby("month")[value_col]
        .agg(clim_mean="mean", clim_std=lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 1.0)
        .reset_index()
    )
    sub = sub.merge(clim, on="month", how="left")
    sub = sub.sort_values(["year", "month"]).reset_index(drop=True)

    sub["anomaly"] = sub[value_col] - sub["clim_mean"]
    safe_std = np.where(sub["clim_std"] > 1e-12, sub["clim_std"], 1.0)
    sub["z_score"] = sub["anomaly"] / safe_std
    return sub


def compute_annual_anomalies(deseason_df: pd.DataFrame, min_months: int = 9) -> pd.DataFrame:
    """
    Acker et al. (2009) Step 2: Average monthly anomalies inside each calendar year
    to produce an annual anomaly series immune to intra-annual seasonal autocorrelation.
    """
    grp = (
        deseason_df.groupby("year")
        .agg(
            annual_anomaly=("anomaly", "mean"),
            annual_raw=(deseason_df.columns[4], "mean"),
            n_months=("anomaly", "count"),
        )
        .reset_index()
    )
    grp = grp[grp["n_months"] >= min_months].sort_values("year").reset_index(drop=True)
    return grp


def compute_climatology_and_seasonal_trends(deseason_df: pd.DataFrame, value_col: str) -> Dict[str, Any]:
    """
    Acker et al. (2009) Step 3: Computes the 12-month climatological baseline profile
    (mean ± 1σ per calendar month) and seasonal sub-series trends (DJF, MAM, JJA, SON)
    to isolate which specific season drives the long-term secular change.
    """
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    clim_rows = []
    for m_idx in range(1, 13):
        m_sub = deseason_df[deseason_df["month"] == m_idx]
        if m_sub.empty:
            continue
        c_mean = float(m_sub["clim_mean"].iloc[0])
        c_std = float(m_sub["clim_std"].iloc[0])
        mk_m = mann_kendall_hamed_rao(m_sub["year"].to_numpy(dtype=float), m_sub["anomaly"].to_numpy(dtype=float))
        clim_rows.append({
            "month": m_idx,
            "month_name": month_names[m_idx - 1],
            "clim_mean": round(c_mean, 4),
            "clim_std": round(c_std, 4),
            "theil_sen_decade": round(mk_m["theil_sen_decade"], 4),
            "p_hamed_rao": round(mk_m["p_hamed_rao"], 5),
        })

    seasons = [
        ("DJF", "Austral Summer / Boreal Winter (Dec–Feb)", [12, 1, 2]),
        ("MAM", "Transition Season (Mar–May)", [3, 4, 5]),
        ("JJA", "Austral Dry / Boreal Summer (Jun–Aug)", [6, 7, 8]),
        ("SON", "Late Dry / Spring Transition (Sep–Nov)", [9, 10, 11]),
    ]
    seasonal_rows = []
    for s_code, s_label, m_list in seasons:
        s_sub = deseason_df[deseason_df["month"].isin(m_list)]
        if len(s_sub) < 15:
            continue
        s_ann = s_sub.groupby("year", as_index=False)["anomaly"].mean().sort_values("year")
        mk_s = mann_kendall_hamed_rao(s_ann["year"].to_numpy(dtype=float), s_ann["anomaly"].to_numpy(dtype=float))
        ols_s = ols_with_santer_correction(s_ann["year"].to_numpy(dtype=float), s_ann["anomaly"].to_numpy(dtype=float))
        seasonal_rows.append({
            "season": s_code,
            "label": s_label,
            "n_years": int(len(s_ann)),
            "theil_sen_decade": round(mk_s["theil_sen_decade"], 4),
            "ols_decade": round(ols_s["slope_decade"], 4),
            "p_hamed_rao": round(mk_s["p_hamed_rao"], 5),
            "significant": bool(mk_s["p_hamed_rao"] < 0.05),
        })

    return {
        "monthly_climatology": clim_rows,
        "seasonal_trends": seasonal_rows,
    }


def ols_with_santer_correction(t_years: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Parametric linear regression y(t) = beta_0 + beta_1 * t, with:
      - Standard OLS t-test, ANOVA F-statistic (F = t^2), R^2, 95% confidence interval
      - Santer et al. (2000) lag-1 autocorrelation correction:
        n_eff = n * (1 - r_1) / (1 + r_1) when r_1 > 0.
    """
    mask = np.isfinite(t_years) & np.isfinite(y)
    t = np.asarray(t_years[mask], dtype=float)
    v = np.asarray(y[mask], dtype=float)
    n = len(v)
    if n < 5:
        return {
            "n": n,
            "slope_yr": 0.0,
            "slope_decade": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "t_stat_naive": 0.0,
            "f_stat_naive": 0.0,
            "p_naive": 1.0,
            "lag1_r": 0.0,
            "n_eff": float(n),
            "t_stat_santer": 0.0,
            "f_stat_santer": 0.0,
            "p_santer": 1.0,
            "ci95_decade_low": 0.0,
            "ci95_decade_high": 0.0,
            "ci95_santer_decade_low": 0.0,
            "ci95_santer_decade_high": 0.0,
        }

    res = stats.linregress(t, v)
    slope = float(res.slope)
    intercept = float(res.intercept)
    r_val = float(res.rvalue)
    r_sq = float(r_val ** 2)
    se_naive = float(res.stderr) if res.stderr and res.stderr > 0 else 1e-12
    t_naive = slope / se_naive
    f_naive = float(t_naive ** 2)
    p_naive = float(res.pvalue)

    t_crit_naive = float(stats.t.ppf(0.975, df=max(1, n - 2)))
    ci_low_naive = (slope - t_crit_naive * se_naive) * 10.0
    ci_high_naive = (slope + t_crit_naive * se_naive) * 10.0

    # Santer et al. (2000) lag-1 residual autocorrelation
    residuals = v - (intercept + slope * t)
    if np.std(residuals[:-1]) > 1e-12 and np.std(residuals[1:]) > 1e-12:
        r1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    else:
        r1 = 0.0

    if r1 > 0.0:
        n_eff = max(4.0, n * (1.0 - r1) / (1.0 + r1))
    else:
        n_eff = float(n)

    df_eff = max(2.0, n_eff - 2.0)
    se_santer = se_naive * np.sqrt((n - 2.0) / df_eff)
    t_santer = float(slope / se_santer) if se_santer > 0 else 0.0
    f_santer = float(t_santer ** 2)
    p_santer = float(2.0 * (1.0 - stats.t.cdf(abs(t_santer), df=df_eff)))

    t_crit_santer = float(stats.t.ppf(0.975, df=df_eff))
    ci_low_santer = (slope - t_crit_santer * se_santer) * 10.0
    ci_high_santer = (slope + t_crit_santer * se_santer) * 10.0

    return {
        "n": int(n),
        "slope_yr": slope,
        "slope_decade": slope * 10.0,
        "intercept": intercept,
        "r_squared": r_sq,
        "t_stat_naive": float(t_naive),
        "f_stat_naive": f_naive,
        "p_naive": p_naive,
        "lag1_r": r1,
        "n_eff": float(n_eff),
        "t_stat_santer": t_santer,
        "f_stat_santer": f_santer,
        "p_santer": p_santer,
        "ci95_decade_low": float(ci_low_naive),
        "ci95_decade_high": float(ci_high_naive),
        "ci95_santer_decade_low": float(ci_low_santer),
        "ci95_santer_decade_high": float(ci_high_santer),
    }


def mann_kendall_hamed_rao(t_years: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Non-parametric Mann-Kendall trend test (both original and Hamed & Rao 1998
    variance-corrected for serial autocorrelation) + Theil-Sen robust slope.
    """
    mask = np.isfinite(t_years) & np.isfinite(y)
    t = np.asarray(t_years[mask], dtype=float)
    x = np.asarray(y[mask], dtype=float)
    n = len(x)
    if n < 5:
        return {
            "n": n,
            "s_stat": 0.0,
            "tau": 0.0,
            "z_original": 0.0,
            "p_original": 1.0,
            "variance_inflation_factor": 1.0,
            "z_hamed_rao": 0.0,
            "p_hamed_rao": 1.0,
            "theil_sen_yr": 0.0,
            "theil_sen_decade": 0.0,
            "theil_sen_intercept": 0.0,
            "theil_sen_ci_low_decade": 0.0,
            "theil_sen_ci_high_decade": 0.0,
        }

    # Pairwise sign matrix
    diff = x[None, :] - x[:, None]
    iu = np.triu_indices(n, k=1)
    s_stat = float(np.sum(np.sign(diff[iu])))

    # Tie-corrected original variance Var_0(S)
    _, counts = np.unique(x, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var_0 = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    var_0 = max(var_0, 1e-12)

    if s_stat > 0:
        z_orig = (s_stat - 1.0) / np.sqrt(var_0)
    elif s_stat < 0:
        z_orig = (s_stat + 1.0) / np.sqrt(var_0)
    else:
        z_orig = 0.0
    p_orig = float(2.0 * (1.0 - stats.norm.cdf(abs(z_orig))))
    tau = float(s_stat / (0.5 * n * (n - 1)))

    # Theil-Sen robust slope
    ts_res = stats.theilslopes(x, t, 0.95)
    ts_slope_yr = float(ts_res.slope)
    ts_intercept = float(ts_res.intercept)
    ts_low_dec = float(ts_res.low_slope * 10.0)
    ts_high_dec = float(ts_res.high_slope * 10.0)

    # Hamed & Rao (1998) modified variance correction on detrended ranks across all lags k = 1..n-1
    slope_idx = float(stats.theilslopes(x)[0])
    detrended = x - slope_idx * np.arange(n, dtype=float)
    ranks = stats.rankdata(detrended)
    r_centered = ranks - np.mean(ranks)
    denom = float(np.sum(r_centered ** 2))
    crit = float(stats.norm.ppf(0.975) / np.sqrt(n))
    corr_sum = 0.0
    if denom > 1e-12:
        for k in range(1, n):
            rk = float(np.sum(r_centered[:-k] * r_centered[k:]) / denom)
            if abs(rk) > crit:
                corr_sum += (n - k) * (n - k - 1) * (n - k - 2) * rk

    vif = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * corr_sum
    vif = max(1.0, float(vif))
    var_star = var_0 * vif

    if s_stat > 0:
        z_hr = (s_stat - 1.0) / np.sqrt(var_star)
    elif s_stat < 0:
        z_hr = (s_stat + 1.0) / np.sqrt(var_star)
    else:
        z_hr = 0.0
    p_hr = float(2.0 * (1.0 - stats.norm.cdf(abs(z_hr))))

    return {
        "n": int(n),
        "s_stat": s_stat,
        "tau": tau,
        "z_original": float(z_orig),
        "p_original": p_orig,
        "variance_inflation_factor": vif,
        "z_hamed_rao": float(z_hr),
        "p_hamed_rao": p_hr,
        "theil_sen_yr": ts_slope_yr,
        "theil_sen_decade": ts_slope_yr * 10.0,
        "theil_sen_intercept": ts_intercept,
        "theil_sen_ci_low_decade": ts_low_dec,
        "theil_sen_ci_high_decade": ts_high_dec,
    }


def pettitt_change_point_test(dates: List[str], y: np.ndarray) -> Dict[str, Any]:
    """
    Pettitt (1979) non-parametric change-point test to detect abrupt regime shifts
    or satellite sensor splices (e.g., Jan 2001 GEWEX SRB -> CERES splice).
    """
    mask = np.isfinite(y)
    v = np.asarray(y[mask], dtype=float)
    valid_dates = [d for d, m in zip(dates, mask) if m]
    n = len(v)
    if n < 12:
        return {
            "change_index": 0,
            "change_date": valid_dates[0] if valid_dates else "N/A",
            "k_stat": 0.0,
            "p_value": 1.0,
            "mean_before": 0.0,
            "mean_after": 0.0,
            "step_delta": 0.0,
            "is_significant_breakpoint": False,
        }

    ranks = stats.rankdata(v)
    cum_ranks = np.cumsum(ranks)
    idx_arr = np.arange(1, n)
    u_t = 2.0 * cum_ranks[:-1] - idx_arr * (n + 1.0)
    abs_u = np.abs(u_t)
    k_idx = int(np.argmax(abs_u))
    k_stat = float(abs_u[k_idx])

    p_val = float(min(1.0, 2.0 * np.exp((-6.0 * (k_stat ** 2)) / (n ** 3 + n ** 2))))
    mean_before = float(np.mean(v[: k_idx + 1]))
    mean_after = float(np.mean(v[k_idx + 1 :]))
    step_delta = mean_after - mean_before

    return {
        "change_index": k_idx,
        "change_date": valid_dates[k_idx],
        "k_stat": k_stat,
        "p_value": p_val,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "step_delta": step_delta,
        "is_significant_breakpoint": bool(p_val < 0.01),
    }


def classify_trend_verdict(
    slope_decade: float,
    p_hamed_rao: float,
    p_annual: float,
    p_santer: float = 0.0,
    lag1_r: float = 0.0,
) -> Dict[str, str]:
    """
    Standardized scientific verdict using autocorrelation-corrected p-values.
    When serial autocorrelation is extreme (lag1_r > 0.85), requires confirmation
    from either Annual Anomaly Mann-Kendall (Acker Step 2) or Santer et al. (2000) OLS.
    """
    if lag1_r > 0.85 and p_santer > 0.10 and p_annual > 0.10:
        return {
            "verdict": "No Statistically Significant Trend (Autocorrelation-Inflated)",
            "badge_color": "#64748b",
            "confidence": f"Rejected False Positive (r₁={lag1_r:.2f}, p_Santer={p_santer:.3f}, p_Annual={p_annual:.3f})",
        }

    p = p_hamed_rao
    if p < 0.05:
        if slope_decade > 0:
            return {
                "verdict": "Statistically Significant Increasing",
                "badge_color": "#ef4444",
                "confidence": "High Confidence (p_HR < 0.05)",
            }
        else:
            return {
                "verdict": "Statistically Significant Decreasing",
                "badge_color": "#3b82f6",
                "confidence": "High Confidence (p_HR < 0.05)",
            }
    elif p < 0.10:
        direction = "Increasing" if slope_decade > 0 else "Decreasing"
        return {
            "verdict": f"Emerging / Marginal Trend ({direction})",
            "badge_color": "#f59e0b",
            "confidence": "Marginal (0.05 ≤ p_HR < 0.10, Acker et al. Criterion)",
        }
    else:
        return {
            "verdict": "No Statistically Significant Monotonic Trend",
            "badge_color": "#64748b",
            "confidence": "Not Significant After Autocorrelation Correction (p_HR ≥ 0.10)",
        }


def benjamini_hochberg_fdr(p_values: List[float]) -> List[float]:
    """
    Benjamini-Hochberg (1995) False Discovery Rate (FDR) adjusted q-values.
    """
    m = len(p_values)
    if m == 0:
        return []
    arr = np.asarray(p_values, dtype=float)
    order = np.argsort(arr)
    ranked = arr[order]
    q_ranked = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        q_val = min(prev, ranked[i] * m / rank)
        q_ranked[i] = q_val
        prev = q_val
    q_out = np.empty(m, dtype=float)
    q_out[order] = q_ranked
    return [float(x) for x in q_out]


def analyze_single_variable(
    df: pd.DataFrame,
    variable: str,
    oni_df: Optional[pd.DataFrame] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    exclude_dates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Runs the complete Acker et al. (2009) + Hamed-Rao (1998) + Santer et al. (2000) + Pettitt (1979)
    pipeline on a single Earth system variable.
    """
    work = df.copy()
    if start_year is not None:
        work = work[work["year"] >= int(start_year)]
    if end_year is not None:
        work = work[work["year"] <= int(end_year)]
    if exclude_dates:
        excl_set = set(exclude_dates)
        work = work[~work["date"].isin(excl_set)]

    des = deseasonalize_acker(work, variable)
    ann = compute_annual_anomalies(des)

    t_m = des["decimal_year"].to_numpy()
    a_m = des["anomaly"].to_numpy()
    dates_m = des["date"].tolist()

    ols_m = ols_with_santer_correction(t_m, a_m)
    mk_m = mann_kendall_hamed_rao(t_m, a_m)
    pettitt_m = pettitt_change_point_test(dates_m, a_m)

    # Annual anomaly trend (Acker Step 2)
    t_y = ann["year"].to_numpy(dtype=float)
    a_y = ann["annual_anomaly"].to_numpy(dtype=float)
    ols_y = ols_with_santer_correction(t_y, a_y)
    mk_y = mann_kendall_hamed_rao(t_y, a_y)

    # ENSO (NOAA ONI) correlation if provided
    enso_corr = {"r_oni": 0.0, "p_oni": 1.0, "spearman_oni": 0.0, "n_overlap": 0}
    if oni_df is not None and not oni_df.empty:
        merged_oni = des.merge(oni_df[["year", "month", "oni"]], on=["year", "month"], how="inner").dropna(
            subset=["anomaly", "oni"]
        )
        if len(merged_oni) >= 12:
            r_p, p_p = stats.pearsonr(merged_oni["anomaly"], merged_oni["oni"])
            r_s, _ = stats.spearmanr(merged_oni["anomaly"], merged_oni["oni"])
            enso_corr = {
                "r_oni": float(r_p),
                "p_oni": float(p_p),
                "spearman_oni": float(r_s),
                "n_overlap": int(len(merged_oni)),
            }

    verdict = classify_trend_verdict(
        mk_m["theil_sen_decade"],
        mk_m["p_hamed_rao"],
        mk_y["p_original"],
        p_santer=ols_m["p_santer"],
        lag1_r=ols_m["lag1_r"],
    )

    clim_seasonal = compute_climatology_and_seasonal_trends(des, variable)

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
        "enso": enso_corr,
        "verdict": verdict,
        "monthly_climatology": clim_seasonal["monthly_climatology"],
        "seasonal_trends": clim_seasonal["seasonal_trends"],
        "monthly_series": des,
        "annual_series": ann,
    }


def compare_regional_dipole(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    variable: str,
    oni_df: Optional[pd.DataFrame] = None,
    start_year: int = 1981,
    end_year: int = 2025,
) -> Dict[str, Any]:
    """
    Compares two regions for opposite secular trends or opposite teleconnection polarity (ENSO dipole).
    """
    res_a = analyze_single_variable(df_a, variable, oni_df=oni_df, start_year=start_year, end_year=end_year)
    res_b = analyze_single_variable(df_b, variable, oni_df=oni_df, start_year=start_year, end_year=end_year)

    ser_a = res_a["monthly_series"][["date", "year", "month", "decimal_year", "anomaly", "z_score"]].rename(
        columns={"anomaly": "anom_a", "z_score": "z_a"}
    )
    ser_b = res_b["monthly_series"][["date", "anomaly", "z_score"]].rename(
        columns={"anomaly": "anom_b", "z_score": "z_b"}
    )
    merged = ser_a.merge(ser_b, on="date", how="inner").dropna()
    if oni_df is not None and not oni_df.empty:
        merged = merged.merge(oni_df[["year", "month", "oni"]], on=["year", "month"], how="left")

    if len(merged) >= 12:
        r_ab, p_ab = stats.pearsonr(merged["anom_a"], merged["anom_b"])
        rho_ab, p_rho_ab = stats.spearmanr(merged["anom_a"], merged["anom_b"])
    else:
        r_ab, p_ab, rho_ab, p_rho_ab = 0.0, 1.0, 0.0, 1.0

    # Filter strong ENSO events (|ONI| >= 1.0) correlation if ONI available
    r_enso_active = 0.0
    p_enso_active = 1.0
    if "oni" in merged.columns:
        strong = merged[merged["oni"].abs() >= 1.0].dropna()
        if len(strong) >= 8:
            r_enso_active, p_enso_active = stats.pearsonr(strong["z_a"], strong["z_b"])

    slope_a = res_a["monthly_mk"]["theil_sen_decade"]
    slope_b = res_b["monthly_mk"]["theil_sen_decade"]
    opposite_secular = bool((slope_a * slope_b) < 0)
    opposite_oni = bool((res_a["enso"]["r_oni"] * res_b["enso"]["r_oni"]) < -0.02)

    return {
        "variable": variable,
        "region_a": res_a,
        "region_b": res_b,
        "pearson_r": float(r_ab),
        "pearson_p": float(p_ab),
        "spearman_rho": float(rho_ab),
        "spearman_p": float(p_rho_ab),
        "r_during_strong_enso": float(r_enso_active),
        "p_during_strong_enso": float(p_enso_active),
        "has_opposite_secular_slopes": opposite_secular,
        "has_opposite_enso_teleconnection": opposite_oni,
        "merged_series": merged,
    }
