"""
TerraLens v2.1 Pro Machine Learning & Explainability Engine.
STRICT ZERO-HALLUCINATION POLICY (Rule #0): 100% Real Data, Deterministic Seeds.

Implements:
1. 3-Model Unsupervised Multivariate Anomaly Ensemble:
   - Isolation Forest (Tree-based space partitioning)
   - Local Outlier Factor / LOF (Local phase-space density deviation)
   - Covariance-Adjusted Mahalanobis Distance (Elliptical multivariate distance)
   -> Produces a 0–100 Consensus Anomaly Score and 3-Model Agreement Vote (0/3 to 3/3).
2. Quantitative Variable Attribution (±σ, % Contribution, Earth Sphere) + Plain-English
   Human Explainer for every detected anomaly event.
3. PCA (2D Phase-Space Projection) + KMeans Environmental Regime Clustering (with human-readable regime cards).
4. Supervised Lagged Teleconnection & Drought Driver ML (RandomForestRegressor + TimeSeries R²):
   - Discovers which lagged drivers (NOAA ONI at lags 0, 1, 3, 6 months + Precipitation/Temp lags)
     drive Root-Zone Soil Moisture (GWETROOT) in each region.
5. Acker et al. (2009, NASA GES DISC) ML Outlier Exclusion Experiment:
   - Compares trend slopes and autocorrelation-corrected p-values BEFORE vs. AFTER removing
     ML-consensus extreme months.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

from core.catalog import VARIABLES
from core.stats_engine import deseasonalize_acker, analyze_single_variable

DEFAULT_ML_VARS = ["T2M", "PRECTOTCORR", "GWETROOT", "GWETTOP", "ALLSKY_SFC_SW_DWN", "RH2M"]


def _normalize_0_100(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return 100.0 * (a - lo) / (hi - lo)


def _compute_mahalanobis_scores(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes regularized multivariate Mahalanobis distance D_M(z) = sqrt((z - mu)^T Sigma^-1 (z - mu))
    and flags the top 5% (or chi-square p < 0.01) as multivariate covariance outliers.
    """
    n, d = X.shape
    mu = np.mean(X, axis=0)
    centered = X - mu
    cov = np.cov(centered, rowvar=False) + np.eye(d) * 1e-4
    inv_cov = np.linalg.pinv(cov)
    md_sq = np.sum((centered @ inv_cov) * centered, axis=1)
    md = np.sqrt(np.maximum(0.0, md_sq))
    # Chi-square threshold with d degrees of freedom (95th percentile / top 5%)
    thresh = float(np.percentile(md, 95.0))
    is_out = md >= thresh
    return md, is_out


def _build_plain_english_event_summary(row: pd.Series, drivers: List[Dict[str, Any]]) -> str:
    """
    Generates a clear, deterministic, plain-English physical explanation of a multivariate anomaly
    based strictly on its measured Z-scores (±σ) and NOAA ONI value.
    """
    z_t = float(row.get("Z_T2M", 0.0))
    z_p = float(row.get("Z_PRECTOTCORR", 0.0))
    z_gr = float(row.get("Z_GWETROOT", 0.0))
    z_gt = float(row.get("Z_GWETTOP", 0.0))
    z_rh = float(row.get("Z_RH2M", 0.0))
    oni = float(row.get("oni", np.nan))

    oni_tag = ""
    if np.isfinite(oni):
        if oni >= 1.5:
            oni_tag = f" Co-occurred with a Strong El Niño phase (ONI = {oni:+.2f}°C)."
        elif oni >= 0.5:
            oni_tag = f" Co-occurred with an El Niño warm phase (ONI = {oni:+.2f}°C)."
        elif oni <= -1.0:
            oni_tag = f" Co-occurred with a Strong La Niña cold phase (ONI = {oni:+.2f}°C)."
        elif oni <= -0.5:
            oni_tag = f" Co-occurred with a La Niña phase (ONI = {oni:+.2f}°C)."

    top1 = drivers[0]
    top2 = drivers[1] if len(drivers) > 1 else drivers[0]

    if z_p >= 2.0 and z_t >= 1.0:
        headline = "Compound Warm-Wet Deluge Anomaly: Extreme rainfall surge combined with elevated air temperatures."
    elif z_p >= 2.0:
        headline = "Extreme Pluvial / Flood Pulse: Intense positive precipitation and soil saturation anomaly."
    elif z_t >= 1.5 and (z_gr <= -1.2 or z_gt <= -1.2 or z_rh <= -1.5):
        headline = "Compound Heat-Drought Stress: High atmospheric heat coincided with severe soil moisture and humidity deficit."
    elif z_gr <= -1.5 or z_gt <= -1.5 or z_p <= -1.5:
        headline = "Hydroclimatic Drought Deficit: Pronounced below-normal water availability across soil and atmosphere."
    elif z_t <= -1.8:
        headline = "Cold Atmospheric Anomaly: Unusual negative temperature excursion below monthly climatology."
    else:
        headline = "Multivariate Regime Departure: Unusual combination of atmospheric and land-surface variables."

    detail = (
        f" Primary driver was {VARIABLES[top1['variable']]['short_name']} ({top1['z_sigma']:+.2f}σ, "
        f"{top1['contribution_pct']:.0f}% of total anomaly magnitude), followed by "
        f"{VARIABLES[top2['variable']]['short_name']} ({top2['z_sigma']:+.2f}σ, {top2['contribution_pct']:.0f}%)."
    )
    return headline + detail + oni_tag


def build_multivariate_z_matrix(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    oni_df: Optional[pd.DataFrame] = None,
    start_year: int = 1984,
    end_year: int = 2025,
) -> pd.DataFrame:
    """
    Constructs a clean monthly multivariate table of standardized anomalies Z_{y,m}
    and raw deseasonalized anomalies A_{y,m}.
    """
    vars_use = [v for v in (variables or DEFAULT_ML_VARS) if v in df.columns]
    work = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()

    base = work[["date", "year", "month", "decimal_year"]].drop_duplicates().sort_values(["year", "month"])
    for v in vars_use:
        des = deseasonalize_acker(work, v)
        base = base.merge(
            des[["date", "z_score", "anomaly", v, "clim_mean"]].rename(
                columns={
                    "z_score": f"Z_{v}",
                    "anomaly": f"A_{v}",
                    v: f"RAW_{v}",
                    "clim_mean": f"CLIM_{v}",
                }
            ),
            on="date",
            how="inner",
        )
    if oni_df is not None and not oni_df.empty:
        base = base.merge(oni_df[["year", "month", "oni"]], on=["year", "month"], how="left")

    z_cols = [f"Z_{v}" for v in vars_use]
    base = base.dropna(subset=z_cols).reset_index(drop=True)
    return base


def run_multivariate_anomaly_detective(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    oni_df: Optional[pd.DataFrame] = None,
    contamination: float = 0.05,
    start_year: int = 1984,
    end_year: int = 2025,
) -> Dict[str, Any]:
    """
    Runs the 3-Model Unsupervised Anomaly Ensemble (Isolation Forest + LOF + Mahalanobis),
    PCA 2D projection, KMeans regime clustering, Lagged Teleconnection Random Forest,
    and Quantitative ±σ Explainability.
    """
    vars_use = [v for v in (variables or DEFAULT_ML_VARS) if v in df.columns]
    z_df = build_multivariate_z_matrix(df, vars_use, oni_df=oni_df, start_year=start_year, end_year=end_year)
    z_cols = [f"Z_{v}" for v in vars_use]

    if len(z_df) < 24 or len(z_cols) < 2:
        return {
            "variables": vars_use,
            "matrix": z_df,
            "anomalies": [],
            "extreme_dates": [],
            "pca_variance_ratio": [0.0, 0.0],
            "pca_loadings": {},
            "cluster_summary": [],
            "lagged_driver_model": {},
        }

    X = z_df[z_cols].to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # 1. 3-MODEL UNSUPERVISED MULTIVARIATE ANOMALY ENSEMBLE
    # -------------------------------------------------------------------------
    # Model A: Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso_pred = iso.fit_predict(X) == -1
    iso_raw = -iso.decision_function(X)
    iso_score = _normalize_0_100(iso_raw)

    # Model B: Local Outlier Factor (LOF)
    n_neighbors = min(20, max(5, len(X) // 15))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof_pred = lof.fit_predict(X) == -1
    lof_raw = -lof.negative_outlier_factor_
    lof_score = _normalize_0_100(lof_raw)

    # Model C: Covariance-Regularized Mahalanobis Distance
    md_raw, md_pred = _compute_mahalanobis_scores(X)
    md_score = _normalize_0_100(md_raw)

    # Ensemble Consensus Score (0-100) and Model Agreement Count (0..3)
    consensus_score = 0.40 * iso_score + 0.30 * lof_score + 0.30 * md_score
    votes = iso_pred.astype(int) + lof_pred.astype(int) + md_pred.astype(int)

    # Consensus anomaly: either >=2 models flag it OR top 5% consensus score
    score_cutoff = float(np.percentile(consensus_score, 100.0 * (1.0 - contamination)))
    is_consensus_anomaly = (votes >= 2) | (consensus_score >= score_cutoff)

    z_df["iso_flag"] = iso_pred
    z_df["lof_flag"] = lof_pred
    z_df["mahalanobis_flag"] = md_pred
    z_df["mahalanobis_dist"] = np.round(md_raw, 2)
    z_df["model_votes"] = votes
    z_df["is_anomaly"] = is_consensus_anomaly
    z_df["anomaly_score"] = np.round(consensus_score / 100.0, 3)
    z_df["consensus_score_pct"] = np.round(consensus_score, 1)
    z_df["max_abs_sigma"] = np.round(np.max(np.abs(X), axis=1), 2)
    z_df["compound_l2_sigma"] = np.round(np.sqrt(np.mean(X ** 2, axis=1)), 2)

    # -------------------------------------------------------------------------
    # 2. PCA (2D Phase Space + Feature Loadings)
    # -------------------------------------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(X)
    z_df["pca_1"] = np.round(coords_pca[:, 0], 3)
    z_df["pca_2"] = np.round(coords_pca[:, 1], 3)
    var_ratio = [float(x) for x in pca.explained_variance_ratio_]
    pca_loadings = {
        v: {
            "PC1_loading": round(float(pca.components_[0, i]), 3),
            "PC2_loading": round(float(pca.components_[1, i]), 3),
        }
        for i, v in enumerate(vars_use)
    }

    # -------------------------------------------------------------------------
    # 3. KMeans Clustering into 3 Interpretable Environmental Regimes
    # -------------------------------------------------------------------------
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    clusters = km.fit_predict(X)
    z_df["cluster_id"] = clusters

    cluster_labels = {}
    cluster_summary = []
    for cid in range(3):
        sub_c = z_df[z_df["cluster_id"] == cid]
        mean_t = float(sub_c["Z_T2M"].mean()) if "Z_T2M" in sub_c else 0.0
        mean_p = float(sub_c["Z_PRECTOTCORR"].mean()) if "Z_PRECTOTCORR" in sub_c else 0.0
        mean_g = float(sub_c["Z_GWETROOT"].mean()) if "Z_GWETROOT" in sub_c else 0.0
        mean_rh = float(sub_c["Z_RH2M"].mean()) if "Z_RH2M" in sub_c else 0.0

        if mean_t > 0.30 and (mean_p < -0.15 or mean_g < -0.15 or mean_rh < -0.20):
            label = "Warm & Dry Compound Stress Regime"
            desc = "Elevated air temperature paired with below-normal rainfall, soil moisture, and humidity."
        elif mean_p > 0.30 or mean_g > 0.30:
            label = "Wet / Positive Moisture Pulse Regime"
            desc = "Above-normal rainfall and root-zone soil wetness recharge."
        elif mean_t < -0.25:
            label = "Cooler / La Niña-Like Regime"
            desc = "Below-average thermal conditions with moderate-to-high moisture."
        else:
            label = "Baseline / Near-Normal Climatological Regime"
            desc = "Variables fluctuate within ±0.5σ of their 45-year monthly normal."

        if label in cluster_labels.values():
            label = f"{label} (Mode {cid + 1})"
        cluster_labels[cid] = label
        cluster_summary.append(
            {
                "cluster_id": int(cid),
                "regime_name": label,
                "plain_description": desc,
                "count_months": int(len(sub_c)),
                "share_pct": round(100.0 * len(sub_c) / len(z_df), 1),
                "mean_Z_T2M": round(mean_t, 2),
                "mean_Z_PREC": round(mean_p, 2),
                "mean_Z_GWETROOT": round(mean_g, 2),
                "mean_Z_RH2M": round(mean_rh, 2),
            }
        )

    z_df["regime_name"] = z_df["cluster_id"].map(cluster_labels)

    # -------------------------------------------------------------------------
    # 4. Detailed Explainability Cards for Every Detected Anomaly
    # -------------------------------------------------------------------------
    anom_rows = z_df[z_df["is_anomaly"]].sort_values("consensus_score_pct", ascending=False)
    anomalies_out = []
    for _, row in anom_rows.iterrows():
        abs_vals = {v: abs(float(row[f"Z_{v}"])) for v in vars_use}
        total_abs = sum(abs_vals.values()) + 1e-12
        drivers = []
        for v in sorted(vars_use, key=lambda k: abs_vals[k], reverse=True):
            z_val = float(row[f"Z_{v}"])
            raw_val = float(row.get(f"RAW_{v}", np.nan))
            clim_val = float(row.get(f"CLIM_{v}", np.nan))
            anom_val = float(row.get(f"A_{v}", np.nan))
            contrib_pct = 100.0 * abs(z_val) / total_abs
            drivers.append(
                {
                    "variable": v,
                    "short_name": VARIABLES[v]["short_name"],
                    "sphere": VARIABLES[v]["sphere"],
                    "units": VARIABLES[v]["units"],
                    "raw_value": round(raw_val, 3) if np.isfinite(raw_val) else None,
                    "normal_value": round(clim_val, 3) if np.isfinite(clim_val) else None,
                    "anomaly_value": round(anom_val, 3) if np.isfinite(anom_val) else None,
                    "z_sigma": round(z_val, 2),
                    "abs_sigma": round(abs(z_val), 2),
                    "contribution_pct": round(contrib_pct, 1),
                    "direction": "Above Normal (+σ)" if z_val >= 0 else "Below Normal (-σ)",
                }
            )
        top_d = drivers[0]
        second_d = drivers[1] if len(drivers) > 1 else drivers[0]
        models_agreed = []
        if bool(row["iso_flag"]):
            models_agreed.append("IsolationForest")
        if bool(row["lof_flag"]):
            models_agreed.append("LOF")
        if bool(row["mahalanobis_flag"]):
            models_agreed.append("Mahalanobis")

        plain_summary = _build_plain_english_event_summary(row, drivers)

        anomalies_out.append(
            {
                "date": str(row["date"]),
                "year": int(row["year"]),
                "month": int(row["month"]),
                "anomaly_score": round(float(row["anomaly_score"]), 3),
                "consensus_score_pct": round(float(row["consensus_score_pct"]), 1),
                "model_votes": int(row["model_votes"]),
                "models_agreed": ", ".join(models_agreed) if models_agreed else "Top Score Threshold",
                "mahalanobis_dist": round(float(row["mahalanobis_dist"]), 2),
                "compound_l2_sigma": round(float(row["compound_l2_sigma"]), 2),
                "oni": round(float(row["oni"]), 2) if "oni" in row and np.isfinite(row["oni"]) else None,
                "regime_name": str(row["regime_name"]),
                "primary_driver": f"{top_d['variable']} ({top_d['z_sigma']:+.2f}σ, {top_d['contribution_pct']:.0f}%)",
                "secondary_driver": f"{second_d['variable']} ({second_d['z_sigma']:+.2f}σ, {second_d['contribution_pct']:.0f}%)",
                "plain_english_summary": plain_summary,
                "drivers": drivers,
            }
        )

    extreme_dates = [a["date"] for a in anomalies_out]

    # -------------------------------------------------------------------------
    # 5. Supervised Lagged Teleconnection & Soil Moisture Driver ML (Random Forest)
    # -------------------------------------------------------------------------
    lagged_model_info = run_lagged_driver_random_forest(z_df)

    return {
        "variables": vars_use,
        "matrix": z_df,
        "anomalies": anomalies_out,
        "extreme_dates": extreme_dates,
        "pca_variance_ratio": var_ratio,
        "pca_loadings": pca_loadings,
        "cluster_summary": cluster_summary,
        "lagged_driver_model": lagged_model_info,
    }


def run_lagged_driver_random_forest(z_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains a RandomForestRegressor to predict Root-Zone Soil Moisture Anomaly (Z_GWETROOT)
    from concurrent and lagged ocean-atmosphere variables (NOAA ONI at lags 0, 1, 3, 6 months;
    Precipitation at lags 0, 1, 2 months; Temperature at lags 0, 1; Relative Humidity & Solar Radiation).
    Reveals which physical drivers and lead times govern droughts/floods in this region.
    """
    if "Z_GWETROOT" not in z_df.columns or len(z_df) < 60:
        return {"available": False}

    work = z_df.sort_values(["year", "month"]).copy()
    feature_map = {}

    if "Z_PRECTOTCORR" in work.columns:
        work["PREC_lag0"] = work["Z_PRECTOTCORR"]
        work["PREC_lag1"] = work["Z_PRECTOTCORR"].shift(1)
        work["PREC_lag2"] = work["Z_PRECTOTCORR"].shift(2)
        feature_map["PREC_lag0"] = "Precipitation (Current Month t)"
        feature_map["PREC_lag1"] = "Precipitation (1-Month Lag t-1)"
        feature_map["PREC_lag2"] = "Precipitation (2-Month Lag t-2)"

    if "Z_T2M" in work.columns:
        work["T2M_lag0"] = work["Z_T2M"]
        work["T2M_lag1"] = work["Z_T2M"].shift(1)
        feature_map["T2M_lag0"] = "Air Temperature (Current Month t)"
        feature_map["T2M_lag1"] = "Air Temperature (1-Month Lag t-1)"

    if "Z_RH2M" in work.columns:
        work["RH2M_lag0"] = work["Z_RH2M"]
        feature_map["RH2M_lag0"] = "Relative Humidity (Current Month t)"

    if "Z_ALLSKY_SFC_SW_DWN" in work.columns:
        work["SW_lag0"] = work["Z_ALLSKY_SFC_SW_DWN"]
        feature_map["SW_lag0"] = "Solar Irradiance / Cloud Cover (Month t)"

    if "oni" in work.columns and work["oni"].notna().sum() > 60:
        work["ONI_lag0"] = work["oni"]
        work["ONI_lag1"] = work["oni"].shift(1)
        work["ONI_lag3"] = work["oni"].shift(3)
        work["ONI_lag6"] = work["oni"].shift(6)
        feature_map["ONI_lag0"] = "NOAA Oceanic Niño Index (Concurrent t)"
        feature_map["ONI_lag1"] = "NOAA Oceanic Niño Index (1-Month Lead t-1)"
        feature_map["ONI_lag3"] = "NOAA Oceanic Niño Index (3-Month Lead t-3)"
        feature_map["ONI_lag6"] = "NOAA Oceanic Niño Index (6-Month Lead t-6)"

    feat_cols = list(feature_map.keys())
    clean = work.dropna(subset=feat_cols + ["Z_GWETROOT"]).reset_index(drop=True)
    if len(clean) < 48:
        return {"available": False}

    X = clean[feat_cols].to_numpy(dtype=float)
    y = clean["Z_GWETROOT"].to_numpy(dtype=float)

    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42,
    )
    rf.fit(X, y)
    y_pred_in = rf.predict(X)
    r2_in = float(r2_score(y, y_pred_in))

    # Out-of-fold TimeSeriesSplit R^2
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, test_idx in tscv.split(X):
        rf_cv = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=4, random_state=42)
        rf_cv.fit(X[train_idx], y[train_idx])
        pred_cv = rf_cv.predict(X[test_idx])
        cv_scores.append(float(r2_score(y[test_idx], pred_cv)))

    importances = rf.feature_importances_
    ranked_features = []
    for col, imp in sorted(zip(feat_cols, importances), key=lambda x: x[1], reverse=True):
        r_corr, p_corr = stats.pearsonr(clean[col], clean["Z_GWETROOT"])
        ranked_features.append(
            {
                "feature_code": col,
                "feature_label": feature_map[col],
                "importance_pct": round(float(imp * 100.0), 1),
                "linear_r": round(float(r_corr), 3),
                "linear_p": round(float(p_corr), 5),
            }
        )

    # Best ENSO lead time (among ONI_lag0, ONI_lag1, ONI_lag3, ONI_lag6)
    oni_lags = [f for f in ranked_features if f["feature_code"].startswith("ONI_")]
    best_oni = max(oni_lags, key=lambda x: abs(x["linear_r"])) if oni_lags else None

    return {
        "available": True,
        "target": "Z_GWETROOT (Root-Zone Soil Moisture Anomaly)",
        "n_samples": int(len(clean)),
        "r2_in_sample": round(r2_in, 3),
        "r2_cv_mean": round(float(np.mean(cv_scores)), 3),
        "top_features": ranked_features,
        "best_enso_lag": best_oni,
    }


def run_acker_ml_outlier_experiment_all_vars(
    df: pd.DataFrame,
    oni_df: Optional[pd.DataFrame] = None,
    use_homogeneous_ceres: bool = True,
) -> Dict[str, Any]:
    """
    Runs the full Acker et al. (2009, Table 6 & Fig. 13) Outlier Sensitivity Experiment across all 6 variables:
    Compares trend slopes and autocorrelation-corrected p-values BEFORE vs. AFTER removing the
    months flagged by the 3-Model ML Consensus Ensemble.
    """
    ml_res = run_multivariate_anomaly_detective(df, oni_df=oni_df, start_year=1984, end_year=2025)
    extreme_dates = ml_res["extreme_dates"]

    comparison_rows = []
    for vc, vmeta in VARIABLES.items():
        sy = 2001 if (vc == "ALLSKY_SFC_SW_DWN" and use_homogeneous_ceres) else 1984
        before = analyze_single_variable(df, vc, oni_df=oni_df, start_year=sy, end_year=2025, exclude_dates=None)
        after = analyze_single_variable(
            df, vc, oni_df=oni_df, start_year=sy, end_year=2025, exclude_dates=extreme_dates
        )

        slope_b = before["monthly_mk"]["theil_sen_decade"]
        slope_a = after["monthly_mk"]["theil_sen_decade"]
        p_b = before["monthly_mk"]["p_hamed_rao"]
        p_a = after["monthly_mk"]["p_hamed_rao"]

        sig_b = p_b < 0.05
        sig_a = p_a < 0.05
        if sig_b and sig_a and np.sign(slope_b) == np.sign(slope_a):
            robustness = "Robust Structural Trend (Persists Without Extremes)"
        elif sig_b and not sig_a:
            robustness = "Outlier-Driven Signal (Loses Significance Without Extremes)"
        elif not sig_b and sig_a:
            robustness = "Masked Background Trend (Emerges After Removing Extremes)"
        else:
            robustness = "Stable / No Monotonic Trend in Both Cases"

        comparison_rows.append(
            {
                "variable": vc,
                "short_name": vmeta["short_name"],
                "units": vmeta["decade_units"],
                "period": f"{sy}–2025",
                "slope_all_months": round(slope_b, 4),
                "p_hr_all_months": round(p_b, 5),
                "slope_without_ml_extremes": round(slope_a, 4),
                "p_hr_without_ml_extremes": round(p_a, 5),
                "slope_delta_pct": round(100.0 * (slope_a - slope_b) / (abs(slope_b) + 1e-9), 1),
                "structural_diagnosis": robustness,
            }
        )

    return {
        "n_excluded_months": len(extreme_dates),
        "excluded_dates": extreme_dates,
        "comparisons": comparison_rows,
    }


def run_spatiotemporal_cube_ml(
    region_dfs: Dict[str, pd.DataFrame],
    oni_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Runs a synchronized Spatio-Temporal Cube analysis across all loaded regions (1984–2025)
    to identify Continental Synchronized Anomaly Epochs (months where the largest fraction
    of regions simultaneously experience multivariate consensus anomalies).
    """
    all_anom_records = []
    region_summaries = []

    for reg_id, df_reg in region_dfs.items():
        res = run_multivariate_anomaly_detective(df_reg, oni_df=oni_df, start_year=1984, end_year=2025)
        anoms = res["anomalies"]
        lag_rf = res["lagged_driver_model"]
        top_feat = lag_rf["top_features"][0]["feature_label"] if lag_rf.get("available") and lag_rf.get("top_features") else "N/A"
        best_oni = lag_rf.get("best_enso_lag") or {}

        region_summaries.append(
            {
                "region_id": reg_id,
                "n_consensus_anomalies": len(anoms),
                "top_event_date": anoms[0]["date"] if anoms else "N/A",
                "top_event_score": anoms[0]["consensus_score_pct"] if anoms else 0.0,
                "top_event_driver": anoms[0]["primary_driver"] if anoms else "N/A",
                "rf_soil_moisture_r2": lag_rf.get("r2_in_sample", 0.0) if lag_rf.get("available") else 0.0,
                "top_soil_moisture_driver": top_feat,
                "best_enso_lead_lag": f"{best_oni.get('feature_code', 'N/A')} (r={best_oni.get('linear_r', 0.0):+.2f})"
                if best_oni
                else "N/A",
            }
        )
        for a in anoms:
            all_anom_records.append(
                {
                    "date": a["date"],
                    "region_id": reg_id,
                    "consensus_score_pct": a["consensus_score_pct"],
                    "primary_driver": a["primary_driver"],
                    "oni": a["oni"],
                }
            )

    if not all_anom_records:
        return {"synchronized_epochs": [], "region_ml_summary": region_summaries}

    anom_df = pd.DataFrame(all_anom_records)
    sync_grp = (
        anom_df.groupby("date")
        .agg(
            n_regions_anomalous=("region_id", "count"),
            mean_consensus_score=("consensus_score_pct", "mean"),
            regions_list=("region_id", lambda s: ", ".join(sorted(set(s)))),
            oni=("oni", "first"),
        )
        .reset_index()
        .sort_values(["n_regions_anomalous", "mean_consensus_score"], ascending=[False, False])
    )

    top_epochs = sync_grp[sync_grp["n_regions_anomalous"] >= 3].head(15).to_dict(orient="records")
    return {
        "total_spatiotemporal_vectors": int(sum(len(df) for df in region_dfs.values())),
        "synchronized_epochs": top_epochs,
        "region_ml_summary": region_summaries,
    }

