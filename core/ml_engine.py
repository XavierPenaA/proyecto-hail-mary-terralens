"""
TerraLens machine learning engine (runs at request time on one region's monthly anomalies).

1. Unsupervised anomaly consensus: Isolation Forest, Local Outlier Factor and Mahalanobis
   distance, each with a data-driven threshold. A month is anomalous when at least two
   detectors agree, so a stable climate can legitimately have very few anomalies.
2. Per-anomaly attribution in standardized units (sigma) and a neutral event code.
3. PCA projection and K-Means regimes, with k chosen by silhouette score.
4. Random Forest attribution of root-zone soil moisture anomalies to concurrent and lagged
   drivers, reported with out-of-sample R2 next to a persistence baseline.
5. Outlier sensitivity experiment (Acker et al., 2009): trends with and without the anomalous months.

All outputs are numbers or stable codes; labels live in the frontend locales.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import LocalOutlierFactor

from core.stats_engine import deseasonalize_acker

DEFAULT_ML_VARS = ["T2M", "PRECTOTCORR", "GWETROOT", "GWETTOP", "ALLSKY_SFC_SW_DWN", "RH2M"]
MOISTURE_VARS = ["PRECTOTCORR", "GWETROOT", "RH2M"]
VOTES_REQUIRED = 2
MAHALANOBIS_ALPHA = 0.01
CLUSTER_K_CANDIDATES = (2, 3, 4)
RANDOM_STATE = 42
CV_SPLITS = 5


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------
def build_multivariate_z_matrix(
    df: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    oni_df: Optional[pd.DataFrame] = None,
    start_year: int = 1984,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Monthly table of standardized (Z_*) and absolute (A_*) anomalies for every variable."""
    vars_use = [v for v in (variables or DEFAULT_ML_VARS) if v in df.columns]
    work = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    base = work[["date", "year", "month", "decimal_year"]].drop_duplicates().sort_values(["year", "month"])
    for v in vars_use:
        des = deseasonalize_acker(work, v)
        base = base.merge(
            des[["date", "z_score", "anomaly", v, "clim_mean"]].rename(
                columns={"z_score": f"Z_{v}", "anomaly": f"A_{v}", v: f"RAW_{v}", "clim_mean": f"CLIM_{v}"}
            ),
            on="date",
            how="inner",
        )
    if oni_df is not None and not oni_df.empty:
        base = base.merge(oni_df[["year", "month", "oni"]], on=["year", "month"], how="left")
    return base.dropna(subset=[f"Z_{v}" for v in vars_use]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """0-100 rank of each value within its own array (higher = more anomalous)."""
    return 100.0 * (stats.rankdata(values) - 1) / max(1, len(values) - 1)


def mahalanobis_outliers(X: np.ndarray, alpha: float = MAHALANOBIS_ALPHA) -> Dict[str, np.ndarray]:
    """Squared Mahalanobis distance against the chi-square(d) quantile 1 - alpha."""
    d = X.shape[1]
    centered = X - X.mean(axis=0)
    inv_cov = np.linalg.pinv(np.cov(centered, rowvar=False) + np.eye(d) * 1e-6)
    md_sq = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
    threshold = float(stats.chi2.ppf(1.0 - alpha, df=d))
    return {"distance": np.sqrt(np.maximum(md_sq, 0.0)), "flags": md_sq > threshold}


def run_anomaly_detectors(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Runs the three detectors with their own data-driven thresholds."""
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=RANDOM_STATE)
    iso_flags = iso.fit_predict(X) == -1
    iso_raw = -iso.score_samples(X)

    lof = LocalOutlierFactor(n_neighbors=min(24, max(5, len(X) // 15)), contamination="auto")
    lof_flags = lof.fit_predict(X) == -1
    lof_raw = -lof.negative_outlier_factor_

    md = mahalanobis_outliers(X)
    votes = iso_flags.astype(int) + lof_flags.astype(int) + md["flags"].astype(int)
    consensus_score = (_percentile_rank(iso_raw) + _percentile_rank(lof_raw) + _percentile_rank(md["distance"])) / 3.0
    return {
        "iso_flags": iso_flags,
        "lof_flags": lof_flags,
        "mahalanobis_flags": md["flags"],
        "mahalanobis_distance": md["distance"],
        "votes": votes,
        "consensus_score": consensus_score,
    }


# ---------------------------------------------------------------------------
# Neutral codes
# ---------------------------------------------------------------------------
def classify_event(z: Dict[str, float]) -> str:
    """Event code from standardized anomalies (no physical cause is implied)."""
    z_t, z_p = z.get("T2M", 0.0), z.get("PRECTOTCORR", 0.0)
    z_gr, z_gt, z_rh = z.get("GWETROOT", 0.0), z.get("GWETTOP", 0.0), z.get("RH2M", 0.0)
    if z_p >= 2.0 and z_t >= 1.0:
        return "warm_wet"
    if z_p >= 2.0:
        return "wet_extreme"
    if z_t >= 1.5 and (z_gr <= -1.2 or z_gt <= -1.2 or z_rh <= -1.5):
        return "hot_dry"
    if z_gr <= -1.5 or z_gt <= -1.5 or z_p <= -1.5:
        return "dry_extreme"
    if z_t <= -1.8:
        return "cold_extreme"
    return "multivariate"


def oni_phase(oni: Optional[float]) -> Optional[str]:
    """NOAA ENSO phase thresholds (strong events at |ONI| >= 1.5)."""
    if oni is None or not np.isfinite(oni):
        return None
    if oni >= 1.5:
        return "strong_el_nino"
    if oni >= 0.5:
        return "el_nino"
    if oni <= -1.5:
        return "strong_la_nina"
    if oni <= -0.5:
        return "la_nina"
    return "neutral"


def regime_code(mean_t: float, mean_moisture: float) -> str:
    """Regime code from a cluster's mean temperature and moisture anomalies (sigma)."""
    if mean_t > 0.3 and mean_moisture < -0.15:
        return "warm_dry"
    if mean_moisture > 0.3:
        return "wet"
    if mean_moisture < -0.3:
        return "dry"
    if mean_t > 0.3:
        return "warm"
    if mean_t < -0.3:
        return "cool"
    return "near_normal"


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------
def select_regimes(X: np.ndarray, z_df: pd.DataFrame, vars_use: List[str]) -> Dict[str, Any]:
    """K-Means with k chosen by silhouette score among CLUSTER_K_CANDIDATES."""
    silhouettes, labels_by_k = {}, {}
    for k in CLUSTER_K_CANDIDATES:
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
        silhouettes[k] = float(silhouette_score(X, labels))
        labels_by_k[k] = labels
    best_k = max(silhouettes, key=silhouettes.get)
    labels = labels_by_k[best_k]

    moisture_cols = [f"Z_{v}" for v in MOISTURE_VARS if v in vars_use]
    clusters = []
    for cid in range(best_k):
        members = z_df[labels == cid]
        mean_t = float(members["Z_T2M"].mean()) if "Z_T2M" in members else 0.0
        mean_moisture = float(members[moisture_cols].mean().mean()) if moisture_cols else 0.0
        clusters.append(
            {
                "cluster_id": cid,
                "regime_code": regime_code(mean_t, mean_moisture),
                "count_months": int(len(members)),
                "share_pct": round(100.0 * len(members) / len(z_df), 1),
                "mean_z": {v: round(float(members[f"Z_{v}"].mean()), 2) for v in vars_use},
            }
        )
    return {
        "k": int(best_k),
        "silhouette_by_k": {str(k): round(s, 4) for k, s in silhouettes.items()},
        "clusters": clusters,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _attribution(row: pd.Series, vars_use: List[str]) -> List[Dict[str, Any]]:
    abs_z = {v: abs(float(row[f"Z_{v}"])) for v in vars_use}
    total = sum(abs_z.values()) + 1e-12

    def _rounded(value: Any, digits: int) -> Optional[float]:
        value = float(value)
        return round(value, digits) if np.isfinite(value) else None

    return [
        {
            "variable": v,
            "raw_value": _rounded(row.get(f"RAW_{v}", np.nan), 3),
            "normal_value": _rounded(row.get(f"CLIM_{v}", np.nan), 3),
            "anomaly_value": _rounded(row.get(f"A_{v}", np.nan), 3),
            "z_sigma": round(float(row[f"Z_{v}"]), 2),
            "contribution_pct": round(100.0 * abs_z[v] / total, 1),
        }
        for v in sorted(vars_use, key=lambda name: abs_z[name], reverse=True)
    ]


def run_multivariate_anomaly_detective(
    df: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    oni_df: Optional[pd.DataFrame] = None,
    start_year: int = 1984,
    end_year: int = 2025,
) -> Dict[str, Any]:
    """Anomaly consensus, attribution, PCA, regimes and the soil-moisture driver model."""
    vars_use = [v for v in (variables or DEFAULT_ML_VARS) if v in df.columns]
    z_df = build_multivariate_z_matrix(df, vars_use, oni_df=oni_df, start_year=start_year, end_year=end_year)
    z_cols = [f"Z_{v}" for v in vars_use]
    if len(z_df) < 60 or len(z_cols) < 2:
        return {
            "variables": vars_use, "matrix": z_df, "n_months": int(len(z_df)), "anomalies": [],
            "extreme_dates": [], "pca": {"variance_ratio": [0.0, 0.0], "loadings": {}},
            "regimes": {"k": 0, "silhouette_by_k": {}, "clusters": []}, "lagged_driver_model": {"available": False},
        }

    X = z_df[z_cols].to_numpy(dtype=float)
    det = run_anomaly_detectors(X)
    z_df = z_df.assign(
        iso_flag=det["iso_flags"],
        lof_flag=det["lof_flags"],
        mahalanobis_flag=det["mahalanobis_flags"],
        mahalanobis_dist=np.round(det["mahalanobis_distance"], 2),
        model_votes=det["votes"],
        consensus_score_pct=np.round(det["consensus_score"], 1),
        is_anomaly=det["votes"] >= VOTES_REQUIRED,
    )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    regimes = select_regimes(X, z_df, vars_use)
    z_df = z_df.assign(pca_1=np.round(coords[:, 0], 3), pca_2=np.round(coords[:, 1], 3), cluster_id=regimes.pop("labels"))

    anomalies = []
    for _, row in z_df[z_df["is_anomaly"]].sort_values("consensus_score_pct", ascending=False).iterrows():
        oni_value = float(row["oni"]) if "oni" in row and pd.notna(row["oni"]) else None
        anomalies.append(
            {
                "date": str(row["date"]),
                "consensus_score_pct": float(row["consensus_score_pct"]),
                "model_votes": int(row["model_votes"]),
                "detectors": {
                    "isolation_forest": bool(row["iso_flag"]),
                    "lof": bool(row["lof_flag"]),
                    "mahalanobis": bool(row["mahalanobis_flag"]),
                },
                "mahalanobis_dist": float(row["mahalanobis_dist"]),
                "oni": round(oni_value, 2) if oni_value is not None else None,
                "oni_phase": oni_phase(oni_value),
                "event_code": classify_event({v: float(row[f"Z_{v}"]) for v in vars_use}),
                "cluster_id": int(row["cluster_id"]),
                "drivers": _attribution(row, vars_use),
            }
        )

    return {
        "variables": vars_use,
        "matrix": z_df,
        "n_months": int(len(z_df)),
        "anomalies": anomalies,
        "extreme_dates": [a["date"] for a in anomalies],
        "pca": {
            "variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
            "loadings": {
                v: {"pc1": round(float(pca.components_[0, i]), 3), "pc2": round(float(pca.components_[1, i]), 3)}
                for i, v in enumerate(vars_use)
            },
        },
        "regimes": regimes,
        "lagged_driver_model": run_lagged_driver_random_forest(z_df),
    }


# ---------------------------------------------------------------------------
# Supervised attribution with an honest baseline
# ---------------------------------------------------------------------------
def _driver_features(work: pd.DataFrame) -> Dict[str, pd.Series]:
    """Concurrent and lagged predictors; codes follow the pattern <SOURCE>_lag<k>."""
    features: Dict[str, pd.Series] = {}
    lag_plan = {"Z_PRECTOTCORR": ("PREC", (0, 1, 2)), "Z_T2M": ("T2M", (0, 1)), "Z_RH2M": ("RH2M", (0,)),
                "Z_ALLSKY_SFC_SW_DWN": ("SW", (0,))}
    for column, (code, lags) in lag_plan.items():
        if column in work.columns:
            for lag in lags:
                features[f"{code}_lag{lag}"] = work[column].shift(lag)
    if "oni" in work.columns and work["oni"].notna().sum() > 60:
        for lag in (0, 1, 3, 6):
            features[f"ONI_lag{lag}"] = work["oni"].shift(lag)
    return features


def _persistence_r2(train_y: np.ndarray, train_prev: np.ndarray, test_y: np.ndarray, test_prev: np.ndarray) -> float:
    """R2 of y_t = phi * y_(t-1), with phi fitted on the training fold."""
    phi = float(np.dot(train_y, train_prev) / max(1e-12, np.dot(train_prev, train_prev)))
    return float(r2_score(test_y, phi * test_prev))


def run_lagged_driver_random_forest(z_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Random Forest that explains root-zone soil moisture anomalies (Z_GWETROOT) from concurrent
    and lagged drivers. The headline metric is the out-of-sample TimeSeriesSplit R2, shown next to
    a persistence baseline that only uses last month's soil moisture.
    """
    if "Z_GWETROOT" not in z_df.columns or len(z_df) < 60:
        return {"available": False}

    work = z_df.sort_values(["year", "month"]).reset_index(drop=True)
    features = _driver_features(work)
    table = pd.DataFrame(features).assign(target=work["Z_GWETROOT"], target_prev=work["Z_GWETROOT"].shift(1)).dropna()
    if len(table) < 48:
        return {"available": False}

    feat_cols = list(features.keys())
    X = table[feat_cols].to_numpy(dtype=float)
    y = table["target"].to_numpy(dtype=float)
    prev = table["target_prev"].to_numpy(dtype=float)

    def _forest(n_estimators: int) -> RandomForestRegressor:
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=6, min_samples_leaf=4, random_state=RANDOM_STATE)

    cv_model, cv_persistence = [], []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=CV_SPLITS).split(X):
        model = _forest(100).fit(X[train_idx], y[train_idx])
        cv_model.append(float(r2_score(y[test_idx], model.predict(X[test_idx]))))
        cv_persistence.append(_persistence_r2(y[train_idx], prev[train_idx], y[test_idx], prev[test_idx]))

    rf = _forest(150).fit(X, y)
    ranked = []
    for code, importance in sorted(zip(feat_cols, rf.feature_importances_), key=lambda item: item[1], reverse=True):
        r_lin = float(np.corrcoef(table[code], y)[0, 1]) if np.std(table[code]) > 1e-12 else 0.0
        ranked.append({"feature_code": code, "importance_pct": round(float(importance) * 100.0, 1), "linear_r": round(r_lin, 3)})

    oni_features = [f for f in ranked if f["feature_code"].startswith("ONI_")]
    return {
        "available": True,
        "target": "GWETROOT",
        "n_samples": int(len(table)),
        "r2_cv_mean": round(float(np.mean(cv_model)), 3),
        "r2_cv_folds": [round(v, 3) for v in cv_model],
        "r2_cv_persistence": round(float(np.mean(cv_persistence)), 3),
        "r2_in_sample": round(float(r2_score(y, rf.predict(X))), 3),
        "top_features": ranked,
        "strongest_enso_lag": max(oni_features, key=lambda f: abs(f["linear_r"])) if oni_features else None,
    }


# ---------------------------------------------------------------------------
# Outlier sensitivity (Acker et al., 2009)
# ---------------------------------------------------------------------------
def outlier_sensitivity_code(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    """robust, outlier_driven, masked or no_trend, from the two verdicts."""
    sig_b, sig_a = before["is_significant"], after["is_significant"]
    if sig_b and sig_a:
        return "robust" if before["direction"] == after["direction"] else "outlier_driven"
    if sig_b:
        return "outlier_driven"
    if sig_a:
        return "masked"
    return "no_trend"


def outlier_sensitivity_row(variable: str, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """
    One row of the outlier exclusion test of Acker et al. (2009): the same variable's trend with all
    months (before) and without the anomalous months (after).
    """
    slope_b = before["monthly_mk"]["theil_sen_decade"]
    slope_a = after["monthly_mk"]["theil_sen_decade"]
    return {
        "variable": variable,
        "period": [before["year_start"], before["year_end"]],
        "slope_all_months": round(slope_b, 4),
        "p_hr_all_months": round(before["monthly_mk"]["p_hamed_rao"], 5),
        "slope_without_extremes": round(slope_a, 4),
        "p_hr_without_extremes": round(after["monthly_mk"]["p_hamed_rao"], 5),
        "slope_delta_pct": round(100.0 * (slope_a - slope_b) / (abs(slope_b) + 1e-9), 1),
        "code": outlier_sensitivity_code(before["verdict"], after["verdict"]),
    }
