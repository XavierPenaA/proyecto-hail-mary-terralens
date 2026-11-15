"""
TerraLens offline machine-learning benchmark.

Evaluates how well root-zone soil moisture anomalies (GWETROOT, standardized per calendar month)
can be estimated across the NASA POWER grid over Ecuador and the 11 benchmark regions, always
next to a persistence baseline (next month = phi x this month). Writes the results to
models/training_manifest.json, which the observatory's Model Performance view displays.

Benchmarks
- Nowcast: uses same-month precipitation and temperature plus lagged values. It measures how
  much the current weather explains, not how well the future can be predicted.
- Forecast (t+1, t+3): uses only information available h months before the target month.
- Regional drivers: per-region Random Forest on concurrent and lagged drivers (no soil-moisture
  lags), the model whose importances the observatory shows.

Splits: cross-validation uses whole months and fits the monthly climatologies on each
training fold. The final fit uses 1981–2018 and the held-out test uses 2019 onward.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from core.catalog import DATA_END_YEAR, ML_START_YEAR, REGIONS
from core.connectors import CACHE_DIR, load_meteomatics_1km_cache, load_noaa_oni, load_region_nasa_power, parse_nasa_power_json

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
TRAIN_END_YEAR = 2018
TRAIN_END_MONTH = f"{TRAIN_END_YEAR}-12"
TEST_START_YEAR = 2019
CV_SPLITS = 5
RANDOM_STATE = 42
FORECAST_HORIZONS = (1, 3)
MAX_LAG = 6
GRID_VARS = ("T2M", "PRECTOTCORR", "GWETROOT")
REGION_VARS = ("T2M", "PRECTOTCORR", "GWETROOT", "GWETTOP", "ALLSKY_SFC_SW_DWN", "RH2M")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def standardize_on_training_era(df: pd.DataFrame, variables, train_through: str = TRAIN_END_MONTH) -> pd.DataFrame:
    """Monthly z-scores fitted only to observations at or before the training cutoff."""
    out = df[["date", "year", "month"]].copy()
    train = df[df["date"] <= train_through]
    for v in variables:
        means = train.groupby("month")[v].mean()
        stds = train.groupby("month")[v].std(ddof=1).fillna(1.0) + 1e-8
        out[f"{v}_z"] = (df[v] - df["month"].map(means)) / df["month"].map(stds)
    return out


def add_lags(table: pd.DataFrame, column: str, code: str, lags) -> None:
    for lag in lags:
        table[f"{code}_lag{lag}"] = table[column].shift(lag)


def load_grid_observations() -> pd.DataFrame:
    """One row of raw values per grid cell and month."""
    grid = json.loads((Path(CACHE_DIR) / "ecuador_regional_grid.json").read_text(encoding="utf-8"))
    cells: Dict[tuple, Dict[str, pd.DataFrame]] = {}
    for variable in GRID_VARS:
        for feat in grid.get(variable, []):
            lon, lat, elev = (round(float(c), 4) for c in feat["geometry"]["coordinates"])
            df_cell, _ = parse_nasa_power_json(feat)
            cells.setdefault((lat, lon, elev), {})[variable] = df_cell[["date", "year", "month", variable]]

    frames = []
    for (lat, lon, elev), parts in cells.items():
        if set(parts) != set(GRID_VARS):
            continue
        merged = parts["T2M"]
        for variable in GRID_VARS[1:]:
            merged = merged.merge(parts[variable][["date", variable]], on="date")
        merged = merged.dropna().sort_values(["year", "month"]).reset_index(drop=True)
        if len(merged) < 120:
            continue
        frames.append(merged.assign(lat=lat, lon=lon, elev=elev))
    return pd.concat(frames, ignore_index=True)


def build_grid_cube(observations: pd.DataFrame, train_through: str = TRAIN_END_MONTH) -> pd.DataFrame:
    """Standardized values and chronological lags per grid cell."""
    frames = []
    for (lat, lon, elev), cell in observations.groupby(["lat", "lon", "elev"], sort=False):
        z = standardize_on_training_era(cell, GRID_VARS, train_through).rename(
            columns={"PRECTOTCORR_z": "PREC_z", "GWETROOT_z": "GWET_z"}
        )
        z = z.assign(lat=lat, lon=lon, elev=elev,
                     sin_m=np.sin(2 * np.pi * z["month"] / 12.0), cos_m=np.cos(2 * np.pi * z["month"] / 12.0))
        add_lags(z, "GWET_z", "GWET", range(1, MAX_LAG + 1))
        add_lags(z, "PREC_z", "PREC", range(1, MAX_LAG + 1))
        add_lags(z, "T2M_z", "T2M", range(1, MAX_LAG + 1))
        for h in range(0, MAX_LAG):
            z[f"PREC_roll3_lag{h}"] = z["PREC_z"].rolling(3).mean().shift(h)
        frames.append(z)
    return pd.concat(frames, ignore_index=True).sort_values(["year", "month", "lat", "lon"]).reset_index(drop=True)


def nowcast_features() -> List[str]:
    return ["lat", "lon", "elev", "sin_m", "cos_m", "T2M_z", "PREC_z", "T2M_lag1", "T2M_lag2",
            "PREC_lag1", "PREC_lag2", "PREC_lag3", "PREC_roll3_lag0", "GWET_lag1", "GWET_lag2"]


def forecast_features(h: int) -> List[str]:
    """Only values known h months before the target month."""
    return ["lat", "lon", "elev", "sin_m", "cos_m", f"GWET_lag{h}", f"GWET_lag{h + 1}", f"PREC_lag{h}",
            f"PREC_lag{h + 1}", f"PREC_lag{h + 2}", f"T2M_lag{h}", f"T2M_lag{h + 1}", f"PREC_roll3_lag{h}"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class PersistenceBaseline:
    """y_t = phi * y_(t-h), phi fitted by least squares on the training data."""

    def __init__(self, lag_column_index: int) -> None:
        self.lag_column_index = lag_column_index
        self.phi = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceBaseline":
        prev = X[:, self.lag_column_index]
        self.phi = float(np.dot(prev, y) / max(1e-12, np.dot(prev, prev)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.phi * X[:, self.lag_column_index]


def build_feature_folds(dates: pd.Series, build: Callable[[str], pd.DataFrame], gap: int = 0) -> list:
    """Split by target month so all grid cells from the same month share a partition."""
    months = np.sort(dates.unique())
    folds = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=CV_SPLITS, gap=gap).split(months):
        train_end = str(months[train_idx[-1]])
        table = build(train_end)
        folds.append((table[table["date"] <= train_end],
                      table[table["date"].between(months[val_idx[0]], months[val_idx[-1]])]))
    return folds


def fold_inputs(folds: list, features: List[str], target: str) -> list:
    pairs = []
    for training, validation in folds:
        train = training.dropna(subset=features + [target])
        val = validation.dropna(subset=features + [target])
        pairs.append((train[features].to_numpy(), train[target].to_numpy(),
                      val[features].to_numpy(), val[target].to_numpy()))
    return pairs


def evaluate(make_model: Callable[[], object], X_dev, y_dev, X_test, y_test, cv_data: list) -> Dict[str, object]:
    """5-fold TimeSeriesSplit on the development years, then a fit on all of them scored on the test years."""
    t0 = time.perf_counter()
    folds = []
    for X_train, y_train, X_val, y_val in cv_data:
        model = make_model().fit(X_train, y_train)
        folds.append(float(r2_score(y_val, model.predict(X_val))))
    model = make_model().fit(X_dev, y_dev)
    pred = model.predict(X_test)
    return {
        "model": model,
        "cv_r2_mean": round(float(np.mean(folds)), 4),
        "cv_r2_std": round(float(np.std(folds)), 4),
        "fold_r2_scores": [round(f, 4) for f in folds],
        "test_r2": round(float(r2_score(y_test, pred)), 4),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
        "test_mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "train_seconds": round(time.perf_counter() - t0, 2),
    }


def run_benchmark(table: pd.DataFrame, folds: list, features: List[str], persistence_feature: str,
                  candidates: Dict[str, Callable[[], object]], horizon: int = 0) -> Dict[str, object]:
    data = table.dropna(subset=features + ["GWET_z"])
    # A forecast made before the final training month must not be scored by that model.
    first_test_month = str(pd.Period(TRAIN_END_MONTH, freq="M") + max(1, horizon))
    dev, test = data[data["year"] <= TRAIN_END_YEAR], data[data["date"] >= first_test_month]
    X_dev, y_dev = dev[features].to_numpy(), dev["GWET_z"].to_numpy()
    X_test, y_test = test[features].to_numpy(), test["GWET_z"].to_numpy()

    models = {"PersistenceBaseline": lambda: PersistenceBaseline(features.index(persistence_feature)), **candidates}
    cv_data = fold_inputs(folds, features, "GWET_z")
    leaderboard, best = [], None
    for name, make_model in models.items():
        result = evaluate(make_model, X_dev, y_dev, X_test, y_test, cv_data)
        model = result.pop("model")
        leaderboard.append({"model_name": name, **result})
        print(f"   {name:28s} CV R2={result['cv_r2_mean']:.4f} (+/-{result['cv_r2_std']:.4f})  test R2={result['test_r2']:.4f}")
        if name != "PersistenceBaseline" and (best is None or result["cv_r2_mean"] > best[1]["cv_r2_mean"]):
            best = (model, result, name)

    baseline = leaderboard[0]
    for row in leaderboard:
        row["test_r2_gain_vs_persistence"] = round(row["test_r2"] - baseline["test_r2"], 4)

    perm = permutation_importance(best[0], X_test, y_test, n_repeats=8, random_state=RANDOM_STATE)
    importance = sorted(
        ({"feature": f, "importance_mean": round(float(m), 4), "importance_std": round(float(s), 4)}
         for f, m, s in zip(features, perm.importances_mean, perm.importances_std)),
        key=lambda r: r["importance_mean"], reverse=True,
    )
    return {
        "n_dev": int(len(dev)), "n_test": int(len(test)),
        "test_period": [int(test["year"].min()), int(test["year"].max())],
        "test_start_month": str(test["date"].min()),
        "best_model": best[2], "leaderboard": leaderboard, "permutation_importance": importance,
        "_best_estimator": best[0], "_features": features,
    }


def candidate_models() -> Dict[str, Callable[[], object]]:
    return {
        "RidgeRegression": lambda: Ridge(alpha=10.0),
        "RandomForestRegressor": lambda: RandomForestRegressor(
            n_estimators=120, max_depth=12, min_samples_leaf=12, max_samples=0.65, n_jobs=-1, random_state=RANDOM_STATE),
        "HistGradientBoostingRegressor": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=1.0,
            random_state=RANDOM_STATE),
    }


# ---------------------------------------------------------------------------
# Regional driver models
# ---------------------------------------------------------------------------
REGION_DRIVERS = ["PRECTOTCORR_z", "PREC_lag1", "PREC_lag2", "PREC_lag3", "PREC_roll3", "PREC_roll6", "RH2M_z",
                  "RH2M_lag1", "T2M_z", "T2M_lag1", "ALLSKY_SFC_SW_DWN_z", "ONI", "ONI_lag1", "sin_m", "cos_m"]


def build_region_table(df: pd.DataFrame, oni_map: Dict[str, float], train_through: str = TRAIN_END_MONTH) -> pd.DataFrame:
    vars_ok = [v for v in REGION_VARS if v in df.columns]
    t = standardize_on_training_era(df, vars_ok, train_through)
    t = t[(t["year"] >= ML_START_YEAR) & (t["year"] <= DATA_END_YEAR)].sort_values(["year", "month"]).reset_index(drop=True)
    add_lags(t, "PRECTOTCORR_z", "PREC", (1, 2, 3))
    add_lags(t, "RH2M_z", "RH2M", (1,))
    add_lags(t, "T2M_z", "T2M", (1,))
    add_lags(t, "GWETROOT_z", "GWET", (1,))
    t["PREC_roll3"] = t["PRECTOTCORR_z"].rolling(3).mean()
    t["PREC_roll6"] = t["PRECTOTCORR_z"].rolling(6).mean()
    t["sin_m"], t["cos_m"] = np.sin(2 * np.pi * t["month"] / 12.0), np.cos(2 * np.pi * t["month"] / 12.0)
    t["ONI"] = t["date"].map(oni_map)
    t["ONI_lag1"] = t["ONI"].shift(1)
    return t.dropna().reset_index(drop=True)


def regional_telemetry(region_id: str, df: pd.DataFrame, oni_map: Dict[str, float]) -> Dict[str, object]:
    t0 = time.perf_counter()
    table = build_region_table(df, oni_map)
    dev, test = table[table["year"] <= TRAIN_END_YEAR], table[table["year"] >= TEST_START_YEAR]
    folds = build_feature_folds(dev["date"], lambda through: build_region_table(df, oni_map, through))
    target = "GWETROOT_z"

    def _forest() -> RandomForestRegressor:
        return RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=4, random_state=RANDOM_STATE, n_jobs=-1)

    model = evaluate(_forest, dev[REGION_DRIVERS].to_numpy(), dev[target].to_numpy(),
                     test[REGION_DRIVERS].to_numpy(), test[target].to_numpy(), fold_inputs(folds, REGION_DRIVERS, target))
    persistence = evaluate(lambda: PersistenceBaseline(0), dev[["GWET_lag1"]].to_numpy(), dev[target].to_numpy(),
                           test[["GWET_lag1"]].to_numpy(), test[target].to_numpy(), fold_inputs(folds, ["GWET_lag1"], target))
    perm = permutation_importance(model["model"], test[REGION_DRIVERS].to_numpy(), test[target].to_numpy(),
                                  n_repeats=10, random_state=RANDOM_STATE)
    top = sorted(({"feature": f, "importance": round(float(m), 4)} for f, m in zip(REGION_DRIVERS, perm.importances_mean)),
                 key=lambda r: r["importance"], reverse=True)[:6]

    X_regimes = StandardScaler().fit_transform(table[[f"{v}_z" for v in REGION_VARS]].to_numpy())
    silhouettes = {
        str(k): round(float(silhouette_score(X_regimes, KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(X_regimes))), 4)
        for k in (2, 3, 4)
    }
    best_k = max(silhouettes, key=silhouettes.get)
    print(f"   {region_id:22s} driver CV R2={model['cv_r2_mean']:+.3f} test={model['test_r2']:+.3f} | "
          f"persistence CV={persistence['cv_r2_mean']:+.3f} test={persistence['test_r2']:+.3f} | k={best_k}")
    return {
        "region_id": region_id,
        "n_samples": int(len(table)),
        "driver_cv_r2_mean": model["cv_r2_mean"],
        "driver_test_r2": model["test_r2"],
        "driver_test_rmse": model["test_rmse"],
        "persistence_cv_r2": persistence["cv_r2_mean"],
        "persistence_test_r2": persistence["test_r2"],
        "top_drivers": top,
        "silhouette_by_k": silhouettes,
        "optimal_k": int(best_k),
        "best_silhouette": silhouettes[best_k],
        "train_seconds": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------------------
# MODIS coupling check
# ---------------------------------------------------------------------------
def modis_coupling(region_frames: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    modis_file = Path(CACHE_DIR) / "nasa_modis_250m_ndvi.json"
    if not modis_file.exists():
        return {}
    out = {}
    for region_id, records in json.loads(modis_file.read_text(encoding="utf-8")).items():
        if not records or region_id not in region_frames:
            continue
        mdf = pd.DataFrame(records).assign(date=lambda d: d["calendar_date"].str.slice(0, 7))
        monthly = mdf.groupby("date", as_index=False)["ndvi"].mean()
        joined = monthly.merge(region_frames[region_id][["date", "GWETROOT", "PRECTOTCORR"]], on="date").dropna()
        if len(joined) >= 12:
            out[region_id] = {
                "n_satellite_passes": len(records),
                "n_matched_months": len(joined),
                "corr_ndvi_gwetroot": round(float(joined["ndvi"].corr(joined["GWETROOT"])), 4),
                "corr_ndvi_precip": round(float(joined["ndvi"].corr(joined["PRECTOTCORR"])), 4),
                "ndvi_mean": round(float(mdf["ndvi"].mean()), 4),
            }
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _public(benchmark: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in benchmark.items() if not k.startswith("_")}


def run_full_training_pipeline(save_model: bool = False) -> None:
    t_start = time.perf_counter()
    oni = load_noaa_oni()
    oni_map = dict(zip(oni["date"], oni["oni"]))

    print("[1/4] Building the Ecuador grid cube...")
    observations = load_grid_observations()
    cube = build_grid_cube(observations)
    dev_months = cube.loc[cube["year"] <= TRAIN_END_YEAR, "date"]
    grid_folds = build_feature_folds(dev_months, lambda through: build_grid_cube(observations, through))
    print(f"      {len(cube):,} cell-months, {cube.groupby(['lat', 'lon']).ngroups} cells")

    print("[2/4] Nowcast benchmark (uses same-month weather)")
    nowcast = run_benchmark(cube, grid_folds, nowcast_features(), "GWET_lag1", candidate_models())

    forecasts = []
    for h in FORECAST_HORIZONS:
        print(f"[3/4] Forecast benchmark t+{h} (only data known {h} month(s) ahead)")
        folds = grid_folds if h == 1 else build_feature_folds(
            dev_months, lambda through: build_grid_cube(observations, through), gap=h - 1,
        )
        result = run_benchmark(cube, folds, forecast_features(h), f"GWET_lag{h}",
                               {k: v for k, v in candidate_models().items() if k != "RandomForestRegressor"}, horizon=h)
        forecasts.append({"horizon_months": h, **_public(result)})

    print("[4/4] Regional driver models")
    region_frames = {rid: load_region_nasa_power(rid)[0] for rid in REGIONS}
    telemetry = {rid: regional_telemetry(rid, df, oni_map) for rid, df in region_frames.items()}

    meteomatics = load_meteomatics_1km_cache()
    modis = json.loads((Path(CACHE_DIR) / "nasa_modis_250m_ndvi.json").read_text(encoding="utf-8"))
    manifest = {
        "pipeline_version": "TerraLens ML benchmark v2.3",
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_training_seconds": round(time.perf_counter() - t_start, 1),
        "library_versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "split": {"development": [int(cube["year"].min()), TRAIN_END_YEAR], "test": [TEST_START_YEAR, int(cube["year"].max())],
                  "cv": f"{CV_SPLITS} temporal folds grouped by month, with climatology fitted separately in each fold"},
        "dataset_summary": {
            "grid_cells": int(cube.groupby(["lat", "lon"]).ngroups),
            "grid_cell_months": int(len(cube)),
            "benchmark_regions": len(REGIONS),
            "coverage": [str(cube["date"].min()), str(cube["date"].max())],
            "modis_250m_passes": int(sum(len(v) for v in modis.values())),
            "meteomatics_1km_records": int(sum(len(v) for v in meteomatics.values())),
        },
        "nowcast_benchmark": _public(nowcast),
        "forecast_benchmarks": forecasts,
        "regional_telemetry": telemetry,
        "modis_250m_coupling": modis_coupling(region_frames),
    }
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if save_model:
        joblib.dump({"nowcast_model": nowcast["_best_estimator"], "features": nowcast["_features"]},
                    MODELS_DIR / "terralens_nowcast_model.joblib", compress=3)
    print(f"Done in {manifest['total_training_seconds']}s -> models/training_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-model", action="store_true", help="also write the best nowcast model with joblib")
    run_full_training_pipeline(save_model=parser.parse_args().save_model)
